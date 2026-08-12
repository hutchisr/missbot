# Missbot

<!-- This file is the project doc; CLAUDE.md is just `@AGENTS.md`. Edit AGENTS.md, not CLAUDE.md. -->

Pydantic AI chat agent with LLM fallback, an optional Redis-backed social credit system, and optional mem0 long-term memory backed by Postgres/pgvector. It serves **two frontends over one shared brain**:

- **Misskey/Fediverse** (`bot/bot.py`) — WebSocket streaming, mentions, timeline auto-replies, autonomous posts
- **ACP** (`bot/acp/`) — Agent Client Protocol over stdio, so ACP clients (Zed, JetBrains, [buzz-acp](https://github.com/block/buzz)) reach the same persona

Both are thin adapters translating their wire format into the neutral `AgentTurn` in `bot/core.py`; `ChatAgent` never sees a platform type. Persona, memories, and scores live in Postgres and Redis, so a separate ACP process pointed at the same backends is genuinely the same bot rather than a copy of it.

## Commands

```bash
# Install
uv sync

# Run (Misskey frontend)
uv run python -m bot -c config.local.yaml   # or: mise run bot

# Run (ACP frontend — stdio JSON-RPC; clients that spawn subprocesses use this)
uv run python -m bot.acp stdio -c config.local.yaml

# Run (ACP frontend — WebSocket, for remote clients via `acpremote mirror`)
uv run python -m bot.acp serve -c config.local.yaml --host 0.0.0.0 --port 8080 --token-env ACP_TOKEN
# Consumer side (e.g. buzz-acp):
#   BUZZ_ACP_AGENT_COMMAND="acpremote mirror ws://<host>:8080/acp/ws --bearer-token $ACP_TOKEN"

# mem0 maintenance (the k8s CronJob runs the destructive form daily)
uv run python -m bot.maintenance cleanup --dry-run -c config.local.yaml
uv run python -m bot.maintenance cleanup -c config.local.yaml
# Embedding-model migration (stop every memory reader/writer before the live form).
# The live form takes full timestamped backups and atomically replaces only the
# vector columns in both the memory and entity tables.
uv run python -m bot.maintenance reembed --dry-run -c config.local.yaml
uv run python -m bot.maintenance reembed -c config.local.yaml

# Test
# OPENROUTER_API_KEY must be set or test collection errors out: importing the agents
# constructs a pydantic-ai OpenRouter provider that fails fast without a key. The value is
# never used (tests mock the network), so any dummy string works.
OPENROUTER_API_KEY=sk-dummy uv run pytest -q

# Lint & format
uv run ruff check bot/
uv run ruff format --check bot/

# Type check (config in pyproject.toml; bot/ and tests/ are both clean, keep them that way)
uv run pyright

# Docker
docker build -t missbot . && docker run -v /path/to/config.yaml:/config.yaml missbot

# Kubernetes
mise run build      # Build and push Docker image
mise run deploy     # Apply K8s manifests and restart
# k8s/config.yaml and k8s/secrets.txt are ignored local inputs; Kustomize emits
# both as Secrets (the runtime config contains the Misskey token/DB credentials).
# Edit k8s/maintenance-settings.yaml for the cleanup schedule/timezone; set
# memory_max_memories_per_author in config.yaml for the per-author limit.

# Production cluster (kubectl context: mercury)
# Memory DB is CloudNativePG, NOT local. Connect via the primary pod with peer auth
# as the postgres OS user (the `grok` app user fails peer auth; never put the password on the
# command line — the safety classifier blocks it, and you don't need it):
kubectl exec -n cnpg pg-cluster-1 -- psql -U postgres -d grok -tAc "SELECT count(*) FROM missbot_memories"
# Bot pod and memory-maintenance CronJob live in the `misskey` namespace (missbot-*), not `default`.

# ACP endpoint (k8s/acp.yaml: Deployment + Service + Tailscale Ingress)
kubectl -n misskey logs deployment/missbot-acp --tail=50
curl -s https://missbot-acp.taile6e57.ts.net/healthz            # unauthenticated probe
curl -s https://missbot-acp.taile6e57.ts.net/acp | jq           # transport metadata
# The WebSocket requires ACP_TOKEN (in k8s/secrets.txt, gitignored). Consumers connect via:
#   acpremote mirror wss://missbot-acp.taile6e57.ts.net/acp/ws --bearer-token "$ACP_TOKEN"
# Private to the tailnet by design — the agent writes memories and moves social credit.
```

**Important:** Always use `uv run` or `.venv/bin/python` — never bare `python`.

## Architecture

| File | Purpose |
|------|---------|
| `bot/core.py` | Frontend-neutral turn types: `AgentTurn`, `HistoryTurn`, `TurnAuthor`, `AutoPost`, `Poll`. The contract every frontend adapter builds or consumes. No platform imports |
| `bot/bot.py` | **Misskey adapter** — WebSocket client, mention handling, context building, reply sending. Owns all Misskey-specific translation: `_note_to_turn()`, `_user_handle()`, `_image_urls_for()` (with SSRF guard), visibility→memory rules, and the note-length budget |
| `bot/acp/agent.py` | **ACP adapter** — `MissbotAgent(acp.Agent)`: `initialize` / `new_session` / `prompt` / `cancel` / `close_session` over stdio |
| `bot/acp/identity.py` | `parse_sender()` — derives a caller identity from an ACP harness's message header, trusting only the region before the first `Content:` line and only the pubkey (never the display name) |
| `bot/acp/session.py` | `AcpSession` + `SessionRegistry` — bounded per-session history and the in-flight task handle `session/cancel` interrupts |
| `bot/acp/ws.py` | WebSocket transport wire-compatible with `acpremote mirror`: frame↔stream bridge, bearer auth, metadata/health routes. No acpremote dependency |
| `bot/acp/__main__.py` | `python -m bot.acp {stdio,serve}` entry points. Routes **all** logging to stderr — stdout is the JSON-RPC channel |
| `bot/ai.py` | `ChatAgent` class — Pydantic AI agent with `FallbackModel`, vision support. Consumes `AgentTurn`; the reply length cap is per-run (`AgentDeps.char_budget`), not baked into the agent |
| `bot/models.py` | Pydantic models: `Config`, `Note`, `User`, `MiFile`, WS message types |
| `bot/tools.py` | `build_tools()` factory — datetime, web search, search_users/notes, social credit tools; `apply_social_credit()` helper |
| `bot/scoring.py` | Injection-resistant message classifier: `build_scoring_spec()` turns `Config.social_credit_categories` into the constrained output type + delta map + hardened instructions; `build_scoring_prompt()` fences untrusted input |
| `bot/memory.py` | Thin async adapter around mem0's `AsyncMemory`; builds the mem0 config, scopes memories to the bot `agent_id`, and exposes the runtime plus maintenance read/delete paths |
| `bot/maintenance.py` | Out-of-process mem0 cleanup CLI; selects expired, duplicate, stale, empty, and per-author overflow note memories, then deletes them through mem0 so entity links stay consistent. Driven by `k8s/maintenance.yaml` |
| `bot/net.py` | `is_safe_media_url()` — SSRF guard for attacker-supplied image URLs (blocks private/reserved IPs and internal hosts); `fetch_image()` — bounded, guarded download used by `vision_image_mode: fetch` |
| `bot/imagegen.py` | `ImageGenerator` + `GeneratedImage` — OpenAI-compatible `/images/generations` client for auto-post images. Validates by magic bytes (PNG/JPEG/GIF/WebP, SVG refused), caps the response body and decoded size, and uses a dedicated client so the Misskey token never reaches the provider |
| `bot/mcp.py` | `build_mcp_toolsets()` + `gate_names()` — streamable-HTTP MCP servers with allow/block and gate filtering |
| `bot/api.py` | HTTP client utilities |
| `bot/cli.py` | CLI entry point and argument parsing |

## Config Schema (`config.yaml`)

Required fields:
- `domain`, `url` (HTTPS), `ws_url` (WebSocket), `token`
- `bot_user_id`, `bot_username`
- `llm_models`: list of model entries — either pydantic-ai strings (e.g. `"openrouter:anthropic/claude-3.5-sonnet"`) or `ModelSpec` dicts. A dict can add metadata such as `vision` to a provider string, or select an explicit API family and optional custom endpoint with `model`, `api_type`, `base_url`, and optional `api_key` / `api_key_env`
- `api_type` on a `ModelSpec` (default unset): explicit wire API, one of `openai-chat`, `openai-responses`, or `anthropic`. With neither `api_type` nor `base_url`, `model` is passed to Pydantic AI as a `provider:model` string and Pydantic selects the provider/API. Setting `base_url` without `api_type` preserves the legacy `openai-chat` behavior. With `api_type` set, `model` is the actual model name sent to that API; `base_url` is optional, so omitting it uses the selected provider's standard endpoint and credentials
- `extra_body` on a `ModelSpec` (default `{}`): arbitrary provider-specific JSON fields merged into only that model's request body. It remains per-model inside fallback chains. For example, OpenRouter Auto Beta's named cost tier is:
  ```yaml
  - model: openrouter:openrouter/auto-beta
    extra_body:
      plugins:
        - id: auto-beta-router
          cost_tier: medium
  ```
- `system_prompt`, `max_retries`

Optional fields:
- `max_tokens` (default unset/`None`): the hard reply/auto-post length cap. When set it's wired into the reply and auto agents' `model_settings` by `ChatAgent._generation_settings` (alongside the sampling knobs below) and only sent then; when unset, models generate unboundedly and long replies get truncated at the Misskey note cap (`max_note_length`), which can make the bot resume/repeat itself on the next turn
- `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` (all default unset/`None`): sampling + anti-repetition knobs for the **reply and auto-post** models, applied via `ChatAgent._generation_settings`. Each is only sent to the model when set (so an unset one keeps the provider default and isn't sent to models that reject it). Positive `frequency_penalty`/`presence_penalty` curb the bot reusing its own phrasing turn-after-turn. Bounds: temperature 0–2, top_p 0–1, penalties −2–2. The social scoring classifier is unaffected (it keeps its own structured-output settings)
- Every model-provider request identifies the app with the HTTPS Radicle Explorer URL for `rad:zLseUdKik1qrsiTonrjSoPGYbC6g` as `HTTP-Referer`, plus versioned `User-Agent: Missbot/<version>` and `X-OpenRouter-Title: missbot-<version>` headers. This covers reply, ACP, autonomous-post, social-scoring, mem0 extraction and embedding, and image-generation calls
- `vision`: bool (default `true`) — pass images directly to the main LLM
- `vision_image_mode` (default `url`): `url` sends the media URL; `fetch` downloads the image and sends it inline as base64. **`fetch` is required by providers that refuse URLs** — Ollama Cloud answers `image URLs are not currently supported, please use base64 encoded data instead`. Fetching means this process retrieves attacker-supplied media, so `bot/net.py:fetch_image` re-checks the SSRF guard, requires an `image/*` content type, refuses redirects, streams with a byte cap, and uses a dedicated client (never `api_client`, which carries the Misskey token). A fetch failure drops that one image rather than the reply
- `vision_max_image_bytes` (default `8388608`): per-image cap in `fetch` mode; the body is abandoned mid-stream once exceeded
- **Model vision flags matter.** A bare string in `llm_models` defaults to `vision: true`. A model that cannot accept images must be declared `{model: ..., vision: false}`, or `ChatAgent` will route image prompts to it and burn a failed call. `_spec_supports_vision` builds a separate vision chain from the models that can (see `k8s/config.yaml`, where only `minimax-m3` accepts images)
- `vision_models`: legacy, unused when `vision=true`
- `system_prompt_auto` + `auto_post_interval`: autonomous posting (interval in seconds)
- `image_gen_enabled` (default `false`): give the autonomous-post agent a `generate_image` tool so it can illustrate its own post. The agent chooses the image first, then may add a caption or publish the image without text. Auto posts only — the tool is appended solely to `auto_tools`, a list passed only to the auto agent (`Agent[AutoDeps, str]`); the reply and ACP agents are built from a separate `tools` list that never receives it. The `RunContext[AutoDeps]` typing does not enforce this by itself (pyright accepts the tool on either agent, since `build_tools()` returns `list[Callable[..., object]]` and pydantic-ai's `tools=` parameter is gradually typed) — the confinement is structural, and `test_reply_agent_never_gets_image_tool` pins it. Requires `image_gen_model`
- `image_gen_model`: image model id sent to the endpoint (e.g. `google/gemini-2.5-flash-image`)
- `image_gen_base_url` (default `https://openrouter.ai/api/v1`): OpenAI-compatible base URL; the request goes to `<base_url>/images/generations`
- `image_gen_api_key` / `image_gen_api_key_env` (default env `OPENROUTER_API_KEY`): same resolution order as `bot/memory.py`'s embedding/extraction keys — explicit key first, then the env var, and if neither resolves the request is sent unauthenticated (a warning is logged, since that's valid for a keyless self-hosted endpoint rather than a misconfiguration)
- `image_gen_size` (default unset/`None`): optional `size` request param (e.g. `1024x1024`); sent only when set, so backends that reject the field are unaffected
- `image_gen_timeout_seconds` (default `120`): HTTP timeout for one generation call; image models are much slower than chat
- `image_gen_max_bytes` (default `8388608`): cap on the decoded image; checked against the base64 length before decoding (so an over-cap image is dropped without ever materializing it) and again after
- `image_gen_mark_sensitive` (default `false`): upload with `isSensitive` set so Misskey blurs the image behind a click
- **Uploaded drive files are never cleaned up.** Every generated image (~5/day at the default interval) becomes a permanent Misskey drive file, plus one orphan per `notes/create` failure that follows a successful upload; the `missbot-maintenance` CronJob only cleans up mem0 memories, not the drive. Misskey drive has a per-user capacity, so this accumulates indefinitely — years away at this rate, but silent: once capacity is hit, `drive/files/create` starts failing and the feature quietly reverts to text-only posts forever. No cleanup is implemented; this is a known, accepted gap
- `searxng_url`, `searxng_user`, `searxng_password`: web search via SearXNG
- `redis_url`, `redis_password`, `redis_db`: Redis for social credit system
- `social_credit_auto_score` (default `true`): score every author's message via an isolated, tool-less classifier whose category is mapped to a fixed delta (−10…+10) in code — users can't dictate their own score (privileged users are scored too; the flag only gates the manual adjust tool)
- `social_credit_score_cooldown` (default `10`): min seconds between automatic score changes per user (bounds farming)
- `social_credit_ignore_threshold` (default unset/`None`): when set (and Redis configured), `Bot.on_mention` drops any author whose score is below it — the note never reaches the LLM, no reply is sent, and the author isn't scored or ingested. Authors with no score yet (`None`) are never ignored; checked via `ChatAgent.get_author_score`
- `score_models`: model chain for the classifier (same forms as `llm_models`); defaults to `llm_models`. Use a cheaper/smaller model — classification is a simple labeling task
- `social_credit_categories`: list of sentiment buckets the classifier may assign, each `{name, delta, description}`. The model only picks a `name` (constrained output); code applies the matching `delta`, so configurability never lets the model choose the number. Defaults to the built-in toxic(−10)/rude(−5)/neutral(0)/good(+5)/exceptional(+10) set. Names must be unique (case-insensitive); `description` is shown to the classifier
- `social_credit_unrestricted_user_ids`: list of user ids; when the note's author is one of these, the bot may manually adjust any user's score by any amount via `adjust_social_credit` (which is refused for everyone else)
- `max_context`: parent notes to include (default 3)
- `ignore_direct_messages` (default `true`): skip direct/private messages (Misskey `specified` visibility); the bot is built for public-timeline threads. Set false to also reply to DMs
- `ignore_bots` (default `true`): skip mentions from accounts flagged as bots (Misskey user `isBot`); prevents bot-to-bot reply loops. Set false to also reply to other bots
- `max_reply_mentions`: cap on total mentions (incl. the author) echoed into a reply (default 5); prevents mention-amplification/harassment relaying
- `max_concurrent_handlers` (default `20`): hard cap on in-flight mention/auto-reply handlers; excess events are dropped before a coroutine is created to bound provider and memory load
- `http_timeout_seconds`: HTTP timeout (default 30.0)
- `mcp_servers`: list of streamable-HTTP MCP servers (see below)
- `memory_enabled` (default `false`): turn on mem0 long-term memory (see below). Requires `postgres_url` and `embedding_model`
- `acp_*`: ACP frontend settings (see below); ignored by `python -m bot`
- `channel`, `debug`

### ACP frontend
`python -m bot.acp` serves the same agent over the [Agent Client Protocol](https://agentclientprotocol.com) in two modes:

- **`stdio`** — JSON-RPC on stdin/stdout, for clients that spawn agents as subprocesses (Zed, JetBrains).
- **`serve`** — the same agent over WebSocket, for remote consumers. ACP's own HTTP transport is still a draft RFD and the SDK ships stdio only, so remote clients bridge via [acpremote](https://github.com/vcoderun/acpkit): `acpremote mirror ws://host:8080/acp/ws` turns the endpoint back into a local stdio ACP command, which is exactly what `BUZZ_ACP_AGENT_COMMAND` wants.

```
Buzz Relay ──WS──→ buzz-acp ──stdio──→ acpremote mirror ──WS──→ python -m bot.acp serve
```

`bot/acp/ws.py` implements acpremote's server contract directly rather than depending on the package — acpremote pins `websockets<16.0`, and adding it would downgrade the library the Misskey streaming client runs on. The contract: **one WebSocket text frame carries exactly one ACP JSON-RPC message** with no trailing newline (the SDK's `Connection` is newline-delimited, so the bridge strips it on send and re-adds it on receive), binary frames are an error, optional `Authorization: Bearer <token>`, plus `GET <mount>` metadata and `GET /healthz`. Routes default to `/acp`, `/acp/ws`, `/healthz` so `acpremote mirror` needs no extra flags. A configured token is stripped once and must remain nonempty; whitespace-only values fail startup instead of disabling authentication. Outbound frames and their queue are bounded, and clients exceeding the backlog are closed.

`serve` builds **one** `ChatAgent`, `SessionRegistry`, and prompt semaphore shared by every connection, plus one `MissbotAgent` adapter *per connection* because the adapter holds the client handle used for `session/update`. The shared `acp_max_sessions` budget caps connections, sessions, and in-flight provider work across all adapters; one session cannot overlap prompts.

Config fields:
- `acp_default_identity` (default `acp`): identity used when no sender header parses. Namespaced `acp:<value>` so it can never collide with a fediverse handle
- `acp_parse_sender_header` (default `true`): derive per-caller identity from the harness's `From:` header. Set false to key every ACP caller on `acp_default_identity`
- `acp_max_history_turns` (default `20`): conversation turns retained per session; ACP sessions are long-lived, so history is bounded
- `acp_max_sessions` (default `8`): shared cap for WebSocket connections, sessions, and concurrent prompts, the ACP analogue of `max_concurrent_handlers`
- `acp_max_prompt_chars` (default `65536`): maximum aggregate text accepted in one prompt before scoring, model, or memory work

**Identity and its limits.** ACP carries no per-sender field, so attribution comes out of the prompt text. `parse_sender()` reads **only the region before the first `Content:` line** — user text lands in `Content:` and after, so a message body structurally cannot reach the parsed region — and keys on the **pubkey**, never the user-settable display name. Failure falls back to `acp_default_identity`, never to an unattributed write. Two limits are real and not papered over: a *batched* prompt concatenates several event blocks, so only the first block's header is structurally safe; and the header format is buzz-acp's internal detail that may change. `acp_parse_sender_header: false` is the kill switch.

**Differences from the Misskey path.** Replies have no character budget (`char_budget=None` — the note limit is Misskey's, not a universal one), while inbound prompts have the independent `acp_max_prompt_chars` safety limit. ACP exposes no `fs/*` or `terminal/*` client capabilities (missbot is conversational, not a coding agent), no `session/load` (history is in-process and does not survive restart), and no auth methods (over stdio the trust boundary is *who spawned the process*). Client-supplied `cwd` and `mcpServers` on `session/new` are ignored — missbot brings its own toolset from config. Social credit scoring, the ignore threshold, memory ingestion, and `NO_REPLY` all behave exactly as on the Misskey path.

**Unsupported methods are declined explicitly.** `acp.Agent` is a `Protocol`, so any method `MissbotAgent` doesn't override is still *inherited* as a stub with an `...` body — and the SDK's router resolves handlers with `getattr`, so those stubs get routed and return `None`, which the connection reports to the client as a **success**. `bot/acp/agent.py` therefore implements `load_session`, `list_sessions`, `set_session_mode`, `set_config_option`, `fork_session`, `resume_session`, and `ext_method` as explicit `method_not_found` (-32601) refusals; `ext_notification` logs and drops, since a notification has no response channel to refuse through. Adding a method to the ACP surface means *replacing* one of these, not adding alongside it.

**stdout is the protocol channel.** `bot/acp/__main__.py` points logfire's console exporter and `logging` at stderr. Anything printed to stdout corrupts the JSON-RPC stream and breaks the connection.

### Long-term memory
Memory is delegated to mem0 OSS through `bot/memory.py:MemoryStore`, backed by Postgres/pgvector. The bot does not maintain its own entity graph, claim schema, agreement counts, or consolidation job. `bot.maintenance reembed` performs controlled embedding-model migrations in place: it validates the new endpoint/dimension, precomputes every memory and entity vector, creates full timestamped backup tables, then replaces both vector columns atomically. Stop all memory readers/writers before running its live form. Off unless `memory_enabled: true`. Config fields:
- `postgres_url` (required when enabled): Postgres DSN for mem0's pgvector backend; the server must have the `vector` extension available
- `memory_collection_name` (default `missbot_memories`): Postgres table/collection mem0 uses for memories
- `memory_history_db_path` (default unset): optional SQLite path for mem0's local message/history database; leave unset for mem0's default
- `embedding_model` (required when enabled): embedding model id, e.g. `perplexity/pplx-embed-v1-0.6b`
- `embedding_dim` (default `1024`): vector dimension for the mem0 pgvector collection; must match the embedding model output
- `embedding_dimensions` (default unset): when set, sent as the OpenAI `dimensions` request param to truncate a Matryoshka model; must equal `embedding_dim`
- `embedding_base_url` (default `https://openrouter.ai/api/v1`) + `embedding_api_key` / `embedding_api_key_env` (default env `OPENROUTER_API_KEY`): OpenAI-compatible embeddings endpoint
- `memory_llm_model` (default first `llm_models` entry with provider prefix stripped): model mem0 uses for memory extraction
- `memory_llm_base_url`, `memory_llm_api_key`, `memory_llm_api_key_env` (default env `OPENROUTER_API_KEY`): optional OpenAI-compatible extraction-LLM settings
- `memory_search_limit` (default `5`), `memory_search_threshold` (default `0.1`): search result count and mem0 score floor
- `memory_custom_instructions`: optional custom instructions appended to mem0's extraction prompt
- `memory_ingest_notes` (default `true`): auto-ingest each incoming user note through mem0
- `memory_trusted_user_ids` (default `[]`): stable platform user ids whose inferred memories are exempt from the HEARSAY label. Misskey uses its user id; ACP uses its namespaced `acp:<pubkey>` identity. Trust is checked from stored `author_user_id` metadata at recall time, so handles/display names never confer trust and removing an id from config revokes the exemption
- `memory_note_retention_days` (default `90`, nullable): expiration/physical-retention window for inferred note memories; explicit `add_memory` entries are exempt
- `memory_max_memories_per_author` (default `50`, nullable): per-author cap for inferred note memories; maintenance removes the oldest overflow rows
- `memory_cleanup_scan_limit` (default `10000`): maximum scoped rows examined in one cleanup run
- `max_fact_length` (default `500`): longer `add_memory` submissions are rejected

**Write path.** `ChatAgent.run` still runs reply generation, social scoring, and memory ingestion concurrently. `_maybe_ingest_note` sends only the latest public author note to mem0 with metadata `{source: "misskey_note", author, author_user_id, source_note_id}` and a configured expiration date; `author_user_id` is omitted only when a frontend cannot provide a stable identity. Specified/private notes are never ingested. The `add_memory` tool likewise refuses writes during private interactions. mem0 owns extraction, deduplication, vector storage, and entity-style linking. Successful tool writes use metadata `{source: "add_memory", author: bot_username}` and do not expire automatically. Memory failures are logged and swallowed so they never cancel a reply.

**Read path.** `search_memory` calls `MemoryStore.search()` with `agent_id=bot_username`, renders mem0 memories with score/source/author/recency metadata when available, and fences the returned text as untrusted data before the model sees it. Memories inferred from user-authored `misskey_note` or `acp_prompt` inputs are additionally marked **HEARSAY** with an explicit warning that they are unverified, not guaranteed true, and must not be presented as established fact without corroboration. The label is omitted when stored `author_user_id` metadata matches `memory_trusted_user_ids`; those memories remain fenced as untrusted data so epistemic trust never becomes permission to follow recalled instructions. Explicit `add_memory` entries are not mislabeled as hearsay. Older inferred memories without `author_user_id` fail closed and remain hearsay.

**Maintenance path.** `python -m bot.maintenance cleanup` lists only the bot's `agent_id`, includes already-expired rows, and physically deletes empty memories, exact duplicates, inferred note memories past retention, and the oldest inferred memories above each author's cap. Explicit `add_memory` rows are protected from retention/cap cleanup. Deletion goes through mem0 after initializing its entity store so `missbot_memories_entities` links are cleaned too. `--dry-run` reports the same candidate summary without deleting. The `missbot-maintenance` CronJob runs with concurrency forbidden; Kustomize reads its schedule and timezone from `k8s/maintenance-settings.yaml`, while `memory_max_memories_per_author` in `config.yaml` controls the cap.

### MCP servers
Each entry in `mcp_servers` takes:
- `name` (required): human-readable id
- `url` (required): streamable-HTTP endpoint
- `headers`: dict of extra HTTP headers (e.g. `Authorization: Bearer ...`)
- `tool_prefix`: prepended to every tool name (e.g. `tavily` → `tavily_search`)
- `allowed_tools`: list; if set, only these are exposed (match **unprefixed** MCP names)
- `blocked_tools`: list of unprefixed names to hide
- `timeout`: connect timeout seconds (default 30)
- `enabled`: toggle off without deleting (default true)
- `gate`: if set, server's tools are hidden until the model calls `enable_<gate>()`. Multiple servers can share a gate.

Gating is progressive disclosure driven by the model itself: each unique `gate` value generates one `enable_<gate>` meta-tool. When the model calls it, that gate's tools become visible on the next model turn within the same run. Auto-agent (autonomous posts) does not get MCP toolsets.

## Key Patterns

### Agent setup (`bot/ai.py`)
- `AgentDeps` is a **dataclass** (not BaseModel) with `username`, `source_note_id`, `social_credit_score`, `adjusted_credit_users`, `social_credit_unrestricted`, `enabled_gates`, and `memory_writes_allowed`
- `adjust_social_credit` is privileged-only: it works only when `deps.social_credit_unrestricted` is set (`ChatAgent.run` sets it when the note's author id is in `social_credit_unrestricted_user_ids`); for everyone else it refuses
- Every author's score moves via `ChatAgent._maybe_score_message` (privileged users included): a separate tool-less classifier (`bot/scoring.py`, model from `score_models` or the reply model) runs concurrently with the reply, returns one of the configured `social_credit_categories` (default toxic/rude/neutral/good/exceptional) that's mapped to its fixed delta in code, applied through `apply_social_credit` and rate-limited by a Redis `score_cooldown:<user>` key. `ChatAgent.__init__` builds a `ScoringSpec` (constrained output type + delta map + instructions) from the configured categories via `build_scoring_spec`. This is the prompt-injection mitigation — the model only picks a category name, never the number
- Agent uses `output_type=str` (plain string output, not structured)
- Tools are built via `build_tools()` in `bot/tools.py` and passed to `Agent(..., tools=tools)`
- When `memory_enabled`, the bot receives a `MemoryStore` adapter around mem0 `AsyncMemory`; the adapter is passed to `build_tools()` so `add_memory` and `search_memory` are exposed
- `ChatAgent.run` runs three coroutines concurrently via `asyncio.gather`: the reply, `_maybe_score_message`, and `_maybe_ingest_note` (mem0 ingestion when `memory_ingest_notes`). Like scoring, note ingestion swallows its own errors, so it never affects the reply
- `FallbackModel` wraps multiple `llm_models` for automatic failover
- Social credit score is injected via a dynamic system prompt function

### Adding tools
Add tools inside `build_tools()` in `bot/tools.py` as plain functions or async functions, then append to the `tools` list:
```python
def my_tool(param: str) -> str:
    """Tool description for LLM."""
    return result

tools.append(my_tool)
```
For tools needing `RunContext`, use the signature `async def my_tool(ctx: RunContext[object], ...) -> str:`.

### Message flow
1. WebSocket mention received → `Bot` ignores own mentions and (by default) direct messages (`specified` visibility)
2. Reply chain traversed (up to `max_context`) to build `message_history`
3. Images passed inline via `ImageUrl` when `vision=true` — each URL is SSRF-checked with `bot/net.py:is_safe_media_url` first (attacker-controlled on federated notes); unsafe ones are dropped
4. `ChatAgent.run()` calls Pydantic AI agent with fallback model
5. Reply sent via Misskey API with proper mention formatting

### Autonomous posting
When `system_prompt_auto` and `auto_post_interval` are configured, `ChatAgent.run_auto()` generates unprompted timeline posts on a timer. `run_auto()` returns `AutoPost(text, image, poll)` rather than a bare string. The auto agent always receives `create_poll`: it accepts 2–10 distinct choices (50 characters each), optional multi-select, and an optional duration in minutes, then stores a neutral `Poll` on `AutoDeps`. The final model output supplies the question text; `Bot.post_autonomous` translates the duration to Misskey's millisecond `expiredAfter` field. Polls may coexist with generated images. When `image_gen_enabled`, `generate_image` is also registered on the auto agent only — both attachment tools are appended to `auto_tools`, a list distinct from the reply agent's `tools` and never passed to it. The `RunContext[AutoDeps]` typing alone does not enforce that split (pyright accepts either tool on the reply agent, since `build_tools()`'s return type erases the deps type and pydantic-ai's `tools=` parameter is gradually typed); the separation is structural, and runtime tests pin it. A successful image call leaves its `GeneratedImage` on `AutoDeps.image` for `run_auto()` to read back after the run completes. The tool is called before the final composition; the model then emits a caption or an internal image-only marker that `run_auto()` normalizes to empty text. As a recovery for smaller models that ignore the tool, a leading bracketed visual description containing an explicit medium (for example `*[meme: ...]*`) is sent to the image generator once and removed from the published caption when generation succeeds; ordinary stage directions remain text, and a failed generation retains the original text fallback. `Bot.post_autonomous` uploads that image to `drive/files/create` before calling `notes/create` only when `post.image` is set (never when `image_gen_enabled` is off, or when the tool declined or failed). Upload failure falls back to text when a caption exists; an image-only post is skipped because it has no valid fallback content.

## Available Tools (runtime)
- `current_datetime_tool` — always available
- `search_web` (async) — when `searxng_url` configured. Returns domain-prefixed snippets
- `search_users`, `search_notes` — Misskey search APIs
- Social credit tools (when Redis configured): `get_social_credit`, `adjust_social_credit` (privileged authors only), `get_social_credit_history`, `get_social_credit_leaderboard`. All users (privileged included) are also scored automatically by the `bot/scoring.py` classifier, separate from any tool call
- Long-term memory tools (when `memory_enabled`): `add_memory` (passes text to mem0 `add`, except in private interactions) and `search_memory` (mem0 `search` results, fenced as untrusted data). Public user notes are also ingested automatically when `memory_ingest_notes=true`; private/specified notes are not. Not given to the auto-agent
- `enable_<gate>` — one per unique `gate` value in `mcp_servers`; model calls it to unlock gated MCP tools
- MCP tools — from each configured `mcp_servers` entry, name-prefixed per `tool_prefix`
- `generate_image` — **auto posts only** (when `image_gen_enabled`). The model writes the image prompt and its alt text; one image per post, over-length prompt/alt refused, failure degrades to a text-only post
- `create_poll` — **auto posts only**. Attaches one Misskey poll with 2–10 distinct choices, optional multi-select, and optional expiration; the model's final text becomes the poll question

## Verification

After making changes, always:
1. Check for IDE/compiler errors on modified files
2. Run `uv run ruff check bot/` and `uv run ruff format --check bot/`
3. Run `uv run pyright` — the tree is at zero errors, so any output is yours
4. Run the tests: `OPENROUTER_API_KEY=sk-dummy uv run pytest -q` (the dummy key is required for collection — see Commands)
5. Fix any issues before considering the task complete
