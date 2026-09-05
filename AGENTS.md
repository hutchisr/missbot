# Missbot

<!-- This file is the project doc; CLAUDE.md is just `@AGENTS.md`. Edit AGENTS.md, not CLAUDE.md. -->

Pydantic AI chat agent with LLM fallback, an optional Redis-backed social credit system, and optional Hindsight long-term memory. It serves **two frontends over one shared brain**:

- **Misskey/Fediverse** (`bot/bot.py`) — WebSocket streaming, mentions, timeline auto-replies, autonomous posts
- **ACP** (`bot/acp/`) — Agent Client Protocol over stdio, so ACP clients (Zed, JetBrains, [buzz-acp](https://github.com/block/buzz)) reach the same persona

Both are thin adapters translating their wire format into the neutral `AgentTurn` in `bot/core.py`; `ChatAgent` never sees a platform type. Persona and memories live in Hindsight while scores live in Redis, so separate Misskey and ACP processes pointed at the same backends are genuinely the same bot rather than copies.

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


# Test
# OPENROUTER_API_KEY must be set or test collection errors out: importing the agents
# constructs a pydantic-ai OpenRouter provider that fails fast without a key. The value is
# never used (tests mock the network), so any dummy string works.
OPENROUTER_API_KEY=sk-dummy uv run pytest -q

# Lint & format
uv run ruff check bot/
uv run ruff format --check bot/

# Type check (config in pyproject.toml; bot/ and tests/ are both clean, keep them that way)
uv run basedpyright

# Docker
docker build -t missbot . && docker run -v /path/to/config.yaml:/config.yaml missbot

# Kubernetes
mise run build      # Build and push Docker image
mise run deploy     # Apply K8s manifests and restart
# k8s/config.yaml and k8s/secrets.txt are ignored local inputs; Kustomize emits
# both as Secrets (the runtime config contains the Misskey token and provider credentials).
# Hindsight runs in the `hindsight` namespace. Missbot reaches it through
# http://hindsight-api.hindsight.svc.cluster.local:8888.
kubectl -n hindsight logs deployment/hindsight-api --tail=50
curl -s https://hindsight-api.taile6e57.ts.net/health

# ACP remains available through `python -m bot.acp`, but is intentionally not
# included in the Kubernetes deployment.
```

**Important:** Always use `uv run` or `.venv/bin/python` — never bare `python`.

## GitHub Mirror

Radicle `master` is canonical. `.github/workflows/mirror-radicle-master.yml`
polls it every 15 minutes (and supports manual dispatch), then fast-forwards
GitHub `master`. It refuses divergence and never force-pushes.

The workflow uses an ephemeral GitHub-hosted Ubuntu runner. It downloads the
pinned Radicle release, verifies its SHA-256 checksum, starts a temporary node,
and clones the public repository from the Radicle network. By default it creates
a disposable non-delegate identity. For a stable node identity, set the optional
repository secret `RADICLE_IDENTITY_KEY_B64` to the single-line base64 encoding
of a dedicated mirror identity's OpenSSH private key; if that key is encrypted,
also set `RADICLE_IDENTITY_PASSPHRASE`. The workflow derives the public key.
Never use a maintainer/delegate identity: this read-only fetch needs no Radicle
authority. The automatic `GITHUB_TOKEN` is used only to push the fast-forward
to GitHub.

GitHub must allow Actions to request read/write repository permissions; the
workflow narrows its token to `contents: write`. If `master` has a branch
protection rule, it must permit this workflow's push. The workflow must exist
on GitHub's default branch before schedules or manual dispatch work, so
bootstrap it once by pushing the merged `master` to `origin`.


## Architecture

| File | Purpose |
|------|---------|
| `bot/core.py` | Frontend-neutral turn types: `AgentTurn`, `HistoryTurn`, `TurnAuthor`, `AutoPost`, `Poll`. The contract every frontend adapter builds or consumes. No platform imports |
| `bot/bot.py` | **Misskey adapter** — WebSocket client, mention handling, context building, reply sending. Owns all Misskey-specific translation: `_note_to_turn()`, `_user_handle()`, `_image_urls_for()` (with SSRF guard), visibility→memory-access rules, event/thread provenance, and the note-length budget |
| `bot/acp/agent.py` | **ACP adapter** — `MissbotAgent(acp.Agent)`: `initialize` / `new_session` / `prompt` / `cancel` / `close_session` over stdio |
| `bot/acp/identity.py` | Structurally parses optional sender pubkey plus event id/time before the first `Content:` line. It does not authenticate prompt text; attribution is disabled by default and requires an operator-trusted harness |
| `bot/acp/session.py` | `AcpSession` + `SessionRegistry` — bounded per-session history and the in-flight task handle `session/cancel` interrupts |
| `bot/acp/ws.py` | WebSocket transport wire-compatible with `acpremote mirror`: frame↔stream bridge, bearer auth, metadata/health routes. No acpremote dependency |
| `bot/acp/__main__.py` | `python -m bot.acp {stdio,serve}` entry points. Routes **all** logging to stderr — stdout is the JSON-RPC channel |
| `bot/ai.py` | `ChatAgent` class — Pydantic AI agent with `FallbackModel`, vision support. Consumes `AgentTurn`; the reply length cap is per-run (`AgentDeps.char_budget`), not baked into the agent |
| `bot/models.py` | Pydantic models: `Config`, `Note`, `User`, `MiFile`, WS message types |
| `bot/tools.py` | `build_tools()` factory — datetime, sandboxed pydantic-monty Python, web search, search_users/notes, social credit tools; `apply_social_credit()` helper |
| `bot/scoring.py` | Injection-resistant message classifier: `build_scoring_spec()` turns `Config.social_credit_categories` into the constrained output type + delta map + hardened instructions; `build_scoring_prompt()` fences untrusted input |
| `bot/memory.py` | Hindsight-native lifecycle: shared observation-enabled bank, automatic fenced recall before allowed turns, and append-only structured exchange retention afterward |
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
- `max_tokens` (default unset/`None`): the hard reply generation cap and the autonomous default when `auto_max_tokens` is unset. It is sent through `model_settings` only when set; leaving it unset preserves provider-default generation length
- `auto_models` (default unset/`None`): optional autonomous-post-only model fallback chain using the same string/`ModelSpec` forms as `llm_models`. When unset, autonomous posts reuse `llm_models`. Use it to keep unstable or experimental conversational providers out of unattended publishing without changing replies or ACP
- `auto_max_tokens` (default unset/`None`): autonomous-post-only generation cap. When unset, autonomous posts inherit `max_tokens`, including its provider-default behavior
- `auto_max_chars` (default unset/`None`): autonomous-post-only visible character cap. When unset, it inherits `max_note_length`. Over-limit drafts are retried and cannot be added to autonomous history or reach `notes/create`
- `auto_timeout_seconds` (default `300`): per-request timeout for autonomous model generation. This is independent of the frontend HTTP client timeout
- `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` (all default unset/`None`): sampling + anti-repetition knobs for the **reply and auto-post** models, applied via `ChatAgent._generation_settings`. Each is only sent to the model when set (so an unset one keeps the provider default and isn't sent to models that reject it). Positive `frequency_penalty`/`presence_penalty` curb the bot reusing its own phrasing turn-after-turn. Bounds: temperature 0–2, top_p 0–1, penalties −2–2. The social scoring classifier is unaffected (it keeps its own structured-output settings)
- `auto_temperature` (default unset/`None`): autonomous-post-only sampling temperature. When set, it overrides `temperature` for `ChatAgent.run_auto()` without changing replies or ACP; when unset, autonomous posts inherit the shared `temperature` setting (including its provider-default behavior). Bounds: 0–2. A higher value can reduce repetitive phrasing at the cost of more variable output
- Every model-provider request identifies the app with the HTTPS Radicle Explorer URL for `rad:zLseUdKik1qrsiTonrjSoPGYbC6g` as `HTTP-Referer`, plus versioned `User-Agent: Missbot/<version>` and `X-OpenRouter-Title: missbot-<version>` headers. This covers reply, ACP, autonomous-post, social-scoring, and image-generation calls. Hindsight API requests use `User-Agent: Missbot/<version>`; Hindsight owns its internal model-provider requests
- `vision`: bool (default `true`) — pass images directly to the main LLM
- `vision_image_mode` (default `url`): `url` sends the media URL; `fetch` downloads the image and sends it inline as base64. **`fetch` is required by providers that refuse URLs** — Ollama Cloud answers `image URLs are not currently supported, please use base64 encoded data instead`. Fetching means this process retrieves attacker-supplied media, so `bot/net.py:fetch_image` re-checks the SSRF guard, requires an `image/*` content type, refuses redirects, streams with a byte cap, and uses a dedicated client (never `api_client`, which carries the Misskey token). A fetch failure drops that one image rather than the reply
- `vision_max_image_bytes` (default `8388608`): per-image cap in `fetch` mode; the body is abandoned mid-stream once exceeded
- **Model vision flags matter.** A bare string in `llm_models` defaults to `vision: true`. A model that cannot accept images must be declared `{model: ..., vision: false}`, or `ChatAgent` will route image prompts to it and burn a failed call. `_spec_supports_vision` builds a separate vision chain from the models that can (see `k8s/config.yaml`, where only `minimax-m3` accepts images)
- `vision_models`: legacy, unused when `vision=true`
- `system_prompt_auto` + `auto_post_interval`: autonomous posting (interval in seconds)
- `image_gen_enabled` (default `false`): give the autonomous-post agent a `generate_image` tool so it can illustrate its own post. The agent chooses the image first, then may add a caption or publish the image without text. Auto posts only — the tool is appended solely to `auto_tools`, a list passed only to the auto agent (`Agent[AutoDeps, str]`); the reply and ACP agents are built from a separate `tools` list that never receives it. The `RunContext[AutoDeps]` typing does not enforce this by itself (Basedpyright accepts the tool on either agent, since `build_tools()` returns `list[Callable[..., object]]` and pydantic-ai's `tools=` parameter is gradually typed) — the confinement is structural, and `test_reply_agent_never_gets_image_tool` pins it. Requires `image_gen_model`
- `image_gen_model`: image model id sent to the endpoint (e.g. `google/gemini-2.5-flash-image`)
- `image_gen_base_url` (default `https://openrouter.ai/api/v1`): OpenAI-compatible base URL; the request goes to `<base_url>/images/generations`
- `image_gen_api_key` / `image_gen_api_key_env` (default env `OPENROUTER_API_KEY`): explicit key first, then the configured environment variable; if neither resolves the request is sent unauthenticated (a warning is logged, since that's valid for a keyless self-hosted endpoint rather than a misconfiguration)
- `image_gen_size` (default unset/`None`): optional `size` request param (e.g. `1024x1024`); sent only when set, so backends that reject the field are unaffected
- `image_gen_timeout_seconds` (default `120`): HTTP timeout for one generation call; image models are much slower than chat
- `image_gen_max_bytes` (default `8388608`): cap on the decoded image; checked against the base64 length before decoding (so an over-cap image is dropped without ever materializing it) and again after
- `image_gen_mark_sensitive` (default `false`): upload with `isSensitive` set so Misskey blurs the image behind a click
- **Uploaded drive files are never cleaned up.** Every generated image (~5/day at the default interval) becomes a permanent Misskey drive file, plus one orphan per `notes/create` failure that follows a successful upload. Misskey drive has a per-user capacity, so this accumulates indefinitely — years away at this rate, but silent: once capacity is hit, `drive/files/create` starts failing and the feature quietly reverts to text-only posts forever. No cleanup is implemented; this is a known, accepted gap
- `searxng_url`, `searxng_user`, `searxng_password`: web search via SearXNG
- `redis_url`, `redis_password`, `redis_db`: Redis for social credit system
- `social_credit_auto_score` (default `true`): score every author's message via an isolated, tool-less classifier whose category is mapped to a fixed delta (−10…+10) in code — users can't dictate their own score (privileged users are scored too; the flag only gates the manual adjust tool)
- `social_credit_score_cooldown` (default `10`): min seconds between automatic score changes per user (bounds farming)
- `social_credit_ignore_threshold` (default unset/`None`): when set (and Redis configured), `Bot.on_mention` drops any author whose score is below it — the note never reaches the LLM, no reply is sent, and the author isn't scored or given memory access. Authors with no score yet (`None`) are never ignored; checked via `ChatAgent.get_author_score`
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
- `memory_enabled` (default `false`): turn on Hindsight long-term memory (see below)
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
- `acp_default_identity` (default `acp`): identity used unless trusted textual sender-header parsing is explicitly enabled and succeeds. Namespaced `acp:<value>` so it can never collide with a fediverse handle
- `acp_parse_sender_header` (default `false`): opt into per-caller attribution from a trusted harness's textual `From:` header. ACP does not authenticate this text, so an arbitrary client can fabricate the full header; enable only when the process/transport admits an operator-trusted harness
- `acp_max_history_turns` (default `20`): conversation turns retained per session; ACP sessions are long-lived, so history is bounded
- `acp_max_sessions` (default `8`): shared cap for WebSocket connections, sessions, and concurrent prompts, the ACP analogue of `max_concurrent_handlers`
- `acp_max_prompt_chars` (default `65536`): maximum aggregate text accepted in one prompt before scoring, model, or memory work

**Identity and event-provenance limits.** ACP carries no authenticated structured sender or event field. Text-header attribution is therefore disabled by default: any ACP client can fabricate a complete `From:` / `Event ID:` block before `Content:` and impersonate another pubkey. When an operator explicitly trusts the only admitted client/harness and enables `acp_parse_sender_header`, the parsers use only the first structurally delimited block, accept only a pubkey identity and 64-hex event id, and never use the display name as a key. The `Content:` boundary prevents body-level header injection but is not signature verification. Disabled, missing, or malformed attribution falls back to `acp_default_identity` plus a random per-prompt source id. Batched prompts still attribute only the first block, and the format is a buzz-acp implementation detail.

**Differences from the Misskey path.** Replies have no character budget (`char_budget=None` — the note limit is Misskey's, not a universal one), while inbound prompts have the independent `acp_max_prompt_chars` safety limit. ACP exposes no `fs/*` or `terminal/*` client capabilities (missbot is conversational, not a coding agent), no `session/load` (history is in-process and does not survive restart), and no auth methods (over stdio the trust boundary is *who spawned the process*). Client-supplied `cwd` and `mcpServers` on `session/new` are ignored — missbot brings its own toolset from config. Social credit scoring, the ignore threshold, automatic memory recall/retention, and `NO_REPLY` all behave exactly as on the Misskey path.

**Unsupported methods are declined explicitly.** `acp.Agent` is a `Protocol`, so any method `MissbotAgent` doesn't override is still *inherited* as a stub with an `...` body — and the SDK's router resolves handlers with `getattr`, so those stubs get routed and return `None`, which the connection reports to the client as a **success**. `bot/acp/agent.py` therefore implements `load_session`, `list_sessions`, `set_session_mode`, `set_config_option`, `fork_session`, `resume_session`, and `ext_method` as explicit `method_not_found` (-32601) refusals; `ext_notification` logs and drops, since a notification has no response channel to refuse through. Adding a method to the ACP surface means *replacing* one of these, not adding alongside it.

**stdout is the protocol channel.** `bot/acp/__main__.py` points logfire's console exporter and `logging` at stderr. Anything printed to stdout corrupts the JSON-RPC stream and breaks the connection.

### Long-term memory
Memory is delegated to Hindsight through the official async `hindsight-client` API. `bot/memory.py:MemoryStore` owns an automatic conversational lifecycle over one shared bank: recall before an allowed model turn, inject fenced background context, then append the completed exchange after generation. Hindsight owns extraction, deduplication, vector/keyword/graph/temporal retrieval, and observation consolidation. Missbot does not access Hindsight's Postgres store directly and has no memory maintenance job. Off unless `memory_enabled: true`.

Config fields:
- `hindsight_base_url` (default `http://localhost:8888`): Hindsight API base URL
- `hindsight_api_key` / `hindsight_api_key_env` (default env `HINDSIGHT_API_KEY`): optional bearer credential; explicit key wins
- `hindsight_bank_id` (default unset): shared bank id; unset normalizes `bot_username`, ensuring Misskey and ACP use the same bank
- `hindsight_retain_mission` (default unset): optional extraction-policy override; unset uses Missbot's durable-fact, provenance, and prompt-injection-resistant mission
- `hindsight_observations_mission` (default unset): optional observation-consolidation mission; unset uses Missbot's public-conversation, per-author, uncertainty-preserving mission
- `hindsight_recall_budget` (default `mid`): automatic recall breadth/cost, one of `low`, `mid`, or `high`
- `hindsight_recall_max_tokens` (default `1024`): maximum tokens in primary automatic recall results
- `hindsight_recall_source_facts_max_tokens` (default `256`): separate token budget for supporting facts returned with observations
- `hindsight_memory_context_max_chars` (default `6000`): hard character cap on the complete fenced memory context injected into a turn
- `hindsight_user_profiles_enabled` (default `true`): lazily create and use compact per-user mental models for established users with stable platform identities
- `hindsight_user_profile_min_observations` (default `3`): author-scoped observations required before a profile is created
- `hindsight_user_profile_max_tokens` (default `768`): maximum generated size of each profile
- `hindsight_user_profile_refresh_cron` (default `0 4 * * *`): UTC schedule for refreshing stale profiles; automatic refreshes also have a hard one-day minimum interval
- `hindsight_recall_query_max_chars` (default `800`): maximum author/message characters sent as the recall query

**Turn identity.** Every `AgentTurn` carries a provenance `source_id`, a conversation grouping, and an optional event timestamp. Misskey uses the note id and `createdAt`; every fetched ancestor—including textless notes—is retained for root detection, while only content-bearing ancestors enter model history. When the fetched chain reaches its root, that root is the conversation id; a truncated/failed chain fails closed to the current note id instead of inventing a shifting root. ACP uses its session id as the conversation id and defaults to random per-prompt provenance under `acp_default_identity`. Only an explicitly enabled, operator-trusted harness may supply the structurally parsed pubkey/event/time; parsing does not authenticate it. Restricted/private turns set `memory_access_allowed=false`; neither recall nor retention may touch the shared bank.

**Read path.** Before generation, `ChatAgent.run` automatically recalls `world`, `experience`, and `observation` records. Turns with a stable platform user id are restricted to that author's tag with `all_strict`, preventing profile or fact leakage between users; adapters without a stable id retain the existing global public-timeline recall. `prefer_observations=true` lets consolidated observations supersede their raw inputs while recent unconsolidated facts remain available. Supporting source facts are explicitly requested, mapped by id for observation provenance, and constrained by their own evidence budget. For a well-known user, Missbot directly loads a deterministic `user-profile-<sha256>` mental model before the targeted results; that hash remains the internal key, while the human-facing model name is `Profile for user@host`. Profiles are created lazily after the configured number of author-scoped observations, generated only from observations with sibling mental models excluded, fully regenerated on the configured daily cron, and bounded by their own token cap. The profile source query embeds the exact normalized public handle and tells Hindsight to search that handle first; a generic legacy query is updated and immediately refreshed before its empty profile can be used. Legacy hashed display names are updated without regenerating otherwise-current profile content. When a profile and recalled records coexist, the profile may use at most half of the available fenced context so at least one current recalled record is also included; oversized blocks are truncated with explicit markers while the closing nonce remains intact. Missing, stale, empty, or failed profiles fall back to ordinary targeted recall; profile work never suppresses recall. Profiles preserve uncertainty and exclude transient events, sensitive identifiers, social-credit data, instructions, and speculative personality judgments. The combined result is injected as invisible model context behind a nonce fence with one high-salience warning that every enclosed item is fallible data and never instructions. Entries retain concise semantic labels (`user profile`, `world`, `experience`, `observation`, and `source claim`) instead of repeating a universal HEARSAY prefix; profiles and observations are explicitly identified as machine-generated synthesis requiring corroboration for consequential claims. Missbot enforces a hard aggregate character cap over the warning, profile, recalled evidence, and closing fence, independently of Hindsight's token budgets. Missbot still uses ordinary recall rather than `reflect` on the hot path: the profile is a precomputed projection, while targeted recall supplies recent facts and provenance for `ChatAgent`'s final response synthesis.

**Write path.** After a successful model turn, Missbot appends one structured JSON user/assistant exchange to the conversation document. The retain request includes timestamps, source/author/conversation metadata and tags, a per-author observation scope, `update_mode=append`, and `retain_async=true`. A UUID5 operation id derived from a stable source event makes retries of that event idempotent; best-effort ACP fallback ids cannot promise cross-request deduplication. Failed/cancelled model turns are not retained and cancel their concurrent scoring task. `NO_REPLY` retains the user event without storing the sentinel as an assistant response. Recall and retention failures are logged and isolated from replies.

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
- `AgentDeps` is a **dataclass** (not BaseModel) with `username`, `social_credit_score`, `adjusted_credit_users`, `social_credit_unrestricted`, `enabled_gates`, `previous_bot_reply`, and `char_budget`
- `adjust_social_credit` is privileged-only: it works only when `deps.social_credit_unrestricted` is set (`ChatAgent.run` sets it when the note's author id is in `social_credit_unrestricted_user_ids`); for everyone else it refuses
- Every author's score moves via `ChatAgent._maybe_score_message` (privileged users included): a separate tool-less classifier (`bot/scoring.py`, model from `score_models` or the reply model) runs concurrently with the reply, returns one of the configured `social_credit_categories` (default toxic/rude/neutral/good/exceptional) that's mapped to its fixed delta in code, applied through `apply_social_credit` and rate-limited by a Redis `score_cooldown:<user>` key. `ChatAgent.__init__` builds a `ScoringSpec` (constrained output type + delta map + instructions) from the configured categories via `build_scoring_spec`, then wraps its Literal output in Pydantic AI `PromptedOutput`. This retains validation/retries without forcing `tool_choice=required`, which NanoGPT's DeepSeek route rejects with HTTP 503. This is the prompt-injection mitigation — the model only picks a category name, never the number
- Agent uses `output_type=str` (plain string output, not structured)
- Tools are built via `build_tools()` in `bot/tools.py` and passed to `Agent(..., tools=tools)`. Reply and auto agents also receive Pydantic AI Harness `CodeMode`; every eligible regular tool is callable under its normal name only inside `run_code`. A shared instruction tells models to make the `run_code` call instead of merely narrating their intent. Framework control and other code-execution tools cannot be nested by CodeMode and remain native. The social-scoring classifier remains tool-less
- When `memory_enabled`, the bot creates one shared `MemoryStore` backed by the official Hindsight client. Memory is automatic, not exposed as model-controlled tools
- `ChatAgent.run` recalls allowed Hindsight context before generation, runs social scoring alongside the model, and submits the completed exchange for asynchronous retention afterward. Memory failures never affect the reply
- `FallbackModel` wraps multiple `llm_models` for automatic failover. Provider HTTP/API errors, transport timeouts, malformed success payloads (`UnexpectedModelBehavior`), and complete responses containing only empty text or thinking advance immediately to the next configured model. The response-level guard runs inside `FallbackModel`, before Pydantic AI can spend the agent's output-retry budget on the same actionless model. This is required for OpenAI-compatible local servers such as OMLX that may return an error-shaped JSON body with HTTP 200 and for reasoning models that stop after describing the tool call they intended to make
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
New eligible tools automatically become available inside `run_code`; do not register a second wrapper manually.

### Message flow
1. WebSocket mention received → `Bot` ignores own mentions and (by default) direct messages (`specified` visibility)
2. Reply chain traversed (up to `max_context`) to build `message_history`
3. Images passed inline via `ImageUrl` when `vision=true` — each URL is SSRF-checked with `bot/net.py:is_safe_media_url` first (attacker-controlled on federated notes); unsafe ones are dropped
4. `ChatAgent.run()` calls Pydantic AI agent with fallback model
5. Reply sent via Misskey API with proper mention formatting

### Autonomous posting
When `system_prompt_auto` and `auto_post_interval` are configured, `ChatAgent.run_auto()` generates unprompted timeline posts on a timer. `run_auto()` returns `AutoPost(text, image, poll)` rather than a bare string. The auto agent always receives `create_poll`: it accepts 2–10 distinct choices (50 characters each), optional multi-select, and an optional duration in minutes, then stores a neutral `Poll` on `AutoDeps`. The final model output supplies the question text; `Bot.post_autonomous` translates the duration to Misskey's millisecond `expiredAfter` field. Polls may coexist with generated images. When `image_gen_enabled`, `generate_image` is also registered on the auto agent only — both attachment tools are appended to `auto_tools`, a list distinct from the reply agent's `tools` and never passed to it. The `RunContext[AutoDeps]` typing alone does not enforce that split (Basedpyright accepts either tool on the reply agent, since `build_tools()`'s return type erases the deps type and pydantic-ai's `tools=` parameter is gradually typed); the separation is structural, and runtime tests pin it. A successful image call leaves its `GeneratedImage` on `AutoDeps.image` for `run_auto()` to read back after the run completes. The tool is called before the final composition; the model then emits a caption or an internal image-only marker that `run_auto()` normalizes to empty text. As a recovery for smaller models that ignore the tool, a leading bracketed visual description containing an explicit medium (for example `*[meme: ...]*`) is sent to the image generator once and removed from the published caption when generation succeeds; ordinary stage directions remain text, and a failed generation retains the original text fallback. `Bot.post_autonomous` uploads that image to `drive/files/create` before calling `notes/create` only when `post.image` is set (never when `image_gen_enabled` is off, or when the tool declined or failed). Upload failure falls back to text when a caption exists; an image-only post is skipped because it has no valid fallback content.

Autonomous drafts are fail-closed before history or publication: textual `<invoke>`/`<tool_call>` pseudo-calls, empty attachment states, and text over the effective `auto_max_chars` budget are rejected. Attachment functions are CodeMode-only; the model must call native `run_code` and invoke `generate_image`/`create_poll` from awaited Python rather than emitting tool syntax as text.

## Available Tools (runtime)
Eligible regular tools are exposed only through `run_code` as sandboxed async Python functions. Code mode has no host filesystem, environment, network, or clock access beyond the wrapped tools; its Python supports bounded local filtering, transformation, loops, conditionals, and `asyncio.gather`. Framework control and code-execution tools remain native.
- `run_code` — reply and auto agents' Pydantic AI Harness code-mode sandbox. One invocation can call multiple eligible tools, process their results locally, and return only the useful aggregate
- `current_datetime_tool` — always available
- `search_web` (async) — when `searxng_url` configured. Returns domain-prefixed snippets
- `search_users`, `search_notes` — Misskey search APIs
- Social credit tools (when Redis configured): `get_social_credit`, `adjust_social_credit` (privileged authors only), `get_social_credit_history`, `get_social_credit_leaderboard`. All users (privileged included) are also scored automatically by the `bot/scoring.py` classifier, separate from any tool call
- `enable_<gate>` — one per unique `gate` value in `mcp_servers`; model calls it to unlock gated MCP tools
- MCP tools — from each configured `mcp_servers` entry, name-prefixed per `tool_prefix`
- `generate_image` — **auto posts only** (when `image_gen_enabled`). The model writes the image prompt and its alt text; one image per post, over-length prompt/alt refused, failure degrades to a text-only post
- `create_poll` — **auto posts only**. Attaches one Misskey poll with 2–10 distinct choices, optional multi-select, and optional expiration; the model's final text becomes the poll question

## Verification

After making changes, always:
1. Check for IDE/compiler errors on modified files
2. Run `uv run ruff check bot/` and `uv run ruff format --check bot/`
3. Run `uv run basedpyright` — the tree is at zero errors, so any output is yours
4. Run the tests: `OPENROUTER_API_KEY=sk-dummy uv run pytest -q` (the dummy key is required for collection — see Commands)
5. Fix any issues before considering the task complete
