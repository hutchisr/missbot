# Missbot

<!-- This file is the project doc; CLAUDE.md is just `@AGENTS.md`. Edit AGENTS.md, not CLAUDE.md. -->

Misskey/Fediverse chatbot using Pydantic AI with LLM fallback, WebSocket streaming, an optional Redis-backed social credit system, and optional mem0 long-term memory backed by Postgres/pgvector.

## Commands

```bash
# Install
uv sync

# Run
uv run python -m bot -c config.local.yaml   # or: mise run bot

# mem0 maintenance (the k8s CronJob runs the destructive form daily)
uv run python -m bot.maintenance cleanup --dry-run -c config.local.yaml
uv run python -m bot.maintenance cleanup -c config.local.yaml

# Test
# OPENROUTER_API_KEY must be set or test collection errors out: importing the agents
# constructs a pydantic-ai OpenRouter provider that fails fast without a key. The value is
# never used (tests mock the network), so any dummy string works.
OPENROUTER_API_KEY=sk-dummy uv run pytest -q

# Lint & format
uv run ruff check bot/
uv run ruff format --check bot/

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
```

**Important:** Always use `uv run` or `.venv/bin/python` — never bare `python`.

## Architecture

| File | Purpose |
|------|---------|
| `bot/bot.py` | WebSocket client, mention handling, context building, reply sending |
| `bot/ai.py` | `ChatAgent` class — Pydantic AI agent with `FallbackModel`, vision support |
| `bot/models.py` | Pydantic models: `Config`, `Note`, `User`, `MiFile`, WS message types |
| `bot/tools.py` | `build_tools()` factory — datetime, web search, search_users/notes, social credit tools; `apply_social_credit()` helper |
| `bot/scoring.py` | Injection-resistant message classifier: `build_scoring_spec()` turns `Config.social_credit_categories` into the constrained output type + delta map + hardened instructions; `build_scoring_prompt()` fences untrusted input |
| `bot/memory.py` | Thin async adapter around mem0's `AsyncMemory`; builds the mem0 config, scopes memories to the bot `agent_id`, and exposes the runtime plus maintenance read/delete paths |
| `bot/maintenance.py` | Out-of-process mem0 cleanup CLI; selects expired, duplicate, stale, empty, and per-author overflow note memories, then deletes them through mem0 so entity links stay consistent. Driven by `k8s/maintenance.yaml` |
| `bot/net.py` | `is_safe_media_url()` — SSRF guard for attacker-supplied image URLs (blocks private/reserved IPs and internal hosts) |
| `bot/mcp.py` | `build_mcp_toolsets()` + `gate_names()` — streamable-HTTP MCP servers with allow/block and gate filtering |
| `bot/api.py` | HTTP client utilities |
| `bot/cli.py` | CLI entry point and argument parsing |

## Config Schema (`config.yaml`)

Required fields:
- `domain`, `url` (HTTPS), `ws_url` (WebSocket), `token`
- `bot_user_id`, `bot_username`
- `llm_models`: list of model entries — either pydantic-ai strings (e.g. `"openrouter:anthropic/claude-3.5-sonnet"`) or dicts for custom OpenAI-compatible endpoints (`model`, `base_url`, optional `api_key` / `api_key_env`)
- `system_prompt`, `max_retries`

Optional fields:
- `max_tokens` (default unset/`None`): the hard reply/auto-post length cap. When set it's wired into the reply and auto agents' `model_settings` by `ChatAgent._generation_settings` (alongside the sampling knobs below) and only sent then; when unset, models generate unboundedly and long replies get truncated at the Misskey note cap (`max_note_length`), which can make the bot resume/repeat itself on the next turn
- `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` (all default unset/`None`): sampling + anti-repetition knobs for the **reply and auto-post** models, applied via `ChatAgent._generation_settings`. Each is only sent to the model when set (so an unset one keeps the provider default and isn't sent to models that reject it). Positive `frequency_penalty`/`presence_penalty` curb the bot reusing its own phrasing turn-after-turn. Bounds: temperature 0–2, top_p 0–1, penalties −2–2. The social scoring classifier is unaffected (it keeps its own structured-output settings)
- `vision`: bool (default `true`) — pass images directly to the main LLM
- `vision_models`: legacy, unused when `vision=true`
- `system_prompt_auto` + `auto_post_interval`: autonomous posting (interval in seconds)
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
- `channel`, `debug`

### Long-term memory
Memory is delegated to mem0 OSS through `bot/memory.py:MemoryStore`, backed by Postgres/pgvector. The bot does not maintain its own entity graph, claim schema, agreement counts, consolidation job, or re-embedding CLI anymore. Off unless `memory_enabled: true`. Config fields:
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
- `memory_note_retention_days` (default `90`, nullable): expiration/physical-retention window for inferred note memories; explicit `add_memory` entries are exempt
- `memory_max_memories_per_author` (default `50`, nullable): per-author cap for inferred note memories; maintenance removes the oldest overflow rows
- `memory_cleanup_scan_limit` (default `10000`): maximum scoped rows examined in one cleanup run
- `max_fact_length` (default `500`): longer `add_memory` submissions are rejected

**Write path.** `ChatAgent.run` still runs reply generation, social scoring, and memory ingestion concurrently. `_maybe_ingest_note` sends only the latest public author note to mem0 with metadata `{source: "misskey_note", author, source_note_id}` and a configured expiration date; specified/private notes are never ingested. The `add_memory` tool likewise refuses writes during private interactions. mem0 owns extraction, deduplication, vector storage, and entity-style linking. Successful tool writes use metadata `{source: "add_memory", author: bot_username}` and do not expire automatically. Memory failures are logged and swallowed so they never cancel a reply.

**Read path.** `search_memory` calls `MemoryStore.search()` with `agent_id=bot_username`, renders mem0 memories with score/source/author/recency metadata when available, and fences the returned text as untrusted data before the model sees it.

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
When `system_prompt_auto` and `auto_post_interval` are configured, `ChatAgent.run_auto()` generates unprompted timeline posts on a timer.

## Available Tools (runtime)
- `current_datetime_tool` — always available
- `search_web` (async) — when `searxng_url` configured. Returns domain-prefixed snippets
- `search_users`, `search_notes` — Misskey search APIs
- Social credit tools (when Redis configured): `get_social_credit`, `adjust_social_credit` (privileged authors only), `get_social_credit_history`, `get_social_credit_leaderboard`. All users (privileged included) are also scored automatically by the `bot/scoring.py` classifier, separate from any tool call
- Long-term memory tools (when `memory_enabled`): `add_memory` (passes text to mem0 `add`, except in private interactions) and `search_memory` (mem0 `search` results, fenced as untrusted data). Public user notes are also ingested automatically when `memory_ingest_notes=true`; private/specified notes are not. Not given to the auto-agent
- `enable_<gate>` — one per unique `gate` value in `mcp_servers`; model calls it to unlock gated MCP tools
- MCP tools — from each configured `mcp_servers` entry, name-prefixed per `tool_prefix`

## Verification

After making changes, always:
1. Check for IDE/compiler errors on modified files
2. Run `uv run ruff check bot/` and `uv run ruff format --check bot/`
3. Run the tests: `OPENROUTER_API_KEY=sk-dummy uv run pytest -q` (the dummy key is required for collection — see Commands)
4. Fix any issues before considering the task complete
