# Missbot

Misskey/Fediverse chatbot using Pydantic AI with LLM fallback, WebSocket streaming, an optional Redis-backed social credit system, and optional Postgres/pgvector long-term memory.

## Commands

```bash
# Install
uv sync

# Run
uv run python -m bot -c config.local.yaml   # or: mise run bot

# Lint & format
uv run ruff check bot/
uv run ruff format --check bot/

# Docker
docker build -t missbot . && docker run -v /path/to/config.yaml:/config.yaml missbot

# Kubernetes
mise run build      # Build and push Docker image
mise run deploy     # Apply K8s manifests and restart
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
| `bot/memory.py` | `MemoryStore` — Postgres/pgvector global long-term memory: embeds facts, cosine search, dedup, provenance. Only built when `memory_enabled` |
| `bot/net.py` | `is_safe_media_url()` — SSRF guard for attacker-supplied image URLs (blocks private/reserved IPs and internal hosts) |
| `bot/mcp.py` | `build_mcp_toolsets()` + `gate_names()` — streamable-HTTP MCP servers with allow/block and gate filtering |
| `bot/api.py` | HTTP client utilities |
| `bot/cli.py` | CLI entry point and argument parsing |

## Config Schema (`config.yaml`)

Required fields:
- `domain`, `url` (HTTPS), `ws_url` (WebSocket), `token`
- `bot_user_id`, `bot_username`
- `llm_models`: list of model entries — either pydantic-ai strings (e.g. `"openrouter:anthropic/claude-3.5-sonnet"`) or dicts for custom OpenAI-compatible endpoints (`model`, `base_url`, optional `api_key` / `api_key_env`)
- `system_prompt`, `max_tokens`, `max_retries`

Optional fields:
- `vision`: bool (default `true`) — pass images directly to the main LLM
- `vision_models`: legacy, unused when `vision=true`
- `system_prompt_auto` + `auto_post_interval`: autonomous posting (interval in seconds)
- `searxng_url`, `searxng_user`, `searxng_password`: web search via SearXNG
- `redis_url`, `redis_password`, `redis_db`: Redis for social credit system
- `social_credit_auto_score` (default `true`): score every author's message via an isolated, tool-less classifier whose category is mapped to a fixed delta (−10…+10) in code — users can't dictate their own score (privileged users are scored too; the flag only gates the manual adjust tool)
- `social_credit_score_cooldown` (default `10`): min seconds between automatic score changes per user (bounds farming)
- `score_models`: model chain for the classifier (same forms as `llm_models`); defaults to `llm_models`. Use a cheaper/smaller model — classification is a simple labeling task
- `social_credit_categories`: list of sentiment buckets the classifier may assign, each `{name, delta, description}`. The model only picks a `name` (constrained output); code applies the matching `delta`, so configurability never lets the model choose the number. Defaults to the built-in toxic(−10)/rude(−5)/neutral(0)/good(+5)/exceptional(+10) set. Names must be unique (case-insensitive); `description` is shown to the classifier
- `social_credit_unrestricted_user_ids`: list of user ids; when the note's author is one of these, the bot may manually adjust any user's score by any amount via `adjust_social_credit` (which is refused for everyone else)
- `max_context`: parent notes to include (default 1)
- `max_reply_mentions`: cap on total mentions (incl. the author) echoed into a reply (default 5); prevents mention-amplification/harassment relaying
- `http_timeout_seconds`: HTTP timeout (default 30.0)
- `mcp_servers`: list of streamable-HTTP MCP servers (see below)
- `memory_enabled` (default `false`): turn on Postgres/pgvector long-term memory (see below). Requires `postgres_url` and `embedding_model`
- `channel`, `debug`

### Long-term memory
Global (non-user-specific) long-term memory backed by Postgres + the `pgvector` extension. **Stage 1 of the memory feature — only the global store exists; per-user memory is planned.** Off unless `memory_enabled: true`. Config fields:
- `postgres_url` (required when enabled): Postgres DSN; the server must have the `vector` extension available. `MemoryStore.create()` runs `CREATE EXTENSION/TABLE IF NOT EXISTS` on startup
- `embedding_model` (required when enabled): embedding model id, e.g. `perplexity/pplx-embed-v1-0.6b`
- `embedding_dim` (default `1024`): vector dimension; **must** match the model and the `global_memory.embedding vector(N)` column. Startup fails fast if an existing column's dimension disagrees (changing models means re-embedding every row). pplx-embed-v1-0.6b is 1024
- `embedding_base_url` (default `https://openrouter.ai/api/v1`) + `embedding_api_key` / `embedding_api_key_env` (default env `OPENROUTER_API_KEY`): the OpenAI-compatible embeddings endpoint (POSTed to `<base_url>/embeddings`)
- `global_recall_k` (default `5`), `global_recall_min_similarity` (default `0.3`): how many facts `search_memory` returns and the cosine-similarity floor
- `global_write_cooldown` (default `60`): min seconds between global writes per author (bounds poisoning rate)
- `global_dedup_threshold` (default `0.95`): a new fact this cosine-similar to an existing one is skipped as a near-duplicate
- `max_fact_length` (default `500`): longer facts are rejected

**Security:** global memory is writable from attacker-controlled message text (open-with-provenance model). Mitigations are mandatory and live in code: every fact stores `author` + `source_note_id` provenance; writes are rate-limited per author and de-duplicated; recalled facts are returned to the model fenced as untrusted data (`_fence_untrusted` in `bot/tools.py`). The embedding model emits **unnormalized** vectors, so all comparisons use cosine distance (`<=>`).

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
- `AgentDeps` is a **dataclass** (not BaseModel) with `username`, `source_note_id`, `social_credit_score`, `adjusted_credit_users`, `social_credit_unrestricted`, `enabled_gates`. `source_note_id` is recorded as provenance on `remember_fact` writes
- `adjust_social_credit` is privileged-only: it works only when `deps.social_credit_unrestricted` is set (`ChatAgent.run` sets it when the note's author id is in `social_credit_unrestricted_user_ids`); for everyone else it refuses
- Every author's score moves via `ChatAgent._maybe_score_message` (privileged users included): a separate tool-less classifier (`bot/scoring.py`, model from `score_models` or the reply model) runs concurrently with the reply, returns one of the configured `social_credit_categories` (default toxic/rude/neutral/good/exceptional) that's mapped to its fixed delta in code, applied through `apply_social_credit` and rate-limited by a Redis `score_cooldown:<user>` key. `ChatAgent.__init__` builds a `ScoringSpec` (constrained output type + delta map + instructions) from the configured categories via `build_scoring_spec`. This is the prompt-injection mitigation — the model only picks a category name, never the number
- Agent uses `output_type=str` (plain string output, not structured)
- Tools are built via `build_tools()` in `bot/tools.py` and passed to `Agent(..., tools=tools)`
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
1. WebSocket mention received → `Bot` ignores own mentions
2. Reply chain traversed (up to `max_context`) to build `message_history`
3. Images passed inline via `ImageUrl` when `vision=true` — each URL is SSRF-checked with `bot/net.py:is_safe_media_url` first (attacker-controlled on federated notes); unsafe ones are dropped
4. `ChatAgent.run()` calls Pydantic AI agent with fallback model
5. Reply sent via Misskey API with proper mention formatting

### Autonomous posting
When `system_prompt_auto` and `auto_post_interval` are configured, `ChatAgent.run_auto()` generates unprompted timeline posts on a timer.

## Available Tools (runtime)
- `current_datetime_tool` — always available
- `search_web` — when `searxng_url` configured
- `search_users`, `search_notes` — Misskey search APIs
- Social credit tools (when Redis configured): `get_social_credit`, `adjust_social_credit` (privileged authors only), `get_social_credit_history`, `get_social_credit_leaderboard`. All users (privileged included) are also scored automatically by the `bot/scoring.py` classifier, separate from any tool call
- Long-term memory tools (when `memory_enabled`): `remember_fact` (save a general fact to shared memory; rate-limited + deduped + provenance-stamped) and `search_memory` (semantic recall, results fenced as untrusted data). Not given to the auto-agent
- `enable_<gate>` — one per unique `gate` value in `mcp_servers`; model calls it to unlock gated MCP tools
- MCP tools — from each configured `mcp_servers` entry, name-prefixed per `tool_prefix`

## Verification

After making changes, always:
1. Check for IDE/compiler errors on modified files
2. Run `uv run ruff check bot/` and `uv run ruff format --check bot/`
3. Fix any issues before considering the task complete
