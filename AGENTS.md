# Missbot

Misskey/Fediverse chatbot using Pydantic AI with LLM fallback, WebSocket streaming, an optional Redis-backed social credit system, and an optional Postgres/pgvector world-knowledge store (claims-with-provenance, not bare facts).

## Commands

```bash
# Install
uv sync

# Run
uv run python -m bot -c config.local.yaml   # or: mise run bot

# World-knowledge maintenance (M4/M5) — out-of-process, also run by a k8s CronJob
uv run python -m bot.maintenance run-all -c config.local.yaml          # consolidate -> detect -> resolve disputes -> decay
uv run python -m bot.maintenance retract-source -c config.local.yaml --name evil.example --kind web
uv run python -m bot.maintenance stats -c config.local.yaml
uv run python -m bot.maintenance calibrate-entities -c config.local.yaml   # tune entity_match_threshold
uv run python -m bot.maintenance unmerge -c config.local.yaml --id 42      # reverse a soft entity-merge

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
| `bot/extract.py` | Injection-resistant claim extractor: `ClaimExtraction` discriminated union (`ExtractedClaim` \| `Skip`) + `EXTRACTION_INSTRUCTIONS` + `build_extraction_prompt()`. The admission gate that structures a submitted fact into a typed subject/predicate/object claim or rejects it |
| `bot/memory.py` | `MemoryStore` — Postgres/pgvector world-knowledge store: `source`/`entity`/`claim` tables, entity resolution, supersession, corroboration-based promotion, read-time conflict resolution + provenance, source retraction, **M4 maintenance** (`consolidate`, `detect_contradictions`, `stats`), legacy migration. Pure helpers (`resolve_conflict`, `tier_rank`, `is_stale`, `normalize_predicate`, `merge_aliases`) are DB-free and unit-tested. Only built when `memory_enabled` |
| `bot/maintenance.py` | Out-of-process maintenance/admin CLI (`python -m bot.maintenance`): `consolidate`, `detect-contradictions`, `resolve-disputes`, `decay`, `run-all`, `retract-source`, `unmerge`, `stats`, `calibrate-entities`. Wires the entity linker (`bot/ai.py:build_entity_linker`) so consolidation's LLM merge pass runs headless. Builds a one-shot `MemoryStore`; driven by a k8s CronJob (`k8s/maintenance.yaml`) |
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
- `ignore_direct_messages` (default `true`): skip direct/private messages (Misskey `specified` visibility); the bot is built for public-timeline threads. Set false to also reply to DMs
- `max_reply_mentions`: cap on total mentions (incl. the author) echoed into a reply (default 5); prevents mention-amplification/harassment relaying
- `http_timeout_seconds`: HTTP timeout (default 30.0)
- `mcp_servers`: list of streamable-HTTP MCP servers (see below)
- `memory_enabled` (default `false`): turn on the Postgres/pgvector world-knowledge store (see below). Requires `postgres_url` and `embedding_model`
- `channel`, `debug`

### World-knowledge store
Global (non-user-specific) knowledge backed by Postgres + the `pgvector` extension. **It never stores bare facts** — every row is a *claim* bound to a source and a time. **Built: M1–M3 (write path with quarantine + provenance, entity resolution, corroboration-based promotion, conflict-resolving read path); M4 consolidation, contradiction detection, autonomous dispute resolution, and decay; M5 retraction CLI + Logfire decision tracing.** Designed for minimal human curation: the maintenance passes are autonomous and non-destructive — even entity merges are **soft/reversible** (`merged_into` marker + `knowledge_entity_merge_log`, undoable via `unmerge`), nothing is hard-deleted. M4 active re-verification and per-user memory are still not built. Off unless `memory_enabled: true`. Config fields:
- `postgres_url` (required when enabled): Postgres DSN; the server must have the `vector` extension available. `MemoryStore.create()` runs `CREATE EXTENSION/TABLE IF NOT EXISTS`, then migrates any legacy `global_memory` rows on startup
- `embedding_model` (required when enabled): embedding model id, e.g. `perplexity/pplx-embed-v1-0.6b`
- `embedding_dim` (default `1024`): vector dimension; **must** match the model and the `knowledge_claim.embedding` / `knowledge_entity.embedding vector(N)` columns. Startup fails fast if an existing column's dimension disagrees (changing models means re-embedding every row). pplx-embed-v1-0.6b is 1024
- `embedding_base_url` (default `https://openrouter.ai/api/v1`) + `embedding_api_key` / `embedding_api_key_env` (default env `OPENROUTER_API_KEY`): the OpenAI-compatible embeddings endpoint (POSTed to `<base_url>/embeddings`)
- `global_recall_k` (default `5`), `global_recall_min_similarity` (default `0.3`): how many claims `search_memory` returns (after conflict resolution) and the cosine-similarity floor
- `global_write_cooldown` (default `60`): min seconds between writes per author (bounds poisoning rate)
- `global_dedup_threshold` (default `0.95`): a new claim from the *same source* this cosine-similar to an existing same-subject+predicate claim is skipped as a near-duplicate
- `max_fact_length` (default `500`): longer submitted facts are rejected
- `corroboration_threshold` (default `2`): independent tier-≥`secondary` sources that must assert the same subject+predicate+object before a claim is promoted `asserted` → `believed`. Model-sourced (`model_quarantine`) claims never count and are never promoted
- `entity_match_threshold` (default `0.82`): cosine-similarity floor for the *deterministic* write-time entity link, used only when no LLM linker is wired (maintenance/headless). With the bot running, a constrained LLM linker (`bot/extract.py:EntityMatch`) decides among the nearest existing entities instead — exact name/alias match always links first regardless
- `entity_merge_threshold` (default `0.90`): cosine-similarity floor for the embedding-based entity merge in `consolidate` (kept higher than `entity_match_threshold`). A deterministic normalized-name pass runs first; tune the embedding floor with `calibrate-entities`
- `entity_merge_llm` (default `true`): in `consolidate`, after the name/embedding passes, run an LLM merge pass that offers each entity's near-neighbours to the entity-link classifier and folds in confirmed same-entity matches (heals fragmentation the deterministic passes miss). Merges are soft/reversible (`unmerge`)
- `volatile_ttl_seconds` (default `86400`): a `volatile` claim older than this (by `valid_from`, else `recorded_at`) is flagged stale on recall so the model re-verifies it live
- `decay_ttl_seconds` (default `2592000`, 30d): the `decay` pass soft-retracts low-trust (tier < secondary), non-`believed`/non-`disputed` claims neither recorded nor recalled within this window; promotable/recently-used/disputed claims are never touched (disputed ones are left to `resolve-disputes`)
- `dispute_grace_seconds` (default `604800`, 7d): a contradiction must stay `disputed` this long before `resolve-disputes` picks the best-supported value and supersedes the rest
- `memory_extract_models` (defaults to `llm_models`): model chain for the claim extractor (`bot/extract.py`); a smaller/cheaper model is usually fine
- `memory_ingest_web` (default `true`): auto-ingest web-search results as `web`/`secondary` claims attributed to their domain (adds an extraction call per result)
- `memory_ingest_notes` (default `true`): auto-ingest each incoming user note as a `user`-tier claim attributed to its author (rate-limited per author by `global_write_cooldown`)

**Source determination (by channel, never by model self-report).** A claim's `source`/`trust_tier` is assigned by code based on where the text provably came from, since the model is an unreliable narrator of its own provenance:
- **Web** — `search_web` (now async) parses each result's domain and ingests it as `kind=web`, `secondary` tier in a background task (`bot/tools.py:_ingest_web_results`, fire-and-forget so it never adds latency to the search). Distinct domains agreeing is what lets a claim reach `believed`.
- **User note** — `ChatAgent._maybe_ingest_note` extracts a claim from the author's note and stores it as `kind=user`, `user` tier, sourced to the author handle (deterministic; never promotable alone, but attributable + retractable by author). The author handle is passed to the extractor so a self-statement ("I use Arch") resolves to a claim about them, and the prior thread (the same `context`/`max_context` chain the reply uses) is supplied as **reference-only** material so cross-note references resolve ("her name is Olive" after "I have a pet lizard") — claims are still extracted only from the latest note. Durable **personal** facts about a named person are allowed, but the extractor skips sensitive categories and a `looks_sensitive` code backstop drops any obvious PII (email/phone/IDs) that slips through.
- **Model** — `remember_fact` (model decides to remember) → `kind=model`, `model_quarantine`.

All three pass through the same `bot/extract.py` admission gate, so web/note text that isn't a durable, entity-bound fact is `Skip`ped. **Write-time entity linking:** when a claim's subject doesn't exactly match an existing entity, `MemoryStore._resolve_entity` offers the nearest entities to a constrained LLM linker (`ChatAgent._link_entity` + `EntityMatch`) which returns one of them (the same real-world entity) or "new" — preventing fragmentation ("Cordillerans" vs "Cordilleran tribes") at the source. It can only pick an offered candidate, the call is made with no DB transaction held, and it falls back to "new" on any error; without a linker the deterministic `entity_match_threshold` applies. Personal facts are scoped only by tier/provenance, not by who's asking — recall is global (per-user-scoped recall is the planned per-user memory).

**Data model.** `knowledge_source` (`name`, `kind` = web|doc|user|model, `default_trust_tier`); `knowledge_entity` (`canonical_name`, `aliases`, `embedding`); `knowledge_claim` (`subject_entity_id`, `predicate`, `object_text`, `source_id`, `trust_tier`, `confidence`, `status` = asserted|believed|disputed|retracted, bitemporal `valid_from`/`valid_to` + `recorded_at`, `superseded_by`/`superseded_at`, `retracted_at`, `corroboration_count`, `volatility`, `embedding`, `author`/`source_note_id` provenance). Trust tiers rank `model_quarantine` < `user` < `secondary` < `primary`; only `secondary`/`primary` are promotable (`PROMOTABLE_TIERS`).

**Safety core (enforced in code):** (1) no bare facts — every claim has a source + time; (2) **model output is quarantined** — `remember_fact` writes at `model_quarantine` and can never auto-promote to `believed`; (3) append-only — updates supersede, never overwrite; (4) promotion needs `corroboration_threshold` independent tier-≥`secondary` sources; (5) conflicts resolved at *read* time (`resolve_conflict`: believed > asserted, then trust tier, then recency), never collapsed at write; (6) provenance always rides along with recalled claims; (7) volatile claims past TTL are flagged stale; (8) `MemoryStore.retract_source()` tombstones every claim from a source and recomputes corroboration everywhere. Writes are rate-limited per author and deduped; the submitted fact passes the `bot/extract.py` admission gate (typed claim or `Skip`); recalled claims reach the model fenced as untrusted data (`_fence_untrusted` in `bot/tools.py`). The embedding model emits **unnormalized** vectors, so all comparisons use cosine distance (`<=>`).

**Maintenance (M4/M5).** Run out-of-process via `python -m bot.maintenance` (the `missbot-maintenance` k8s CronJob runs `run-all` daily). *Consolidation* (`consolidate`) folds duplicate entities into the lowest-id keeper (repointing claims, unioning aliases) in three passes — a high-precision **name** pass (`normalize_entity_name`: accent/case/punctuation/plural normalization only), a conservative **embedding** pass at `entity_merge_threshold`, and (when `entity_merge_llm` + a linker is wired) an **LLM** pass that offers each entity's near-neighbours to the entity-link classifier and folds in confirmed matches (healing fragmentation like "Cordillerans"/"Cordilleran tribes"). All merges are **soft/reversible** (the dup is marked `merged_into` and logged, not deleted; `unmerge` reverses it); the LLM pass makes its decisions outside any transaction. Afterwards it recomputes corroboration everywhere (so a merge can newly promote/demote). *Contradiction detection* (`detect_contradictions`) marks every live claim in a subject+predicate group with ≥2 distinct object values as `disputed` (never deletes; just demotes and excludes them from corroboration until resolved), stamps `disputed_at`, and clears the flag when a group narrows back to one value. `disputed` claims don't count toward corroboration. *Dispute resolution* (`resolve_disputes`) acts autonomously on disputes older than `dispute_grace_seconds`: `rank_dispute_values` picks the best-supported value (independent sources, then tier, then recency) and **supersedes** the losing values (archived, recoverable via an as-of read, not re-flagged), then re-promotes the winner. *Decay* (`decay`) soft-retracts low-trust, non-`believed`/non-`disputed`, never-recently-recalled claims past `decay_ttl_seconds` — the autonomous GC that replaces human curation (recall updates `last_recalled_at` on the winning claim only). The legacy migration seeds `last_recalled_at=now()` so imported facts aren't decayed on the first run. Active re-verification is the one M4 pass not implemented. `stats` prints store counts; promotion/merge/contradiction/dispute/decay/retraction decisions are traced to Logfire.

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
- When `memory_enabled`, `ChatAgent.__init__` builds a tool-less **claim-extraction agent** (`output_type=ClaimExtraction`, model from `memory_extract_models` or the reply model). Its `_extract_claim` method is passed into `build_tools(..., extractor=...)`; `remember_fact` is only exposed when that extractor exists, and uses it to structure (or `Skip`) a submitted fact before writing it as a `model_quarantine` claim. Like scoring, this is an injection-mitigation: untrusted text can only pick a union branch, never free-form a stored fact
- `ChatAgent.run` runs three coroutines concurrently via `asyncio.gather`: the reply, `_maybe_score_message`, and `_maybe_ingest_note` (extract a `user`-tier claim from the author's note when `memory_ingest_notes`). Like scoring, note ingestion swallows its own errors and is rate-limited per author, so it never affects the reply
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
- `search_web` (async) — when `searxng_url` configured. Returns domain-prefixed snippets; when `memory_enabled` + `memory_ingest_web`, also ingests each result as a `web`/`secondary` claim attributed to its domain in a background task (doesn't block the search)
- `search_users`, `search_notes` — Misskey search APIs
- Social credit tools (when Redis configured): `get_social_credit`, `adjust_social_credit` (privileged authors only), `get_social_credit_history`, `get_social_credit_leaderboard`. All users (privileged included) are also scored automatically by the `bot/scoring.py` classifier, separate from any tool call
- World-knowledge tools (when `memory_enabled`): `remember_fact` (extract a submitted fact into a typed claim and store it at the quarantined `model_quarantine` tier; rate-limited + deduped + provenance-stamped; only present when the extractor is wired) and `search_memory` (semantic recall, conflict-resolved, returned with full provenance and fenced as untrusted data). Not given to the auto-agent
- `enable_<gate>` — one per unique `gate` value in `mcp_servers`; model calls it to unlock gated MCP tools
- MCP tools — from each configured `mcp_servers` entry, name-prefixed per `tool_prefix`

## Verification

After making changes, always:
1. Check for IDE/compiler errors on modified files
2. Run `uv run ruff check bot/` and `uv run ruff format --check bot/`
3. Fix any issues before considering the task complete
