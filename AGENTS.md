# Missbot

<!-- This file is the project doc; CLAUDE.md is just `@AGENTS.md`. Edit AGENTS.md, not CLAUDE.md. -->

Misskey/Fediverse chatbot using Pydantic AI with LLM fallback, WebSocket streaming, an optional Redis-backed social credit system, and an optional Postgres/pgvector world-knowledge store (claims ranked by user agreement).

## Commands

```bash
# Install
uv sync

# Run
uv run python -m bot -c config.local.yaml   # or: mise run bot

# World-knowledge maintenance — out-of-process, also run by a k8s CronJob
uv run python -m bot.maintenance consolidate -c config.local.yaml          # merge duplicate entities
uv run python -m bot.maintenance stats -c config.local.yaml
uv run python -m bot.maintenance calibrate-entities -c config.local.yaml   # tune entity_match_threshold

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

# Production cluster (kubectl context: mercury)
# World-knowledge DB is CloudNativePG, NOT local. Connect via the primary pod with peer auth
# as the postgres OS user (the `grok` app user fails peer auth; never put the password on the
# command line — the safety classifier blocks it, and you don't need it):
kubectl exec -n cnpg pg-cluster-1 -- psql -U postgres -d grok -tAc "SELECT count(*) FROM knowledge_claim"
# Bot pod + maintenance CronJob live in the `misskey` namespace (missbot-*), not `default`.
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
| `bot/memory.py` | `MemoryStore` — Postgres/pgvector world-knowledge store: `entity`/`claim` tables, subject entity resolution + write-time object linking, per-author upsert (`add_claim`), agreement-ranked recall (`search_claims`), entity consolidation/merge + `stats`. Pure helpers (`resolve_conflict`, `normalize_predicate`, `normalize_entity_name`, `normalize_object`, `object_group_key`, `merge_aliases`) are DB-free and unit-tested. Only built when `memory_enabled` |
| `bot/maintenance.py` | Out-of-process maintenance/admin CLI (`python -m bot.maintenance`): `consolidate`, `stats`, `calibrate-entities`. Wires the entity linker (`bot/ai.py:build_entity_linker`) so consolidation's LLM merge pass runs headless. Builds a one-shot `MemoryStore`; driven by a k8s CronJob (`k8s/maintenance.yaml`) |
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
- `temperature`, `top_p`, `frequency_penalty`, `presence_penalty` (all default unset/`None`): sampling + anti-repetition knobs for the **reply and auto-post** models, applied via `ChatAgent._generation_settings`. Each is only sent to the model when set (so an unset one keeps the provider default and isn't sent to models that reject it). Positive `frequency_penalty`/`presence_penalty` curb the bot reusing its own phrasing turn-after-turn. Bounds: temperature 0–2, top_p 0–1, penalties −2–2. The scoring/extraction/entity-linker agents are unaffected (they keep their own structured-output settings)
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
- `max_context`: parent notes to include (default 3)
- `ignore_direct_messages` (default `true`): skip direct/private messages (Misskey `specified` visibility); the bot is built for public-timeline threads. Set false to also reply to DMs
- `ignore_bots` (default `true`): skip mentions from accounts flagged as bots (Misskey user `isBot`); prevents bot-to-bot reply loops. Set false to also reply to other bots
- `max_reply_mentions`: cap on total mentions (incl. the author) echoed into a reply (default 5); prevents mention-amplification/harassment relaying
- `http_timeout_seconds`: HTTP timeout (default 30.0)
- `mcp_servers`: list of streamable-HTTP MCP servers (see below)
- `memory_enabled` (default `false`): turn on the Postgres/pgvector world-knowledge store (see below). Requires `postgres_url` and `embedding_model`
- `channel`, `debug`

### World-knowledge store
Global (non-user-specific) knowledge backed by Postgres + the `pgvector` extension, **ranked by user agreement**: every row is a `(subject, predicate, object)` claim attributed to the user who asserted it, and recall returns the value that the most *distinct* users independently assert. Deliberately minimal — trust tiers, model-output quarantine, append-only supersession/bitemporal reads, contradiction/dispute passes, decay, and source retraction were all dropped (see git history for the prior rich-claims design). Off unless `memory_enabled: true`. Config fields:
- `postgres_url` (required when enabled): Postgres DSN; the server must have the `vector` extension available. `MemoryStore.create()` runs `CREATE EXTENSION/TABLE IF NOT EXISTS`. It **fails fast** if it finds the legacy rich-schema (`knowledge_claim.trust_tier` column): drop the `knowledge_*` tables to migrate (the bot re-learns from the timeline; the store is not column-compatible with the old one)
- `embedding_model` (required when enabled): embedding model id, e.g. `perplexity/pplx-embed-v1-0.6b`
- `embedding_dim` (default `1024`): vector dimension; **must** match the model and the `knowledge_claim.embedding` / `knowledge_entity.embedding vector(N)` columns. Startup fails fast if an existing column's dimension disagrees (changing models means re-embedding every row). pplx-embed-v1-0.6b is 1024
- `embedding_base_url` (default `https://openrouter.ai/api/v1`) + `embedding_api_key` / `embedding_api_key_env` (default env `OPENROUTER_API_KEY`): the OpenAI-compatible embeddings endpoint (POSTed to `<base_url>/embeddings`)
- `global_recall_k` (default `5`), `global_recall_min_similarity` (default `0.3`): how many claims `search_memory` returns and the cosine-similarity floor
- `global_write_cooldown` (default `60`): min seconds between writes per author (bounds poisoning rate)
- `max_fact_length` (default `500`): longer submitted facts are rejected
- `entity_match_threshold` (default `0.82`): cosine-similarity floor for the *deterministic* write-time subject-entity link, used only when no LLM linker is wired (maintenance/headless). With the bot running, a constrained LLM linker (`bot/extract.py:EntityMatch`) decides among the nearest existing entities instead — exact name/alias match always links first regardless
- `entity_merge_threshold` (default `0.90`): cosine-similarity floor for the embedding-based entity merge in `consolidate` (kept higher than `entity_match_threshold`). A deterministic normalized-name pass runs first; tune the embedding floor with `calibrate-entities`
- `entity_merge_llm` (default `true`): in `consolidate`, after the name/embedding passes, run an LLM merge pass that offers each entity's near-neighbours to the entity-link classifier and folds in confirmed same-entity matches (heals fragmentation the deterministic passes miss)
- `memory_extract_models` (defaults to `llm_models`): model chain for the claim extractor (`bot/extract.py`); a smaller/cheaper model is usually fine
- `memory_ingest_notes` (default `true`): auto-ingest each incoming user note as a claim attributed to its author (rate-limited per author by `global_write_cooldown`)

**Write path (user notes only).** `ChatAgent._maybe_ingest_note` extracts a claim from the author's note and stores it via a **per-author upsert** keyed `UNIQUE(author, subject_entity_id, predicate)` — each user holds one current value per (subject, predicate); re-asserting overwrites their own row. The author handle is passed to the extractor so a self-statement ("I use Arch") resolves to a claim about them, and the prior thread (the same `context`/`max_context` chain the reply uses) is supplied as **reference-only** material so cross-note references resolve ("her name is Olive" after "I have a pet lizard") — claims are extracted only from the latest note. Durable **personal** facts about a named person are allowed, but the extractor skips sensitive categories and a `looks_sensitive` code backstop drops obvious PII (email/phone/IDs). The note text passes the `bot/extract.py` gate (typed claim or `Skip`). The bot may also write via the `remember_fact` tool (durable facts it wants to recall); those claims are stored under the **bot author**, which is **excluded from the agreement count** at recall (`MemoryStore.bot_author`, wired from `bot_username`) — so the bot can persist and recall its own facts but never inflates corroboration, and a single human assertion always outranks a bot-only one.

**Entity linking (stable grouping for the agreement count).** *Subjects:* when a claim's subject doesn't exactly match an existing entity, `MemoryStore._resolve_entity` offers the nearest entities to a constrained LLM linker (`ChatAgent._link_entity` + `EntityMatch`) which returns one of them or "new" — preventing fragmentation ("Cordillerans" vs "Cordilleran tribes"). It can only pick an offered candidate, runs with no DB transaction held, and falls back to "new" on error; without a linker the deterministic `entity_match_threshold` applies. *Objects:* linked to an entity **exact-match only** (`_match_entity_exact`: case-insensitive canonical-name/alias hit) and **link-only — never creates** an entity, so an object naming a known entity (`uses_os: "Arch Linux"`) gets an `object_entity_id` while free-text values (`born 1990`, `likes pizza`) stay literal. Recall is global, not scoped by who's asking.

**Data model.** `knowledge_entity` (`canonical_name`, `aliases`, `embedding`, `merged_into`); `knowledge_claim` (`subject_entity_id`, `predicate`, `object_text`, `object_key`, `object_entity_id`, `author`, `embedding`, `updated_at`, `UNIQUE(author, subject_entity_id, predicate)`). **Agreement grouping:** `search_claims` recalls candidate (subject, predicate) groups by embedding ANN, then tallies `COUNT(DISTINCT author)` per value across *all* live claims of those subjects (not just the candidate pool), grouping values by `object_group_key` = the linked `object_entity_id` if present, else `object_key` (normalized `object_text` from `normalize_object`: lowercase + accent-strip + whitespace-collapse, deliberately *no* plural-fold or punctuation-strip so `3.13`≠`3 13`). The SQL mirror is `_OBJECT_GROUP_SQL`. So surface variants (`Arch Linux`/`arch  linux`) and entity-linked synonyms don't fragment agreement. `resolve_conflict` (pure/DB-free) picks the value with the most distinct authors, recency (`updated_at`) breaking ties; losing values ride along as `conflicts`.

**Safety properties (enforced in code):** writes are per-author (one opinion each, so a single user can't inflate agreement by repeating) and rate-limited per author by `global_write_cooldown`; the agreement count is `COUNT(DISTINCT author)`, computed at read time, never a stored status; recalled claims reach the model fenced as untrusted data (`_fence_untrusted` in `bot/tools.py`); the extractor admission gate + `looks_sensitive` PII backstop still apply. The embedding model emits **unnormalized** vectors, so all comparisons use cosine distance (`<=>`).

**Maintenance.** Run out-of-process via `python -m bot.maintenance` (the `missbot-maintenance` k8s CronJob runs `consolidate` daily). *Consolidation* (`consolidate`) folds duplicate entities into the lowest-id keeper (repointing claims as subject and object, unioning aliases, soft-marking the dup `merged_into`) in three passes — a high-precision **name** pass (`normalize_entity_name`: accent/case/punctuation/plural normalization only), a conservative **embedding** pass at `entity_merge_threshold`, and (when `entity_merge_llm` + a linker is wired) an **LLM** pass. On repoint a dup-side claim that would collide with a keep-side claim from the same author+predicate is dropped first (the `UNIQUE` constraint). A final **object-link backfill** pass (`_backfill_object_links`) then links any unlinked claim object that now exactly names a live entity (canonical/alias, case-insensitive) — healing the write-time-only staleness where a claim's object names an entity created/aliased after the claim was written; exact-match and link-only, so zero overcount risk, and run last so it sees merged-in aliases. No corroboration recompute is needed — agreement re-tallies on the next recall. `stats` prints store counts; `calibrate-entities` lists the most-similar entity pairs to tune `entity_match_threshold`. Merge decisions are traced to Logfire.

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
- `AgentDeps` is a **dataclass** (not BaseModel) with `username`, `source_note_id`, `social_credit_score`, `adjusted_credit_users`, `social_credit_unrestricted`, `enabled_gates`
- `adjust_social_credit` is privileged-only: it works only when `deps.social_credit_unrestricted` is set (`ChatAgent.run` sets it when the note's author id is in `social_credit_unrestricted_user_ids`); for everyone else it refuses
- Every author's score moves via `ChatAgent._maybe_score_message` (privileged users included): a separate tool-less classifier (`bot/scoring.py`, model from `score_models` or the reply model) runs concurrently with the reply, returns one of the configured `social_credit_categories` (default toxic/rude/neutral/good/exceptional) that's mapped to its fixed delta in code, applied through `apply_social_credit` and rate-limited by a Redis `score_cooldown:<user>` key. `ChatAgent.__init__` builds a `ScoringSpec` (constrained output type + delta map + instructions) from the configured categories via `build_scoring_spec`. This is the prompt-injection mitigation — the model only picks a category name, never the number
- Agent uses `output_type=str` (plain string output, not structured)
- Tools are built via `build_tools()` in `bot/tools.py` and passed to `Agent(..., tools=tools)`
- When `memory_enabled`, `ChatAgent.__init__` builds a tool-less **claim-extraction agent** (`output_type=ClaimExtraction`, model from `memory_extract_models` or the reply model). Its `_extract_claim` method is used by `_maybe_ingest_note` to structure each incoming note into a typed claim (or `Skip` it) before the per-author upsert. Like scoring, this is an injection-mitigation: untrusted text can only pick a union branch, never free-form a stored fact
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
- `search_web` (async) — when `searxng_url` configured. Returns domain-prefixed snippets
- `search_users`, `search_notes` — Misskey search APIs
- Social credit tools (when Redis configured): `get_social_credit`, `adjust_social_credit` (privileged authors only), `get_social_credit_history`, `get_social_credit_leaderboard`. All users (privileged included) are also scored automatically by the `bot/scoring.py` classifier, separate from any tool call
- World-knowledge tools (when `memory_enabled`): `search_memory` (agreement-ranked recall — the most-agreed value per subject+predicate with its `agreed by N` count and any competing values, fenced as untrusted data) and `remember_fact` (the bot stores a durable fact via the same `bot/extract.py` admission gate + `looks_sensitive` PII backstop; written under the bot author, which is excluded from the agreement count — present only when the extractor is wired). User notes are also ingested automatically. Not given to the auto-agent
- `enable_<gate>` — one per unique `gate` value in `mcp_servers`; model calls it to unlock gated MCP tools
- MCP tools — from each configured `mcp_servers` entry, name-prefixed per `tool_prefix`

## Verification

After making changes, always:
1. Check for IDE/compiler errors on modified files
2. Run `uv run ruff check bot/` and `uv run ruff format --check bot/`
3. Run the tests: `OPENROUTER_API_KEY=sk-dummy uv run pytest -q` (the dummy key is required for collection — see Commands)
4. Fix any issues before considering the task complete
