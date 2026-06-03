# Missbot

<!-- This file is the project doc; CLAUDE.md is just `@AGENTS.md`. Edit AGENTS.md, not CLAUDE.md. -->

Misskey/Fediverse chatbot using Pydantic AI with LLM fallback, WebSocket streaming, an optional Redis-backed social credit system, and an optional Postgres/pgvector world-knowledge **graph** (entities as vertices, facts as edges, ranked by user agreement).

## Commands

```bash
# Install
uv sync

# Run
uv run python -m bot -c config.local.yaml   # or: mise run bot

# World-knowledge maintenance — out-of-process, also run by a k8s CronJob
uv run python -m bot.maintenance consolidate -c config.local.yaml          # merge duplicate entities
uv run python -m bot.maintenance stats -c config.local.yaml
uv run python -m bot.maintenance reembed -c config.local.yaml              # regenerate ALL embeddings (model swap / seed cleanup)
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
kubectl exec -n cnpg pg-cluster-1 -- psql -U postgres -d grok -tAc "SELECT count(*) FROM knowledge_edge"
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
| `bot/memory.py` | `MemoryStore` — Postgres/pgvector world-knowledge **graph**: `knowledge_entity` (vertices) / `knowledge_relation` (one recall embedding per `(src, predicate)` "question") / `knowledge_edge` (per-author values) tables, source-entity resolution + write-time destination linking, per-author upsert (`add_edge`), agreement-ranked group-level recall (`search_edges`), entity consolidation/merge + `stats`. Pure helpers (`resolve_conflict`, `normalize_predicate`, `normalize_entity_name`, `normalize_value`, `value_group_key`, `render_relation`, `merge_aliases`) are DB-free and unit-tested. Only built when `memory_enabled` |
| `bot/maintenance.py` | Out-of-process maintenance/admin CLI (`python -m bot.maintenance`): `consolidate`, `stats`, `reembed`, `calibrate-entities`. Wires the entity linker (`bot/ai.py:build_entity_linker`) so consolidation's LLM merge pass runs headless. Builds a one-shot `MemoryStore`; driven by a k8s CronJob (`k8s/maintenance.yaml`) |
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
- `social_credit_ignore_threshold` (default unset/`None`): when set (and Redis configured), `Bot.on_mention` drops any author whose score is below it — the note never reaches the LLM, no reply is sent, and the author isn't scored or ingested. Authors with no score yet (`None`) are never ignored; checked via `ChatAgent.get_author_score`
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

### World-knowledge graph
Global (non-user-specific) knowledge backed by Postgres + the `pgvector` extension as an explicit **property graph**, **ranked by user agreement**: `knowledge_entity` rows are *vertices*, `knowledge_relation` rows are `(src_entity, predicate)` "questions" (one recall embedding each), and `knowledge_edge` rows are the per-author *values* hanging off a relation — each a `(src_entity, predicate, value)` fact attributed to the user who asserted it. Recall returns the value that the most *distinct* users independently assert. An edge is a **relationship edge** when its value names another entity (`dst_entity_id` set) or an **attribute edge** when the value is a literal (`dst_entity_id` NULL). Recall is **group-level**: the ANN runs over relation embeddings (one per question, not per author), so the candidate pool is naturally one-row-per-group and isn't crowded by a popular fact's many author rows. Deliberately minimal — multi-hop traversal recall, trust tiers, model-output quarantine, append-only supersession/bitemporal reads, contradiction/dispute passes, decay, and source retraction were all dropped (see git history for the prior rich-claims design). Off unless `memory_enabled: true`. Config fields:
- `postgres_url` (required when enabled): Postgres DSN; the server must have the `vector` extension available. `MemoryStore.create()` runs `CREATE EXTENSION/TABLE IF NOT EXISTS` and **migrates older layouts in place, losslessly**: a minimal `knowledge_claim` store → `knowledge_edge` (column rename), legacy snake_case predicates → phrases, and the per-edge recall embedding → a group-level `knowledge_relation` (one embedding per `(src, predicate)`, each seeded from one of its edges' existing vectors so the migration never calls the embeddings API). It **fails fast** if it finds the legacy rich-schema (`trust_tier` column): drop the `knowledge_*` tables to migrate (the bot re-learns from the timeline; the store is not column-compatible with that old one)
- `embedding_model` (required when enabled): embedding model id, e.g. `perplexity/pplx-embed-v1-0.6b`
- `embedding_dim` (default `1024`): vector dimension; **must** match what the model returns (see `embedding_dimensions`) and the `knowledge_relation.embedding` / `knowledge_entity.embedding vector(N)` columns. Startup fails fast if an existing column's dimension disagrees (changing it means re-embedding every row — `python -m bot.maintenance reembed`). pplx-embed-v1-0.6b is 1024. **pgvector HNSW indexes cap a plain `vector` column at 2000 dimensions** (use `halfvec` for more)
- `embedding_dimensions` (default unset): when set, sent as the OpenAI `dimensions` request param to truncate a Matryoshka (MRL) model's native output to the column size — e.g. pplx-embed-v1-4b returns its native 2560 unless you ask for 1024. **Must equal `embedding_dim`** (enforced). Leave unset when the model's native output already equals `embedding_dim`
- `embedding_base_url` (default `https://openrouter.ai/api/v1`) + `embedding_api_key` / `embedding_api_key_env` (default env `OPENROUTER_API_KEY`): the OpenAI-compatible embeddings endpoint (POSTed to `<base_url>/embeddings`)
- `global_recall_k` (default `5`), `global_recall_min_similarity` (default `0.3`): how many edges `search_memory` returns and the cosine-similarity floor
- `global_write_cooldown` (default `60`): min seconds between writes per author (bounds poisoning rate)
- `max_fact_length` (default `500`): longer submitted facts are rejected
- `entity_match_threshold` (default `0.82`): cosine-similarity floor for the *deterministic* write-time source-entity link, used only when no LLM linker is wired (maintenance/headless). With the bot running, a constrained LLM linker (`bot/extract.py:EntityMatch`) decides among the nearest existing entities instead — exact name/alias match always links first regardless
- `entity_merge_threshold` (default `0.90`): cosine-similarity floor for the embedding-based entity merge in `consolidate` (kept higher than `entity_match_threshold`). A deterministic normalized-name pass runs first; tune the embedding floor with `calibrate-entities`
- `entity_merge_llm` (default `true`): in `consolidate`, after the name/embedding passes, run an LLM merge pass that offers each entity's near-neighbours to the entity-link classifier and folds in confirmed same-entity matches (heals fragmentation the deterministic passes miss)
- `memory_extract_models` (defaults to `llm_models`): model chain for the claim extractor (`bot/extract.py`); a smaller/cheaper model is usually fine
- `memory_ingest_notes` (default `true`): auto-ingest each incoming user note as an edge attributed to its author (rate-limited per author by `global_write_cooldown`)

**Write path (user notes only).** `ChatAgent._maybe_ingest_note` extracts a claim from the author's note and stores it as an edge via a **per-author upsert** keyed `UNIQUE(author, relation_id)` — each user holds one current value per relation (source, predicate); re-asserting overwrites their own row. Embeddings are **lazy**: the subject name is embedded only when there's no exact entity match, and the relation "question" only when the relation doesn't already exist — so a repeat assertion about a known subject costs zero embedding calls. The author handle is passed to the extractor so a self-statement ("I use Arch") resolves to an edge about them, and the prior thread (the same `context`/`max_context` chain the reply uses) is supplied as **reference-only** material so cross-note references resolve ("her name is Olive" after "I have a pet lizard") — edges are extracted only from the latest note. Durable **personal** facts about a named person are allowed, but the extractor skips sensitive categories and a `looks_sensitive` code backstop drops obvious PII (email/phone/IDs). The note text passes the `bot/extract.py` gate (typed claim or `Skip`). The bot may also write via the `remember_fact` tool (durable facts it wants to recall); those edges are stored under the **bot author**, which is **excluded from the agreement count** at recall (`MemoryStore.bot_author`, wired from `bot_username`) — so the bot can persist and recall its own facts but never inflates corroboration, and a single human assertion always outranks a bot-only one.

**Entity linking (stable grouping for the agreement count).** *Sources:* an exact name/alias match (`_match_entity_exact`) links with no embedding; otherwise `MemoryStore._resolve_entity_nearest` offers the nearest entities to a constrained LLM linker (`build_entity_linker` + `EntityMatch`) which returns one of them or "new" — preventing fragmentation ("Cordillerans" vs "Cordilleran tribes"). It can only pick an offered candidate, runs with no DB transaction held, and falls back to "new" on error; without a linker the deterministic `entity_match_threshold` applies. *Values:* linked to a destination entity **exact-match only** (`_match_entity_exact`: case-insensitive canonical-name/alias hit) and **link-only — never creates** an entity, so a value naming a known entity (`uses_os: "Arch Linux"`) becomes a relationship edge with a `dst_entity_id` while free-text values (`born 1990`, `likes pizza`) stay attribute edges. This property-graph split keeps the entity (vertex) set small and meaningful so automated consolidation/dedup only ever reasons about real entities. Recall is global, not scoped by who's asking.

**Data model.** `knowledge_entity` (`canonical_name`, `aliases`, `embedding`, `merged_into`); `knowledge_relation` (`src_entity_id`, `predicate`, `embedding`, `UNIQUE(src_entity_id, predicate)`) — the question embedded as `render_relation` = `"subject — predicate"`; `knowledge_edge` (`relation_id`, `value_text`, `value_key`, `dst_entity_id`, `author`, `updated_at`, `UNIQUE(author, relation_id)`) — no embedding, recall is over the relation. **Agreement grouping:** `search_edges` recalls candidate relations by embedding ANN (one row per group), then tallies `COUNT(DISTINCT author)` per value across *all* live edges of those relations (not just the candidate pool), grouping values by `value_group_key` = the linked `dst_entity_id` if present, else `value_key` (normalized `value_text` from `normalize_value`: lowercase + accent-strip + whitespace-collapse, deliberately *no* plural-fold or punctuation-strip so `3.13`≠`3 13`). The SQL mirror is `_VALUE_GROUP_SQL`. So surface variants (`Arch Linux`/`arch  linux`) and entity-linked synonyms don't fragment agreement. Predicates are stored as natural lowercase **phrases** (`normalize_predicate`: lowercase, non-alphanumeric→single space, trim — e.g. `latest version`, `capital of`), so `latest version`/`Latest Version`/legacy `latest_version` all group together; `create()` folds any legacy snake_case predicates to the phrase form in place on startup. `resolve_conflict` (pure/DB-free) picks the value with the most distinct authors, recency (`updated_at`) breaking ties; losing values ride along as `conflicts`.

**Safety properties (enforced in code):** writes are per-author (one opinion each, so a single user can't inflate agreement by repeating) and rate-limited per author by `global_write_cooldown`; the agreement count is `COUNT(DISTINCT author)`, computed at read time, never a stored status; recalled edges reach the model fenced as untrusted data (`_fence_untrusted` in `bot/tools.py`); the extractor admission gate + `looks_sensitive` PII backstop still apply. The embedding model emits **unnormalized** vectors, so all comparisons use cosine distance (`<=>`).

**Maintenance.** Run out-of-process via `python -m bot.maintenance` (the `missbot-maintenance` k8s CronJob runs `consolidate` daily). *Consolidation* (`consolidate`) folds duplicate entities into the lowest-id keeper (folding the dup's relations onto the keeper, repointing edges' destination links, unioning aliases, soft-marking the dup `merged_into`) in three passes — a high-precision **name** pass (`normalize_entity_name`: accent/case/punctuation/plural normalization only), a conservative **embedding** pass at `entity_merge_threshold`, and (when `entity_merge_llm` + a linker is wired) an **LLM** pass. When the keeper already has a relation for the same predicate, the dup's edges move onto it and a dup-side edge that would collide with a keep-side edge from the same author is dropped first (the `UNIQUE(author, relation_id)` constraint). A final **relationship-link backfill** pass (`_backfill_relationship_links`) then links any unlinked edge value that now exactly names a live entity (canonical/alias, case-insensitive) — healing the write-time-only staleness where an edge's value names an entity created/aliased after the edge was written; exact-match and link-only, so zero overcount risk, and run last so it sees merged-in aliases. No corroboration recompute is needed — agreement re-tallies on the next recall. `stats` prints store counts; `calibrate-entities` lists the most-similar entity pairs to tune `entity_match_threshold`. `reembed` (`MemoryStore.reembed_all`) regenerates **all** embeddings (entity names + relation questions) with the configured model in batches (`embed_batch`) and in place — use it after a model swap or to upgrade migration-seeded relation vectors to clean question vectors; it is resumable and, if the model's vector **dimension** changed, re-dimensions the columns first (drop index → clear/retype → repopulate → restore `NOT NULL` + index). Because the bot's own startup fails fast on a dimension mismatch, a dim-changing swap means: update config → run `reembed` (built with `skip_dim_check=True`) while the bot is down → the bot starts once vectors are rebuilt. Merge decisions are traced to Logfire.

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
