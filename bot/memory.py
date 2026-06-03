"""World-knowledge store for Missbot (Postgres + pgvector), modelled as a graph.

This module owns the bot's shared, non-user-specific knowledge as an explicit **property
graph**: ``knowledge_entity`` rows are *vertices* (entities) and ``knowledge_claim`` rows are
*attributed statements* — each a ``(src_entity, predicate, value)`` fact attributed to the user
who asserted it. A claim is one of two kinds, by whether its value is itself a vertex:

* **relationship** — ``dst_entity_id`` set: the value names another entity ("uses_os" -> the
  *Arch Linux* vertex), so the claim forms a true graph **edge** between two vertices. Grouped
  by the destination vertex identity.
* **attribute** — ``dst_entity_id`` NULL: the value is a literal ("born 1990", "latest version:
  3.13"), a property of the source vertex. Grouped by ``value_key`` (normalized literal text).

The one ranking signal is **agreement**: a value that more *distinct* users independently
assert outranks one a single user asserts.

Recall is **group-level**. Every ``(src_entity, predicate)`` "question" is one
``knowledge_relation`` row holding a single embedding of the question text (``subject —
predicate``); the per-author values hang off it as claims. So the recall vector is shared by
every author's value, isn't biased by any one asserted answer, and the ANN candidate pool is
naturally one-row-per-group (no per-author duplicates crowding it). Embeddings: ``embed`` the
*question* once per relation, and ``embed`` an entity's *name* once for resolution.

Model (enforced in code):

1. **One opinion per user.** A claim is keyed ``UNIQUE(author, relation_id)`` and written by
   upsert, so each user holds at most one current value per relation; re-asserting overwrites
   their own row ("changed my mind").
2. **Agreement = distinct authors.** Recall counts ``COUNT(DISTINCT author)`` per value within
   a relation; the most-agreed value wins (recency breaks ties). The count is computed at read
   time, never stored as a status.
3. **Stable grouping.** Sources resolve to entities (write-time linker); values group by
   :func:`value_group_key` (linked destination entity, else normalized literal text) so
   surface variants ("Arch" / "Arch Linux") don't fragment the agreement count.

Deliberately *not* modelled (dropped for simplicity): multi-hop traversal recall, trust
tiers, model-output quarantine, append-only supersession / bitemporal reads,
contradiction/dispute passes, decay, and source retraction. The embedding model produces
*unnormalized* vectors, so all comparisons use cosine distance (``<=>``).
"""

import os
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Optional, Union

import asyncpg
import httpx
import logfire
from asyncpg.pool import PoolConnectionProxy

from .models import Config

# Write-time entity linking: an injected async classifier that, given a source name and the
# nearest existing entities, returns the id of the one it's the SAME entity as (or None for
# "new"). Set by the bot wiring; None in maintenance/headless contexts (deterministic
# embedding fallback then applies).
EntityLinker = Callable[[str, list[tuple[int, str]]], Awaitable[Optional[int]]]
# How many nearest entities to offer the linker, and the (broad) similarity floor below
# which a candidate isn't even a plausible duplicate.
_LINK_CANDIDATES = 8
_LINK_FLOOR = 0.4

# Pool-acquired connections are proxies, not bare Connections; helpers accept either.
_Conn = Union[asyncpg.Connection, PoolConnectionProxy]


def normalize_predicate(predicate: str) -> str:
    """Normalize a predicate to a stable lowercase **phrase** for grouping/agreement.

    Two relations only group together (and so corroborate each other) when their predicates
    match exactly, so we canonicalize to a natural-language phrase: lowercase, reduce any run
    of non-alphanumeric characters to a single space, and trim. So "Latest Version!!",
    "latest_version", and "latest version" all collapse to "latest version" — predicates read
    as ordinary phrases while legacy snake_case still folds onto the same key.
    """
    p = re.sub(r"[^a-z0-9]+", " ", (predicate or "").strip().lower()).strip()
    return p or "fact"


def normalize_entity_name(name: str) -> str:
    """Canonicalize an entity name for high-precision, embedding-free dedup.

    This is the merge key for the consolidation name pass: strip accents, lowercase, drop a
    leading ``@``, reduce punctuation to spaces, and apply a light plural fold (drop a
    trailing ``s`` on tokens longer than 3 chars, but not ``ss``). Two names with the same
    result differ only in formatting, so merging them is safe. It is deliberately
    conservative — it only collapses formatting/accent/plural differences ("@anemone" /
    "anemone"), not granularity differences ("Tausug" / "Tausug people"). A folding miss
    (e.g. ``-ies`` plurals) only costs recall: a lone mis-stemmed name matches nothing,
    never the wrong thing.
    """
    decomposed = unicodedata.normalize("NFKD", name or "")
    ascii_ish = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = ascii_ish.strip().lower().lstrip("@")
    tokens = re.sub(r"[^a-z0-9]+", " ", lowered).split()
    folded = [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in tokens]
    return " ".join(folded)


def normalize_value(value_text: str) -> str:
    """Canonicalize a claim's literal value into the grouping key for the agreement count.

    Two attribute claims agree (count as the same value) only when their values share this key,
    so it must fold away the formatting noise two writers can differ on for the *same* value while
    keeping genuinely different values apart. Deliberately **more conservative** than
    :func:`normalize_entity_name`: it lowercases, strips accents (NFKD), and collapses internal
    whitespace runs — but does **not** fold plurals or drop internal punctuation, because literal
    values are heterogeneous (versions like ``3.13``, dates, free phrases) where ``3.13`` vs
    ``3 13`` or ``Windows`` vs ``Window`` are distinct, and a wrong merge silently miscounts
    agreement. A folding miss only undercounts agreement (the safe failure); a wrong merge would
    credit agreement to the wrong value. Synonyms ("Manila" vs "City of Manila") are out of scope
    here — that needs entity resolution (relationship claims via ``dst_entity_id``), not string
    normalization.
    """
    decomposed = unicodedata.normalize("NFKD", value_text or "")
    ascii_ish = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_ish.lower()).strip()


# SQL expression yielding a claim's value grouping key: the linked destination entity if present
# (relationship claim), else the normalized literal text (attribute claim). The ``e``/``k`` prefixes
# namespace the two so an entity id can never collide with a literal value that happens to be
# numeric. MUST stay in lockstep with :func:`value_group_key` (the Python mirror) — the agreement
# count groups by both.
_VALUE_GROUP_SQL = "COALESCE('e' || dst_entity_id::text, 'k' || value_key)"


def value_group_key(dst_entity_id: Optional[int], value_key: str) -> str:
    """Grouping key for the agreement count — the Python mirror of :data:`_VALUE_GROUP_SQL`.

    Two claims count as the same value (so they agree) when they point at the same destination
    entity (relationship claims), or — when neither is linked — when their normalized literal text
    matches (attribute claims). A value is only ever *linked* to an entity that already exists (see
    :meth:`MemoryStore._match_entity_exact`), so this never invents entities for free-text values;
    unlinked values fall back to ``value_key``.
    """
    if dst_entity_id is not None:
        return f"e{dst_entity_id}"
    return f"k{value_key}"


def render_relation(subject: str, predicate: str) -> str:
    """Text embedded for a relation's recall vector: the ``(subject, predicate)`` "question".

    Recall matches a query against this question — not the asserted value — so the recall vector
    is one per ``(subject, predicate)`` group, shared by every author's value, and isn't biased by
    any one answer.
    """
    return f"{subject} — {predicate}"


def resolve_conflict(values: list[dict]) -> dict:
    """Pick the winning value among those sharing a relation. Pure / DB-free.

    The value more *distinct users* assert wins (``agreed_by``); ties break by recency
    (``recency`` — the latest assertion of the value). Callers keep the full set so the
    losing alternatives surface alongside the winner. Each input dict carries at least
    ``agreed_by`` (int) and ``recency`` (datetime).
    """
    return max(values, key=lambda v: (int(v["agreed_by"]), v["recency"]))


def merge_aliases(
    keep_canonical: str, keep_aliases: list[str], dup_canonical: str, dup_aliases: list[str]
) -> list[str]:
    """Alias list for an entity that absorbs a duplicate (consolidation helper).

    The duplicate's canonical name and aliases become aliases of the keeper, deduped
    (case-insensitive) and excluding the keeper's own canonical name. Pure / DB-free.
    """
    merged: list[str] = []
    seen: set[str] = {keep_canonical.lower()}
    for alias in [*keep_aliases, dup_canonical, *dup_aliases]:
        key = (alias or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(alias)
    return merged


def _vector_literal(vec: list[float]) -> str:
    """Render an embedding as a pgvector text literal (e.g. '[0.1,0.2,...]').

    We pass vectors as text and cast to ``vector`` in SQL, which avoids depending on a
    binary pgvector codec and never needs to decode vectors back in Python.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


@dataclass
class ConflictingClaim:
    """An alternative value (fewer users assert it) that disagrees with the recalled winner."""

    value_text: str
    agreed_by: int


@dataclass
class RecalledClaim:
    """A recalled claim: the most-agreed value for a (src, predicate), with alternatives."""

    subject: str
    predicate: str
    value_text: str
    agreed_by: int
    similarity: float
    conflicts: list[ConflictingClaim]


@dataclass
class ClaimWriteResult:
    """Outcome of an :meth:`MemoryStore.add_claim` call."""

    stored: bool
    claim_id: Optional[int]
    updated: bool
    subject: str
    predicate: str


@dataclass
class EntityNeighbor:
    """A pair of entities and their cosine similarity (for threshold calibration)."""

    name_a: str
    name_b: str
    similarity: float


class MemoryStore:
    """Async Postgres-backed store for the world-knowledge graph (entities, relations, claims)."""

    def __init__(self, pool: asyncpg.Pool, http: httpx.AsyncClient, config: Config):
        self._pool = pool
        self._http = http
        self._config = config
        self._embed_url = f"{str(config.embedding_base_url).rstrip('/')}/embeddings"
        self._embed_model = config.embedding_model
        # When set, sent as the `dimensions` request param to truncate a Matryoshka model's output
        # to the stored column size (validated == embedding_dim in Config).
        self._embed_dimensions = config.embedding_dimensions
        key = config.embedding_api_key or os.environ.get(config.embedding_api_key_env)
        self._embed_headers = {"Authorization": f"Bearer {key}"} if key else {}
        # Optional write-time entity-linking classifier, injected by the bot wiring.
        self.entity_linker: Optional[EntityLinker] = None
        # Author whose claims are stored and recalled but excluded from the agreement count
        # (the bot itself, via remember_fact), injected by the bot wiring. None in
        # maintenance/headless contexts, where every author counts. See `search_claims`.
        self.bot_author: Optional[str] = None

    @classmethod
    async def create(cls, config: Config, *, skip_dim_check: bool = False) -> "MemoryStore":
        """Connect, ensure/migrate the schema/extension exist, return store.

        Schema: ``knowledge_entity`` (vertices), ``knowledge_relation`` (one recall embedding per
        ``(src_entity, predicate)`` "question"), and ``knowledge_claim`` (one per-author value for a
        relation). A claim whose value names another entity (``dst_entity_id`` set) forms a graph
        *edge* (a relationship); a literal-valued claim is an *attribute*.

        Migrates older layouts in place, losslessly and idempotently, converging on that schema from
        any prior state:

        * an interim ``knowledge_edge`` value table -> ``knowledge_claim`` (table rename);
        * a legacy minimal store's ``subject/object`` columns -> ``src/dst``/``value`` (column rename);
        * legacy snake_case predicates -> natural phrases;
        * a per-row recall embedding -> a group-level ``knowledge_relation`` (one embedding per
          ``(src, predicate)``), seeding each relation from one of its claims' existing vectors so no
          row is re-embedded against the API at startup.

        Fails fast if it finds the legacy rich-claims schema (``trust_tier`` column), or if an
        existing relation-embedding column dimension disagrees with ``config.embedding_dim`` —
        that mismatch silently returns garbage neighbors, so it must surface loudly (it means the
        corpus needs re-embedding). ``skip_dim_check`` suppresses only that dimension guard, for the
        ``reembed`` maintenance path which is *about* to re-dimension and repopulate the vectors.
        """
        assert config.postgres_url, "postgres_url is required to build a MemoryStore"
        dim = config.embedding_dim

        # Create the extension and tables on a plain connection *before* opening the
        # pool, so the `vector` type exists by the time any pooled connection uses it.
        conn = await asyncpg.connect(config.postgres_url)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            async def _has_column(table: str, column: str) -> bool:
                """Whether ``table.column`` exists — used to tell schema generations apart."""
                return bool(
                    await conn.fetchval(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = $1 AND column_name = $2)",
                        table,
                        column,
                    )
                )

            # Fail fast on the legacy rich-claims schema (not column-compatible with this store);
            # check every historical value-table name so the guard holds at any migration stage.
            legacy = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name IN ('knowledge_claim', 'knowledge_edge') AND column_name = 'trust_tier'"
            )
            if legacy is not None:
                raise RuntimeError(
                    "knowledge_claim/knowledge_edge has the legacy 'trust_tier' column — the old rich claims "
                    "store, which this agreement-ranked store is not column-compatible with. Drop the "
                    "knowledge_* tables (the bot re-learns from the timeline) before starting."
                )

            # --- In-place, idempotent migration converging on the current schema ----------------
            # The value table was briefly named `knowledge_edge`; normalise it back to
            # `knowledge_claim` first so every later step reasons about one name (column checks then
            # tell the generations apart). Runs only when the interim name exists and the final
            # one does not.
            if await conn.fetchval(
                "SELECT to_regclass('knowledge_edge') IS NOT NULL AND to_regclass('knowledge_claim') IS NULL"
            ):
                await conn.execute("ALTER TABLE knowledge_edge RENAME TO knowledge_claim")
                for _old, _new in (
                    ("knowledge_edge_embedding_idx", "knowledge_claim_embedding_idx"),
                    ("knowledge_edge_sp_idx", "knowledge_claim_sp_idx"),
                    ("knowledge_edge_relation_idx", "knowledge_claim_relation_idx"),
                    ("knowledge_edge_dst_idx", "knowledge_claim_dst_idx"),
                ):
                    await conn.execute(f"ALTER INDEX IF EXISTS {_old} RENAME TO {_new}")
                logfire.info("Renamed knowledge_edge -> knowledge_claim")

            # Legacy minimal store: rename subject/object columns to src/dst/value.
            if await _has_column("knowledge_claim", "subject_entity_id"):
                await conn.execute("ALTER TABLE knowledge_claim RENAME COLUMN subject_entity_id TO src_entity_id")
                await conn.execute("ALTER TABLE knowledge_claim RENAME COLUMN object_entity_id TO dst_entity_id")
                await conn.execute("ALTER TABLE knowledge_claim RENAME COLUMN object_text TO value_text")
                await conn.execute("ALTER TABLE knowledge_claim RENAME COLUMN object_key TO value_key")
                logfire.info("Renamed legacy subject/object columns to src/dst/value")

            # Fold legacy snake_case predicates to natural phrases (while `predicate` still lives on
            # the value table, i.e. before the relation split). Safe: pre-change predicates never
            # contain spaces, so no rewrite can collide. Idempotent (matches only rows holding '_').
            if await _has_column("knowledge_claim", "predicate"):
                await conn.execute(
                    "UPDATE knowledge_claim SET predicate = replace(predicate, '_', ' ') WHERE strpos(predicate, '_') > 0"
                )

            # Split a per-row recall embedding out into a group-level `knowledge_relation` (one
            # embedding per (src, predicate)). Seeds each relation from one of its claims' existing
            # vectors (DISTINCT ON) — no embeddings API call; the seed self-corrects to the question
            # vector the next time that relation is written. Runs only while the value table still
            # carries an `embedding` column and no relation table exists.
            if await conn.fetchval("SELECT to_regclass('knowledge_relation') IS NULL") and await _has_column(
                "knowledge_claim", "embedding"
            ):
                async with conn.transaction():
                    await conn.execute(
                        f"""
                        CREATE TABLE knowledge_relation (
                            id            BIGSERIAL PRIMARY KEY,
                            src_entity_id BIGINT NOT NULL REFERENCES knowledge_entity(id),
                            predicate     TEXT NOT NULL,
                            embedding     vector({dim}) NOT NULL,
                            updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                            UNIQUE (src_entity_id, predicate)
                        )
                        """
                    )
                    await conn.execute(
                        "INSERT INTO knowledge_relation (src_entity_id, predicate, embedding, created_at, updated_at) "
                        "SELECT DISTINCT ON (src_entity_id, predicate) "
                        "src_entity_id, predicate, embedding, created_at, updated_at "
                        "FROM knowledge_claim ORDER BY src_entity_id, predicate, updated_at DESC"
                    )
                    await conn.execute(
                        "ALTER TABLE knowledge_claim ADD COLUMN relation_id BIGINT REFERENCES knowledge_relation(id)"
                    )
                    await conn.execute(
                        "UPDATE knowledge_claim c SET relation_id = r.id FROM knowledge_relation r "
                        "WHERE r.src_entity_id = c.src_entity_id AND r.predicate = c.predicate"
                    )
                    await conn.execute("ALTER TABLE knowledge_claim ALTER COLUMN relation_id SET NOT NULL")
                    # Drop the superseded per-row embedding + the src/predicate columns (now on the
                    # relation). CASCADE on src_entity_id removes the old UNIQUE(author, src, predicate)
                    # constraint and its index.
                    await conn.execute("ALTER TABLE knowledge_claim DROP COLUMN embedding")
                    await conn.execute("ALTER TABLE knowledge_claim DROP COLUMN src_entity_id CASCADE")
                    await conn.execute("ALTER TABLE knowledge_claim DROP COLUMN predicate")
                    await conn.execute(
                        "ALTER TABLE knowledge_claim "
                        "ADD CONSTRAINT knowledge_claim_author_relation_key UNIQUE (author, relation_id)"
                    )
                logfire.info("Split per-row embedding into knowledge_relation (group-level recall)")

            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_entity (
                    id             BIGSERIAL PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    aliases        TEXT[] NOT NULL DEFAULT '{{}}',
                    embedding      vector({dim}) NOT NULL,
                    merged_into    BIGINT REFERENCES knowledge_entity(id),
                    merged_at      TIMESTAMPTZ,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # The (src, predicate) "question": one recall embedding per group, shared by all of its
            # claims. UNIQUE(src, predicate) also serves the write-path relation lookup.
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_relation (
                    id            BIGSERIAL PRIMARY KEY,
                    src_entity_id BIGINT NOT NULL REFERENCES knowledge_entity(id),
                    predicate     TEXT NOT NULL,
                    embedding     vector({dim}) NOT NULL,
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (src_entity_id, predicate)
                )
                """
            )
            # One per-author value (claim) per relation; no embedding — recall is over the relation.
            # A claim whose value names another entity (dst_entity_id) is a relationship (a graph
            # edge); a literal-valued claim is an attribute.
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_claim (
                    id            BIGSERIAL PRIMARY KEY,
                    relation_id   BIGINT NOT NULL REFERENCES knowledge_relation(id),
                    value_text    TEXT NOT NULL,
                    value_key     TEXT NOT NULL,
                    dst_entity_id BIGINT REFERENCES knowledge_entity(id),
                    author        TEXT NOT NULL,
                    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (author, relation_id)
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS knowledge_entity_embedding_idx "
                "ON knowledge_entity USING hnsw (embedding vector_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS knowledge_entity_name_idx ON knowledge_entity (lower(canonical_name))"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS knowledge_relation_embedding_idx "
                "ON knowledge_relation USING hnsw (embedding vector_cosine_ops)"
            )
            # Agreement is tallied per relation; this serves the agg lookup by relation_id (the
            # UNIQUE(author, relation_id) index leads with author, so it can't).
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS knowledge_claim_relation_idx ON knowledge_claim (relation_id)"
            )
            # Speeds entity merges (repointing relationship claims' destinations) and is traversal-ready.
            await conn.execute("CREATE INDEX IF NOT EXISTS knowledge_claim_dst_idx ON knowledge_claim (dst_entity_id)")

            existing_dim = await conn.fetchval(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'knowledge_relation'::regclass AND attname = 'embedding' AND NOT attisdropped"
            )
            if not skip_dim_check and existing_dim is not None and existing_dim > 0 and existing_dim != dim:
                raise RuntimeError(
                    f"knowledge_relation.embedding has dimension {existing_dim} but embedding_dim={dim}. "
                    "Changing the embedding model requires re-embedding every row (the vectors must "
                    "share one space); run `python -m bot.maintenance reembed` (it re-dimensions and "
                    "repopulates), or drop/migrate the tables before changing embedding_dim."
                )
        finally:
            await conn.close()

        # A single-replica chatbot needs only a couple of connections; keep the pool
        # small so idle connections don't count against the server's max_connections.
        pool = await asyncpg.create_pool(config.postgres_url, min_size=1, max_size=5)
        assert pool is not None
        http = httpx.AsyncClient(timeout=httpx.Timeout(config.http_timeout_seconds))
        store = cls(pool, http, config)
        logfire.info("World-knowledge store ready", embedding_model=config.embedding_model, embedding_dim=dim)
        return store

    async def close(self) -> None:
        await self._pool.close()
        await self._http.aclose()

    def _embed_body(self, inp: Union[str, list[str]]) -> dict:
        """Request body for the embeddings endpoint, adding ``dimensions`` when configured.

        ``dimensions`` truncates a Matryoshka model's native vector to the stored column size; it is
        only sent when ``embedding_dimensions`` is set, so non-MRL models that reject the param are
        unaffected.
        """
        body: dict = {"model": self._embed_model, "input": inp}
        if self._embed_dimensions is not None:
            body["dimensions"] = self._embed_dimensions
        return body

    @logfire.instrument(extract_args=False)
    async def embed(self, text: str) -> list[float]:
        """Embed text via the OpenAI-compatible embeddings endpoint.

        Returns the raw (unnormalized) vector; cosine distance handles normalization.
        """
        resp = await self._http.post(self._embed_url, json=self._embed_body(text), headers=self._embed_headers)
        resp.raise_for_status()
        data = resp.json()
        embedding = data["data"][0]["embedding"]
        return [float(x) for x in embedding]

    @logfire.instrument(extract_args=False)
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts in one request (for bulk re-embedding); order is preserved.

        The OpenAI-compatible endpoint accepts a list ``input`` and returns one item per input;
        we sort by the returned ``index`` defensively before mapping back to the input order.
        """
        if not texts:
            return []
        resp = await self._http.post(self._embed_url, json=self._embed_body(texts), headers=self._embed_headers)
        resp.raise_for_status()
        items = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [[float(x) for x in it["embedding"]] for it in items]

    # --- Write path --------------------------------------------------------------

    async def _match_entity_exact(self, conn: _Conn, name: str) -> Optional[int]:
        """Id of the live entity whose canonical_name or an alias equals ``name`` (case-insensitive).

        The cheap, deterministic, embedding-free, LLM-free half of resolution. Used as the fast
        path in :meth:`add_claim` and as the *only* linker for a claim **value** (link-only: a value
        becomes a relationship claim only when it already names a known entity, never by creating one
        — so free-text values like "born 1990" stay attribute claims).
        """
        row = await conn.fetchval(
            "SELECT id FROM knowledge_entity "
            "WHERE merged_into IS NULL AND (lower(canonical_name) = lower($1) "
            "OR EXISTS (SELECT 1 FROM unnest(aliases) a WHERE lower(a) = lower($1))) LIMIT 1",
            name,
        )
        return int(row) if row is not None else None

    async def _resolve_entity_nearest(self, name: str, name_vec_literal: str) -> Optional[int]:
        """Nearest existing entity for a source name with no exact match, or None ("new").

        The exact name/alias fast path is the caller's; this is the embedding/LLM half. The nearest
        existing entities are offered to ``entity_linker`` (the LLM linker), which returns the id of
        the same-real-world-entity match or None; without a linker, the single nearest entity within
        ``entity_match_threshold`` is linked. Runs with NO DB transaction held — the caller creates
        the entity in its own txn.
        """
        async with self._pool.acquire() as conn:
            candidates = await conn.fetch(
                "SELECT id, canonical_name, embedding <=> $1::vector AS dist FROM knowledge_entity "
                "WHERE merged_into IS NULL ORDER BY dist LIMIT $2",
                name_vec_literal,
                _LINK_CANDIDATES,
            )
        if not candidates:
            return None

        if self.entity_linker is not None:
            # Offer only plausibly-duplicate neighbours (above a broad floor) to the linker.
            offered = [
                (int(c["id"]), c["canonical_name"]) for c in candidates if (1.0 - float(c["dist"])) >= _LINK_FLOOR
            ]
            if not offered:
                return None
            chosen = await self.entity_linker(name, offered)
            # Trust only an id the linker was actually offered.
            return chosen if chosen in {oid for oid, _ in offered} else None

        # No linker (maintenance/headless): deterministic nearest-within-threshold link.
        nearest = candidates[0]
        if (1.0 - float(nearest["dist"])) >= self._config.entity_match_threshold:
            return int(nearest["id"])
        return None

    @logfire.instrument(extract_args=["subject", "predicate", "author"])
    async def add_claim(self, *, subject: str, predicate: str, object_text: str, author: str) -> ClaimWriteResult:
        """Record that ``author`` asserts ``object_text`` for (``subject``, ``predicate``).

        **Per-author upsert:** each user holds at most one current value per relation; re-asserting
        a new value overwrites their own row ("changed my mind"). So ``COUNT(DISTINCT author)`` per
        value — the agreement count tallied at recall — reflects who *currently* asserts it. The
        source is resolved (or created) as an entity; the ``(src, predicate)`` relation is resolved
        (or created); the value is linked to a destination entity only on an exact name/alias match
        (a relationship claim), otherwise it stays a literal grouped by its normalized text (an
        attribute claim).

        Embeddings are **lazy**: the subject name is embedded only when there's no exact entity
        match, and the relation "question" only when the relation doesn't already exist — so a
        repeat assertion about a known subject costs zero embedding calls.
        """
        subject = subject.strip()
        predicate = normalize_predicate(predicate)
        object_text = object_text.strip()
        value_key = normalize_value(object_text)

        # 1. Resolve the source entity. An exact name/alias match needs no embedding; only with no
        #    exact hit do we embed the name (for the nearest-neighbour search / to seed a new entity).
        #    Resolution runs before any txn so a (possibly LLM) link never holds locks.
        async with self._pool.acquire() as conn:
            src_entity_id = await self._match_entity_exact(conn, subject)
        name_vec: Optional[str] = None
        if src_entity_id is None:
            name_vec = _vector_literal(await self.embed(subject))
            src_entity_id = await self._resolve_entity_nearest(subject, name_vec)

        # 2. Resolve the relation. If it already exists we reuse its recall embedding — so the
        #    question is embedded once per group, not once per write.
        relation_id: Optional[int] = None
        if src_entity_id is not None:
            async with self._pool.acquire() as conn:
                relation_id = await conn.fetchval(
                    "SELECT id FROM knowledge_relation WHERE src_entity_id = $1 AND predicate = $2",
                    src_entity_id,
                    predicate,
                )
        question_vec: Optional[str] = None
        if relation_id is None:
            question_vec = _vector_literal(await self.embed(render_relation(subject, predicate)))

        # 3. Write the entity (if new), relation (if new), and the per-author claim in one txn.
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if src_entity_id is None:
                    assert name_vec is not None  # set above whenever there was no exact match
                    created = await conn.fetchval(
                        "INSERT INTO knowledge_entity (canonical_name, embedding) VALUES ($1, $2::vector) RETURNING id",
                        subject,
                        name_vec,
                    )
                    assert created is not None  # INSERT ... RETURNING always yields the id
                    src_entity_id = int(created)

                if relation_id is None:
                    assert question_vec is not None  # set above whenever the relation was missing
                    # ON CONFLICT covers the race where a concurrent write created the relation first.
                    relation_id = int(
                        await conn.fetchval(
                            "INSERT INTO knowledge_relation (src_entity_id, predicate, embedding) "
                            "VALUES ($1, $2, $3::vector) "
                            "ON CONFLICT (src_entity_id, predicate) DO UPDATE SET "
                            "embedding = EXCLUDED.embedding, updated_at = now() "
                            "RETURNING id",
                            src_entity_id,
                            predicate,
                            question_vec,
                        )
                    )

                # Link the value to a destination entity only if it *exactly* names a known one
                # (link-only, no creation, no embedding/LLM) — entity-valued facts become
                # relationship claims that group by identity while free-text values stay attribute
                # claims. A cheap indexed lookup inside the txn.
                dst_entity_id = await self._match_entity_exact(conn, object_text)

                # Upsert this author's value for the relation. xmax <> 0 means the row already
                # existed and we updated it (the author changed their mind).
                row = await conn.fetchrow(
                    "INSERT INTO knowledge_claim (relation_id, value_text, value_key, dst_entity_id, author) "
                    "VALUES ($1, $2, $3, $4, $5) "
                    "ON CONFLICT (author, relation_id) DO UPDATE SET "
                    "value_text = EXCLUDED.value_text, value_key = EXCLUDED.value_key, "
                    "dst_entity_id = EXCLUDED.dst_entity_id, updated_at = now() "
                    "RETURNING id, (xmax <> 0) AS updated",
                    relation_id,
                    object_text,
                    value_key,
                    dst_entity_id,
                    author,
                )
                assert row is not None  # INSERT ... RETURNING always yields a row

        return ClaimWriteResult(
            stored=True,
            claim_id=int(row["id"]),
            updated=bool(row["updated"]),
            subject=subject,
            predicate=predicate,
        )

    async def seconds_since_last_write(self, author: str) -> Optional[float]:
        """Seconds since ``author``'s most recent claim write, or None if never."""
        async with self._pool.acquire() as conn:
            last = await conn.fetchval(
                "SELECT EXTRACT(EPOCH FROM (now() - max(updated_at))) FROM knowledge_claim WHERE author = $1",
                author,
            )
        return float(last) if last is not None else None

    # --- Read path ---------------------------------------------------------------

    @logfire.instrument(extract_args=["query"])
    async def search_claims(self, query: str, k: int) -> list[RecalledClaim]:
        """Recall the most-agreed value for each relevant relation.

        Embedding recall runs over the **relation** vectors — one per ``(src, predicate)`` question,
        not per author — so the ANN candidate pool is naturally group-level (no per-author
        duplicates crowding it). For each candidate relation the **agreement count** —
        ``COUNT(DISTINCT author)`` per value over *all* its live claims, excluding ``self.bot_author``
        — picks the winner via :func:`resolve_conflict` (recency breaks ties), with the losing values
        riding along as ``conflicts``. Results are ordered by recall similarity and capped at ``k``.
        """
        vec = _vector_literal(await self.embed(query))
        max_dist = 1.0 - self._config.global_recall_min_similarity
        # Over-fetch relative to k so approximate ANN still surfaces the true top-k groups.
        candidate_limit = max(k * 4, 20)

        async with self._pool.acquire() as conn:
            candidates = await conn.fetch(
                """
                SELECT r.id AS relation_id, e.canonical_name AS subject, r.predicate,
                       r.embedding <=> $1::vector AS dist
                FROM knowledge_relation r
                JOIN knowledge_entity e ON e.id = r.src_entity_id
                WHERE r.embedding <=> $1::vector <= $2
                ORDER BY dist LIMIT $3
                """,
                vec,
                max_dist,
                candidate_limit,
            )
            if not candidates:
                return []
            # One row per relation already — no per-author collapse needed.
            sim_by_rel: dict[int, float] = {}
            subject_by_rel: dict[int, str] = {}
            predicate_by_rel: dict[int, str] = {}
            for r in candidates:
                rid = int(r["relation_id"])
                sim_by_rel[rid] = 1.0 - float(r["dist"])
                subject_by_rel[rid] = r["subject"]
                predicate_by_rel[rid] = r["predicate"]

            # Tally agreement (distinct authors per value) across ALL live claims of the candidate
            # relations. The bot author (when set) is excluded via ``IS DISTINCT FROM`` — which
            # counts every (non-null) author when ``bot_author`` is None, so maintenance/headless
            # recall is unaffected. Bot-authored claims (from remember_fact) still group and surface
            # (recency/value aggregate over all asserters); they just don't inflate corroboration,
            # so a human asserting the same value always outranks a bot-only one.
            agg = await conn.fetch(
                f"""
                SELECT relation_id,
                       count(DISTINCT author) FILTER (WHERE author IS DISTINCT FROM $2) AS agreed_by,
                       max(updated_at) AS recency,
                       (array_agg(value_text ORDER BY updated_at DESC))[1] AS value_text
                FROM knowledge_claim
                WHERE relation_id = ANY($1)
                GROUP BY relation_id, {_VALUE_GROUP_SQL}
                """,
                list(sim_by_rel.keys()),
                self.bot_author,
            )

        # One entry per distinct value within a relation.
        values_by_rel: dict[int, list[dict]] = {}
        for r in agg:
            values_by_rel.setdefault(int(r["relation_id"]), []).append(
                {"value_text": r["value_text"], "agreed_by": int(r["agreed_by"]), "recency": r["recency"]}
            )

        claims: list[RecalledClaim] = []
        for rid, sim in sim_by_rel.items():
            values = values_by_rel.get(rid)
            if not values:
                continue
            winner = resolve_conflict(values)
            conflicts = [
                ConflictingClaim(value_text=v["value_text"], agreed_by=v["agreed_by"])
                for v in values
                if v is not winner
            ]
            claims.append(
                RecalledClaim(
                    subject=subject_by_rel[rid],
                    predicate=predicate_by_rel[rid],
                    value_text=winner["value_text"],
                    agreed_by=winner["agreed_by"],
                    similarity=sim,
                    conflicts=conflicts,
                )
            )

        claims.sort(key=lambda c: c.similarity, reverse=True)
        return claims[:k]

    # --- Maintenance (M4) --------------------------------------------------------

    async def _merge_entity(self, conn: _Conn, keep: int, dup: int) -> None:
        """Fold entity ``dup`` into ``keep``: merge its relations + repoint claims, union aliases, mark merged.

        The duplicate row is **not deleted** — it is marked ``merged_into = keep`` so it drops out
        of resolution/recall/consolidation but stays auditable. Each of ``dup``'s relations is
        folded onto ``keep``: if ``keep`` has no relation for that predicate the dup relation is
        repointed wholesale; otherwise ``dup``'s claims move onto ``keep``'s relation — dropping any
        that would collide with an existing keep-side claim from the same author
        (``UNIQUE(author, relation_id)``) — and the now-empty dup relation is deleted. Edges whose
        value pointed at ``dup`` (relationship claims) are repointed to ``keep``.
        """
        dup_relations = await conn.fetch("SELECT id, predicate FROM knowledge_relation WHERE src_entity_id = $1", dup)
        for rel in dup_relations:
            keep_rel = await conn.fetchval(
                "SELECT id FROM knowledge_relation WHERE src_entity_id = $1 AND predicate = $2", keep, rel["predicate"]
            )
            if keep_rel is None:
                await conn.execute("UPDATE knowledge_relation SET src_entity_id = $1 WHERE id = $2", keep, rel["id"])
            else:
                # One opinion per author per relation: drop dup-side claims whose author already has a
                # keep-side claim, move the rest onto the keep relation, then drop the empty dup.
                await conn.execute(
                    "DELETE FROM knowledge_claim WHERE relation_id = $1 "
                    "AND author IN (SELECT author FROM knowledge_claim WHERE relation_id = $2)",
                    rel["id"],
                    keep_rel,
                )
                await conn.execute(
                    "UPDATE knowledge_claim SET relation_id = $1 WHERE relation_id = $2", keep_rel, rel["id"]
                )
                await conn.execute("DELETE FROM knowledge_relation WHERE id = $1", rel["id"])
        # Repoint relationship-claim destinations that named the duplicate.
        await conn.execute("UPDATE knowledge_claim SET dst_entity_id = $1 WHERE dst_entity_id = $2", keep, dup)
        krow = await conn.fetchrow("SELECT canonical_name, aliases FROM knowledge_entity WHERE id = $1", keep)
        drow = await conn.fetchrow("SELECT canonical_name, aliases FROM knowledge_entity WHERE id = $1", dup)
        assert krow is not None and drow is not None
        new_aliases = merge_aliases(
            krow["canonical_name"], list(krow["aliases"] or []), drow["canonical_name"], list(drow["aliases"] or [])
        )
        await conn.execute("UPDATE knowledge_entity SET aliases = $1 WHERE id = $2", new_aliases, keep)
        await conn.execute("UPDATE knowledge_entity SET merged_into = $1, merged_at = now() WHERE id = $2", keep, dup)
        logfire.info("Merged duplicate entity", keep_id=keep, dup_id=dup, dup_name=drow["canonical_name"])

    async def _merge_by_name(self, conn: _Conn) -> int:
        """Merge entities with the same :func:`normalize_entity_name` into the lowest-id keeper.

        High precision and embedding-free: it only collapses names that differ by
        formatting/accents/punctuation/plural ("@anemone" / "anemone"). Returns the number
        of duplicates folded away.
        """
        rows = await conn.fetch("SELECT id, canonical_name FROM knowledge_entity WHERE merged_into IS NULL ORDER BY id")
        groups: dict[str, list[int]] = {}
        for r in rows:
            key = normalize_entity_name(r["canonical_name"])
            if key:
                groups.setdefault(key, []).append(int(r["id"]))
        merged = 0
        for ids in groups.values():
            if len(ids) < 2:
                continue
            keep = ids[0]  # lowest id (rows came back id-ascending)
            for dup in ids[1:]:
                await self._merge_entity(conn, keep=keep, dup=dup)
                merged += 1
        return merged

    async def _merge_by_embedding(self, conn: _Conn) -> int:
        """Merge entities within ``entity_merge_threshold`` cosine similarity (conservative).

        Secondary to :meth:`_merge_by_name`; this is the destructive, embedding-based pass,
        so its threshold is deliberately high. Each duplicate is folded into the lowest-id
        keeper. Returns the number folded away.
        """
        entities = await conn.fetch(
            "SELECT id, embedding::text AS emb FROM knowledge_entity WHERE merged_into IS NULL ORDER BY id"
        )
        merged_away: set[int] = set()
        max_dist = 1.0 - self._config.entity_merge_threshold
        merged = 0
        for ent in entities:
            if ent["id"] in merged_away:
                continue
            dups = await conn.fetch(
                "SELECT id FROM knowledge_entity "
                "WHERE merged_into IS NULL AND id <> $1 AND embedding <=> $2::vector <= $3",
                ent["id"],
                ent["emb"],
                max_dist,
            )
            for d in dups:
                if d["id"] <= ent["id"] or d["id"] in merged_away:
                    continue  # keep the lowest id; lower-id matches were handled already
                await self._merge_entity(conn, keep=ent["id"], dup=d["id"])
                merged_away.add(d["id"])
                merged += 1
        return merged

    async def _backfill_relationship_links(self, conn: _Conn) -> int:
        """Link unlinked claim values that now *exactly* name a live entity. Returns the count.

        A claim's destination link is computed write-time-only (:meth:`add_claim`) against the
        entities that exist at that instant and is **never** otherwise revisited — so a claim whose
        value names an entity created (or aliased, via a merge) *after* the claim was written stays
        an attribute claim forever, fragmenting the agreement count against sibling claims that did
        link as relationship claims. This pass heals that staleness using the *same* exact-match rule
        the write path applies (:meth:`_match_entity_exact`: case-insensitive canonical-name/alias
        hit, live entities only, lowest id on ties): no embeddings, no LLM, link-only (never
        creates). Because it is exact match it carries zero overcount risk — it can only fold a stale
        literal back onto the entity the write path would have linked it to. Run last in
        :meth:`consolidate`, after merges have unioned duplicate names into keeper aliases (which
        exposes more exact matches).
        """
        match = (
            "SELECT id FROM knowledge_entity e WHERE e.merged_into IS NULL "
            "AND (lower(e.canonical_name) = lower(c.value_text) "
            "OR EXISTS (SELECT 1 FROM unnest(e.aliases) a WHERE lower(a) = lower(c.value_text))) "
            "ORDER BY e.id LIMIT 1"
        )
        # The EXISTS guard keeps the UPDATE to rows that actually match (so the rowcount reflects
        # real backfills, and a non-matching row is never rewritten to NULL).
        status = await conn.execute(
            f"UPDATE knowledge_claim c SET dst_entity_id = ({match}) WHERE c.dst_entity_id IS NULL AND EXISTS ({match})"
        )
        # asyncpg returns a command tag like "UPDATE 3"; the trailing int is the affected rowcount.
        return int(status.rsplit(" ", 1)[-1]) if status else 0

    async def _merge_by_llm(self) -> int:
        """Heal fragmentation the deterministic passes miss, via the entity linker (LLM).

        For each active entity, offers its near-neighbours (within ``_LINK_FLOOR``) to
        ``entity_linker``; on a confirmed same-entity match the two are merged (lowest id
        kept). Decisions are made with NO transaction held — only each individual merge is
        transactional (and re-checks both entities are still active). No-op without a linker.
        Returns the number folded away.
        """
        if self.entity_linker is None:
            return 0
        async with self._pool.acquire() as conn:
            entities = await conn.fetch(
                "SELECT id, canonical_name, embedding::text AS emb FROM knowledge_entity "
                "WHERE merged_into IS NULL ORDER BY id"
            )
        merged_away: set[int] = set()
        merged = 0
        for ent in entities:
            ent_id = int(ent["id"])
            if ent_id in merged_away:
                continue
            async with self._pool.acquire() as conn:
                neighbors = await conn.fetch(
                    "SELECT id, canonical_name FROM knowledge_entity "
                    "WHERE merged_into IS NULL AND id <> $1 AND embedding <=> $2::vector <= $3 "
                    "ORDER BY embedding <=> $2::vector LIMIT $4",
                    ent_id,
                    ent["emb"],
                    1.0 - _LINK_FLOOR,
                    _LINK_CANDIDATES,
                )
            offered = [(int(n["id"]), n["canonical_name"]) for n in neighbors if int(n["id"]) not in merged_away]
            if not offered:
                continue
            chosen = await self.entity_linker(ent["canonical_name"], offered)
            if chosen not in {oid for oid, _ in offered}:
                continue
            keep, dup = (chosen, ent_id) if chosen < ent_id else (ent_id, chosen)
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    still_active = await conn.fetchval(
                        "SELECT count(*) FROM knowledge_entity WHERE id = ANY($1) AND merged_into IS NULL",
                        [keep, dup],
                    )
                    if still_active == 2:
                        await self._merge_entity(conn, keep=keep, dup=dup)
                        merged += 1
            merged_away.add(dup)
        return merged

    @logfire.instrument()
    async def consolidate(self) -> dict[str, int]:
        """Merge duplicate entities (name, then embedding, then optional LLM), then backfill links.

        Pass 1 (:meth:`_merge_by_name`) collapses entities with an identical normalized name
        key — high precision, no embeddings. Pass 2 (:meth:`_merge_by_embedding`) is a
        conservative embedding fallback at ``entity_merge_threshold``. Pass 3
        (:meth:`_merge_by_llm`, when a linker is wired and ``entity_merge_llm``) heals
        fragmentation the deterministic passes miss. Merges fold relations/claims onto the keeper, so
        agreement counts re-tally automatically on the next recall — no recompute needed.
        Finally :meth:`_backfill_relationship_links` exact-matches any unlinked claim value onto
        a live entity (heals write-time link staleness; run last so it sees the merged aliases).
        """
        # Deterministic passes are atomic; the LLM pass makes decisions outside any
        # transaction (each merge is its own short txn) so no transaction is held across an
        # LLM call.
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                merged_by_name = await self._merge_by_name(conn)
                merged_by_embedding = await self._merge_by_embedding(conn)

        merged_by_llm = await self._merge_by_llm() if self._config.entity_merge_llm else 0

        # Backfill last: merges above may have unioned duplicate names into keeper aliases,
        # exposing more exact value matches than existed at the claims' write time.
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                relationship_links_backfilled = await self._backfill_relationship_links(conn)

        summary = {
            "merged_by_name": merged_by_name,
            "merged_by_embedding": merged_by_embedding,
            "merged_by_llm": merged_by_llm,
            "relationship_links_backfilled": relationship_links_backfilled,
        }
        logfire.info("Consolidation complete", summary=summary)
        return summary

    async def entity_neighbors(self, limit: int = 50, min_similarity: float = 0.5) -> list[EntityNeighbor]:
        """Most-similar entity pairs in the store, for calibrating ``entity_match_threshold``.

        For each entity, finds its single nearest other entity (by cosine), drops pairs
        below ``min_similarity``, dedupes symmetric pairs, and returns them most-similar
        first (capped at ``limit``). The top of this list is where true duplicates and
        genuinely-distinct entities meet — set the threshold just above the most-similar
        pair that should stay separate.
        """
        max_dist = 1.0 - min_similarity
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.id AS a_id, e.canonical_name AS a, n.id AS b_id, n.name AS b, n.dist AS dist
                FROM knowledge_entity e
                CROSS JOIN LATERAL (
                    SELECT o.id, o.canonical_name AS name, o.embedding <=> e.embedding AS dist
                    FROM knowledge_entity o
                    WHERE o.id <> e.id AND o.merged_into IS NULL
                    ORDER BY o.embedding <=> e.embedding
                    LIMIT 1
                ) n
                WHERE e.merged_into IS NULL AND n.dist <= $1
                ORDER BY n.dist
                """,
                max_dist,
            )
        seen: set[tuple[int, int]] = set()
        pairs: list[EntityNeighbor] = []
        for r in rows:
            key = (min(int(r["a_id"]), int(r["b_id"])), max(int(r["a_id"]), int(r["b_id"])))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(EntityNeighbor(name_a=r["a"], name_b=r["b"], similarity=1.0 - float(r["dist"])))
            if len(pairs) >= limit:
                break
        return pairs

    async def stats(self) -> dict[str, object]:
        """Snapshot counts for observability (entities, relations, claims, distinct sources/authors)."""
        async with self._pool.acquire() as conn:
            entities = await conn.fetchval("SELECT count(*) FROM knowledge_entity WHERE merged_into IS NULL")
            merged_entities = await conn.fetchval("SELECT count(*) FROM knowledge_entity WHERE merged_into IS NOT NULL")
            relations = await conn.fetchval("SELECT count(*) FROM knowledge_relation")
            claims = await conn.fetchval("SELECT count(*) FROM knowledge_claim")
            sources = await conn.fetchval("SELECT count(DISTINCT src_entity_id) FROM knowledge_relation")
            authors = await conn.fetchval("SELECT count(DISTINCT author) FROM knowledge_claim")
        return {
            "entities": int(entities or 0),
            "merged_entities": int(merged_entities or 0),
            "relations": int(relations or 0),
            "claims": int(claims or 0),
            "sources": int(sources or 0),
            "authors": int(authors or 0),
        }

    # --- Re-embedding (maintenance) ----------------------------------------------

    # The two text-bearing embedding columns: (table, hnsw index name).
    _EMBEDDING_TABLES = (
        ("knowledge_entity", "knowledge_entity_embedding_idx"),
        ("knowledge_relation", "knowledge_relation_embedding_idx"),
    )

    @logfire.instrument()
    async def reembed_all(self, *, batch_size: int = 64) -> dict[str, int]:
        """Regenerate every entity-name and relation-question embedding with the configured model.

        Use after changing ``embedding_model`` (the stored vectors are from the old model and no
        longer share a space with new queries), or to upgrade migration-seeded relation vectors
        (full-triple seeds) to clean question vectors. Re-embeds in place in batches, so it is
        **resumable** — a re-run simply regenerates again (and, after a dimension change, fills any
        rows a prior interrupted run left ``NULL``).

        If ``config.embedding_dim`` differs from the live column dimension this also re-dimensions
        the columns first (drop the HNSW index, clear + retype the column), repopulates, then
        restores ``NOT NULL`` and the index — so it covers a model swap that changes the vector size.
        Build the store with ``skip_dim_check=True`` (the ``reembed`` CLI does) so the dimension
        guard in :meth:`create` doesn't abort before this runs. Every row (including merged-away
        entities) is re-embedded so the ``NOT NULL`` constraint can be restored afterwards.
        """
        await self._redimension_if_needed()
        entities = await self._reembed_entities(batch_size)
        relations = await self._reembed_relations(batch_size)
        await self._restore_embedding_constraints()
        summary = {"entities": entities, "relations": relations}
        logfire.info("Re-embedding complete", summary=summary)
        return summary

    async def _redimension_if_needed(self) -> None:
        """When the configured dim differs from a column's, drop its index and clear+retype it.

        Leaves the column nullable and all-``NULL``; :meth:`reembed_all` then repopulates it and
        :meth:`_restore_embedding_constraints` restores ``NOT NULL`` + the index. No-op when dims
        already match (so a same-model re-embed never drops indexes or touches the column type).
        """
        dim = self._config.embedding_dim
        async with self._pool.acquire() as conn:
            for table, idx in self._EMBEDDING_TABLES:
                current = await conn.fetchval(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = $1::regclass AND attname = 'embedding' AND NOT attisdropped",
                    table,
                )
                if current is not None and current > 0 and current != dim:
                    logfire.info("Re-dimensioning embedding column", table=table, old_dim=current, new_dim=dim)
                    await conn.execute(f"DROP INDEX IF EXISTS {idx}")
                    await conn.execute(f"ALTER TABLE {table} ALTER COLUMN embedding DROP NOT NULL")
                    await conn.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector({dim}) USING NULL")

    async def _restore_embedding_constraints(self) -> None:
        """Restore ``NOT NULL`` (only when no NULLs remain) and the HNSW index on each column.

        Idempotent: ``SET NOT NULL`` is a no-op when already set, ``CREATE INDEX IF NOT EXISTS``
        when the index already exists — so a same-dim re-embed (which never relaxed either) is
        unaffected, and an interrupted re-dimension is healed on the next full run.
        """
        async with self._pool.acquire() as conn:
            for table, idx in self._EMBEDDING_TABLES:
                nulls = await conn.fetchval(f"SELECT count(*) FROM {table} WHERE embedding IS NULL")
                if not nulls:
                    await conn.execute(f"ALTER TABLE {table} ALTER COLUMN embedding SET NOT NULL")
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {idx} ON {table} USING hnsw (embedding vector_cosine_ops)"
                )

    async def _reembed_entities(self, batch_size: int) -> int:
        """Re-embed every entity's ``canonical_name`` (merged-away rows included). Returns the count."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT id, canonical_name FROM knowledge_entity ORDER BY id")
        return await self._reembed_rows(
            "knowledge_entity", [(int(r["id"]), r["canonical_name"]) for r in rows], batch_size
        )

    async def _reembed_relations(self, batch_size: int) -> int:
        """Re-embed every relation's ``render_relation`` question (current canonical name). Returns the count."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT r.id, e.canonical_name AS subject, r.predicate FROM knowledge_relation r "
                "JOIN knowledge_entity e ON e.id = r.src_entity_id ORDER BY r.id"
            )
        items = [(int(r["id"]), render_relation(r["subject"], r["predicate"])) for r in rows]
        return await self._reembed_rows("knowledge_relation", items, batch_size)

    async def _reembed_rows(self, table: str, items: list[tuple[int, str]], batch_size: int) -> int:
        """Embed ``items`` (id, text) in batches and write each vector back to ``table``. Returns the count."""
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            vectors = await self.embed_batch([text for _, text in chunk])
            params = [(_vector_literal(vec), row_id) for (row_id, _), vec in zip(chunk, vectors)]
            async with self._pool.acquire() as conn:
                await conn.executemany(f"UPDATE {table} SET embedding = $1::vector WHERE id = $2", params)
        return len(items)
