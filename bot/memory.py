"""World-knowledge store for Missbot (Postgres + pgvector), ranked by user agreement.

This module owns the bot's shared, non-user-specific knowledge as a flat set of *claims*
— ``(subject, predicate, object)`` triples, each attributed to the user who asserted it.
The one ranking signal is **agreement**: a value that more *distinct* users independently
assert outranks one a single user asserts.

Model (enforced in code):

1. **One opinion per user.** A claim is keyed ``UNIQUE(author, subject_entity_id,
   predicate)`` and written by upsert, so each user holds at most one current value per
   (subject, predicate); re-asserting overwrites their own row ("changed my mind").
2. **Agreement = distinct authors.** Recall counts ``COUNT(DISTINCT author)`` per object
   value within a (subject, predicate) group; the most-agreed value wins (recency breaks
   ties). The count is computed at read time, never stored as a status.
3. **Stable grouping.** Subjects resolve to entities (write-time linker); objects group by
   :func:`object_group_key` (linked entity, else normalized text) so surface variants
   ("Arch" / "Arch Linux") don't fragment the agreement count.

Deliberately *not* modelled (dropped for simplicity): trust tiers, model-output quarantine,
append-only supersession / bitemporal reads, contradiction/dispute passes, decay, and
source retraction. The embedding model produces *unnormalized* vectors, so all comparisons
use cosine distance (``<=>``).
"""

import asyncio
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

# Write-time entity linking: an injected async classifier that, given a subject name and
# the nearest existing entities, returns the id of the one it's the SAME entity as (or None
# for "new"). Set by the bot wiring; None in maintenance/headless contexts (deterministic
# embedding fallback then applies).
EntityLinker = Callable[[str, list[tuple[int, str]]], Awaitable[Optional[int]]]
# How many nearest entities to offer the linker, and the (broad) similarity floor below
# which a candidate isn't even a plausible duplicate.
_LINK_CANDIDATES = 8
_LINK_FLOOR = 0.4

# Pool-acquired connections are proxies, not bare Connections; helpers accept either.
_Conn = Union[asyncpg.Connection, PoolConnectionProxy]


def normalize_predicate(predicate: str) -> str:
    """Normalize a predicate to a stable snake_case key for grouping/agreement.

    Two claims only group together (and so corroborate each other) when their predicates
    match exactly, so we canonicalize aggressively (lowercase, non-alphanumeric -> ``_``).
    """
    p = re.sub(r"[^a-z0-9]+", "_", (predicate or "").strip().lower()).strip("_")
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


def normalize_object(object_text: str) -> str:
    """Canonicalize a claim's object value into the grouping key for the agreement count.

    Two claims agree (count as the same value) only when their objects share this key, so it
    must fold away the formatting noise two writers can differ on for the *same* value while
    keeping genuinely different values apart. Deliberately **more conservative** than
    :func:`normalize_entity_name`: it lowercases, strips accents (NFKD), and collapses internal
    whitespace runs — but does **not** fold plurals or drop internal punctuation, because object
    values are heterogeneous (versions like ``3.13``, dates, free phrases) where ``3.13`` vs
    ``3 13`` or ``Windows`` vs ``Window`` are distinct, and a wrong merge silently miscounts
    agreement. A folding miss only undercounts agreement (the safe failure); a wrong merge would
    credit agreement to the wrong value. Synonyms ("Manila" vs "City of Manila") are out of scope
    here — that needs entity resolution (object linking via ``object_entity_id``), not string
    normalization.
    """
    decomposed = unicodedata.normalize("NFKD", object_text or "")
    ascii_ish = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_ish.lower()).strip()


# SQL expression yielding a claim's object grouping value: the linked object entity if present,
# else the normalized object text. The ``e``/``k`` prefixes namespace the two so an entity id can
# never collide with a literal value that happens to be numeric. MUST stay in lockstep with
# :func:`object_group_key` (the Python mirror) — the agreement count groups by both.
_OBJECT_GROUP_SQL = "COALESCE('e' || object_entity_id::text, 'k' || object_key)"


def object_group_key(object_entity_id: Optional[int], object_key: str) -> str:
    """Grouping key for the agreement count — the Python mirror of :data:`_OBJECT_GROUP_SQL`.

    Two claims count as the same value (so they agree) when they link to the same object entity,
    or — when neither is linked — when their normalized object text matches. An object is only
    ever *linked* to an entity that already exists (see :meth:`MemoryStore._match_entity_exact`),
    so this never invents entities for free-text values; unlinked objects fall back to ``object_key``.
    """
    if object_entity_id is not None:
        return f"e{object_entity_id}"
    return f"k{object_key}"


def render_claim(subject: str, predicate: str, object_text: str) -> str:
    """Human/embedding rendering of a claim's content (no provenance)."""
    return f"{subject} — {predicate.replace('_', ' ')}: {object_text}"


def resolve_conflict(values: list[dict]) -> dict:
    """Pick the winning value among those sharing a (subject, predicate). Pure / DB-free.

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

    object_text: str
    agreed_by: int


@dataclass
class RecalledClaim:
    """A recalled claim: the most-agreed value for a (subject, predicate), with alternatives."""

    subject: str
    predicate: str
    object_text: str
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
    """Async Postgres-backed store for the world-knowledge claim graph."""

    def __init__(self, pool: asyncpg.Pool, http: httpx.AsyncClient, config: Config):
        self._pool = pool
        self._http = http
        self._config = config
        self._embed_url = f"{str(config.embedding_base_url).rstrip('/')}/embeddings"
        self._embed_model = config.embedding_model
        key = config.embedding_api_key or os.environ.get(config.embedding_api_key_env)
        self._embed_headers = {"Authorization": f"Bearer {key}"} if key else {}
        # Optional write-time entity-linking classifier, injected by the bot wiring.
        self.entity_linker: Optional[EntityLinker] = None

    @classmethod
    async def create(cls, config: Config) -> "MemoryStore":
        """Connect, ensure the schema/extension exist, migrate legacy rows, return store.

        Fails fast if an existing claim-embedding column dimension disagrees with
        ``config.embedding_dim`` — that mismatch silently returns garbage neighbors, so
        it must surface loudly (it means the corpus needs re-embedding).
        """
        assert config.postgres_url, "postgres_url is required to build a MemoryStore"
        dim = config.embedding_dim

        # Create the extension and tables on a plain connection *before* opening the
        # pool, so the `vector` type exists by the time any pooled connection uses it.
        conn = await asyncpg.connect(config.postgres_url)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Fail fast on a pre-existing rich-schema deployment: the minimal store is not
            # column-compatible with the old claims store (no trust_tier/status/supersession),
            # so refuse loudly rather than silently break on the first upsert.
            legacy = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'knowledge_claim' AND column_name = 'trust_tier'"
            )
            if legacy is not None:
                raise RuntimeError(
                    "knowledge_claim has the legacy 'trust_tier' column — this is the old rich claims "
                    "store, which the minimal agreement-ranked store is not column-compatible with. "
                    "Drop the knowledge_* tables (the bot re-learns from the timeline) before starting."
                )
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
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_claim (
                    id                BIGSERIAL PRIMARY KEY,
                    subject_entity_id BIGINT NOT NULL REFERENCES knowledge_entity(id),
                    predicate         TEXT NOT NULL,
                    object_text       TEXT NOT NULL,
                    object_key        TEXT NOT NULL,
                    object_entity_id  BIGINT REFERENCES knowledge_entity(id),
                    author            TEXT NOT NULL,
                    embedding         vector({dim}) NOT NULL,
                    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
                    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (author, subject_entity_id, predicate)
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
                "CREATE INDEX IF NOT EXISTS knowledge_claim_embedding_idx "
                "ON knowledge_claim USING hnsw (embedding vector_cosine_ops)"
            )
            # Agreement is tallied per (subject, predicate); the UNIQUE(author, subject, predicate)
            # index also serves author-prefix lookups (the write cooldown).
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS knowledge_claim_sp_idx ON knowledge_claim (subject_entity_id, predicate)"
            )

            existing_dim = await conn.fetchval(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'knowledge_claim'::regclass AND attname = 'embedding' AND NOT attisdropped"
            )
            if existing_dim is not None and existing_dim > 0 and existing_dim != dim:
                raise RuntimeError(
                    f"knowledge_claim.embedding has dimension {existing_dim} but embedding_dim={dim}. "
                    "Changing the embedding model requires re-embedding every row (the vectors must "
                    "share one space); drop/migrate the table before changing embedding_dim."
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

    @logfire.instrument(extract_args=False)
    async def embed(self, text: str) -> list[float]:
        """Embed text via the OpenAI-compatible embeddings endpoint.

        Returns the raw (unnormalized) vector; cosine distance handles normalization.
        """
        resp = await self._http.post(
            self._embed_url,
            json={"model": self._embed_model, "input": text},
            headers=self._embed_headers,
        )
        resp.raise_for_status()
        data = resp.json()
        embedding = data["data"][0]["embedding"]
        return [float(x) for x in embedding]

    # --- Write path --------------------------------------------------------------

    async def _match_entity_exact(self, conn: _Conn, name: str) -> Optional[int]:
        """Id of the live entity whose canonical_name or an alias equals ``name`` (case-insensitive).

        The cheap, deterministic, embedding-free, LLM-free half of resolution. Used as the fast
        path in :meth:`_resolve_entity` and as the *only* linker for claim **objects** (link-only:
        an object becomes an entity edge only when it already names a known entity, never by
        creating one — so free-text objects like "born 1990" never pollute the entity table).
        """
        row = await conn.fetchval(
            "SELECT id FROM knowledge_entity "
            "WHERE merged_into IS NULL AND (lower(canonical_name) = lower($1) "
            "OR EXISTS (SELECT 1 FROM unnest(aliases) a WHERE lower(a) = lower($1))) LIMIT 1",
            name,
        )
        return int(row) if row is not None else None

    async def _resolve_entity(self, name: str, name_vec_literal: str) -> Optional[int]:
        """Decide which existing entity a subject maps to, or None to create a new one.

        Exact (case-insensitive) name/alias match wins immediately. Otherwise the nearest
        existing entities are offered to ``entity_linker`` (the LLM linker), which returns
        the id of the same-real-world-entity match or None; without a linker, the single
        nearest entity within ``entity_match_threshold`` is linked. The (possibly LLM) call
        is made with NO DB transaction held — the caller creates the entity in its own txn.
        """
        async with self._pool.acquire() as conn:
            exact = await self._match_entity_exact(conn, name)
            if exact is not None:
                return exact
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

        **Per-author upsert:** each user holds at most one current value per (subject,
        predicate); re-asserting a new value overwrites their own row ("changed my mind"). So
        ``COUNT(DISTINCT author)`` per value — the agreement count tallied at recall — reflects
        who *currently* asserts it. The subject is resolved (or created) as an entity; the object
        is linked to an existing entity only on an exact name/alias match, otherwise it stays a
        literal grouped by its normalized text.
        """
        subject = subject.strip()
        predicate = normalize_predicate(predicate)
        object_text = object_text.strip()
        object_key = normalize_object(object_text)

        # Independent embeddings — run them concurrently (every claim write pays both).
        name_emb, claim_emb = await asyncio.gather(
            self.embed(subject), self.embed(render_claim(subject, predicate, object_text))
        )
        name_vec = _vector_literal(name_emb)
        claim_vec = _vector_literal(claim_emb)

        # Resolve the subject entity BEFORE opening the transaction, so the (possibly LLM)
        # linking decision never holds a DB transaction/locks open.
        resolved_entity_id = await self._resolve_entity(subject, name_vec)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if resolved_entity_id is not None:
                    subject_entity_id = resolved_entity_id
                else:
                    created = await conn.fetchval(
                        "INSERT INTO knowledge_entity (canonical_name, embedding) VALUES ($1, $2::vector) RETURNING id",
                        subject,
                        name_vec,
                    )
                    assert created is not None  # INSERT ... RETURNING always yields the id
                    subject_entity_id = int(created)

                # Link the object to an entity only if it *exactly* names a known one (link-only,
                # no creation, no embedding/LLM) — entity-valued objects group by identity while
                # free-text objects stay literal. A cheap indexed lookup inside the txn.
                object_entity_id = await self._match_entity_exact(conn, object_text)

                # Upsert this author's value for (subject, predicate). xmax <> 0 means the row
                # already existed and we updated it (the author changed their mind).
                row = await conn.fetchrow(
                    "INSERT INTO knowledge_claim "
                    "(subject_entity_id, predicate, object_text, object_key, object_entity_id, author, embedding) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7::vector) "
                    "ON CONFLICT (author, subject_entity_id, predicate) DO UPDATE SET "
                    "object_text = EXCLUDED.object_text, object_key = EXCLUDED.object_key, "
                    "object_entity_id = EXCLUDED.object_entity_id, embedding = EXCLUDED.embedding, "
                    "updated_at = now() "
                    "RETURNING id, (xmax <> 0) AS updated",
                    subject_entity_id,
                    predicate,
                    object_text,
                    object_key,
                    object_entity_id,
                    author,
                    claim_vec,
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
        """Recall the most-agreed value for each relevant (subject, predicate).

        Embedding recall over claim text surfaces candidate (subject, predicate) groups; for
        each, the **agreement count** — ``COUNT(DISTINCT author)`` per value over *all* live
        claims for that group (not just the candidate pool, which would undercount) — picks the
        winner via :func:`resolve_conflict` (recency breaks ties), with the losing values riding
        along as ``conflicts``. Results are ordered by recall similarity and capped at ``k``.
        """
        vec = _vector_literal(await self.embed(query))
        max_dist = 1.0 - self._config.global_recall_min_similarity
        # Pull a wider candidate pool than k so each relevant group's nearest member surfaces.
        candidate_limit = max(k * 4, 20)

        async with self._pool.acquire() as conn:
            candidates = await conn.fetch(
                """
                SELECT c.subject_entity_id, e.canonical_name AS subject, c.predicate,
                       c.embedding <=> $1::vector AS dist
                FROM knowledge_claim c
                JOIN knowledge_entity e ON e.id = c.subject_entity_id
                WHERE c.embedding <=> $1::vector <= $2
                ORDER BY dist LIMIT $3
                """,
                vec,
                max_dist,
                candidate_limit,
            )
            if not candidates:
                return []
            # Best similarity + display name per candidate (subject, predicate) group.
            best_sim: dict[tuple[int, str], float] = {}
            subject_name: dict[int, str] = {}
            for r in candidates:
                sid = int(r["subject_entity_id"])
                subject_name[sid] = r["subject"]
                gkey = (sid, r["predicate"])
                sim = 1.0 - float(r["dist"])
                if sim > best_sim.get(gkey, -1.0):
                    best_sim[gkey] = sim

            # Tally agreement (distinct authors per value) across ALL live claims of the
            # candidate subjects — the candidate pool alone would undercount agreement.
            agg = await conn.fetch(
                f"""
                SELECT subject_entity_id, predicate,
                       count(DISTINCT author) AS agreed_by,
                       max(updated_at) AS recency,
                       (array_agg(object_text ORDER BY updated_at DESC))[1] AS object_text
                FROM knowledge_claim
                WHERE subject_entity_id = ANY($1)
                GROUP BY subject_entity_id, predicate, {_OBJECT_GROUP_SQL}
                """,
                list({sid for sid, _ in best_sim}),
            )

        # One entry per distinct value within a (subject, predicate) group.
        values_by_group: dict[tuple[int, str], list[dict]] = {}
        for r in agg:
            gkey = (int(r["subject_entity_id"]), r["predicate"])
            values_by_group.setdefault(gkey, []).append(
                {"object_text": r["object_text"], "agreed_by": int(r["agreed_by"]), "recency": r["recency"]}
            )

        claims: list[RecalledClaim] = []
        for gkey, sim in best_sim.items():
            values = values_by_group.get(gkey)
            if not values:
                continue
            winner = resolve_conflict(values)
            conflicts = [
                ConflictingClaim(object_text=v["object_text"], agreed_by=v["agreed_by"])
                for v in values
                if v is not winner
            ]
            claims.append(
                RecalledClaim(
                    subject=subject_name[gkey[0]],
                    predicate=gkey[1],
                    object_text=winner["object_text"],
                    agreed_by=winner["agreed_by"],
                    similarity=sim,
                    conflicts=conflicts,
                )
            )

        claims.sort(key=lambda c: c.similarity, reverse=True)
        return claims[:k]

    # --- Maintenance (M4) --------------------------------------------------------

    async def _merge_entity(self, conn: _Conn, keep: int, dup: int) -> None:
        """Fold entity ``dup`` into ``keep``: repoint its claims, union aliases, soft-mark it merged.

        The duplicate row is **not deleted** — it is marked ``merged_into = keep`` so it drops out
        of resolution/recall/consolidation but stays auditable. Claims pointing at ``dup`` (as
        subject or object) are repointed to ``keep``. If an author asserted the same predicate
        under both names, the dup-side row is dropped first (the ``UNIQUE(author, subject,
        predicate)`` constraint allows one opinion per author per group), keeping the keep-side value.
        """
        await conn.execute(
            "DELETE FROM knowledge_claim WHERE subject_entity_id = $1 "
            "AND (author, predicate) IN (SELECT author, predicate FROM knowledge_claim WHERE subject_entity_id = $2)",
            dup,
            keep,
        )
        await conn.execute("UPDATE knowledge_claim SET subject_entity_id = $1 WHERE subject_entity_id = $2", keep, dup)
        await conn.execute("UPDATE knowledge_claim SET object_entity_id = $1 WHERE object_entity_id = $2", keep, dup)
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
        """Merge duplicate entities: a name pass, then embedding, then an optional LLM pass.

        Pass 1 (:meth:`_merge_by_name`) collapses entities with an identical normalized name
        key — high precision, no embeddings. Pass 2 (:meth:`_merge_by_embedding`) is a
        conservative embedding fallback at ``entity_merge_threshold``. Pass 3
        (:meth:`_merge_by_llm`, when a linker is wired and ``entity_merge_llm``) heals
        fragmentation the deterministic passes miss. Merges repoint claims to the keeper, so
        agreement counts re-tally automatically on the next recall — no recompute needed.
        """
        # Deterministic passes are atomic; the LLM pass makes decisions outside any
        # transaction (each merge is its own short txn) so no transaction is held across an
        # LLM call.
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                merged_by_name = await self._merge_by_name(conn)
                merged_by_embedding = await self._merge_by_embedding(conn)

        merged_by_llm = await self._merge_by_llm() if self._config.entity_merge_llm else 0

        summary = {
            "merged_by_name": merged_by_name,
            "merged_by_embedding": merged_by_embedding,
            "merged_by_llm": merged_by_llm,
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
        """Snapshot counts for observability (entities, claims, distinct subjects/authors)."""
        async with self._pool.acquire() as conn:
            entities = await conn.fetchval("SELECT count(*) FROM knowledge_entity WHERE merged_into IS NULL")
            merged_entities = await conn.fetchval("SELECT count(*) FROM knowledge_entity WHERE merged_into IS NOT NULL")
            claims = await conn.fetchval("SELECT count(*) FROM knowledge_claim")
            subjects = await conn.fetchval("SELECT count(DISTINCT subject_entity_id) FROM knowledge_claim")
            authors = await conn.fetchval("SELECT count(DISTINCT author) FROM knowledge_claim")
        return {
            "entities": int(entities or 0),
            "merged_entities": int(merged_entities or 0),
            "claims": int(claims or 0),
            "subjects": int(subjects or 0),
            "authors": int(authors or 0),
        }
