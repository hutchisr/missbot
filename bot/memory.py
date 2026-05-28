"""Evolving world-knowledge store for Missbot (Postgres + pgvector).

This module owns the bot's shared, non-user-specific knowledge. It does **not** store
bare facts: every row is a *claim* bound to a source and a time ("source S asserted X
at time T"), so the model and callers always see who said something, when, and how
well corroborated it is. Nothing here is treated as an oracle.

Safety core (enforced in code, not convention):

1. **No bare facts.** Every claim carries a ``source_id``, ``trust_tier``, and times.
2. **Model output is quarantined.** LLM-generated claims enter at ``model_quarantine``
   (the lowest tier) and can never be auto-promoted to ``believed`` — that blocks the
   confabulation-laundering loop.
3. **Append-only.** Updates insert a newer claim and set ``superseded_by`` on the old
   one; we never destructively overwrite a value.
4. **Promotion requires corroboration.** ``asserted`` -> ``believed`` only when
   ``>= corroboration_threshold`` *independent* sources of tier ``>= secondary`` agree.
5. **Conflict resolution is read-time policy** (see :func:`resolve_conflict`), not a
   write-time guess; all conflicting claims are kept.
6. **Provenance travels with the answer.** Recall returns claims *with* source, tier,
   recency, confidence, and corroboration count.
7. **Volatile facts expire.** Claims carry a volatility; stale volatile claims are
   flagged on recall so the model re-verifies rather than trusting them.
8. **Retraction is one query.** :meth:`MemoryStore.retract_source` tombstones every
   claim from a compromised source and recomputes corroboration everywhere.

The embedding model produces *unnormalized* vectors, so all comparisons use cosine
distance (``<=>``).
"""

import asyncio
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union

import asyncpg
import httpx
import logfire
from asyncpg.pool import PoolConnectionProxy

from .models import Config

# Pool-acquired connections are proxies, not bare Connections; helpers accept either.
_Conn = Union[asyncpg.Connection, PoolConnectionProxy]

# --- Trust tiers -----------------------------------------------------------------
# Higher rank = more trusted. Only claims at tier >= ``secondary`` can be corroborated
# into ``believed``; ``model_quarantine`` (LLM output) and ``user`` claims never can.
TRUST_TIERS: dict[str, int] = {
    "model_quarantine": 0,
    "user": 1,
    "secondary": 2,
    "primary": 3,
}
# Tiers eligible for corroboration-based promotion to ``believed`` (rank >= secondary).
PROMOTABLE_TIERS: tuple[str, ...] = tuple(t for t, r in TRUST_TIERS.items() if r >= TRUST_TIERS["secondary"])

# Read-time conflict ranking: a believed claim beats an asserted one, etc.
STATUS_RANK: dict[str, int] = {"retracted": -1, "disputed": 0, "asserted": 1, "believed": 2}

VOLATILITIES: frozenset[str] = frozenset({"stable", "slow", "volatile"})
SOURCE_KINDS: frozenset[str] = frozenset({"web", "doc", "user", "model"})


def tier_rank(tier: str) -> int:
    """Trust rank for a tier name (unknown tiers rank lowest)."""
    return TRUST_TIERS.get(tier, 0)


def normalize_predicate(predicate: str) -> str:
    """Normalize a predicate to a stable snake_case key for dedup/corroboration.

    Two claims only corroborate or supersede each other when their predicates match
    exactly, so we canonicalize aggressively (lowercase, non-alphanumeric -> ``_``).
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


def render_claim(subject: str, predicate: str, object_text: str) -> str:
    """Human/embedding rendering of a claim's content (no provenance)."""
    return f"{subject} — {predicate.replace('_', ' ')}: {object_text}"


def is_stale(volatility: str, reference_time: Optional[datetime], now: datetime, ttl_seconds: int) -> bool:
    """Whether a volatile claim is past its TTL and should be re-verified.

    Only ``volatile`` claims expire; ``stable``/``slow`` never go stale here.
    """
    if volatility != "volatile" or reference_time is None:
        return False
    return (now - reference_time).total_seconds() > ttl_seconds


def _conflict_sort_key(claim: dict) -> tuple[int, int, datetime]:
    """Sort key for read-time conflict resolution.

    Prefer (1) higher status (believed > asserted > disputed), then (2) higher trust
    tier, then (3) the most recent valid_from (falling back to recorded_at).
    """
    ref = claim.get("valid_from") or claim["recorded_at"]
    return (STATUS_RANK.get(claim["status"], 0), tier_rank(claim["trust_tier"]), ref)


def resolve_conflict(claims: list[dict]) -> dict:
    """Pick the winning claim from a set sharing the same subject+predicate.

    Pure policy (no DB): believed > asserted, then trust tier, then recency. The full
    set is preserved by the caller so conflicts surface with provenance.
    """
    return max(claims, key=_conflict_sort_key)


def rank_dispute_values(claims: list[dict]) -> str:
    """Pick the winning object value among disputed claims (autonomous resolution).

    Pure policy (no DB): group by object value and rank by (1) number of independent
    tier-≥``secondary`` sources, then (2) the highest trust tier present, then (3) recency
    (``valid_from`` else ``recorded_at``). Recency is the final tiebreaker so a resolution
    always exists. Returns the winning value's original-cased text.
    """
    by_value: dict[str, dict] = {}
    for c in claims:
        key = c["object_text"].strip().lower()
        v = by_value.setdefault(key, {"object_text": c["object_text"], "sources": set(), "tier": 0, "recency": None})
        if tier_rank(c["trust_tier"]) >= TRUST_TIERS["secondary"]:
            v["sources"].add(c["source_id"])
        v["tier"] = max(v["tier"], tier_rank(c["trust_tier"]))
        ref = c.get("valid_from") or c["recorded_at"]
        v["recency"] = ref if v["recency"] is None else max(v["recency"], ref)
    best = max(by_value.values(), key=lambda v: (len(v["sources"]), v["tier"], v["recency"]))
    return best["object_text"]


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
    """An alternative value that disagrees with a recalled claim's winner."""

    object_text: str
    source_name: str
    source_kind: str
    trust_tier: str
    status: str


@dataclass
class RecalledClaim:
    """A recalled claim with full provenance (what the read path returns)."""

    subject: str
    predicate: str
    object_text: str
    status: str
    trust_tier: str
    source_name: str
    source_kind: str
    confidence: float
    corroboration_count: int
    volatility: str
    recorded_at: datetime
    valid_from: Optional[datetime]
    similarity: float
    stale: bool
    conflicts: list[ConflictingClaim]


@dataclass
class ClaimWriteResult:
    """Outcome of an :meth:`MemoryStore.add_claim` call."""

    stored: bool
    claim_id: Optional[int]
    status: str
    promoted: bool
    duplicate: bool
    superseded_claim_id: Optional[int]
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
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_source (
                    id                 BIGSERIAL PRIMARY KEY,
                    name               TEXT NOT NULL,
                    kind               TEXT NOT NULL,
                    default_trust_tier TEXT NOT NULL,
                    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (name, kind)
                )
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_entity (
                    id             BIGSERIAL PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    aliases        TEXT[] NOT NULL DEFAULT '{{}}',
                    type           TEXT,
                    embedding      vector({dim}) NOT NULL,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_claim (
                    id                  BIGSERIAL PRIMARY KEY,
                    subject_entity_id   BIGINT NOT NULL REFERENCES knowledge_entity(id),
                    predicate           TEXT NOT NULL,
                    object_text         TEXT NOT NULL,
                    object_entity_id    BIGINT REFERENCES knowledge_entity(id),
                    source_id           BIGINT NOT NULL REFERENCES knowledge_source(id),
                    trust_tier          TEXT NOT NULL,
                    confidence          REAL NOT NULL DEFAULT 0.5,
                    status              TEXT NOT NULL DEFAULT 'asserted',
                    valid_from          TIMESTAMPTZ,
                    valid_to            TIMESTAMPTZ,
                    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
                    superseded_by       BIGINT REFERENCES knowledge_claim(id),
                    superseded_at       TIMESTAMPTZ,
                    retracted_at        TIMESTAMPTZ,
                    corroboration_count INTEGER NOT NULL DEFAULT 0,
                    volatility          TEXT NOT NULL DEFAULT 'stable',
                    embedding           vector({dim}) NOT NULL,
                    author              TEXT,
                    source_note_id      TEXT,
                    last_recalled_at    TIMESTAMPTZ,
                    disputed_at         TIMESTAMPTZ,
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # Columns added after the initial schema — ALTER for already-deployed databases
            # (CREATE TABLE IF NOT EXISTS won't add them to an existing table).
            await conn.execute("ALTER TABLE knowledge_claim ADD COLUMN IF NOT EXISTS last_recalled_at TIMESTAMPTZ")
            await conn.execute("ALTER TABLE knowledge_claim ADD COLUMN IF NOT EXISTS disputed_at TIMESTAMPTZ")
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
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS knowledge_claim_sp_idx ON knowledge_claim (subject_entity_id, predicate)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS knowledge_claim_author_idx ON knowledge_claim (author)")
            await conn.execute("CREATE INDEX IF NOT EXISTS knowledge_claim_source_idx ON knowledge_claim (source_id)")

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
        await store._migrate_legacy_global_memory()
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

    async def _get_or_create_source(self, conn: _Conn, name: str, kind: str, tier: str) -> int:
        """Return the id of a source, creating it (with ``tier``) if new.

        The trust tier is fixed at creation: a source's tier never silently changes on
        a later write (that would let an attacker upgrade their own trust by re-asserting).
        """
        row = await conn.fetchval(
            "INSERT INTO knowledge_source (name, kind, default_trust_tier) VALUES ($1, $2, $3) "
            "ON CONFLICT (name, kind) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            name,
            kind,
            tier,
        )
        assert row is not None  # INSERT ... RETURNING always yields the id
        return int(row)

    async def _resolve_entity(self, conn: _Conn, name: str, name_vec_literal: str) -> int:
        """Resolve a subject name to an entity id, linking or creating as needed.

        Exact (case-insensitive) name/alias match wins; otherwise the nearest entity
        within ``entity_match_threshold`` cosine similarity is linked; otherwise a new
        entity is created.
        """
        exact = await conn.fetchval(
            "SELECT id FROM knowledge_entity "
            "WHERE lower(canonical_name) = lower($1) "
            "OR EXISTS (SELECT 1 FROM unnest(aliases) a WHERE lower(a) = lower($1)) LIMIT 1",
            name,
        )
        if exact is not None:
            return int(exact)

        nearest = await conn.fetchrow(
            "SELECT id, embedding <=> $1::vector AS dist FROM knowledge_entity ORDER BY dist LIMIT 1",
            name_vec_literal,
        )
        if nearest is not None and (1.0 - float(nearest["dist"])) >= self._config.entity_match_threshold:
            return int(nearest["id"])

        created = await conn.fetchval(
            "INSERT INTO knowledge_entity (canonical_name, embedding) VALUES ($1, $2::vector) RETURNING id",
            name,
            name_vec_literal,
        )
        assert created is not None  # INSERT ... RETURNING always yields the id
        return int(created)

    @logfire.instrument(extract_args=["subject", "predicate", "trust_tier", "author", "source_note_id"])
    async def add_claim(
        self,
        *,
        subject: str,
        predicate: str,
        object_text: str,
        source_name: str,
        source_kind: str,
        trust_tier: str,
        author: Optional[str] = None,
        source_note_id: Optional[str] = None,
        volatility: str = "stable",
        confidence: float = 0.5,
        valid_from: Optional[datetime] = None,
    ) -> ClaimWriteResult:
        """Insert a claim with provenance; supersede stale same-source values; corroborate.

        - LLM-sourced claims must pass ``trust_tier='model_quarantine'`` and can never be
          promoted to ``believed`` (they are not in ``PROMOTABLE_TIERS``).
        - A new value for the same subject+predicate from the *same source* supersedes that
          source's prior value (append-only update).
        - An identical/near-identical value from the same source is skipped as a duplicate.
        - After insert, the matching subject+predicate+object group is re-evaluated for
          corroboration-based promotion across *independent* tier-``>= secondary`` sources.
        """
        subject = subject.strip()
        predicate = normalize_predicate(predicate)
        object_text = object_text.strip()
        if trust_tier not in TRUST_TIERS:
            raise ValueError(f"unknown trust_tier {trust_tier!r}")
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"unknown source_kind {source_kind!r}")
        if volatility not in VOLATILITIES:
            volatility = "stable"

        # Independent embeddings — run them concurrently (every claim write pays both).
        name_emb, claim_emb = await asyncio.gather(
            self.embed(subject), self.embed(render_claim(subject, predicate, object_text))
        )
        name_vec = _vector_literal(name_emb)
        claim_vec = _vector_literal(claim_emb)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                source_id = await self._get_or_create_source(conn, source_name, source_kind, trust_tier)
                subject_entity_id = await self._resolve_entity(conn, subject, name_vec)

                # Existing live claims from this same source for this subject+predicate.
                existing = await conn.fetch(
                    "SELECT id, object_text, embedding <=> $4::vector AS dist FROM knowledge_claim "
                    "WHERE source_id = $1 AND subject_entity_id = $2 AND predicate = $3 "
                    "AND retracted_at IS NULL AND superseded_by IS NULL",
                    source_id,
                    subject_entity_id,
                    predicate,
                    claim_vec,
                )
                supersede_ids: list[int] = []
                for row in existing:
                    same_value = row["object_text"].strip().lower() == object_text.lower()
                    near_dup = (1.0 - float(row["dist"])) >= self._config.global_dedup_threshold
                    if same_value or near_dup:
                        logfire.info("Skipping duplicate claim", subject=subject, predicate=predicate)
                        return ClaimWriteResult(
                            stored=False,
                            claim_id=int(row["id"]),
                            status="asserted",
                            promoted=False,
                            duplicate=True,
                            superseded_claim_id=None,
                            subject=subject,
                            predicate=predicate,
                        )
                    supersede_ids.append(int(row["id"]))

                claim_id = int(
                    await conn.fetchval(
                        "INSERT INTO knowledge_claim "
                        "(subject_entity_id, predicate, object_text, source_id, trust_tier, confidence, "
                        " status, valid_from, volatility, embedding, author, source_note_id) "
                        "VALUES ($1, $2, $3, $4, $5, $6, 'asserted', $7, $8, $9::vector, $10, $11) RETURNING id",
                        subject_entity_id,
                        predicate,
                        object_text,
                        source_id,
                        trust_tier,
                        float(confidence),
                        valid_from,
                        volatility,
                        claim_vec,
                        author,
                        source_note_id,
                    )
                )

                superseded_claim_id: Optional[int] = None
                if supersede_ids:
                    await conn.execute(
                        "UPDATE knowledge_claim SET superseded_by = $1, superseded_at = now() WHERE id = ANY($2)",
                        claim_id,
                        supersede_ids,
                    )
                    superseded_claim_id = supersede_ids[-1]
                    # A superseded value may have been holding up a corroboration group.
                    for old in existing:
                        await self._recompute_corroboration(conn, subject_entity_id, predicate, old["object_text"])

                promoted = await self._recompute_corroboration(conn, subject_entity_id, predicate, object_text)

        if promoted:
            logfire.info(
                "Claim promoted to believed (corroborated)",
                subject=subject,
                predicate=predicate,
                trust_tier=trust_tier,
            )
        status = "believed" if promoted else "asserted"
        return ClaimWriteResult(
            stored=True,
            claim_id=claim_id,
            status=status,
            promoted=promoted,
            duplicate=False,
            superseded_claim_id=superseded_claim_id,
            subject=subject,
            predicate=predicate,
        )

    async def _recompute_corroboration(
        self, conn: _Conn, subject_entity_id: int, predicate: str, object_text: str
    ) -> bool:
        """Recount independent tier->=secondary sources for a value and (de)promote.

        A value asserted by ``>= corroboration_threshold`` *distinct* promotable-tier
        sources becomes ``believed``; if it drops below that (e.g. after retraction) it
        falls back to ``asserted``. ``model_quarantine``/``user`` claims are never touched,
        so an LLM-sourced claim can never reach ``believed``. Returns True if the value
        is believed afterwards.
        """
        count = await conn.fetchval(
            "SELECT count(DISTINCT source_id) FROM knowledge_claim "
            "WHERE subject_entity_id = $1 AND predicate = $2 AND lower(object_text) = lower($3) "
            "AND retracted_at IS NULL AND superseded_by IS NULL AND status <> 'disputed' "
            "AND trust_tier = ANY($4)",
            subject_entity_id,
            predicate,
            object_text,
            list(PROMOTABLE_TIERS),
        )
        n = int(count or 0)
        promote = n >= self._config.corroboration_threshold
        await conn.execute(
            "UPDATE knowledge_claim SET corroboration_count = $4, "
            "status = CASE WHEN $5 THEN 'believed' ELSE 'asserted' END "
            "WHERE subject_entity_id = $1 AND predicate = $2 AND lower(object_text) = lower($3) "
            "AND retracted_at IS NULL AND superseded_by IS NULL AND trust_tier = ANY($6) "
            "AND status IN ('asserted', 'believed')",
            subject_entity_id,
            predicate,
            object_text,
            n,
            promote,
            list(PROMOTABLE_TIERS),
        )
        return promote

    async def seconds_since_last_write(self, author: str) -> Optional[float]:
        """Seconds since ``author``'s most recent claim write, or None if never."""
        async with self._pool.acquire() as conn:
            last = await conn.fetchval(
                "SELECT EXTRACT(EPOCH FROM (now() - max(recorded_at))) FROM knowledge_claim WHERE author = $1",
                author,
            )
        return float(last) if last is not None else None

    @logfire.instrument(extract_args=["source_name", "source_kind"])
    async def retract_source(self, source_name: str, source_kind: Optional[str] = None) -> int:
        """Tombstone every claim from a source and recompute corroboration everywhere.

        Killing a compromised source removes its influence in one operation (invariant 8):
        its claims are marked ``retracted`` and any ``believed`` value that depended on it
        is re-evaluated (and demoted if it no longer clears the threshold). Returns the
        number of claims retracted.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                if source_kind is not None:
                    source_ids = [
                        int(r["id"])
                        for r in await conn.fetch(
                            "SELECT id FROM knowledge_source WHERE name = $1 AND kind = $2", source_name, source_kind
                        )
                    ]
                else:
                    source_ids = [
                        int(r["id"])
                        for r in await conn.fetch("SELECT id FROM knowledge_source WHERE name = $1", source_name)
                    ]
                if not source_ids:
                    return 0
                affected = await conn.fetch(
                    "UPDATE knowledge_claim SET status = 'retracted', retracted_at = now() "
                    "WHERE source_id = ANY($1) AND retracted_at IS NULL "
                    "RETURNING subject_entity_id, predicate, object_text",
                    source_ids,
                )
                seen: set[tuple[int, str, str]] = set()
                for row in affected:
                    key = (int(row["subject_entity_id"]), row["predicate"], row["object_text"].lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    await self._recompute_corroboration(
                        conn, int(row["subject_entity_id"]), row["predicate"], row["object_text"]
                    )
        return len(affected)

    # --- Read path ---------------------------------------------------------------

    @logfire.instrument(extract_args=["query"])
    async def search_claims(self, query: str, k: int, as_of: Optional[datetime] = None) -> list[RecalledClaim]:
        """Recall claims relevant to a query, conflict-resolved and provenance-attached.

        Embedding recall produces a candidate set; candidates are grouped by
        subject+predicate, the winner of each group is chosen by :func:`resolve_conflict`,
        and disagreeing alternatives ride along as ``conflicts``. Retracted and superseded
        claims are excluded from the default (current) view. Pass ``as_of`` to recover the
        value believed as of a past instant (bitemporal read). Volatile winners past their
        TTL are flagged ``stale``.
        """
        vec = _vector_literal(await self.embed(query))
        min_sim = self._config.global_recall_min_similarity
        max_dist = 1.0 - min_sim
        # Pull a wider candidate pool than k so conflict groups are complete before we
        # collapse each to a single winner.
        candidate_limit = max(k * 4, 20)

        if as_of is None:
            visibility = "c.retracted_at IS NULL AND c.superseded_by IS NULL"
            params: list = [vec, max_dist, candidate_limit]
        else:
            visibility = (
                "c.recorded_at <= $4 "
                "AND (c.retracted_at IS NULL OR c.retracted_at > $4) "
                "AND (c.superseded_at IS NULL OR c.superseded_at > $4)"
            )
            params = [vec, max_dist, candidate_limit, as_of]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT c.id, e.canonical_name AS subject, c.predicate, c.object_text, c.status,
                       c.trust_tier, c.confidence, c.corroboration_count, c.volatility,
                       c.valid_from, c.recorded_at, s.name AS source_name, s.kind AS source_kind,
                       c.embedding <=> $1::vector AS dist
                FROM knowledge_claim c
                JOIN knowledge_entity e ON e.id = c.subject_entity_id
                JOIN knowledge_source s ON s.id = c.source_id
                WHERE {visibility} AND c.embedding <=> $1::vector <= $2
                ORDER BY dist LIMIT $3
                """,
                *params,
            )

        # Group candidates by subject+predicate so conflicting values resolve together.
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            d = dict(row)
            key = (d["subject"].strip().lower(), d["predicate"])
            groups.setdefault(key, []).append(d)

        now = datetime.now(timezone.utc)
        ttl = self._config.volatile_ttl_seconds
        # Each entry pairs the recalled claim with its winning row id, so we can record usage
        # (last_recalled_at) on the claim actually chosen for the model.
        entries: list[tuple[RecalledClaim, int]] = []
        for members in groups.values():
            winner = resolve_conflict(members)
            conflicts = [
                ConflictingClaim(
                    object_text=m["object_text"],
                    source_name=m["source_name"],
                    source_kind=m["source_kind"],
                    trust_tier=m["trust_tier"],
                    status=m["status"],
                )
                for m in members
                if m["object_text"].strip().lower() != winner["object_text"].strip().lower()
            ]
            best_dist = min(float(m["dist"]) for m in members)
            claim = RecalledClaim(
                subject=winner["subject"],
                predicate=winner["predicate"],
                object_text=winner["object_text"],
                status=winner["status"],
                trust_tier=winner["trust_tier"],
                source_name=winner["source_name"],
                source_kind=winner["source_kind"],
                confidence=float(winner["confidence"]),
                corroboration_count=int(winner["corroboration_count"]),
                volatility=winner["volatility"],
                recorded_at=winner["recorded_at"],
                valid_from=winner["valid_from"],
                similarity=1.0 - best_dist,
                stale=is_stale(winner["volatility"], winner["valid_from"] or winner["recorded_at"], now, ttl),
                conflicts=conflicts,
            )
            entries.append((claim, int(winner["id"])))

        entries.sort(key=lambda e: e[0].similarity, reverse=True)
        top = entries[:k]

        # Record usage so the decay pass can prune never-recalled claims. Stamp only the
        # winning claim of each returned group — not the losing alternatives, or a perpetual
        # low-trust loser would have its decay clock reset every time the winner is recalled.
        # Current reads only — an as-of (historical) read shouldn't count as current use.
        if as_of is None and top:
            recalled_ids = [winner_id for _, winner_id in top]
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE knowledge_claim SET last_recalled_at = now() WHERE id = ANY($1)", recalled_ids
                )

        return [claim for claim, _ in top]

    # --- Maintenance (M4) --------------------------------------------------------

    async def _merge_entity(self, conn: _Conn, keep: int, dup: int) -> None:
        """Fold entity ``dup`` into ``keep``: repoint claims, union aliases, delete dup."""
        await conn.execute("UPDATE knowledge_claim SET subject_entity_id = $1 WHERE subject_entity_id = $2", keep, dup)
        await conn.execute("UPDATE knowledge_claim SET object_entity_id = $1 WHERE object_entity_id = $2", keep, dup)
        krow = await conn.fetchrow("SELECT canonical_name, aliases FROM knowledge_entity WHERE id = $1", keep)
        drow = await conn.fetchrow("SELECT canonical_name, aliases FROM knowledge_entity WHERE id = $1", dup)
        assert krow is not None and drow is not None
        new_aliases = merge_aliases(
            krow["canonical_name"], list(krow["aliases"] or []), drow["canonical_name"], list(drow["aliases"] or [])
        )
        await conn.execute("UPDATE knowledge_entity SET aliases = $1 WHERE id = $2", new_aliases, keep)
        await conn.execute("DELETE FROM knowledge_entity WHERE id = $1", dup)
        logfire.info("Merged duplicate entity", keep_id=keep, dup_id=dup, dup_name=drow["canonical_name"])

    async def _merge_by_name(self, conn: _Conn) -> int:
        """Merge entities with the same :func:`normalize_entity_name` into the lowest-id keeper.

        High precision and embedding-free: it only collapses names that differ by
        formatting/accents/punctuation/plural ("@anemone" / "anemone"). Returns the number
        of duplicates folded away.
        """
        rows = await conn.fetch("SELECT id, canonical_name FROM knowledge_entity ORDER BY id")
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
        entities = await conn.fetch("SELECT id, embedding::text AS emb FROM knowledge_entity ORDER BY id")
        merged_away: set[int] = set()
        max_dist = 1.0 - self._config.entity_merge_threshold
        merged = 0
        for ent in entities:
            if ent["id"] in merged_away:
                continue
            dups = await conn.fetch(
                "SELECT id FROM knowledge_entity WHERE id <> $1 AND embedding <=> $2::vector <= $3",
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

    @logfire.instrument()
    async def consolidate(self) -> dict[str, int]:
        """Merge duplicate entities (name pass, then embedding pass), then recompute.

        Pass 1 (:meth:`_merge_by_name`) collapses entities with an identical normalized
        name key — high precision, any name length, no embeddings. Pass 2
        (:meth:`_merge_by_embedding`) is a conservative embedding fallback at
        ``entity_merge_threshold``. Both repoint claims, union aliases, and delete the
        duplicate. Afterwards every live promotable-tier subject+predicate+object group is
        re-evaluated, so a merge that newly satisfies the corroboration threshold promotes
        to ``believed`` (and one that no longer does is demoted).
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                merged_by_name = await self._merge_by_name(conn)
                merged_by_embedding = await self._merge_by_embedding(conn)
                groups = await conn.fetch(
                    "SELECT DISTINCT subject_entity_id, predicate, object_text FROM knowledge_claim "
                    "WHERE retracted_at IS NULL AND superseded_by IS NULL AND status <> 'disputed' "
                    "AND trust_tier = ANY($1)",
                    list(PROMOTABLE_TIERS),
                )
                for g in groups:
                    await self._recompute_corroboration(conn, g["subject_entity_id"], g["predicate"], g["object_text"])

        summary = {
            "merged_by_name": merged_by_name,
            "merged_by_embedding": merged_by_embedding,
            "groups_recomputed": len(groups),
        }
        logfire.info("Consolidation complete", summary=summary)
        return summary

    @logfire.instrument()
    async def detect_contradictions(self) -> dict[str, int]:
        """Flag same-subject+predicate disagreements as ``disputed`` (and self-heal).

        For each live subject+predicate group: if two or more distinct object values
        coexist it's a contradiction — every live claim in the group is marked
        ``disputed`` (never deleted), which both demotes any ``believed`` value and drops
        the group from corroboration counting until it's resolved. If a previously-disputed
        group now has a single value, the flag is cleared back to ``asserted`` and the value
        re-evaluated for promotion. Overlapping validity is approximated as "both live"
        (claims rarely set ``valid_to``). It records ``disputed_at`` so the autonomous
        ``resolve_disputes`` pass can act after a grace period; here it only flags, it does
        not pick a winner.
        """
        conflicting = 0
        cleared = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                groups = await conn.fetch(
                    "SELECT subject_entity_id, predicate, "
                    "count(DISTINCT lower(object_text)) AS values, "
                    "count(*) FILTER (WHERE status = 'disputed') AS disputed "
                    "FROM knowledge_claim WHERE retracted_at IS NULL AND superseded_by IS NULL "
                    "GROUP BY subject_entity_id, predicate"
                )
                for g in groups:
                    if g["values"] >= 2:
                        # Mark disputed and stamp disputed_at, preserving any existing
                        # timestamp (COALESCE) so the grace period runs from first dispute.
                        # The `disputed_at IS NULL` arm also backfills claims disputed before
                        # the column existed, so they become eligible for resolution.
                        await conn.execute(
                            "UPDATE knowledge_claim SET status = 'disputed', disputed_at = COALESCE(disputed_at, now()) "
                            "WHERE subject_entity_id = $1 AND predicate = $2 AND retracted_at IS NULL "
                            "AND superseded_by IS NULL AND (status <> 'disputed' OR disputed_at IS NULL)",
                            g["subject_entity_id"],
                            g["predicate"],
                        )
                        conflicting += 1
                        logfire.info(
                            "Contradiction flagged",
                            subject_entity_id=g["subject_entity_id"],
                            predicate=g["predicate"],
                            distinct_values=g["values"],
                        )
                    elif g["disputed"] > 0:
                        # Conflict resolved (one value remains) — clear the flag and re-promote.
                        await conn.execute(
                            "UPDATE knowledge_claim SET status = 'asserted', disputed_at = NULL "
                            "WHERE subject_entity_id = $1 AND predicate = $2 AND retracted_at IS NULL "
                            "AND superseded_by IS NULL AND status = 'disputed'",
                            g["subject_entity_id"],
                            g["predicate"],
                        )
                        obj = await conn.fetchval(
                            "SELECT object_text FROM knowledge_claim WHERE subject_entity_id = $1 "
                            "AND predicate = $2 AND retracted_at IS NULL AND superseded_by IS NULL LIMIT 1",
                            g["subject_entity_id"],
                            g["predicate"],
                        )
                        if obj is not None:
                            await self._recompute_corroboration(conn, g["subject_entity_id"], g["predicate"], obj)
                        cleared += 1

        summary = {"conflicting_groups": conflicting, "cleared_groups": cleared}
        logfire.info("Contradiction detection complete", summary=summary)
        return summary

    @logfire.instrument()
    async def resolve_disputes(self) -> dict[str, int]:
        """Autonomously resolve contradictions that have stayed disputed past the grace period.

        For each disputed subject+predicate group whose dispute is older than
        ``dispute_grace_seconds``, :func:`rank_dispute_values` picks the best-supported value
        (independent sources, then trust tier, then recency). Claims asserting the losing
        values are **superseded** by the winning claim (archived, recoverable via an as-of
        read, and no longer re-flagged by ``detect_contradictions``); the winning value's
        claims are un-disputed and re-evaluated for promotion. Self-correcting: a later claim
        for a losing value re-opens the dispute. No human in the loop.
        """
        grace = self._config.dispute_grace_seconds
        resolved = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                groups = await conn.fetch(
                    "SELECT subject_entity_id, predicate FROM knowledge_claim "
                    "WHERE retracted_at IS NULL AND superseded_by IS NULL AND status = 'disputed' "
                    "AND disputed_at IS NOT NULL "
                    "GROUP BY subject_entity_id, predicate "
                    "HAVING min(disputed_at) < now() - make_interval(secs => $1)",
                    float(grace),
                )
                for g in groups:
                    claims = await conn.fetch(
                        "SELECT id, object_text, trust_tier, source_id, valid_from, recorded_at FROM knowledge_claim "
                        "WHERE subject_entity_id = $1 AND predicate = $2 AND retracted_at IS NULL "
                        "AND superseded_by IS NULL",
                        g["subject_entity_id"],
                        g["predicate"],
                    )
                    if not claims:
                        continue
                    winner = rank_dispute_values([dict(c) for c in claims]).strip().lower()
                    winner_ids = [c["id"] for c in claims if c["object_text"].strip().lower() == winner]
                    loser_ids = [c["id"] for c in claims if c["object_text"].strip().lower() != winner]
                    if not winner_ids:
                        continue
                    if loser_ids:
                        await conn.execute(
                            "UPDATE knowledge_claim SET superseded_by = $1, superseded_at = now() WHERE id = ANY($2)",
                            winner_ids[0],
                            loser_ids,
                        )
                    await conn.execute(
                        "UPDATE knowledge_claim SET status = 'asserted', disputed_at = NULL WHERE id = ANY($1)",
                        winner_ids,
                    )
                    winning_value = next(c["object_text"] for c in claims if c["id"] == winner_ids[0])
                    await self._recompute_corroboration(conn, g["subject_entity_id"], g["predicate"], winning_value)
                    resolved += 1
                    logfire.info(
                        "Dispute resolved",
                        subject_entity_id=g["subject_entity_id"],
                        predicate=g["predicate"],
                        superseded=len(loser_ids),
                    )

        summary = {"resolved_groups": resolved}
        logfire.info("Dispute resolution complete", summary=summary)
        return summary

    @logfire.instrument()
    async def decay(self) -> dict[str, int]:
        """Autonomously prune low-value claims (soft-retract, never delete).

        Tombstones (``retracted_at``) claims that are low-trust (tier < secondary, i.e.
        ``model_quarantine``/``user``), neither ``believed`` nor ``disputed``, and neither
        recorded nor recalled within ``decay_ttl_seconds``. Secondary/primary, believed,
        disputed (left to ``resolve_disputes``), and recently-used claims are never touched.
        Low-trust claims never count toward corroboration, so no recompute is needed.
        Soft-retract keeps the row auditable and recoverable (append-only).
        """
        low_tiers = [t for t, r in TRUST_TIERS.items() if r < TRUST_TIERS["secondary"]]
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE knowledge_claim SET retracted_at = now() "
                # Leave 'disputed' claims to resolve_disputes — decaying one side of a live
                # contradiction would silently decide it by attrition.
                "WHERE retracted_at IS NULL AND superseded_by IS NULL AND status NOT IN ('believed', 'disputed') "
                "AND trust_tier = ANY($1) "
                "AND recorded_at < now() - make_interval(secs => $2) "
                "AND (last_recalled_at IS NULL OR last_recalled_at < now() - make_interval(secs => $2))",
                low_tiers,
                float(self._config.decay_ttl_seconds),
            )
        tombstoned = int(result.split()[-1]) if result else 0
        summary = {"tombstoned": tombstoned}
        logfire.info("Decay complete", summary=summary)
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
                    WHERE o.id <> e.id
                    ORDER BY o.embedding <=> e.embedding
                    LIMIT 1
                ) n
                WHERE n.dist <= $1
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
        """Snapshot counts for observability (entities, sources, claims by status/tier)."""
        async with self._pool.acquire() as conn:
            entities = await conn.fetchval("SELECT count(*) FROM knowledge_entity")
            sources = await conn.fetchval("SELECT count(*) FROM knowledge_source")
            live = "retracted_at IS NULL AND superseded_by IS NULL"
            by_status = await conn.fetch(
                f"SELECT status, count(*) AS n FROM knowledge_claim WHERE {live} GROUP BY status ORDER BY status"
            )
            by_tier = await conn.fetch(
                f"SELECT trust_tier, count(*) AS n FROM knowledge_claim WHERE {live} GROUP BY trust_tier"
            )
            retracted = await conn.fetchval("SELECT count(*) FROM knowledge_claim WHERE retracted_at IS NOT NULL")
            superseded = await conn.fetchval("SELECT count(*) FROM knowledge_claim WHERE superseded_by IS NOT NULL")
        return {
            "entities": int(entities or 0),
            "sources": int(sources or 0),
            "live_by_status": {r["status"]: int(r["n"]) for r in by_status},
            "live_by_tier": {r["trust_tier"]: int(r["n"]) for r in by_tier},
            "retracted": int(retracted or 0),
            "superseded": int(superseded or 0),
        }

    # --- Migration ---------------------------------------------------------------

    async def _migrate_legacy_global_memory(self) -> None:
        """Backfill rows from the legacy ``global_memory`` table as low-tier claims.

        One-time, idempotent: runs only when a ``global_memory`` table exists and the new
        claim table is still empty. Each old fact becomes a claim on a generic
        ``(legacy memory)`` entity with predicate ``note``, attributed to its original
        author (``user`` tier) or to ``model`` (``model_quarantine``) when authorless. The
        original ``global_memory`` table is left intact as a backup; drop it manually once
        the migration is verified.
        """
        async with self._pool.acquire() as conn:
            has_legacy = await conn.fetchval("SELECT to_regclass('public.global_memory')")
            if has_legacy is None:
                return
            already = await conn.fetchval("SELECT EXISTS (SELECT 1 FROM knowledge_claim)")
            if already:
                return
            rows = await conn.fetch(
                "SELECT fact, embedding::text AS embedding, embedding_model, author, source_note_id, created_at "
                "FROM global_memory ORDER BY id"
            )
            if not rows:
                return

            name_vec = _vector_literal(await self.embed("legacy memory"))
            async with conn.transaction():
                entity_id = int(
                    await conn.fetchval(
                        "INSERT INTO knowledge_entity (canonical_name, embedding) VALUES ($1, $2::vector) RETURNING id",
                        "(legacy memory)",
                        name_vec,
                    )
                )
                migrated = 0
                for row in rows:
                    author = row["author"]
                    if author:
                        src_name, src_kind, tier = author, "user", "user"
                    else:
                        src_name, src_kind, tier = "model", "model", "model_quarantine"
                    source_id = await self._get_or_create_source(conn, src_name, src_kind, tier)
                    # Reuse the old embedding only if it came from the current model (same
                    # vector space); otherwise re-embed the fact text under the new model.
                    if row["embedding_model"] == self._embed_model:
                        claim_vec = row["embedding"]
                    else:
                        claim_vec = _vector_literal(await self.embed(row["fact"]))
                    # Seed last_recalled_at = now() so the imported facts get a fresh decay
                    # window: they carry their original (old) recorded_at, so without this the
                    # decay pass running later in the same `run-all` would immediately
                    # tombstone the whole migrated corpus.
                    await conn.execute(
                        "INSERT INTO knowledge_claim "
                        "(subject_entity_id, predicate, object_text, source_id, trust_tier, status, "
                        " recorded_at, last_recalled_at, volatility, embedding, author, source_note_id) "
                        "VALUES ($1, 'note', $2, $3, $4, 'asserted', $5, now(), 'stable', $6::vector, $7, $8)",
                        entity_id,
                        row["fact"],
                        source_id,
                        tier,
                        row["created_at"],
                        claim_vec,
                        author,
                        row["source_note_id"],
                    )
                    migrated += 1
            logfire.info("Migrated legacy global_memory rows into claims", count=migrated)
