"""Persistent long-term memory for Missbot (Postgres + pgvector).

This module owns the *global* (non-user-specific) memory store: shared facts the
bot has learned, retrieved by semantic similarity. Per-user memory will live in a
separate keyed table (a later stage) and does not need embeddings.

Security note: global memory is writable from attacker-controlled message text, so
every fact carries provenance (author + source note) for auditing/purging, writes
are rate-limited per author and de-duplicated, and recalled facts are returned to
the model as untrusted data (see bot/tools.py). The embedding model produces
*unnormalized* int8 vectors, so all comparisons use cosine distance (`<=>`).
"""

import os
from dataclasses import dataclass
from typing import Optional

import asyncpg
import httpx
import logfire

from .models import Config


@dataclass
class GlobalFact:
    """A recalled global memory fact with its cosine similarity to the query."""

    fact: str
    similarity: float


def _vector_literal(vec: list[float]) -> str:
    """Render an embedding as a pgvector text literal (e.g. '[0.1,0.2,...]').

    We pass vectors as text and cast to ``vector`` in SQL, which avoids depending
    on a binary pgvector codec and never needs to decode vectors back in Python.
    """
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class MemoryStore:
    """Async Postgres-backed store for global long-term memory."""

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
        """Connect, ensure the schema/extension exist, and return a ready store.

        Fails fast if the existing embedding column dimension disagrees with
        ``config.embedding_dim`` — that mismatch silently returns garbage neighbors,
        so it must surface loudly (it means the corpus needs re-embedding).
        """
        assert config.postgres_url, "postgres_url is required to build a MemoryStore"
        dim = config.embedding_dim

        # Create the extension and tables on a plain connection *before* opening the
        # pool, so the `vector` type exists by the time any pooled connection uses it.
        conn = await asyncpg.connect(config.postgres_url)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS global_memory (
                    id              BIGSERIAL PRIMARY KEY,
                    fact            TEXT NOT NULL,
                    embedding       vector({dim}) NOT NULL,
                    author          TEXT,
                    source_note_id  TEXT,
                    embedding_model TEXT NOT NULL,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS global_memory_embedding_idx "
                "ON global_memory USING hnsw (embedding vector_cosine_ops)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS global_memory_author_idx ON global_memory (author)")

            existing_dim = await conn.fetchval(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'global_memory'::regclass AND attname = 'embedding' AND NOT attisdropped"
            )
            if existing_dim is not None and existing_dim > 0 and existing_dim != dim:
                raise RuntimeError(
                    f"global_memory.embedding has dimension {existing_dim} but embedding_dim={dim}. "
                    "Changing the embedding model requires re-embedding every row (the vectors must "
                    "share one space); drop/migrate the table before changing embedding_dim."
                )
        finally:
            await conn.close()

        pool = await asyncpg.create_pool(config.postgres_url)
        assert pool is not None
        http = httpx.AsyncClient(timeout=httpx.Timeout(config.http_timeout_seconds))
        logfire.info("Memory store ready", embedding_model=config.embedding_model, embedding_dim=dim)
        return cls(pool, http, config)

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

    @logfire.instrument(extract_args=["author", "source_note_id"])
    async def add_global_fact(self, fact: str, author: Optional[str], source_note_id: Optional[str]) -> bool:
        """Embed and store a global fact, skipping near-duplicates.

        Returns True if stored, False if it was dropped as a near-duplicate of an
        existing fact (cosine similarity >= global_dedup_threshold).
        """
        vec = await self.embed(fact)
        literal = _vector_literal(vec)
        async with self._pool.acquire() as conn:
            # Cosine distance of the nearest existing fact; similarity = 1 - distance.
            nearest = await conn.fetchval(
                "SELECT embedding <=> $1::vector AS dist FROM global_memory ORDER BY dist LIMIT 1",
                literal,
            )
            if nearest is not None and (1.0 - float(nearest)) >= self._config.global_dedup_threshold:
                logfire.info("Skipping near-duplicate global fact", similarity=1.0 - float(nearest))
                return False
            await conn.execute(
                "INSERT INTO global_memory (fact, embedding, author, source_note_id, embedding_model) "
                "VALUES ($1, $2::vector, $3, $4, $5)",
                fact,
                literal,
                author,
                source_note_id,
                self._embed_model,
            )
        return True

    async def seconds_since_last_write(self, author: str) -> Optional[float]:
        """Seconds since ``author``'s most recent global write, or None if never."""
        async with self._pool.acquire() as conn:
            last = await conn.fetchval(
                "SELECT EXTRACT(EPOCH FROM (now() - max(created_at))) FROM global_memory WHERE author = $1",
                author,
            )
        return float(last) if last is not None else None

    @logfire.instrument(extract_args=["query"])
    async def search_global(self, query: str, k: int) -> list[GlobalFact]:
        """Return up to k global facts most cosine-similar to the query.

        Results below config.global_recall_min_similarity are dropped.
        """
        vec = await self.embed(query)
        literal = _vector_literal(vec)
        min_sim = self._config.global_recall_min_similarity
        # similarity >= min_sim  <=>  distance <= 1 - min_sim
        max_dist = 1.0 - min_sim
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT fact, embedding <=> $1::vector AS dist FROM global_memory "
                "WHERE embedding <=> $1::vector <= $2 ORDER BY dist LIMIT $3",
                literal,
                max_dist,
                k,
            )
        return [GlobalFact(fact=r["fact"], similarity=1.0 - float(r["dist"])) for r in rows]
