#!/usr/bin/env python3
"""One-shot migration: rich claims store -> minimal agreement-ranked store.

THROWAWAY / STANDALONE. Not part of the bot package — run it once against the live
Postgres, then delete it. Depends only on the stdlib + asyncpg (already a project dep).

It rewrites ``knowledge_claim`` from the old rich schema (trust tiers, status,
supersession, bitemporal, source FK) into the new minimal schema expected by
``bot/memory.py``:

    knowledge_claim(subject_entity_id, predicate, object_text, object_key,
                    object_entity_id, author, embedding, updated_at,
                    UNIQUE(author, subject_entity_id, predicate))

Mapping decisions:
- Only LIVE claims migrate (``retracted_at IS NULL AND superseded_by IS NULL``).
- ``author`` := COALESCE(old author, source name). So old user-note claims keep their
  handle, and old web/doc/model claims survive as a distinct *source* "vote" (the new
  agreement count is "distinct asserters", which generalizes users + sources). Nothing
  is dropped as long as every claim has a source (it does — source_id is NOT NULL).
- One row per (author, subject_entity_id, predicate), keeping the most recent
  (COALESCE(valid_from, recorded_at, created_at)) — the new UNIQUE key.
- ``object_key`` := normalize_object(object_text) (same rule as bot/memory.py).
- ``knowledge_entity`` is left as-is (its extra legacy ``type`` column is harmless; the
  new code never reads it). ``knowledge_source`` and ``knowledge_entity_merge_log`` are
  dropped (folded into ``author`` / no longer used).

Usage:
    # dry run (default): reports what WOULD happen, changes nothing
    uv run python migrate_memory_store.py --dsn "postgresql://user:pass@host:5432/db"
    # apply it (all-or-nothing transaction):
    uv run python migrate_memory_store.py --dsn "postgresql://..." --apply

The DSN is the same one in your config's ``postgres_url``. The cluster DSN is only
reachable in-cluster, so run this from a one-off pod (or via ``kubectl port-forward``
to the pg service and point --dsn at localhost). Stop the bot first so nothing writes
concurrently.
"""

import argparse
import asyncio
import re
import unicodedata

import asyncpg

# --- inlined from bot/memory.py (kept identical so migrated keys match runtime keys) ---


def normalize_object(object_text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", object_text or "")
    ascii_ish = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_ish.lower()).strip()


# Live, authored (or source-attributable), newest-per-(author,subject,predicate) claims.
SELECT_LIVE = """
SELECT DISTINCT ON (eff_author, subject_entity_id, predicate)
       subject_entity_id, predicate, object_text, object_entity_id, eff_author, emb, updated_at
FROM (
    SELECT c.subject_entity_id,
           c.predicate,
           c.object_text,
           c.object_entity_id,
           COALESCE(NULLIF(btrim(c.author), ''), s.name) AS eff_author,
           c.embedding::text AS emb,
           COALESCE(c.valid_from, c.recorded_at, c.created_at) AS updated_at
    FROM knowledge_claim c
    JOIN knowledge_source s ON s.id = c.source_id
    WHERE c.retracted_at IS NULL AND c.superseded_by IS NULL
) t
WHERE eff_author IS NOT NULL
ORDER BY eff_author, subject_entity_id, predicate, updated_at DESC
"""


async def run(dsn: str, apply: bool) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        legacy = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'knowledge_claim' AND column_name = 'trust_tier'"
        )
        if legacy is None:
            print("No legacy 'trust_tier' column found — knowledge_claim is not the old schema.")
            print("Nothing to migrate (already migrated, or fresh DB). Aborting.")
            return

        dim = await conn.fetchval(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'knowledge_claim'::regclass AND attname = 'embedding' AND NOT attisdropped"
        )
        dim = int(dim) if dim and int(dim) > 0 else 1024

        total = await conn.fetchval("SELECT count(*) FROM knowledge_claim")
        live = await conn.fetchval(
            "SELECT count(*) FROM knowledge_claim WHERE retracted_at IS NULL AND superseded_by IS NULL"
        )
        rows = await conn.fetch(SELECT_LIVE)

        print(f"embedding dim:                 {dim}")
        print(f"total claims:                  {total}")
        print(f"live claims (kept candidates): {live}")
        print(f"after dedup -> rows to insert: {len(rows)}  (one per author+subject+predicate, newest wins)")
        print(f"dropped (retracted/superseded/non-attributable): {total - len(rows)}")

        if not apply:
            print("\nDRY RUN — nothing changed. Re-run with --apply to perform the migration.")
            return

        records = [
            (
                r["subject_entity_id"],
                r["predicate"],
                r["object_text"],
                normalize_object(r["object_text"]),
                r["object_entity_id"],
                r["eff_author"],
                r["emb"],
                r["updated_at"],
            )
            for r in rows
        ]

        async with conn.transaction():
            await conn.execute("DROP TABLE IF EXISTS knowledge_claim_new")
            await conn.execute(
                f"""
                CREATE TABLE knowledge_claim_new (
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
            await conn.executemany(
                "INSERT INTO knowledge_claim_new "
                "(subject_entity_id, predicate, object_text, object_key, object_entity_id, author, embedding, updated_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8)",
                records,
            )
            # Swap in the new table (CASCADE drops the old source FK), then drop now-unused tables.
            await conn.execute("DROP TABLE knowledge_claim CASCADE")
            await conn.execute("ALTER TABLE knowledge_claim_new RENAME TO knowledge_claim")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS knowledge_claim_embedding_idx "
                "ON knowledge_claim USING hnsw (embedding vector_cosine_ops)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS knowledge_claim_sp_idx ON knowledge_claim (subject_entity_id, predicate)"
            )
            await conn.execute("DROP TABLE IF EXISTS knowledge_source CASCADE")
            await conn.execute("DROP TABLE IF EXISTS knowledge_entity_merge_log")

        migrated = await conn.fetchval("SELECT count(*) FROM knowledge_claim")
        entities = await conn.fetchval("SELECT count(*) FROM knowledge_entity WHERE merged_into IS NULL")
        authors = await conn.fetchval("SELECT count(DISTINCT author) FROM knowledge_claim")
        print(
            f"\nDONE. knowledge_claim now has {migrated} rows across {authors} distinct authors, "
            f"{entities} live entities. knowledge_source / merge_log dropped."
        )
        print("The bot's create() will accept this schema (no trust_tier column). Restart the bot.")
    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate the world-knowledge store to the minimal schema.")
    parser.add_argument("--dsn", required=True, help="Postgres DSN (same as config postgres_url).")
    parser.add_argument("--apply", action="store_true", help="Perform the migration (default is a dry run).")
    args = parser.parse_args()
    asyncio.run(run(args.dsn, args.apply))


if __name__ == "__main__":
    main()
