"""Out-of-process maintenance for mem0 long-term memory.

Run locally with::

    uv run python -m bot.maintenance cleanup --dry-run -c config.local.yaml
    uv run python -m bot.maintenance cleanup -c config.local.yaml
    uv run python -m bot.maintenance reembed --dry-run -c config.local.yaml
    uv run python -m bot.maintenance reembed -c config.local.yaml

The Kubernetes CronJob runs cleanup. Cleanup is scoped to the bot's ``agent_id``
and preserves explicit ``add_memory`` entries from age/author caps. Re-embedding
rewrites only the vectors in the configured memory and entity tables after taking
full timestamped backups; callers must stop every memory reader/writer first.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import click
import logfire
import psycopg
import yaml
from psycopg import sql

from .memory import MemoryStore, StoredMemory
from .models import Config

CleanupReason = Literal["empty", "expired", "duplicate", "retention", "author_limit"]
_REEMBED_PROBE_TEXT = "Missbot re-embedding preflight"


@dataclass(frozen=True)
class CleanupCandidate:
    """One memory selected for deletion and the policy that selected it."""

    memory_id: str
    reason: CleanupReason


@dataclass(frozen=True)
class VectorRecord:
    """One vector-store row and the exact text its vector represents."""

    table: str
    memory_id: str
    text: str
    current_dim: int


def load_config(path: str) -> Config:
    """Load and validate the same YAML configuration used by the bot."""
    with open(path) as config_file:
        return Config(**yaml.safe_load(config_file))


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _timestamp(memory: StoredMemory) -> datetime:
    # Missing timestamps are kept ahead of ordinary rows so malformed metadata
    # alone never makes a memory the first one removed by an author cap.
    return (
        _parse_datetime(memory.updated_at)
        or _parse_datetime(memory.created_at)
        or datetime.max.replace(tzinfo=UTC)
    )


def _source(memory: StoredMemory) -> str:
    return str(memory.metadata.get("source") or "").strip().lower()


def _is_inferred(memory: StoredMemory) -> bool:
    """Whether a memory was inferred from a conversation rather than explicitly saved.

    Keyed on "not an explicit add_memory write" rather than on a specific frontend's
    label, so every frontend's memories (misskey_note, acp_prompt, ...) stay subject to
    retention and per-author caps while explicit `add_memory` entries stay protected.
    """
    return _source(memory) != "add_memory"


def _author(memory: StoredMemory) -> str:
    return str(memory.metadata.get("author") or "").strip().lower()


def _duplicate_key(memory: StoredMemory) -> str:
    return " ".join(memory.memory.split()).casefold()


def plan_cleanup(
    memories: Iterable[StoredMemory],
    *,
    now: datetime,
    note_retention_days: int | None,
    max_memories_per_author: int | None,
) -> list[CleanupCandidate]:
    """Select deletions without touching the store.

    Order matters: malformed/expired rows are selected first, then exact text
    duplicates, age retention, and finally per-author overflow among surviving
    auto-ingested note memories.
    """
    records = list(memories)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    now = now.astimezone(UTC)
    selected: dict[str, CleanupReason] = {}

    def select(memory: StoredMemory, reason: CleanupReason) -> None:
        selected.setdefault(memory.id, reason)

    for memory in records:
        if not memory.memory.strip():
            select(memory, "empty")
            continue
        expiration_date = _parse_date(memory.expiration_date)
        # Match mem0's own expiration semantics: a date becomes expired after
        # that date has fully elapsed, not at the start of the named date.
        if expiration_date is not None and expiration_date < now.date():
            select(memory, "expired")

    duplicate_groups: dict[str, list[StoredMemory]] = defaultdict(list)
    for memory in records:
        if memory.id not in selected:
            duplicate_groups[_duplicate_key(memory)].append(memory)

    for group in duplicate_groups.values():
        if len(group) < 2:
            continue
        # Explicitly remembered facts outrank inferred note memories. Within the
        # same source type, retain the most recently updated copy.
        keeper = max(group, key=lambda item: (_source(item) == "add_memory", _timestamp(item)))
        for memory in group:
            if memory.id != keeper.id:
                select(memory, "duplicate")

    if note_retention_days is not None:
        cutoff = now - timedelta(days=note_retention_days)
        for memory in records:
            created_at = _parse_datetime(memory.created_at)
            if memory.id not in selected and _is_inferred(memory) and created_at is not None and created_at < cutoff:
                select(memory, "retention")

    if max_memories_per_author is not None:
        by_author: dict[str, list[StoredMemory]] = defaultdict(list)
        for memory in records:
            author = _author(memory)
            if memory.id not in selected and _is_inferred(memory) and author:
                by_author[author].append(memory)

        for author_memories in by_author.values():
            newest_first = sorted(author_memories, key=_timestamp, reverse=True)
            for memory in newest_first[max_memories_per_author:]:
                select(memory, "author_limit")

    return [CleanupCandidate(memory_id=memory_id, reason=reason) for memory_id, reason in selected.items()]


async def cleanup(store: MemoryStore, config: Config, *, dry_run: bool = False) -> dict[str, Any]:
    """Plan and optionally execute one scoped mem0 cleanup pass."""
    memories = await store.list_all(config.memory_cleanup_scan_limit)
    candidates = plan_cleanup(
        memories,
        now=datetime.now(UTC),
        note_retention_days=config.memory_note_retention_days,
        max_memories_per_author=config.memory_max_memories_per_author,
    )
    reasons = Counter(candidate.reason for candidate in candidates)
    deleted = 0
    failures: list[dict[str, str]] = []

    if not dry_run:
        for candidate in candidates:
            try:
                await store.delete(candidate.memory_id)
                deleted += 1
            except Exception as exc:
                logfire.warning(
                    "mem0 maintenance deletion failed",
                    memory_id=candidate.memory_id,
                    reason=candidate.reason,
                    error=str(exc),
                )
                failures.append(
                    {
                        "memory_id": candidate.memory_id,
                        "reason": candidate.reason,
                        "error": str(exc),
                    }
                )

    return {
        "scanned": len(memories),
        "scan_limit": config.memory_cleanup_scan_limit,
        "scan_limit_reached": len(memories) >= config.memory_cleanup_scan_limit,
        "candidates": len(candidates),
        "reasons": dict(sorted(reasons.items())),
        "dry_run": dry_run,
        "deleted": deleted,
        "failed": len(failures),
        "failures": failures,
    }


def _reembed_table_names(config: Config) -> tuple[str, str]:
    base = config.memory_collection_name
    return base, f"{base}_entities"


def _backup_table_name(table: str, suffix: str) -> str:
    name = f"{table}_backup_{suffix}"
    if len(name.encode()) > 63:
        raise ValueError(f"backup table name exceeds PostgreSQL's 63-byte limit: {name}")
    return name


def _load_vector_records(postgres_url: str, tables: tuple[str, str]) -> dict[str, list[VectorRecord]]:
    records: dict[str, list[VectorRecord]] = {}
    with psycopg.connect(postgres_url) as connection, connection.cursor() as cursor:
        for table in tables:
            cursor.execute(
                sql.SQL("SELECT id::text, payload->>'data', vector_dims(vector) FROM {} ORDER BY id").format(
                    sql.Identifier(table)
                )
            )
            table_records: list[VectorRecord] = []
            for memory_id, text, current_dim in cursor.fetchall():
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{table} row {memory_id} has no embeddable payload.data text")
                if not isinstance(current_dim, int):
                    raise ValueError(f"{table} row {memory_id} has no vector dimension")
                table_records.append(
                    VectorRecord(
                        table=table,
                        memory_id=str(memory_id),
                        text=text,
                        current_dim=current_dim,
                    )
                )
            records[table] = table_records
    return records


def _validate_vectors(vectors: list[list[float]], expected_count: int, expected_dim: int) -> None:
    if len(vectors) != expected_count:
        raise ValueError(f"embedding endpoint returned {len(vectors)} vectors for {expected_count} texts")
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dim:
            raise ValueError(
                f"embedding endpoint returned dimension {len(vector)} for item {index}; expected {expected_dim}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"embedding endpoint returned a non-finite value for item {index}")


def _replace_vectors(
    postgres_url: str,
    records: dict[str, list[VectorRecord]],
    vectors: dict[str, list[list[float]]],
    *,
    expected_dim: int,
    backup_suffix: str,
) -> dict[str, str]:
    """Back up and replace both collections atomically."""
    tables = tuple(records)
    backup_tables = {table: _backup_table_name(table, backup_suffix) for table in tables}

    with psycopg.connect(postgres_url) as connection, connection.cursor() as cursor:
        # The connection context commits all backups and vector replacements together,
        # or rolls all of them back. Locks also make a missed active writer fail the
        # pre-update row-count check instead of silently leaving a mixed embedding space.
        for table in tables:
            cursor.execute(sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(sql.Identifier(table)))

        for table in tables:
            cursor.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
            live_count = cursor.fetchone()
            if live_count is None or live_count[0] != len(records[table]):
                raise RuntimeError(f"{table} changed after the migration snapshot; expected {len(records[table])} rows")

            cursor.execute(
                sql.SQL("CREATE TABLE {} AS TABLE {}").format(
                    sql.Identifier(backup_tables[table]),
                    sql.Identifier(table),
                )
            )

        for table in tables:
            parameters = [
                (json.dumps(vector, separators=(",", ":")), record.memory_id)
                for record, vector in zip(records[table], vectors[table], strict=True)
            ]
            cursor.executemany(
                sql.SQL("UPDATE {} SET vector = %s::vector WHERE id = %s::uuid").format(sql.Identifier(table)),
                parameters,
            )
            cursor.execute(
                sql.SQL("SELECT count(*) FILTER (WHERE vector IS NULL OR vector_dims(vector) <> %s) FROM {}").format(
                    sql.Identifier(table)
                ),
                (expected_dim,),
            )
            invalid = cursor.fetchone()
            if invalid is None or invalid[0] != 0:
                raise RuntimeError(f"{table} contains invalid vectors after replacement")

    return backup_tables


async def reembed(
    store: MemoryStore,
    config: Config,
    *,
    dry_run: bool = False,
    batch_size: int = 64,
    backup_suffix: str | None = None,
) -> dict[str, Any]:
    """Re-embed the configured memory and entity tables without changing payloads."""
    if not config.postgres_url:
        raise ValueError("postgres_url is required to re-embed memories")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    tables = _reembed_table_names(config)
    records = await asyncio.to_thread(_load_vector_records, config.postgres_url, tables)
    wrong_current_dims = {
        table: sum(record.current_dim != config.embedding_dim for record in table_records)
        for table, table_records in records.items()
    }
    if any(wrong_current_dims.values()):
        raise ValueError(
            f"existing vector dimensions do not match configured embedding_dim={config.embedding_dim}: "
            f"{wrong_current_dims}"
        )

    probe = await store.embed_batch([_REEMBED_PROBE_TEXT], action="update")
    _validate_vectors(probe, 1, config.embedding_dim)
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim,
        "tables": {table: len(table_records) for table, table_records in records.items()},
        "total_rows": sum(len(table_records) for table_records in records.values()),
        "probe_ok": True,
    }
    if dry_run:
        summary["backup_tables"] = {}
        summary["updated"] = 0
        return summary

    replacement_vectors: dict[str, list[list[float]]] = {table: [] for table in tables}
    flattened = [record for table in tables for record in records[table]]
    for offset in range(0, len(flattened), batch_size):
        batch = flattened[offset : offset + batch_size]
        embedded = await store.embed_batch([record.text for record in batch], action="update")
        _validate_vectors(embedded, len(batch), config.embedding_dim)
        for record, vector in zip(batch, embedded, strict=True):
            replacement_vectors[record.table].append(vector)
        logfire.info(
            "mem0 re-embedding progress",
            completed=min(offset + len(batch), len(flattened)),
            total=len(flattened),
            embedding_model=config.embedding_model,
        )

    suffix = backup_suffix or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_tables = await asyncio.to_thread(
        _replace_vectors,
        config.postgres_url,
        records,
        replacement_vectors,
        expected_dim=config.embedding_dim,
        backup_suffix=suffix,
    )
    summary["backup_tables"] = backup_tables
    summary["updated"] = len(flattened)
    return summary


def _run(config_path: str, action: Callable[[MemoryStore, Config], Awaitable[Any]]) -> Any:
    async def _main() -> Any:
        config = load_config(config_path)
        if not config.memory_enabled:
            raise click.ClickException("memory_enabled must be true in the config to run maintenance")
        store = await MemoryStore.create(config)
        try:
            return await action(store, config)
        finally:
            await store.close()

    return asyncio.run(_main())


_config_option = click.option(
    "-c",
    "--config",
    "config_path",
    default="config.local.yaml",
    show_default=True,
    help="Path to the bot configuration file.",
)


@click.group()
def cli() -> None:
    """Maintain mem0 long-term memory."""
    logfire.configure(console=logfire.ConsoleOptions(min_log_level="info"))


@cli.command("cleanup")
@_config_option
@click.option("--dry-run", is_flag=True, help="Report candidates without deleting them.")
def cleanup_command(config_path: str, dry_run: bool) -> None:
    """Delete expired, duplicate, stale, empty, and over-cap note memories."""
    summary = _run(config_path, lambda store, config: cleanup(store, config, dry_run=dry_run))
    click.echo(json.dumps(summary, indent=2))


@cli.command("reembed")
@_config_option
@click.option("--dry-run", is_flag=True, help="Validate the endpoint and report rows without changing vectors.")
@click.option("--batch-size", type=click.IntRange(min=1, max=256), default=64, show_default=True)
@click.option("--backup-suffix", help="Suffix for backup tables; defaults to a UTC timestamp.")
def reembed_command(config_path: str, dry_run: bool, batch_size: int, backup_suffix: str | None) -> None:
    """Back up and re-embed the current memory and entity vector tables."""
    try:
        summary = _run(
            config_path,
            lambda store, config: reembed(
                store,
                config,
                dry_run=dry_run,
                batch_size=batch_size,
                backup_suffix=backup_suffix,
            ),
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    cli()
