"""Out-of-process maintenance for mem0 long-term memory.

Run locally with::

    uv run python -m bot.maintenance cleanup --dry-run -c config.local.yaml
    uv run python -m bot.maintenance cleanup -c config.local.yaml

The Kubernetes CronJob runs the second form. Cleanup is scoped to the bot's
``agent_id`` and preserves explicit ``add_memory`` entries from age/author caps.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

import click
import logfire
import yaml

from .memory import MemoryStore, StoredMemory
from .models import Config


CleanupReason = Literal["empty", "expired", "duplicate", "retention", "author_limit"]


@dataclass(frozen=True)
class CleanupCandidate:
    """One memory selected for deletion and the policy that selected it."""

    memory_id: str
    reason: CleanupReason


def load_config(path: str) -> Config:
    """Load and validate the same YAML configuration used by the bot."""
    with open(path) as config_file:
        return Config(**yaml.safe_load(config_file))


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
        or datetime.max.replace(tzinfo=timezone.utc)
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
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
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
        now=datetime.now(timezone.utc),
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


if __name__ == "__main__":
    cli()
