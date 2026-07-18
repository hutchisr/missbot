"""Tests for mem0 maintenance selection and CLI wiring."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from bot.maintenance import cleanup, cli, plan_cleanup
from bot.memory import StoredMemory


NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)


def _memory(
    memory_id: str,
    text: str,
    *,
    created_at: str = "2026-07-17T00:00:00+00:00",
    updated_at: str | None = None,
    expiration_date: str | None = None,
    source: str = "misskey_note",
    author: str = "alice",
) -> StoredMemory:
    return StoredMemory(
        id=memory_id,
        memory=text,
        created_at=created_at,
        updated_at=updated_at,
        expiration_date=expiration_date,
        metadata={"source": source, "author": author},
    )


def _memory_cfg(make_config, **extra):
    return make_config(
        memory_enabled=True,
        postgres_url="postgres://u:p@db/x",
        embedding_model="embed/model",
        **extra,
    )


def test_plan_cleanup_applies_safe_policy_in_order():
    memories = [
        _memory("empty", "  "),
        _memory("expired", "expired fact", expiration_date="2026-07-16"),
        _memory(
            "explicit",
            "Same fact",
            created_at="2025-01-01T00:00:00+00:00",
            source="add_memory",
            author="grok",
        ),
        _memory("duplicate", " same   FACT ", updated_at="2026-07-17T11:00:00+00:00"),
        _memory("stale", "old note", created_at="2026-01-01T00:00:00+00:00", author="bob"),
        _memory("newest", "newest", created_at="2026-07-17T03:00:00+00:00"),
        _memory("middle", "middle", created_at="2026-07-17T02:00:00+00:00"),
        _memory("oldest", "oldest", created_at="2026-07-17T01:00:00+00:00"),
    ]

    candidates = plan_cleanup(
        memories,
        now=NOW,
        note_retention_days=90,
        max_memories_per_author=2,
    )

    assert {candidate.memory_id: candidate.reason for candidate in candidates} == {
        "empty": "empty",
        "expired": "expired",
        "duplicate": "duplicate",
        "stale": "retention",
        "oldest": "author_limit",
    }


def test_plan_cleanup_can_disable_retention_and_author_cap():
    memories = [_memory("old", "old note", created_at="2020-01-01T00:00:00+00:00")]

    candidates = plan_cleanup(
        memories,
        now=NOW,
        note_retention_days=None,
        max_memories_per_author=None,
    )

    assert candidates == []


@pytest.mark.anyio
async def test_cleanup_dry_run_does_not_delete(make_config):
    store = AsyncMock()
    store.list_all.return_value = [_memory("empty", "")]
    config = _memory_cfg(make_config, memory_cleanup_scan_limit=100)

    summary = await cleanup(store, config, dry_run=True)

    store.list_all.assert_awaited_once_with(100)
    store.delete.assert_not_awaited()
    assert summary == {
        "scanned": 1,
        "scan_limit": 100,
        "scan_limit_reached": False,
        "candidates": 1,
        "reasons": {"empty": 1},
        "dry_run": True,
        "deleted": 0,
        "failed": 0,
        "failures": [],
    }


@pytest.mark.anyio
async def test_cleanup_deletes_candidates_and_reports_failures(make_config):
    store = AsyncMock()
    store.list_all.return_value = [_memory("empty-1", ""), _memory("empty-2", "")]
    store.delete.side_effect = [None, RuntimeError("database unavailable")]
    config = _memory_cfg(make_config)

    summary = await cleanup(store, config)

    assert store.delete.await_count == 2
    assert summary["deleted"] == 1
    assert summary["failed"] == 1
    assert summary["failures"][0]["memory_id"] == "empty-2"


def test_cleanup_cli_runs_and_closes_store(make_config):
    config = _memory_cfg(make_config)
    store = AsyncMock()
    store.list_all.return_value = []
    runner = CliRunner()

    with (
        patch("bot.maintenance.logfire.configure"),
        patch("bot.maintenance.load_config", return_value=config),
        patch("bot.maintenance.MemoryStore.create", AsyncMock(return_value=store)),
    ):
        result = runner.invoke(cli, ["cleanup", "--dry-run", "-c", "x.yaml"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dry_run"] is True
    store.close.assert_awaited_once()


def test_cleanup_cli_requires_memory_enabled(make_config):
    runner = CliRunner()
    with (
        patch("bot.maintenance.logfire.configure"),
        patch("bot.maintenance.load_config", return_value=make_config()),
        patch("bot.maintenance.MemoryStore.create", AsyncMock()) as create,
    ):
        result = runner.invoke(cli, ["cleanup", "-c", "x.yaml"])

    assert result.exit_code != 0
    assert "memory_enabled" in result.output
    create.assert_not_awaited()
