"""Tests for mem0 maintenance selection and CLI wiring."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from bot.maintenance import VectorRecord, cleanup, cli, plan_cleanup, reembed
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


def _vector_records() -> dict[str, list[VectorRecord]]:
    return {
        "missbot_memories": [
            VectorRecord("missbot_memories", "00000000-0000-0000-0000-000000000001", "one", 1024),
            VectorRecord("missbot_memories", "00000000-0000-0000-0000-000000000002", "two", 1024),
        ],
        "missbot_memories_entities": [
            VectorRecord("missbot_memories_entities", "00000000-0000-0000-0000-000000000003", "entity", 1024)
        ],
    }


@pytest.mark.anyio
async def test_reembed_dry_run_probes_without_writing(make_config):
    store = AsyncMock()
    store.embed_batch.return_value = [[0.0] * 1024]
    config = _memory_cfg(make_config)

    with (
        patch("bot.maintenance._load_vector_records", return_value=_vector_records()),
        patch("bot.maintenance._replace_vectors") as replace_vectors,
    ):
        summary = await reembed(store, config, dry_run=True)

    store.embed_batch.assert_awaited_once_with(["Missbot re-embedding preflight"], action="update")
    replace_vectors.assert_not_called()
    assert summary["tables"] == {"missbot_memories": 2, "missbot_memories_entities": 1}
    assert summary["updated"] == 0


@pytest.mark.anyio
async def test_reembed_batches_both_tables_and_writes_once(make_config):
    store = AsyncMock()
    store.embed_batch.side_effect = [
        [[0.0] * 1024],
        [[0.1] * 1024, [0.2] * 1024],
        [[0.3] * 1024],
    ]
    config = _memory_cfg(make_config)
    backup_tables = {
        "missbot_memories": "missbot_memories_backup_test",
        "missbot_memories_entities": "missbot_memories_entities_backup_test",
    }

    with (
        patch("bot.maintenance._load_vector_records", return_value=_vector_records()),
        patch("bot.maintenance._replace_vectors", return_value=backup_tables) as replace_vectors,
    ):
        summary = await reembed(store, config, batch_size=2, backup_suffix="test")

    assert store.embed_batch.await_count == 3
    replace_vectors.assert_called_once()
    call = replace_vectors.call_args
    assert call.args[0] == config.postgres_url
    assert call.kwargs["expected_dim"] == 1024
    assert call.kwargs["backup_suffix"] == "test"
    assert summary["updated"] == 3
    assert summary["backup_tables"] == backup_tables


@pytest.mark.anyio
async def test_reembed_refuses_wrong_endpoint_dimension(make_config):
    store = AsyncMock()
    store.embed_batch.return_value = [[0.0] * 768]
    config = _memory_cfg(make_config)

    with (
        patch("bot.maintenance._load_vector_records", return_value=_vector_records()),
        patch("bot.maintenance._replace_vectors") as replace_vectors,
    ):
        with pytest.raises(ValueError, match="dimension 768"):
            await reembed(store, config)

    replace_vectors.assert_not_called()


def test_reembed_cli_runs_and_closes_store(make_config):
    config = _memory_cfg(make_config)
    store = AsyncMock()
    runner = CliRunner()
    summary = {
        "dry_run": True,
        "embedding_model": "embed/model",
        "embedding_dim": 1024,
        "tables": {},
        "total_rows": 0,
        "probe_ok": True,
        "backup_tables": {},
        "updated": 0,
    }

    with (
        patch("bot.maintenance.logfire.configure"),
        patch("bot.maintenance.load_config", return_value=config),
        patch("bot.maintenance.MemoryStore.create", AsyncMock(return_value=store)),
        patch("bot.maintenance.reembed", AsyncMock(return_value=summary)) as reembed_mock,
    ):
        result = runner.invoke(cli, ["reembed", "--dry-run", "--batch-size", "32", "-c", "x.yaml"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dry_run"] is True
    assert reembed_mock.await_args is not None
    assert reembed_mock.await_args.kwargs["batch_size"] == 32
    store.close.assert_awaited_once()


# --- provenance labels across frontends -------------------------------------


def test_retention_covers_every_inferred_source():
    """A new frontend's source label must not become retention-exempt by accident."""
    memories = [
        _memory("misskey", "old misskey fact", created_at="2026-01-01T00:00:00+00:00"),
        _memory("acp", "old acp fact", created_at="2026-01-01T00:00:00+00:00", source="acp_prompt", author="acp:abc"),
    ]

    candidates = plan_cleanup(memories, now=NOW, note_retention_days=90, max_memories_per_author=None)

    assert {c.memory_id for c in candidates} == {"misskey", "acp"}
    assert {c.reason for c in candidates} == {"retention"}


def test_retention_still_protects_explicit_add_memory():
    memories = [
        _memory("explicit", "durable fact", created_at="2025-01-01T00:00:00+00:00", source="add_memory", author="grok")
    ]

    assert plan_cleanup(memories, now=NOW, note_retention_days=90, max_memories_per_author=None) == []


def test_author_cap_covers_acp_memories():
    memories = [
        _memory(
            f"m{i}",
            f"acp fact {i}",
            created_at=f"2026-07-1{i}T00:00:00+00:00",
            source="acp_prompt",
            author="acp:abc",
        )
        for i in range(1, 5)
    ]

    candidates = plan_cleanup(memories, now=NOW, note_retention_days=None, max_memories_per_author=2)

    # Oldest two beyond the per-author cap are selected.
    assert {c.memory_id for c in candidates} == {"m1", "m2"}
    assert {c.reason for c in candidates} == {"author_limit"}
