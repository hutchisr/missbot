"""Tests for the maintenance CLI (bot.maintenance) — store calls are mocked, no live PG."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from bot.maintenance import cli
from bot.memory import EntityNeighbor


@pytest.fixture
def fake_store() -> AsyncMock:
    store = AsyncMock()
    store.consolidate.return_value = {"merged_by_name": 2, "merged_by_embedding": 0, "merged_by_llm": 1}
    store.stats.return_value = {"entities": 10, "edges": 40, "sources": 12, "authors": 8}
    store.reembed_all.return_value = {"entities": 10, "relations": 14}
    store.entity_neighbors.return_value = [
        EntityNeighbor("Python", "Python (programming language)", 0.94),
        EntityNeighbor("Java", "Java (island)", 0.71),
    ]
    return store


def _memory_cfg(make_config):
    return make_config(memory_enabled=True, postgres_url="postgres://u:p@db/x", embedding_model="m")


def _invoke(args, cfg, fake_store):
    runner = CliRunner()
    with (
        patch("bot.maintenance.logfire.configure"),
        patch("bot.maintenance.load_config", return_value=cfg),
        patch("bot.maintenance.build_entity_linker", return_value=None),
        patch("bot.maintenance.MemoryStore.create", AsyncMock(return_value=fake_store)),
    ):
        return runner.invoke(cli, args)


def test_consolidate_runs_and_closes(make_config, fake_store):
    result = _invoke(["consolidate", "-c", "x.yaml"], _memory_cfg(make_config), fake_store)
    assert result.exit_code == 0, result.output
    fake_store.consolidate.assert_awaited_once()
    fake_store.close.assert_awaited_once()
    assert "merged_by_name" in result.output


def test_stats_prints_json(make_config, fake_store):
    result = _invoke(["stats", "-c", "x.yaml"], _memory_cfg(make_config), fake_store)
    assert result.exit_code == 0, result.output
    fake_store.stats.assert_awaited_once()
    assert json.loads(result.output) == {"entities": 10, "edges": 40, "sources": 12, "authors": 8}


def test_reembed_runs_and_prints_json(make_config, fake_store):
    result = _invoke(["reembed", "-c", "x.yaml"], _memory_cfg(make_config), fake_store)
    assert result.exit_code == 0, result.output
    fake_store.reembed_all.assert_awaited_once()
    fake_store.close.assert_awaited_once()
    assert json.loads(result.output) == {"entities": 10, "relations": 14}


def test_calibrate_entities_lists_pairs_and_threshold(make_config, fake_store):
    cfg = _memory_cfg(make_config)  # entity_match_threshold defaults to 0.82
    result = _invoke(["calibrate-entities", "-c", "x.yaml", "--min-similarity", "0.5"], cfg, fake_store)
    assert result.exit_code == 0, result.output
    fake_store.entity_neighbors.assert_awaited_once_with(50, 0.5)
    assert "current entity_match_threshold = 0.82" in result.output
    assert "0.9400" in result.output and "Python (programming language)" in result.output
    # The 0.71 pair is below the 0.82 threshold and is annotated as such.
    assert "below threshold" in result.output


def test_calibrate_entities_handles_no_pairs(make_config, fake_store):
    fake_store.entity_neighbors.return_value = []
    result = _invoke(["calibrate-entities", "-c", "x.yaml"], _memory_cfg(make_config), fake_store)
    assert result.exit_code == 0, result.output
    assert "No entity pairs" in result.output


def test_requires_memory_enabled(make_config, fake_store):
    # memory_enabled defaults False -> the command errors before building a store.
    result = _invoke(["consolidate", "-c", "x.yaml"], make_config(), fake_store)
    assert result.exit_code != 0
    assert "memory_enabled" in result.output
    fake_store.consolidate.assert_not_awaited()
