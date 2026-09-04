"""Behavioral tests for the Hindsight MemoryStore adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.memory import MemorySearchResult, MemoryStore
from bot.provider import PROJECT_VERSION


def _memory_cfg(make_config, **extra):
    return make_config(memory_enabled=True, **extra)


def _client(*, retained: bool = True) -> MagicMock:
    client = MagicMock()
    client.acreate_bank = AsyncMock()
    client.aretain = AsyncMock(return_value=SimpleNamespace(success=retained, items_count=int(retained)))
    client.arecall = AsyncMock(return_value=SimpleNamespace(results=[]))
    client.aclose = AsyncMock()
    return client


def _recall_result(
    text: str,
    *,
    score: float | None = None,
    memory_type: str = "world",
    metadata: dict[str, str] | None = None,
    mentioned_at: str | None = None,
):
    return SimpleNamespace(
        text=text,
        scores=SimpleNamespace(final=score) if score is not None else None,
        mentioned_at=mentioned_at,
        occurred_start=None,
        type=memory_type,
        metadata=metadata,
    )


def test_bank_defaults_to_normalized_bot_username(make_config):
    store = MemoryStore(_client(), _memory_cfg(make_config, bot_username="@Grok"))

    assert store.bank_id == "grok"


def test_explicit_bank_id_is_preserved(make_config):
    store = MemoryStore(_client(), _memory_cfg(make_config, hindsight_bank_id="shared-missbot"))

    assert store.bank_id == "shared-missbot"


@pytest.mark.anyio
async def test_create_initializes_shared_bank_with_missbot_identity(make_config, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_KEY", "hindsight-secret")
    config = _memory_cfg(
        make_config,
        hindsight_base_url="https://hindsight.example.test",
        hindsight_bank_id="shared-missbot",
        hindsight_retain_mission="Keep durable instance lore.",
    )
    client = _client()

    with patch("bot.memory.Hindsight", return_value=client) as hindsight:
        store = await MemoryStore.create(config)

    assert store.bank_id == "shared-missbot"
    hindsight.assert_called_once_with(
        base_url="https://hindsight.example.test",
        api_key="hindsight-secret",
        timeout=config.http_timeout_seconds,
        user_agent=f"Missbot/{PROJECT_VERSION}",
    )
    client.acreate_bank.assert_awaited_once_with(
        bank_id="shared-missbot",
        name="grok",
        retain_mission="Keep durable instance lore.",
    )


@pytest.mark.anyio
async def test_explicit_api_key_wins_over_environment(make_config, monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_KEY", "environment-secret")
    config = _memory_cfg(make_config, hindsight_api_key="explicit-secret")
    client = _client()

    with patch("bot.memory.Hindsight", return_value=client) as hindsight:
        await MemoryStore.create(config)

    assert hindsight.call_args.kwargs["api_key"] == "explicit-secret"


@pytest.mark.anyio
async def test_create_closes_client_when_bank_initialization_fails(make_config):
    client = _client()
    client.acreate_bank.side_effect = RuntimeError("bank unavailable")

    with (
        patch("bot.memory.Hindsight", return_value=client),
        pytest.raises(RuntimeError, match="bank unavailable"),
    ):
        await MemoryStore.create(_memory_cfg(make_config))

    client.aclose.assert_awaited_once_with()


@pytest.mark.anyio
async def test_add_note_retains_provenance_and_idempotent_document_id(make_config):
    client = _client()
    store = MemoryStore(client, _memory_cfg(make_config))
    note = {
        "text": "I use Arch btw",
        "author": "Alice@Remote.Example",
        "author_user_id": "user-1",
        "note_id": "note-1",
        "source": "misskey_note",
    }

    assert await store.add_note(**note) is True
    assert await store.add_note(**note) is True
    first = client.aretain.await_args_list[0].kwargs
    second = client.aretain.await_args_list[1].kwargs

    assert first["bank_id"] == "grok"
    assert first["content"] == "alice@remote.example: I use Arch btw"
    assert first["context"] == "Public misskey_note message authored by @alice@remote.example"
    assert first["metadata"] == {
        "source": "misskey_note",
        "author": "alice@remote.example",
        "author_user_id": "user-1",
        "source_note_id": "note-1",
    }
    assert first["document_id"].startswith("misskey_note:note-1:")
    assert first["document_id"] == second["document_id"]
    assert first["update_mode"] == "replace"
    assert first["retain_async"] is False


@pytest.mark.anyio
async def test_add_note_document_id_changes_with_content(make_config):
    client = _client()
    store = MemoryStore(client, _memory_cfg(make_config))

    await store.add_note(text="first", author="alice", note_id="acp:session-1", source="acp_prompt")
    await store.add_note(text="second", author="alice", note_id="acp:session-1", source="acp_prompt")

    first_id = client.aretain.await_args_list[0].kwargs["document_id"]
    second_id = client.aretain.await_args_list[1].kwargs["document_id"]
    assert first_id != second_id


@pytest.mark.anyio
async def test_add_explicit_memory_is_marked_as_explicit(make_config):
    client = _client()
    store = MemoryStore(client, _memory_cfg(make_config))

    assert await store.add("The instance mascot is a shrimp") is True

    call = client.aretain.await_args.kwargs
    assert call["content"] == "The instance mascot is a shrimp"
    assert call["metadata"] == {"source": "add_memory", "author": "grok"}
    assert call["document_id"].startswith("add_memory:")
    assert call["update_mode"] == "replace"
    assert call["retain_async"] is False


@pytest.mark.anyio
async def test_search_recalls_raw_provenance_bearing_facts(make_config):
    client = _client()
    client.arecall.return_value = SimpleNamespace(
        results=[
            _recall_result(
                "Alice uses Arch",
                score=0.91,
                memory_type="world",
                metadata={"source": "misskey_note", "author": "alice"},
                mentioned_at="2026-09-04T12:00:00Z",
            ),
            _recall_result("Alice installed it yesterday", memory_type="experience"),
            _recall_result("limit excludes this"),
        ]
    )
    config = _memory_cfg(make_config, hindsight_recall_budget="high", hindsight_recall_max_tokens=2048)
    store = MemoryStore(client, config)

    memories = await store.search("What does Alice use?", limit=2)

    client.arecall.assert_awaited_once_with(
        bank_id="grok",
        query="What does Alice use?",
        types=["world", "experience"],
        max_tokens=2048,
        budget="high",
    )
    assert memories == [
        MemorySearchResult(
            memory="Alice uses Arch",
            score=0.91,
            created_at="2026-09-04T12:00:00Z",
            memory_type="world",
            metadata={"source": "misskey_note", "author": "alice"},
        ),
        MemorySearchResult(memory="Alice installed it yesterday", memory_type="experience"),
    ]


@pytest.mark.anyio
async def test_search_drops_blank_results(make_config):
    client = _client()
    client.arecall.return_value = SimpleNamespace(results=[_recall_result("  ")])
    store = MemoryStore(client, _memory_cfg(make_config))

    assert await store.search("anything", limit=5) == []


@pytest.mark.anyio
async def test_close_closes_official_client(make_config):
    client = _client()
    store = MemoryStore(client, _memory_cfg(make_config))

    await store.close()

    client.aclose.assert_awaited_once_with()
