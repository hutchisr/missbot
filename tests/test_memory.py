"""Tests for the mem0 MemoryStore adapter."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.memory import MemorySearchResult, MemoryStore, _mem0_config, _suppress_openrouter_autodetect


def _memory_cfg(make_config, **extra):
    return make_config(
        memory_enabled=True,
        postgres_url="postgres://u:p@db/x",
        embedding_model="perplexity/pplx-embed-v1-0.6b",
        **extra,
    )


def test_mem0_config_uses_pgvector_and_openrouter_embedding(make_config):
    cfg = _memory_cfg(make_config, memory_llm_model="anthropic/claude-3-haiku")

    mem0_cfg = _mem0_config(cfg)

    assert mem0_cfg["vector_store"]["provider"] == "pgvector"
    assert mem0_cfg["vector_store"]["config"]["connection_string"] == "postgres://u:p@db/x"
    assert mem0_cfg["vector_store"]["config"]["collection_name"] == "missbot_memories"
    assert mem0_cfg["vector_store"]["config"]["embedding_model_dims"] == 1024
    assert mem0_cfg["embedder"]["config"]["model"] == "perplexity/pplx-embed-v1-0.6b"
    assert mem0_cfg["llm"]["config"]["model"] == "anthropic/claude-3-haiku"


def test_mem0_config_strips_pydantic_ai_provider_prefix(make_config):
    cfg = _memory_cfg(make_config, llm_models=["openrouter:anthropic/claude-3-haiku"])

    assert _mem0_config(cfg)["llm"]["config"]["model"] == "anthropic/claude-3-haiku"


def test_mem0_config_passes_embedding_dimensions_only_when_set(make_config):
    cfg = _memory_cfg(make_config)
    assert "embedding_dims" not in _mem0_config(cfg)["embedder"]["config"]

    cfg = _memory_cfg(make_config, embedding_dimensions=1024)
    assert _mem0_config(cfg)["embedder"]["config"]["embedding_dims"] == 1024


def test_mem0_config_always_sets_explicit_llm_base_url(make_config):
    # mem0's "openai" LLM provider ignores config.api_key/openai_base_url whenever
    # OPENROUTER_API_KEY is set in the environment (which it always is for this bot) and
    # falls back to its own OpenRouter auto-detection instead. _mem0_config must never leave
    # openai_base_url unset, or that auto-detection silently overrides our config.
    cfg = _memory_cfg(make_config)
    assert _mem0_config(cfg)["llm"]["config"]["openai_base_url"] == "https://openrouter.ai/api/v1"

    cfg = _memory_cfg(make_config, memory_llm_base_url="https://my-endpoint.internal/v1")
    assert _mem0_config(cfg)["llm"]["config"]["openai_base_url"] == "https://my-endpoint.internal/v1"


def test_mem0_config_inherits_first_custom_model_literal_api_key(make_config, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    cfg = _memory_cfg(
        make_config,
        llm_models=[
            {
                "model": "custom:model-v1",
                "base_url": "https://custom.example/v1",
                "api_key": "sk-custom",
            }
        ],
    )

    llm = _mem0_config(cfg)["llm"]["config"]
    assert llm["model"] == "custom:model-v1"
    assert llm["openai_base_url"] == "https://custom.example/v1"
    assert llm["api_key"] == "sk-custom"


def test_mem0_config_inherits_first_custom_model_api_key_env(make_config, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    monkeypatch.setenv("CUSTOM_LLM_API_KEY", "sk-custom-env")
    cfg = _memory_cfg(
        make_config,
        llm_models=[
            {
                "model": "custom/model-v1",
                "base_url": "https://custom.example/v1",
                "api_key_env": "CUSTOM_LLM_API_KEY",
            }
        ],
    )

    llm = _mem0_config(cfg)["llm"]["config"]
    assert llm == {
        "model": "custom/model-v1",
        "temperature": 0.1,
        "api_key": "sk-custom-env",
        "openai_base_url": "https://custom.example/v1",
    }


def test_mem0_config_does_not_send_default_openrouter_key_to_custom_endpoint(make_config, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    cfg = _memory_cfg(
        make_config,
        llm_models=[{"model": "custom/model-v1", "base_url": "https://custom.example/v1"}],
    )

    llm = _mem0_config(cfg)["llm"]["config"]
    assert llm["openai_base_url"] == "https://custom.example/v1"
    assert "api_key" not in llm


def test_mem0_config_explicit_memory_llm_overrides_inherited_connection(make_config):
    cfg = _memory_cfg(
        make_config,
        llm_models=[
            {
                "model": "custom/model-v1",
                "base_url": "https://custom.example/v1",
                "api_key": "sk-custom",
            }
        ],
        memory_llm_model="override/model-v2",
        memory_llm_base_url="https://override.example/v1",
        memory_llm_api_key="sk-override",
    )

    llm = _mem0_config(cfg)["llm"]["config"]
    assert llm == {
        "model": "override/model-v2",
        "temperature": 0.1,
        "api_key": "sk-override",
        "openai_base_url": "https://override.example/v1",
    }


def test_suppress_openrouter_autodetect_hides_and_restores_env(make_config, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-outer")
    with _suppress_openrouter_autodetect():
        assert "OPENROUTER_API_KEY" not in os.environ
    assert os.environ["OPENROUTER_API_KEY"] == "sk-outer"


def test_suppress_openrouter_autodetect_noop_when_unset(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with _suppress_openrouter_autodetect():
        assert "OPENROUTER_API_KEY" not in os.environ
    assert "OPENROUTER_API_KEY" not in os.environ


@pytest.mark.anyio
async def test_create_builds_async_memory(make_config):
    cfg = _memory_cfg(make_config)
    client = MagicMock()

    with patch("bot.memory.AsyncMemory.from_config", return_value=client) as from_config:
        store = await MemoryStore.create(cfg)

    assert isinstance(store, MemoryStore)
    from_config.assert_called_once()
    assert from_config.call_args.args[0]["vector_store"]["provider"] == "pgvector"


@pytest.mark.anyio
async def test_create_hides_openrouter_key_from_mem0_construction(make_config, monkeypatch):
    # Regression test: MemoryStore.create must suppress OPENROUTER_API_KEY for the duration of
    # AsyncMemory.from_config so mem0's OpenAI provider honors our resolved api_key/base_url
    # instead of silently routing to OpenRouter with the raw env var.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-outer")
    cfg = _memory_cfg(make_config, memory_llm_base_url="https://my-endpoint.internal/v1", memory_llm_api_key="sk-inner")
    seen_env: dict[str, object] = {}

    def _fake_from_config(mem0_config):
        seen_env["during"] = os.environ.get("OPENROUTER_API_KEY")
        return MagicMock()

    with patch("bot.memory.AsyncMemory.from_config", side_effect=_fake_from_config):
        await MemoryStore.create(cfg)

    assert seen_env["during"] is None
    assert os.environ["OPENROUTER_API_KEY"] == "sk-outer"


@pytest.mark.anyio
async def test_add_note_scopes_to_bot_agent_and_records_author(make_config):
    client = AsyncMock()
    store = MemoryStore(client, _memory_cfg(make_config))

    await store.add_note(text="I use Arch btw", author="Alice@Remote.Example", note_id="note-1")

    client.add.assert_awaited_once()
    kwargs = client.add.await_args.kwargs
    assert kwargs["agent_id"] == "grok"
    assert kwargs["infer"] is True
    assert kwargs["metadata"] == {
        "source": "misskey_note",
        "author": "alice@remote.example",
        "source_note_id": "note-1",
    }
    assert client.add.await_args.args[0][0]["content"] == "Alice@Remote.Example: I use Arch btw"


@pytest.mark.anyio
async def test_add_saves_bot_authored_memory(make_config):
    client = AsyncMock()
    store = MemoryStore(client, _memory_cfg(make_config))

    await store.add("the instance mascot is a shrimp")

    client.add.assert_awaited_once()
    assert client.add.await_args.kwargs["agent_id"] == "grok"
    assert client.add.await_args.kwargs["metadata"] == {"source": "add_memory", "author": "grok"}


@pytest.mark.anyio
async def test_search_maps_mem0_results(make_config):
    client = AsyncMock()
    client.search.return_value = {
        "results": [
            {
                "memory": "Python's latest version is 3.13",
                "score": 0.92,
                "updated_at": "2026-03-01T00:00:00Z",
                "metadata": {"author": "alice"},
            },
            {"memory": ""},
        ]
    }
    store = MemoryStore(client, _memory_cfg(make_config, memory_search_threshold=0.4))

    results = await store.search("python version", 3)

    client.search.assert_awaited_once_with(
        "python version",
        top_k=3,
        filters={"agent_id": "grok"},
        threshold=0.4,
    )
    assert results == [
        MemorySearchResult(
            memory="Python's latest version is 3.13",
            score=0.92,
            updated_at="2026-03-01T00:00:00Z",
            metadata={"author": "alice"},
        )
    ]
