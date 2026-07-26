"""Tests for pure helper functions in bot.ai."""

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai import ImageUrl, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel

from bot.ai import (
    AgentDeps,
    ChatAgent,
    _CLASSIFIER_MODEL_SETTINGS,
    _enforce_length,
    _history_content,
    _normalize_for_repeat,
    _resolve_model_spec,
    _spec_supports_vision,
    _strip_leading_mentions,
)
from bot.core import HistoryTurn
from bot.models import CustomOpenAIModel


def test_history_content_text_only():
    turn = HistoryTurn(role="user", text="hi", author="alice")
    assert _history_content(turn, vision=True) == "alice: hi"


def test_history_content_with_images():
    turn = HistoryTurn(
        role="user",
        text="look",
        author="bob@remote.host",
        images=[ImageUrl(url="https://media.example/1.png")],
    )
    content = _history_content(turn, vision=True)
    assert isinstance(content, list)
    assert content[0] == "bob@remote.host: look"
    assert isinstance(content[1], ImageUrl)


def test_history_content_drops_images_when_vision_off():
    turn = HistoryTurn(
        role="user",
        text="look",
        author="alice",
        images=[ImageUrl(url="https://media.example/1.png")],
    )
    assert _history_content(turn, vision=False) == "alice: look"


def test_history_content_without_author_is_unprefixed():
    """Assistant turns carry no author, so they render as bare text."""
    turn = HistoryTurn(role="assistant", text="my prior reply")
    assert _history_content(turn, vision=True) == "my prior reply"


def test_history_content_empty_text():
    turn = HistoryTurn(role="user", text="", author="alice")
    assert _history_content(turn, vision=True) == "alice: "


def test_resolve_model_spec_passes_through_strings():
    assert _resolve_model_spec("openrouter:foo/bar") == "openrouter:foo/bar"


def test_resolve_model_spec_builds_openai_chat_model():
    spec = CustomOpenAIModel(
        model="Qwen/Qwen3",
        base_url="https://example.modal.run/v1",  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        api_key="literal-key",
    )
    model = _resolve_model_spec(spec)
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "Qwen/Qwen3"
    # base_url comes from the AsyncOpenAI client; trailing slash is normal
    assert str(model.client.base_url).startswith("https://example.modal.run/v1")
    assert model.client.api_key == "literal-key"


def test_resolve_model_spec_reads_api_key_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODAL_API_KEY", "env-key")
    spec = CustomOpenAIModel(
        model="Qwen/Qwen3",
        base_url="https://example.modal.run/v1",  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        api_key_env="MODAL_API_KEY",
    )
    model = _resolve_model_spec(spec)
    assert isinstance(model, OpenAIChatModel)
    assert model.client.api_key == "env-key"


def test_resolve_model_spec_dict_without_base_url_returns_string():
    spec = CustomOpenAIModel(model="openrouter:foo/bar", vision=False)
    assert _resolve_model_spec(spec) == "openrouter:foo/bar"


def test_spec_supports_vision_string_defaults_true():
    assert _spec_supports_vision("openrouter:foo/bar") is True


def test_spec_supports_vision_respects_dict_flag():
    spec = CustomOpenAIModel(model="openrouter:foo/bar", vision=False)
    assert _spec_supports_vision(spec) is False
    spec_default = CustomOpenAIModel(
        model="Qwen/Qwen3",
        base_url="https://example.modal.run/v1",  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    )
    assert _spec_supports_vision(spec_default) is True


def test_model_settings_identify_missbot(config):
    agent = ChatAgent(config)
    with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as project_file:
        project_version = tomllib.load(project_file)["project"]["version"]
    expected_headers = {"User-Agent": f"Missbot/{project_version}"}

    assert agent._generation_settings(30.0).get("extra_headers") == expected_headers
    assert _CLASSIFIER_MODEL_SETTINGS.get("extra_headers") == expected_headers


def test_enforce_length_passes_through_within_budget():
    validate = _enforce_length(10)
    assert validate("") == ""
    assert validate("exactly10!") == "exactly10!"  # len == limit is allowed


def test_enforce_length_retries_over_budget():
    validate = _enforce_length(10)
    with pytest.raises(ModelRetry) as exc:
        validate("this is far too long")
    # The overage and the limit are surfaced so the model can self-correct.
    assert "20" in str(exc.value)
    assert "10" in str(exc.value)


def _budget_ctx(char_budget):
    return SimpleNamespace(deps=AgentDeps(username="alice", char_budget=char_budget))


def test_enforce_budget_applies_turn_budget():
    with pytest.raises(ModelRetry) as exc:
        ChatAgent._enforce_budget(None, _budget_ctx(10), "this is far too long")
    assert "20" in str(exc.value)
    assert "10" in str(exc.value)


def test_enforce_budget_passes_within_turn_budget():
    assert ChatAgent._enforce_budget(None, _budget_ctx(10), "exactly10!") == "exactly10!"


def test_enforce_budget_uncapped_when_budget_is_none():
    """Frontends without a platform cap (ACP) pass no budget, so nothing is gated."""
    long_output = "x" * 100_000
    assert ChatAgent._enforce_budget(None, _budget_ctx(None), long_output) == long_output


def test_strip_leading_mentions():
    assert _strip_leading_mentions("@alice hi there") == "hi there"
    assert _strip_leading_mentions("@alice @bob@remote.host hello") == "hello"
    assert _strip_leading_mentions("no mention here") == "no mention here"
    # Only leading mentions are stripped; an in-body @handle stays.
    assert _strip_leading_mentions("@alice ping @bob later") == "ping @bob later"


def test_normalize_for_repeat_ignores_mentions_whitespace_case():
    # Replies that differ only by mention prefix, spacing, or case normalize equal.
    assert _normalize_for_repeat("@alice OMG OLIVE!!!") == _normalize_for_repeat("@bob   omg olive!!!")
    assert _normalize_for_repeat("hello world") != _normalize_for_repeat("hello there")


def _repeat_ctx(previous_bot_reply):
    return SimpleNamespace(deps=AgentDeps(username="alice", previous_bot_reply=previous_bot_reply))


def test_reject_verbatim_repeat_blocks_identical_reply():
    ctx = _repeat_ctx("@alice OMG OLIVE!!! what a cutie")
    # A new reply that only differs by the mention prefix is still a verbatim repeat.
    with pytest.raises(ModelRetry):
        ChatAgent._reject_verbatim_repeat(None, ctx, "@bob OMG OLIVE!!! what a cutie")


def test_reject_verbatim_repeat_allows_fresh_reply():
    ctx = _repeat_ctx("@alice OMG OLIVE!!! what a cutie")
    out = "@alice a chameleon gecko?! tell me everything"
    assert ChatAgent._reject_verbatim_repeat(None, ctx, out) == out


def test_reject_verbatim_repeat_noop_without_previous():
    ctx = _repeat_ctx(None)
    assert ChatAgent._reject_verbatim_repeat(None, ctx, "anything") == "anything"
