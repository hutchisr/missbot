"""Tests for pure helper functions in bot.ai."""

import pytest
from pydantic_ai import ImageUrl
from pydantic_ai.models.openai import OpenAIChatModel

from bot.ai import (
    _build_user_content,
    _image_urls_for,
    _resolve_model_spec,
    _spec_supports_vision,
    _user_handle,
)
from bot.models import CustomOpenAIModel, MiFile


def test_user_handle_local(make_user):
    assert _user_handle(make_user(username="alice", host=None)) == "alice"


def test_user_handle_remote(make_user):
    assert _user_handle(make_user(username="alice", host="remote.host")) == "alice@remote.host"


def test_image_urls_for_vision_off(make_note):
    note = make_note(files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://x/1.png")])
    assert _image_urls_for(note, vision=False) == []


def test_image_urls_for_filters_non_images(make_note):
    files = [
        MiFile(id="f1", type="image/png", thumbnailUrl="https://x/1.png"),
        MiFile(id="f2", type="video/mp4", thumbnailUrl="https://x/2.mp4"),
        MiFile(id="f3", type="image/jpeg", thumbnailUrl=None),
    ]
    note = make_note(files=files)
    urls = _image_urls_for(note, vision=True)
    assert len(urls) == 1
    assert isinstance(urls[0], ImageUrl)
    assert urls[0].url == "https://x/1.png"


def test_image_urls_for_no_files(make_note):
    note = make_note(files=None)
    assert _image_urls_for(note, vision=True) == []


def test_build_user_content_text_only(make_note, make_user):
    note = make_note(user=make_user(username="alice"), text="hi")
    content = _build_user_content(note, vision=True)
    assert content == "alice: hi"


def test_build_user_content_with_images(make_note, make_user):
    note = make_note(
        user=make_user(username="bob", host="remote.host"),
        text="look",
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://x/1.png")],
    )
    content = _build_user_content(note, vision=True)
    assert isinstance(content, list)
    assert content[0] == "bob@remote.host: look"
    assert isinstance(content[1], ImageUrl)


def test_build_user_content_empty_text(make_note):
    note = make_note(text=None)
    content = _build_user_content(note, vision=True)
    assert content == "alice: "


def test_resolve_model_spec_passes_through_strings():
    assert _resolve_model_spec("openrouter:foo/bar") == "openrouter:foo/bar"


def test_resolve_model_spec_builds_openai_chat_model():
    spec = CustomOpenAIModel(
        model="Qwen/Qwen3",
        base_url="https://example.modal.run/v1",  # type: ignore[arg-type]
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
        base_url="https://example.modal.run/v1",  # type: ignore[arg-type]
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
        base_url="https://example.modal.run/v1",  # type: ignore[arg-type]
    )
    assert _spec_supports_vision(spec_default) is True
