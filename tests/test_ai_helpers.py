"""Tests for pure helper functions in bot.ai."""

from types import SimpleNamespace

import pytest
from pydantic_ai import ImageUrl, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel

from bot.ai import (
    AgentDeps,
    ChatAgent,
    _build_user_content,
    _enforce_length,
    _image_urls_for,
    _normalize_for_repeat,
    _resolve_model_spec,
    _spec_supports_vision,
    _strip_leading_mentions,
    _user_handle,
)
from bot.models import CustomOpenAIModel, MiFile


def test_user_handle_local(make_user):
    assert _user_handle(make_user(username="alice", host=None)) == "alice"


def test_user_handle_remote(make_user):
    assert _user_handle(make_user(username="alice", host="remote.host")) == "alice@remote.host"


def test_image_urls_for_vision_off(make_note):
    note = make_note(files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")])
    assert _image_urls_for(note, vision=False) == []


def test_image_urls_for_includes_image_and_video_thumbnails(make_note):
    files = [
        # Image: thumbnail used.
        MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png"),
        # Video: Misskey renders an image thumbnail — use it (NOT the raw video url).
        MiFile(
            id="f2",
            type="video/mp4",
            thumbnailUrl="https://media.example/2-thumb.jpg",
            url="https://media.example/2.mp4",
        ),
        # Video with no thumbnail: skipped (we must not feed the video file as an image).
        MiFile(id="f3", type="video/mp4", thumbnailUrl=None, url="https://media.example/3.mp4"),
        # Non-visual media: skipped.
        MiFile(id="f4", type="audio/mpeg", thumbnailUrl="https://media.example/4.png"),
        # Image with no thumbnail or url: skipped.
        MiFile(id="f5", type="image/jpeg", thumbnailUrl=None),
    ]
    note = make_note(files=files)
    urls = _image_urls_for(note, vision=True)
    assert all(isinstance(u, ImageUrl) for u in urls)
    assert [u.url for u in urls] == ["https://media.example/1.png", "https://media.example/2-thumb.jpg"]


def test_image_urls_for_no_files(make_note):
    note = make_note(files=None)
    assert _image_urls_for(note, vision=True) == []


def test_image_urls_for_drops_ssrf_urls(make_note):
    """Attacker-controlled internal URLs are dropped even with a spoofed image type."""
    files = [
        MiFile(id="f1", type="image/png", thumbnailUrl="http://169.254.169.254/latest/meta-data/"),
        MiFile(id="f2", type="image/png", url="http://missbot-redis.misskey.svc.cluster.local:6379/"),
        MiFile(id="f3", type="image/png", thumbnailUrl="https://media.example/ok.png"),
    ]
    note = make_note(files=files)
    urls = _image_urls_for(note, vision=True)
    # Only the public URL survives.
    assert [u.url for u in urls] == ["https://media.example/ok.png"]


def test_build_user_content_text_only(make_note, make_user):
    note = make_note(user=make_user(username="alice"), text="hi")
    content = _build_user_content(note, vision=True)
    assert content == "alice: hi"


def test_build_user_content_with_images(make_note, make_user):
    note = make_note(
        user=make_user(username="bob", host="remote.host"),
        text="look",
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
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
