"""Tests for pydantic models in bot.models."""

import pytest
from pydantic import ValidationError

from bot.models import Config, CustomOpenAIModel, MiFile, Note, User


def test_config_valid(make_config):
    cfg = make_config()
    assert cfg.domain == "example.test"
    assert cfg.max_retries == 2
    assert cfg.vision is True
    assert cfg.max_context == 1
    assert cfg.http_timeout_seconds == 30.0


def test_config_missing_required_field():
    with pytest.raises(ValidationError):
        Config(domain="example.test")  # type: ignore[call-arg]


def test_config_max_tokens_must_be_positive(make_config):
    with pytest.raises(ValidationError):
        make_config(max_tokens=0)


def test_config_auto_post_requires_auto_prompt(make_config):
    with pytest.raises(ValidationError) as exc:
        make_config(auto_post_interval=60)
    assert "system_prompt_auto" in str(exc.value)


def test_config_auto_post_with_prompt_ok(make_config):
    cfg = make_config(auto_post_interval=60, system_prompt_auto="post something")
    assert cfg.auto_post_interval == 60
    assert cfg.system_prompt_auto == "post something"


def test_config_auto_reply_interval_must_be_positive(make_config):
    with pytest.raises(ValidationError):
        make_config(auto_reply_interval=0)


def test_user_extra_fields_allowed():
    u = User(id="1", username="alice", unknown_field="ok")  # type: ignore[call-arg]
    assert u.username == "alice"


def test_note_nested_reply(make_user):
    inner_user = make_user(id="2", username="bob")
    inner = Note(id="inner", text="hi", userId=inner_user.id, user=inner_user)
    outer_user = make_user()
    outer = Note(
        id="outer",
        text="reply",
        userId=outer_user.id,
        user=outer_user,
        reply=inner,
    )
    assert outer.reply is not None
    assert outer.reply.user.username == "bob"


def test_note_visibility_literal_validation(make_user):
    user = make_user()
    with pytest.raises(ValidationError):
        Note(id="n", text="t", userId=user.id, user=user, visibility="bogus")  # type: ignore[arg-type]


def test_mifile_optional_fields():
    f = MiFile(id="f", type="image/png")
    assert f.url is None
    assert f.thumbnailUrl is None


def test_config_llm_models_mixes_strings_and_custom_endpoints(make_config):
    cfg = make_config(
        llm_models=[
            "openrouter:test/model",
            {
                "model": "Qwen/Qwen3",
                "base_url": "https://example.modal.run/v1",
                "api_key_env": "MODAL_API_KEY",
            },
        ]
    )
    assert cfg.llm_models[0] == "openrouter:test/model"
    custom = cfg.llm_models[1]
    assert isinstance(custom, CustomOpenAIModel)
    assert custom.model == "Qwen/Qwen3"
    assert str(custom.base_url) == "https://example.modal.run/v1"
    assert custom.api_key_env == "MODAL_API_KEY"
    assert custom.api_key is None


def test_config_custom_endpoint_without_base_url_passes_through_string(make_config):
    cfg = make_config(llm_models=[{"model": "openrouter:foo/bar", "vision": False}])
    entry = cfg.llm_models[0]
    assert isinstance(entry, CustomOpenAIModel)
    assert entry.base_url is None
    assert entry.vision is False


def test_custom_openai_model_vision_defaults_true():
    spec = CustomOpenAIModel(
        model="Qwen/Qwen3",
        base_url="https://example.modal.run/v1",  # type: ignore[arg-type]
    )
    assert spec.vision is True


def test_memory_disabled_by_default(make_config):
    cfg = make_config()
    assert cfg.memory_enabled is False
    assert cfg.embedding_dim == 1024
    assert cfg.embedding_api_key_env == "OPENROUTER_API_KEY"
    assert "openrouter.ai" in str(cfg.embedding_base_url)


def test_memory_enabled_requires_postgres_url(make_config):
    with pytest.raises(ValidationError) as exc:
        make_config(memory_enabled=True, embedding_model="perplexity/pplx-embed-v1-0.6b")
    assert "postgres_url" in str(exc.value)


def test_memory_enabled_requires_embedding_model(make_config):
    with pytest.raises(ValidationError) as exc:
        make_config(memory_enabled=True, postgres_url="postgres://u:p@db/x")
    assert "embedding_model" in str(exc.value)


def test_memory_enabled_with_required_fields_ok(make_config):
    cfg = make_config(
        memory_enabled=True,
        postgres_url="postgres://u:p@db/x",
        embedding_model="perplexity/pplx-embed-v1-0.6b",
    )
    assert cfg.memory_enabled is True
    assert cfg.embedding_model == "perplexity/pplx-embed-v1-0.6b"
