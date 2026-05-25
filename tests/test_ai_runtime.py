"""Tests for ChatAgent runtime behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import ImageUrl

from bot.ai import ChatAgent
from bot.models import MiFile


@pytest.mark.anyio
async def test_run_accepts_image_only_note(config, make_note):
    agent = ChatAgent(config)
    note = make_note(
        text=None,
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
    )

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="vision reply"))) as run_mock,
    ):
        result = await agent.run(note)

    assert result == "vision reply"
    assert run_mock.await_args is not None
    prompt = run_mock.await_args.args[0]
    assert isinstance(prompt, list)
    assert prompt[0] == "alice: "
    assert isinstance(prompt[1], ImageUrl)


@pytest.mark.anyio
async def test_run_routes_image_prompt_to_vision_model(make_config, make_note):
    cfg = make_config(
        llm_models=[
            {"model": "openrouter:vision/model"},
            {"model": "openrouter:text/model", "vision": False},
        ]
    )
    agent = ChatAgent(cfg)
    assert agent._has_vision_model
    assert agent._vision_model is not None

    note = make_note(
        text=None,
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
    )

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["model"] is agent._vision_model


@pytest.mark.anyio
async def test_run_skips_model_override_when_no_images(make_config, make_note):
    cfg = make_config(
        llm_models=[
            {"model": "openrouter:vision/model"},
            {"model": "openrouter:text/model", "vision": False},
        ]
    )
    agent = ChatAgent(cfg)
    note = make_note(text="hi")

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    assert "model" not in run_mock.await_args.kwargs


@pytest.mark.anyio
async def test_run_drops_images_when_no_vision_model(make_config, make_note):
    cfg = make_config(llm_models=[{"model": "openrouter:text/model", "vision": False}])
    agent = ChatAgent(cfg)
    assert not agent._has_vision_model

    note = make_note(
        text="look",
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
    )

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    prompt = run_mock.await_args.args[0]
    # Image was dropped — prompt collapses to a single string, no ImageUrl.
    assert isinstance(prompt, str)
    assert "model" not in run_mock.await_args.kwargs


@pytest.mark.anyio
async def test_run_flags_unrestricted_for_privileged_author(make_config, make_note, make_user):
    cfg = make_config(social_credit_unrestricted_user_ids=["admin-id"])
    agent = ChatAgent(cfg)
    note = make_note(text="judge bob", user=make_user(id="admin-id", username="boss"))

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    assert run_mock.await_args.kwargs["deps"].social_credit_unrestricted is True


@pytest.mark.anyio
async def test_run_restricted_for_non_privileged_author(make_config, make_note, make_user):
    cfg = make_config(social_credit_unrestricted_user_ids=["admin-id"])
    agent = ChatAgent(cfg)
    note = make_note(text="hi", user=make_user(id="rando-id", username="rando"))

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    assert run_mock.await_args.kwargs["deps"].social_credit_unrestricted is False


# ---------------------------------------------------------------------------
# Automatic, injection-resistant message scoring
# ---------------------------------------------------------------------------


def test_score_agent_built_only_with_redis_and_enabled(make_config, fake_redis):
    assert ChatAgent(make_config())._score_agent is None  # no redis
    assert ChatAgent(make_config(), redis_client=fake_redis)._score_agent is not None
    disabled = ChatAgent(make_config(social_credit_auto_score=False), redis_client=fake_redis)
    assert disabled._score_agent is None


def test_score_model_defaults_to_main_model(make_config, fake_redis):
    # With no score_models, the classifier reuses the main reply model chain.
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    assert agent._score_model == "openrouter:test/model"


def test_score_model_uses_separate_chain_when_configured(make_config, fake_redis):
    cfg = make_config(score_models=["openrouter:cheap/classifier"])
    agent = ChatAgent(cfg, redis_client=fake_redis)
    assert agent._score_model == "openrouter:cheap/classifier"
    assert agent._score_agent is not None


@pytest.mark.anyio
async def test_run_auto_scores_non_privileged_message(make_config, make_note, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    note = make_note(text="what a lovely day")  # author = alice

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="good"))) as score_mock,
    ):
        out = await agent.run(note)

    assert out == "reply"
    assert score_mock.await_count == 1
    # "good" -> +5 (see QUALITY_DELTAS); applied to the author and cooldown claimed.
    assert await fake_redis.get("score:alice") == "5"
    assert await fake_redis.exists("score_cooldown:alice")


@pytest.mark.anyio
async def test_run_scoring_respects_cooldown(make_config, make_note, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    await fake_redis.set("score_cooldown:alice", "1")  # already on cooldown
    note = make_note(text="another banger")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="good"))) as score_mock,
    ):
        await agent.run(note)

    # Classifier is not even consulted while on cooldown, and no score is applied.
    assert score_mock.await_count == 0
    assert await fake_redis.get("score:alice") is None


@pytest.mark.anyio
async def test_run_scoring_skips_neutral_without_consuming_cooldown(make_config, make_note, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    note = make_note(text="ok")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="neutral"))),
    ):
        await agent.run(note)

    # Neutral => delta 0 => nothing written and cooldown NOT claimed (free retry).
    assert await fake_redis.get("score:alice") is None
    assert not await fake_redis.exists("score_cooldown:alice")


@pytest.mark.anyio
async def test_run_skips_scoring_for_privileged_author(make_config, make_note, make_user, fake_redis):
    cfg = make_config(social_credit_unrestricted_user_ids=["user-1"])
    agent = ChatAgent(cfg, redis_client=fake_redis)
    note = make_note(text="trust me", user=make_user(id="user-1", username="operator"))

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(
            agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="exceptional"))
        ) as score_mock,
    ):
        await agent.run(note)

    # Privileged authors are scored manually, never auto-scored.
    assert score_mock.await_count == 0
    assert await fake_redis.get("score:operator") is None


@pytest.mark.anyio
async def test_run_scoring_failure_does_not_break_reply(make_config, make_note, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    note = make_note(text="hi")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(side_effect=RuntimeError("classifier down"))),
    ):
        out = await agent.run(note)

    # Scoring swallows its own errors; the reply still comes back.
    assert out == "reply"
    assert await fake_redis.get("score:alice") is None
