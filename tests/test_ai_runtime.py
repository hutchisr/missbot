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
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://x/1.png")],
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
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://x/1.png")],
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
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://x/1.png")],
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
