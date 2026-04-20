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
