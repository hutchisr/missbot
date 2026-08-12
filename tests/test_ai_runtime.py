"""Tests for ChatAgent runtime behavior.

`ChatAgent` speaks frontend-neutral `AgentTurn`s. The Misskey-specific translation
(attachments, visibility, privileged user ids) is covered in `test_bot_utils.py`.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import ImageUrl, ModelRetry

from bot.ai import _IMAGE_ONLY_OUTPUT, AutoDeps, ChatAgent, _make_create_poll_tool, _make_generate_image_tool
from bot.core import HistoryTurn, Poll
from bot.imagegen import GeneratedImage

_IMAGE = ImageUrl(url="https://media.example/1.png")


@pytest.mark.anyio
async def test_run_accepts_image_only_turn(config, make_turn):
    agent = ChatAgent(config)
    turn = make_turn(text="", images=[_IMAGE])

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="vision reply"))) as run_mock,
    ):
        result = await agent.run(turn)

    assert result == "vision reply"
    assert run_mock.await_args is not None
    prompt = run_mock.await_args.args[0]
    assert isinstance(prompt, list)
    assert prompt[0] == "alice: "
    assert isinstance(prompt[1], ImageUrl)


@pytest.mark.anyio
async def test_run_rejects_turn_with_no_text_or_images(config, make_turn):
    agent = ChatAgent(config)
    with pytest.raises(ValueError):
        await agent.run(make_turn(text=""))


@pytest.mark.anyio
async def test_run_routes_image_prompt_to_vision_model(make_config, make_turn):
    cfg = make_config(
        llm_models=[
            {"model": "openrouter:vision/model"},
            {"model": "openrouter:text/model", "vision": False},
        ]
    )
    agent = ChatAgent(cfg)
    assert agent._has_vision_model
    assert agent._vision_model is not None

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(make_turn(text="", images=[_IMAGE]))

    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["model"] is agent._vision_model


@pytest.mark.anyio
async def test_run_skips_model_override_when_no_images(make_config, make_turn):
    cfg = make_config(
        llm_models=[
            {"model": "openrouter:vision/model"},
            {"model": "openrouter:text/model", "vision": False},
        ]
    )
    agent = ChatAgent(cfg)

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(make_turn(text="hi"))

    assert run_mock.await_args is not None
    assert "model" not in run_mock.await_args.kwargs


@pytest.mark.anyio
async def test_run_drops_images_when_no_vision_model(make_config, make_turn):
    cfg = make_config(llm_models=[{"model": "openrouter:text/model", "vision": False}])
    agent = ChatAgent(cfg)
    assert not agent._has_vision_model

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(make_turn(text="look", images=[_IMAGE]))

    assert run_mock.await_args is not None
    prompt = run_mock.await_args.args[0]
    # Image was dropped — prompt collapses to a single string, no ImageUrl.
    assert isinstance(prompt, str)
    assert "model" not in run_mock.await_args.kwargs


@pytest.mark.anyio
async def test_run_prepends_author_location(config, make_turn):
    agent = ChatAgent(config)

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(make_turn(text="hi", location="Berlin"))

    assert run_mock.await_args is not None
    prompt = run_mock.await_args.args[0]
    assert prompt == ["User location: Berlin", "alice: hi"]


@pytest.mark.anyio
async def test_run_uses_display_for_prompt_and_handle_for_state(make_config, make_turn, fake_redis):
    """The prompt shows the readable handle; scoring keys off the identity handle."""
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    turn = make_turn(text="hi", handle="acp:abc123", display="alice (acp)")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="good"))),
    ):
        await agent.run(turn)

    assert run_mock.await_args is not None
    assert run_mock.await_args.args[0] == "alice (acp): hi"
    assert run_mock.await_args.kwargs["deps"].username == "alice (acp)"
    assert await fake_redis.get("score:acp:abc123") == "5"


@pytest.mark.anyio
async def test_run_passes_char_budget_to_deps(config, make_turn):
    agent = ChatAgent(config)

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(make_turn(text="hi", char_budget=2720))

    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["deps"].char_budget == 2720


@pytest.mark.anyio
async def test_run_uncapped_when_turn_has_no_budget(config, make_turn):
    agent = ChatAgent(config)

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(make_turn(text="hi"))

    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["deps"].char_budget is None


@pytest.mark.anyio
async def test_run_builds_message_history_from_turn(config, make_turn):
    agent = ChatAgent(config)
    history = [
        HistoryTurn(role="user", text="first", author="bob"),
        HistoryTurn(role="assistant", text="my answer"),
    ]

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(make_turn(text="second", history=history))

    assert run_mock.await_args is not None
    messages = run_mock.await_args.kwargs["message_history"]
    assert [m.parts[0].content for m in messages] == ["bob: first", "my answer"]


@pytest.mark.anyio
async def test_run_flags_unrestricted_for_privileged_author(config, make_turn):
    agent = ChatAgent(config)

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(make_turn(text="judge bob", handle="boss", privileged=True))

    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["deps"].social_credit_unrestricted is True


@pytest.mark.anyio
async def test_run_restricted_for_non_privileged_author(config, make_turn):
    agent = ChatAgent(config)

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(make_turn(text="hi", handle="rando"))

    assert run_mock.await_args is not None
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
async def test_run_auto_scores_non_privileged_message(make_config, make_turn, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="good"))) as score_mock,
    ):
        out = await agent.run(make_turn(text="what a lovely day"))

    assert out == "reply"
    assert score_mock.await_count == 1
    # "good" -> +5 (see QUALITY_DELTAS); applied to the author and cooldown claimed.
    assert await fake_redis.get("score:alice") == "5"
    assert await fake_redis.exists("score_cooldown:alice")


@pytest.mark.anyio
async def test_run_scoring_respects_cooldown(make_config, make_turn, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    await fake_redis.set("score_cooldown:alice", "1")  # already on cooldown

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="good"))) as score_mock,
    ):
        await agent.run(make_turn(text="another banger"))

    # Classifier is not even consulted while on cooldown, and no score is applied.
    assert score_mock.await_count == 0
    assert await fake_redis.get("score:alice") is None


@pytest.mark.anyio
async def test_run_scoring_skips_neutral_without_consuming_cooldown(make_config, make_turn, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="neutral"))),
    ):
        await agent.run(make_turn(text="ok"))

    # Neutral => delta 0 => nothing written and cooldown NOT claimed (free retry).
    assert await fake_redis.get("score:alice") is None
    assert not await fake_redis.exists("score_cooldown:alice")


@pytest.mark.anyio
async def test_run_scoring_passes_history_as_context(make_config, make_turn, fake_redis):
    """The classifier sees the prior conversation so tone is judged in context."""
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    history = [
        HistoryTurn(role="user", text="earlier question", author="bob"),
        HistoryTurn(role="assistant", text="earlier answer"),
    ]

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="good"))) as score_mock,
    ):
        await agent.run(make_turn(text="short", history=history))

    assert score_mock.await_args is not None
    scoring_prompt = score_mock.await_args.args[0]
    assert "bob: earlier question" in scoring_prompt
    # The bot's own turns render under the configured bot handle.
    assert "grok: earlier answer" in scoring_prompt


@pytest.mark.anyio
async def test_run_auto_scores_privileged_author_too(make_config, make_turn, fake_redis):
    """Privileged users are also auto-scored; the flag only gates the manual tool."""
    agent = ChatAgent(make_config(), redis_client=fake_redis)

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(
            agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="exceptional"))
        ) as score_mock,
    ):
        await agent.run(make_turn(text="great thread", handle="operator", privileged=True))

    assert score_mock.await_count == 1
    # "exceptional" -> +10 (see QUALITY_DELTAS).
    assert await fake_redis.get("score:operator") == "10"


@pytest.mark.anyio
async def test_run_scoring_failure_does_not_break_reply(make_config, make_turn, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(side_effect=RuntimeError("classifier down"))),
    ):
        out = await agent.run(make_turn(text="hi"))

    # Scoring swallows its own errors; the reply still comes back.
    assert out == "reply"
    assert await fake_redis.get("score:alice") is None
    # A failed classification must not burn the cooldown — it should retry next message.
    assert not await fake_redis.exists("score_cooldown:alice")


# --- turn -> mem0 ingestion ---


def _memory_cfg(make_config, **extra):
    return make_config(
        memory_enabled=True,
        postgres_url="postgres://u:p@db/x",
        embedding_model="perplexity/pplx-embed-v1-0.6b",
        **extra,
    )


@pytest.mark.anyio
async def test_run_ingests_turn_with_mem0(make_config, make_turn):
    mem = AsyncMock()
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)

    with patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))):
        out = await agent.run(make_turn(text="Python's latest version is 3.13"))

    assert out == "reply"
    mem.add_note.assert_awaited_once_with(
        text="Python's latest version is 3.13",
        author="alice",
        author_user_id=None,
        note_id="note-1",
        source="unknown",
    )


@pytest.mark.anyio
async def test_run_ingestion_disabled_by_flag(make_config, make_turn):
    mem = AsyncMock()
    agent = ChatAgent(_memory_cfg(make_config, memory_ingest_notes=False), memory=mem)

    with patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))):
        await agent.run(make_turn(text="Python's latest version is 3.13"))

    mem.add_note.assert_not_awaited()


@pytest.mark.anyio
async def test_run_ingestion_skips_turns_that_disallow_memory_writes(make_config, make_turn):
    mem = AsyncMock()
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    turn = make_turn(text="restricted account recovery code", memory_writes_allowed=False)

    with patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="private reply"))) as run_mock:
        out = await agent.run(turn)

    assert out == "private reply"
    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["deps"].memory_writes_allowed is False
    mem.add_note.assert_not_awaited()


@pytest.mark.anyio
async def test_run_ingestion_skips_empty_text(make_config, make_turn):
    mem = AsyncMock()
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)

    with patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))):
        await agent.run(make_turn(text="  "))

    mem.add_note.assert_not_awaited()


@pytest.mark.anyio
async def test_run_ingestion_swallows_mem0_errors(make_config, make_turn):
    mem = AsyncMock()
    mem.add_note.side_effect = RuntimeError("mem0 down")
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)

    with patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))):
        out = await agent.run(make_turn(text="some durable world fact"))

    assert out == "reply"
    mem.add_note.assert_awaited_once()


@pytest.mark.anyio
async def test_run_ingestion_normalizes_author_handle(make_config, make_turn):
    mem = AsyncMock()
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)

    with patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))):
        await agent.run(make_turn(text="I use Arch btw", handle="Alice@Remote.Example"))

    mem.add_note.assert_awaited_once_with(
        text="I use Arch btw",
        author="alice@remote.example",
        author_user_id=None,
        note_id="note-1",
        source="unknown",
    )


@pytest.mark.anyio
async def test_run_ingestion_passes_the_turn_source_label(make_config, make_turn):
    """Provenance follows the frontend, so ACP memories aren't tagged as Misskey notes."""
    mem = AsyncMock()
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    turn = make_turn(text="i keep three shrimp tanks", handle="acp:abc", source="acp_prompt")

    with patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))):
        await agent.run(turn)

    assert mem.add_note.await_args is not None
    assert mem.add_note.await_args.kwargs["source"] == "acp_prompt"


@pytest.mark.anyio
async def test_run_ingestion_passes_stable_author_user_id(make_config, make_turn):
    mem = AsyncMock()
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    turn = make_turn(text="the operator prefers concise replies", user_id="trusted-user-id")

    with patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))):
        await agent.run(turn)

    assert mem.add_note.await_args is not None
    assert mem.add_note.await_args.kwargs["author_user_id"] == "trusted-user-id"


_GENERATED = GeneratedImage(data=b"\x89PNG\r\n\x1a\nbytes", media_type="image/png", prompt="p", alt_text="a")


class _StubGenerator:
    """Stands in for ImageGenerator; records calls and returns a canned result."""

    def __init__(self, image=None):
        self.image = image
        self.calls: list[tuple[str, str]] = []

    async def generate(self, prompt: str, alt_text: str):
        self.calls.append((prompt, alt_text))
        return self.image


def _auto_config(make_config, **overrides):
    return make_config(
        system_prompt_auto="Post something.",
        image_gen_enabled=True,
        image_gen_model="test/image-model",
        **overrides,
    )


@pytest.mark.anyio
async def test_create_poll_tool_stashes_normalized_poll_on_deps():
    tool = _make_create_poll_tool()
    deps = AutoDeps()

    result = await tool(  # type: ignore[arg-type]
        SimpleNamespace(deps=deps),
        ["  Tea ", "Coffee"],
        multiple=True,
        duration_minutes=90,
    )

    assert deps.poll == Poll(choices=("Tea", "Coffee"), multiple=True, duration_minutes=90)
    assert "attached" in result.lower()
    assert "90 minutes" in result.lower()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("choices", "duration_minutes"),
    [
        (["only one"], None),
        (["same", " same "], None),
        (["x" * 51, "fine"], None),
        (["yes", "no"], 0),
    ],
)
async def test_create_poll_tool_refuses_invalid_poll(choices, duration_minutes):
    tool = _make_create_poll_tool()
    deps = AutoDeps()

    result = await tool(  # type: ignore[arg-type]
        SimpleNamespace(deps=deps),
        choices,
        duration_minutes=duration_minutes,
    )

    assert deps.poll is None
    assert "refused" in result.lower()


@pytest.mark.anyio
async def test_create_poll_tool_refuses_second_poll():
    tool = _make_create_poll_tool()
    original = Poll(choices=("cats", "dogs"))
    deps = AutoDeps(poll=original)

    result = await tool(SimpleNamespace(deps=deps), ["tea", "coffee"])  # type: ignore[arg-type]

    assert deps.poll is original
    assert "already" in result.lower()


@pytest.mark.anyio
async def test_generate_image_tool_stashes_image_on_deps():
    generator = _StubGenerator(_GENERATED)
    tool = _make_generate_image_tool(generator)  # type: ignore[arg-type]
    deps = AutoDeps()

    result = await tool(SimpleNamespace(deps=deps), "a shrimp in a hat", "a shrimp")  # type: ignore[arg-type]

    assert deps.image is _GENERATED
    assert generator.calls == [("a shrimp in a hat", "a shrimp")]
    assert "attached" in result.lower()
    assert "without text" in result.lower()


@pytest.mark.anyio
async def test_generate_image_tool_refuses_second_call():
    """One image per post: a retry loop must not run up the provider bill."""
    generator = _StubGenerator(_GENERATED)
    tool = _make_generate_image_tool(generator)  # type: ignore[arg-type]
    deps = AutoDeps(image=_GENERATED)

    result = await tool(SimpleNamespace(deps=deps), "another one", "another")  # type: ignore[arg-type]

    assert generator.calls == []
    assert deps.image is _GENERATED
    assert "already" in result.lower()


class _RacyStubGenerator:
    """Like `_StubGenerator`, but actually yields control mid-`generate()`.

    `_StubGenerator.generate` never awaits anything real, so calling it under
    `asyncio.gather` never lets a second concurrent call interleave — the first task runs to
    completion in one scheduler step before the second ever starts. A real provider call does
    yield (it's a network round trip), which is exactly what lets two `generate_image` tool
    calls emitted in the same model turn race past the tool's "already generated" check before
    either writes `ctx.deps.image`. This stub reproduces that yield with `asyncio.sleep(0)` so
    a test can actually exercise the race window instead of merely asserting sequential calls.
    """

    def __init__(self, image):
        self.image = image
        self.calls: list[tuple[str, str]] = []

    async def generate(self, prompt: str, alt_text: str):
        self.calls.append((prompt, alt_text))
        await asyncio.sleep(0)
        return self.image


@pytest.mark.anyio
async def test_generate_image_tool_serializes_concurrent_calls():
    """Two generate_image calls fired concurrently against one AutoDeps (as pydantic-ai's own
    scheduler can do for two tool calls emitted in the same model turn) must not both slip
    past the check-then-set guard: only one generation call happens and only one image is
    stored. A test that merely calls the tool twice in sequence would not catch a regression
    here, since the race only shows up under real concurrency.
    """
    generator = _RacyStubGenerator(_GENERATED)
    tool = _make_generate_image_tool(generator)  # type: ignore[arg-type]
    deps = AutoDeps()
    ctx = SimpleNamespace(deps=deps)

    results = await asyncio.gather(
        tool(ctx, "a shrimp in a hat", "a shrimp"),  # type: ignore[arg-type]
        tool(ctx, "a different shrimp entirely", "another shrimp"),  # type: ignore[arg-type]
    )

    assert len(generator.calls) == 1, "generator.generate() must be called at most once per run"
    assert deps.image is _GENERATED
    # Both messages contain "attached" ("already attached" vs. "will be attached"), so
    # distinguish on "already" (refusal) vs. "generated" (the one call that actually ran).
    assert sum(1 for r in results if "already" in r.lower()) == 1
    assert sum(1 for r in results if "generated" in r.lower()) == 1


@pytest.mark.anyio
async def test_generate_image_tool_refuses_overlong_prompt():
    generator = _StubGenerator(_GENERATED)
    tool = _make_generate_image_tool(generator)  # type: ignore[arg-type]
    deps = AutoDeps()

    result = await tool(SimpleNamespace(deps=deps), "x" * 1001, "a shrimp")  # type: ignore[arg-type]

    assert generator.calls == []
    assert deps.image is None
    assert "refused" in result.lower()


@pytest.mark.anyio
async def test_generate_image_tool_refuses_overlong_alt_text():
    generator = _StubGenerator(_GENERATED)
    tool = _make_generate_image_tool(generator)  # type: ignore[arg-type]
    deps = AutoDeps()

    result = await tool(SimpleNamespace(deps=deps), "a shrimp", "y" * 513)  # type: ignore[arg-type]

    assert generator.calls == []
    assert deps.image is None
    assert "refused" in result.lower()


@pytest.mark.anyio
async def test_generate_image_tool_reports_failure():
    generator = _StubGenerator(None)
    tool = _make_generate_image_tool(generator)  # type: ignore[arg-type]
    deps = AutoDeps()

    result = await tool(SimpleNamespace(deps=deps), "a shrimp", "a shrimp")  # type: ignore[arg-type]

    assert deps.image is None
    assert "without an image" in result.lower()


@pytest.mark.anyio
async def test_generate_image_tool_does_not_retry_provider_after_failure():
    generator = _StubGenerator(None)
    tool = _make_generate_image_tool(generator)  # type: ignore[arg-type]
    deps = AutoDeps()
    ctx = SimpleNamespace(deps=deps)

    first = await tool(ctx, "a shrimp", "a shrimp")  # type: ignore[arg-type]
    second = await tool(ctx, "a different shrimp", "another shrimp")  # type: ignore[arg-type]

    assert generator.calls == [("a shrimp", "a shrimp")]
    assert "without an image" in first.lower()
    assert "already attempted" in second.lower()


def test_auto_agent_gets_image_tool_when_enabled(make_config):
    agent = ChatAgent(_auto_config(make_config))

    assert agent._image_generator is not None
    assert agent._auto_agent is not None
    assert "generate_image" in agent._auto_agent._function_toolset.tools


def test_auto_agent_has_no_image_tool_when_disabled(make_config):
    agent = ChatAgent(make_config(system_prompt_auto="Post something."))

    assert agent._image_generator is None
    assert agent._auto_agent is not None
    assert "generate_image" not in agent._auto_agent._function_toolset.tools


def test_reply_agent_never_gets_image_tool(make_config):
    """Auto-post-only is the point; the reply agent must not see the tool."""
    agent = ChatAgent(_auto_config(make_config))

    assert "generate_image" not in agent._agent._function_toolset.tools


def test_auto_agent_gets_poll_tool(make_config):
    agent = ChatAgent(make_config(system_prompt_auto="Post something."))

    assert agent._auto_agent is not None
    assert "create_poll" in agent._auto_agent._function_toolset.tools


def test_reply_agent_never_gets_poll_tool(make_config):
    agent = ChatAgent(make_config(system_prompt_auto="Post something."))

    assert "create_poll" not in agent._agent._function_toolset.tools


@pytest.mark.anyio
async def test_run_auto_returns_text_only_when_no_image(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None

    with patch.object(agent._auto_agent, "run", AsyncMock(return_value=SimpleNamespace(output="post text"))):
        post = await agent.run_auto()

    assert post.text == "post text"
    assert post.image is None


@pytest.mark.anyio
async def test_run_auto_carries_image_generated_during_the_run(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None

    async def fake_run(prompt, **kwargs):
        # What the tool does mid-run: stash the image on the deps object.
        kwargs["deps"].image = _GENERATED
        return SimpleNamespace(output="post text")

    with patch.object(agent._auto_agent, "run", AsyncMock(side_effect=fake_run)):
        post = await agent.run_auto()

    assert post.text == "post text"
    assert post.image is _GENERATED


@pytest.mark.anyio
async def test_run_auto_carries_poll_created_during_the_run(make_config):
    agent = ChatAgent(make_config(system_prompt_auto="Post something."))
    assert agent._auto_agent is not None
    poll = Poll(choices=("Tea", "Coffee"), duration_minutes=30)

    async def fake_run(prompt, **kwargs):
        kwargs["deps"].poll = poll
        return SimpleNamespace(output="What should I drink?")

    with patch.object(agent._auto_agent, "run", AsyncMock(side_effect=fake_run)):
        post = await agent.run_auto()

    assert post.text == "What should I drink?"
    assert post.poll is poll
    assert list(agent._auto_history) == ["What should I drink?\n[poll choices: Tea | Coffee]"]


@pytest.mark.anyio
async def test_run_auto_turns_leaked_meme_description_into_image(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None
    assert agent._image_generator is not None
    output = '*[meme: tired office worker vs smug robot, text "I DIDN\'T FIRE THEM. THE MARKET DID."]* lol true'

    with (
        patch.object(agent._auto_agent, "run", AsyncMock(return_value=SimpleNamespace(output=output))),
        patch.object(agent._image_generator, "generate", AsyncMock(return_value=_GENERATED)) as generate_mock,
    ):
        post = await agent.run_auto()

    assert post.text == "lol true"
    assert post.image is _GENERATED
    generate_mock.assert_awaited_once_with(
        'meme: tired office worker vs smug robot, text "I DIDN\'T FIRE THEM. THE MARKET DID."',
        'meme: tired office worker vs smug robot, text "I DIDN\'T FIRE THEM. THE MARKET DID."',
    )


@pytest.mark.anyio
async def test_run_auto_leaves_nonvisual_stage_direction_as_text(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None
    assert agent._image_generator is not None
    output = "*[checks watch]* still waiting lol"

    with (
        patch.object(agent._auto_agent, "run", AsyncMock(return_value=SimpleNamespace(output=output))),
        patch.object(agent._image_generator, "generate", AsyncMock(return_value=_GENERATED)) as generate_mock,
    ):
        post = await agent.run_auto()

    assert post.text == output
    assert post.image is None
    generate_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_run_auto_does_not_treat_image_word_in_an_action_as_a_visual_description(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None
    assert agent._image_generator is not None
    output = "*[deletes image cache]* fixed"

    with (
        patch.object(agent._auto_agent, "run", AsyncMock(return_value=SimpleNamespace(output=output))),
        patch.object(agent._image_generator, "generate", AsyncMock(return_value=_GENERATED)) as generate_mock,
    ):
        post = await agent.run_auto()

    assert post.text == output
    assert post.image is None
    generate_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_run_auto_strips_redundant_description_after_tool_generated_image(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None
    assert agent._image_generator is not None
    output = "*[photo: a shrimp at a keyboard]* working"

    async def fake_run(prompt, **kwargs):
        kwargs["deps"].image = _GENERATED
        kwargs["deps"].image_attempted = True
        return SimpleNamespace(output=output)

    with (
        patch.object(agent._auto_agent, "run", AsyncMock(side_effect=fake_run)),
        patch.object(agent._image_generator, "generate", AsyncMock(return_value=_GENERATED)) as generate_mock,
    ):
        post = await agent.run_auto()

    assert post.text == "working"
    assert post.image is _GENERATED
    generate_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_run_auto_does_not_retry_description_after_tool_generation_failed(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None
    assert agent._image_generator is not None
    output = "*[meme: an expensive failure]* still text"

    async def fake_run(prompt, **kwargs):
        kwargs["deps"].image_attempted = True
        return SimpleNamespace(output=output)

    with (
        patch.object(agent._auto_agent, "run", AsyncMock(side_effect=fake_run)),
        patch.object(agent._image_generator, "generate", AsyncMock(return_value=_GENERATED)) as generate_mock,
    ):
        post = await agent.run_auto()

    assert post.text == output
    assert post.image is None
    generate_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_run_auto_keeps_description_when_recovery_generation_fails(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None
    assert agent._image_generator is not None
    output = "[boomer meme comparing regulation to a virus] lol"

    with (
        patch.object(agent._auto_agent, "run", AsyncMock(return_value=SimpleNamespace(output=output))),
        patch.object(agent._image_generator, "generate", AsyncMock(return_value=None)) as generate_mock,
    ):
        post = await agent.run_auto()

    assert post.text == output
    assert post.image is None
    generate_mock.assert_awaited_once()


@pytest.mark.anyio
async def test_run_auto_keeps_description_when_recovery_generation_raises(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None
    assert agent._image_generator is not None
    output = "*[picture: a rocket made of office chairs]* ship it"

    with (
        patch.object(agent._auto_agent, "run", AsyncMock(return_value=SimpleNamespace(output=output))),
        patch.object(agent._image_generator, "generate", AsyncMock(side_effect=RuntimeError("boom"))) as generate_mock,
    ):
        post = await agent.run_auto()

    assert post.text == output
    assert post.image is None
    generate_mock.assert_awaited_once()


@pytest.mark.anyio
async def test_run_auto_normalizes_image_only_output_and_records_image_history(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None

    async def fake_run(prompt, **kwargs):
        kwargs["deps"].image = _GENERATED
        return SimpleNamespace(output=_IMAGE_ONLY_OUTPUT)

    with patch.object(agent._auto_agent, "run", AsyncMock(side_effect=fake_run)):
        post = await agent.run_auto()

    assert post.text == ""
    assert post.image is _GENERATED
    assert list(agent._auto_history) == ["[image-only post: a]"]


def test_auto_output_validator_allows_image_without_caption():
    ctx = SimpleNamespace(deps=AutoDeps(image=_GENERATED))

    assert ChatAgent._validate_auto_output(ctx, _IMAGE_ONLY_OUTPUT) == ""  # type: ignore[arg-type]


def test_auto_output_validator_retries_image_only_marker_with_poll():
    poll = Poll(choices=("Tea", "Coffee"))
    ctx = SimpleNamespace(deps=AutoDeps(image=_GENERATED, poll=poll))

    with pytest.raises(ModelRetry, match="poll needs question text"):
        ChatAgent._validate_auto_output(ctx, _IMAGE_ONLY_OUTPUT)  # type: ignore[arg-type]


def test_auto_output_validator_retries_image_only_marker_without_image():
    ctx = SimpleNamespace(deps=AutoDeps())

    with pytest.raises(ModelRetry, match="valid only after generate_image succeeds"):
        ChatAgent._validate_auto_output(ctx, _IMAGE_ONLY_OUTPUT)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_run_auto_rejects_image_only_marker_without_image(make_config):
    """Keep the publication boundary safe even when a mocked/custom Agent skips validators."""
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None

    with patch.object(
        agent._auto_agent,
        "run",
        AsyncMock(return_value=SimpleNamespace(output=_IMAGE_ONLY_OUTPUT)),
    ), pytest.raises(ValueError, match="without generating an image"):
        await agent.run_auto()


@pytest.mark.anyio
async def test_run_auto_rejects_poll_without_question_when_agent_skips_validators(make_config):
    agent = ChatAgent(_auto_config(make_config))
    assert agent._auto_agent is not None
    poll = Poll(choices=("Tea", "Coffee"))

    async def fake_run(prompt, **kwargs):
        kwargs["deps"].image = _GENERATED
        kwargs["deps"].poll = poll
        return SimpleNamespace(output=_IMAGE_ONLY_OUTPUT)

    with patch.object(agent._auto_agent, "run", AsyncMock(side_effect=fake_run)):
        with pytest.raises(ValueError, match="poll has no question text"):
            await agent.run_auto()
