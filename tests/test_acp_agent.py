"""Tests for the ACP frontend: session lifecycle, prompt flow, cancellation."""

import asyncio
from unittest.mock import AsyncMock, patch

import acp
import pytest

from bot.acp.agent import MissbotAgent, _text_from_blocks
from bot.acp.session import SessionRegistry
from bot.core import AgentTurn, HistoryTurn

_HEX = "a" * 64


@pytest.fixture
def agent(config):
    return MissbotAgent(config=config)


def _text(text: str):
    return acp.text_block(text)


def _block(content: str, *, hex_key: str = _HEX) -> str:
    return f"Event ID: deadbeef\nChannel: general (#0198)\nKind: 9\nFrom: alice (hex: {hex_key})\nContent: {content}"


async def _new_session(agent) -> str:
    return (await agent.new_session(cwd="/tmp")).session_id


def _last_turn(run_mock) -> AgentTurn:
    """Pull the AgentTurn the adapter most recently handed to ChatAgent.run."""
    assert run_mock.await_args is not None
    return run_mock.await_args.args[0]


# --- initialize -------------------------------------------------------------


@pytest.mark.anyio
async def test_initialize_advertises_text_only_and_no_session_loading(agent):
    resp = await agent.initialize(protocol_version=acp.PROTOCOL_VERSION)

    assert resp.protocol_version == acp.PROTOCOL_VERSION
    caps = resp.agent_capabilities
    assert caps is not None
    # No cross-restart persistence, so clients must not try to resume.
    assert caps.load_session is False
    assert caps.prompt_capabilities is not None
    assert caps.prompt_capabilities.image is False
    assert caps.prompt_capabilities.audio is False
    # stdio's trust boundary is process spawn, so no auth methods are offered.
    assert resp.auth_methods == []
    assert resp.agent_info is not None
    assert resp.agent_info.name == "missbot"


@pytest.mark.anyio
async def test_initialize_negotiates_down_to_older_client(agent):
    """An older client must not be told to speak a version it doesn't know."""
    resp = await agent.initialize(protocol_version=0)
    assert resp.protocol_version == 0


# --- sessions ---------------------------------------------------------------


@pytest.mark.anyio
async def test_new_session_returns_distinct_ids(agent):
    first = await _new_session(agent)
    second = await _new_session(agent)
    assert first != second


@pytest.mark.anyio
async def test_new_session_rejects_past_capacity(make_config):
    agent = MissbotAgent(config=make_config(acp_max_sessions=1))
    await _new_session(agent)

    with pytest.raises(acp.RequestError):
        await agent.new_session(cwd="/tmp")


@pytest.mark.anyio
async def test_close_session_frees_capacity(make_config):
    agent = MissbotAgent(config=make_config(acp_max_sessions=1))
    session_id = await _new_session(agent)

    await agent.close_session(session_id)
    # Capacity was released, so a new session succeeds.
    assert await _new_session(agent)


@pytest.mark.anyio
async def test_prompt_on_unknown_session_is_invalid_params(agent):
    with pytest.raises(acp.RequestError):
        await agent.prompt(session_id="nope", prompt=[_text("hi")])


# --- prompt flow ------------------------------------------------------------


@pytest.mark.anyio
async def test_prompt_runs_turn_and_pushes_session_update(agent):
    session_id = await _new_session(agent)
    conn = AsyncMock()
    agent.on_connect(conn)

    with patch.object(agent._agent, "run", AsyncMock(return_value="hi there")) as run_mock:
        resp = await agent.prompt(session_id=session_id, prompt=[_text(_block("hello"))])

    assert resp.stop_reason == "end_turn"
    turn = _last_turn(run_mock)
    # The full harness block reaches the model, metadata included.
    assert "Content: hello" in turn.text
    assert turn.author.handle == f"acp:{_HEX}"
    # No platform length cap on this frontend — that budget is Misskey's.
    assert turn.char_budget is None
    assert turn.source_id == f"acp:{session_id}"

    conn.session_update.assert_awaited_once()
    assert conn.session_update.await_args.args[0] == session_id


@pytest.mark.anyio
async def test_prompt_falls_back_to_configured_identity(make_config):
    agent = MissbotAgent(config=make_config(acp_default_identity="buzz"))
    session_id = await _new_session(agent)
    agent.on_connect(AsyncMock())

    with patch.object(agent._agent, "run", AsyncMock(return_value="ok")) as run_mock:
        await agent.prompt(session_id=session_id, prompt=[_text("bare text, no header")])

    assert _last_turn(run_mock).author.handle == "acp:buzz"


@pytest.mark.anyio
async def test_prompt_accumulates_session_history(agent):
    session_id = await _new_session(agent)
    agent.on_connect(AsyncMock())

    with patch.object(agent._agent, "run", AsyncMock(side_effect=["first reply", "second reply"])) as run_mock:
        await agent.prompt(session_id=session_id, prompt=[_text(_block("one"))])
        await agent.prompt(session_id=session_id, prompt=[_text(_block("two"))])

    # The second turn sees the first exchange.
    second_turn = _last_turn(run_mock)
    assert [(h.role, h.text) for h in second_turn.history][-1] == ("assistant", "first reply")
    assert second_turn.previous_reply == "first reply"


@pytest.mark.anyio
async def test_session_history_is_bounded(make_config):
    agent = MissbotAgent(config=make_config(acp_max_history_turns=2))
    session_id = await _new_session(agent)
    agent.on_connect(AsyncMock())

    with patch.object(agent._agent, "run", AsyncMock(return_value="reply")):
        for i in range(5):
            await agent.prompt(session_id=session_id, prompt=[_text(_block(f"msg {i}"))])

    session = agent._sessions.get(session_id)
    assert session is not None
    # 2 turns => 2 user + 2 assistant messages retained.
    assert len(session.history) == 4
    assert "msg 4" in session.history[-2].text


@pytest.mark.anyio
async def test_prompt_no_reply_sentinel_sends_no_update(agent):
    session_id = await _new_session(agent)
    conn = AsyncMock()
    agent.on_connect(conn)

    with patch.object(agent._agent, "run", AsyncMock(return_value="NO_REPLY")):
        resp = await agent.prompt(session_id=session_id, prompt=[_text(_block("meh"))])

    assert resp.stop_reason == "end_turn"
    conn.session_update.assert_not_awaited()


@pytest.mark.anyio
async def test_prompt_with_no_text_blocks_ends_turn(agent):
    session_id = await _new_session(agent)
    agent.on_connect(AsyncMock())

    with patch.object(agent._agent, "run", AsyncMock()) as run_mock:
        resp = await agent.prompt(session_id=session_id, prompt=[])

    assert resp.stop_reason == "end_turn"
    run_mock.assert_not_awaited()


# --- social credit floor ----------------------------------------------------


@pytest.mark.anyio
async def test_prompt_refuses_caller_below_score_threshold(make_config):
    agent = MissbotAgent(config=make_config(social_credit_ignore_threshold=-50))
    session_id = await _new_session(agent)
    conn = AsyncMock()
    agent.on_connect(conn)

    with (
        patch.object(agent._agent, "get_score", AsyncMock(return_value=-60)),
        patch.object(agent._agent, "run", AsyncMock()) as run_mock,
    ):
        resp = await agent.prompt(session_id=session_id, prompt=[_text(_block("i am a menace"))])

    assert resp.stop_reason == "refusal"
    run_mock.assert_not_awaited()
    conn.session_update.assert_not_awaited()


@pytest.mark.anyio
async def test_prompt_allows_unscored_caller(make_config):
    """No score yet (None) is never below the floor — same rule as the Misskey path."""
    agent = MissbotAgent(config=make_config(social_credit_ignore_threshold=0))
    session_id = await _new_session(agent)
    agent.on_connect(AsyncMock())

    with (
        patch.object(agent._agent, "get_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value="hello")) as run_mock,
    ):
        resp = await agent.prompt(session_id=session_id, prompt=[_text(_block("first time"))])

    assert resp.stop_reason == "end_turn"
    run_mock.assert_awaited_once()


@pytest.mark.anyio
async def test_prompt_skips_score_lookup_without_threshold(agent):
    session_id = await _new_session(agent)
    agent.on_connect(AsyncMock())

    with (
        patch.object(agent._agent, "get_score", AsyncMock()) as score_mock,
        patch.object(agent._agent, "run", AsyncMock(return_value="hi")),
    ):
        await agent.prompt(session_id=session_id, prompt=[_text(_block("hello"))])

    score_mock.assert_not_awaited()


# --- unimplemented protocol surface -----------------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda a: a.load_session(cwd="/tmp", session_id="s1"), id="session/load"),
        pytest.param(lambda a: a.list_sessions(), id="session/list"),
        pytest.param(lambda a: a.set_session_mode(session_id="s1", mode_id="fast"), id="session/set_mode"),
        pytest.param(
            lambda a: a.set_config_option(config_id="c", session_id="s1", value=True),
            id="session/set_config_option",
        ),
        pytest.param(lambda a: a.fork_session(session_id="s1", cwd="/tmp"), id="session/fork"),
        pytest.param(lambda a: a.resume_session(session_id="s1", cwd="/tmp"), id="session/resume"),
        pytest.param(lambda a: a.ext_method("example/thing", {}), id="ext_method"),
    ],
)
async def test_unsupported_requests_raise_method_not_found(agent, call):
    """`acp.Agent` is a Protocol, so an unimplemented method is still *inherited*.

    The SDK's router resolves handlers with `getattr`, so those stubs get routed and
    return None, which it reports to the client as a success. A client probing
    `session/load` would read that as "session restored". Answer with a real error.
    """
    with pytest.raises(acp.RequestError) as excinfo:
        await call(agent)

    assert excinfo.value.code == -32601


@pytest.mark.anyio
async def test_unsupported_notification_is_a_noop(agent):
    """Notifications have no response channel — raising would only break the connection."""
    await agent.ext_notification("example/ping", {})


# --- cancellation -----------------------------------------------------------


@pytest.mark.anyio
async def test_cancel_stops_in_flight_turn(agent):
    session_id = await _new_session(agent)
    agent.on_connect(AsyncMock())
    started = asyncio.Event()

    async def slow_run(_turn):
        started.set()
        await asyncio.sleep(30)
        return "never"

    with patch.object(agent._agent, "run", AsyncMock(side_effect=slow_run)):
        prompt_task = asyncio.create_task(agent.prompt(session_id=session_id, prompt=[_text(_block("wait"))]))
        await started.wait()
        await agent.cancel(session_id)
        resp = await prompt_task

    # ACP requires the pending prompt to answer, not raise.
    assert resp.stop_reason == "cancelled"


@pytest.mark.anyio
async def test_cancel_unknown_session_is_a_noop(agent):
    await agent.cancel("nope")


# --- helpers ----------------------------------------------------------------


def test_text_from_blocks_joins_text_and_drops_others():
    blocks = [_text("one"), acp.image_block(data="ZmFrZQ==", mime_type="image/png"), _text("two")]
    assert _text_from_blocks(blocks) == "one\ntwo"


def test_session_registry_previous_reply_ignores_blank_assistant_turns():
    registry = SessionRegistry(max_sessions=2, max_history_turns=5)
    session = registry.create()
    session.history.append(HistoryTurn(role="assistant", text="   "))
    session.history.append(HistoryTurn(role="user", text="hi", author="acp:x"))
    assert session.previous_reply() is None
