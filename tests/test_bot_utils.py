"""Tests for Bot helper methods that don't need a live WebSocket."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from bot.bot import Bot


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def bot(config, event_loop):
    """Build a Bot without spinning up the websocket."""
    return Bot(config=config, loop=event_loop)


def test_format_handle_local(bot, make_user):
    assert bot._format_handle(make_user(username="alice", host=None)) == "@alice"


def test_format_handle_remote(bot, make_user):
    user = make_user(username="bob", host="remote.example")
    assert bot._format_handle(user) == "@bob@remote.example"


def test_unique_ordered_preserves_first_occurrence(bot):
    assert bot._unique_ordered(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_unique_ordered_empty(bot):
    assert bot._unique_ordered([]) == []


def test_strip_leading_mentions_strips_all(bot):
    result = bot._strip_leading_mentions("@alice @bob@remote.example hello there")
    assert result == "hello there"


def test_strip_leading_mentions_no_op(bot):
    assert bot._strip_leading_mentions("hello @alice") == "hello @alice"


def test_compute_auto_reply_delay_no_jitter(make_config, event_loop):
    cfg = make_config(auto_reply_interval=300, auto_reply_jitter=0)
    bot = Bot(cfg, loop=event_loop)
    assert bot._compute_auto_reply_delay() == 300


def test_compute_auto_reply_delay_with_jitter(make_config, event_loop):
    cfg = make_config(auto_reply_interval=300, auto_reply_jitter=30)
    bot = Bot(cfg, loop=event_loop)
    delay = bot._compute_auto_reply_delay()
    assert 270 <= delay <= 330


@pytest.mark.anyio
async def test_normalize_note_mention_empty(bot):
    assert await bot._normalize_note_mention("") is None
    assert await bot._normalize_note_mention("   ") is None
    assert await bot._normalize_note_mention("@") is None


@pytest.mark.anyio
async def test_normalize_note_mention_with_host(bot):
    assert await bot._normalize_note_mention("@bob@remote.host") == "@bob@remote.host"


@pytest.mark.anyio
async def test_normalize_note_mention_resolves_local_id(bot):
    """When given a bare id (no @host), the bot resolves via the API."""
    with patch.object(bot, "_resolve_user_handle", AsyncMock(return_value="@alice")):
        assert await bot._normalize_note_mention("alice") == "@alice"


@pytest.mark.anyio
async def test_normalize_note_mention_falls_back_when_unresolved(bot):
    with patch.object(bot, "_resolve_user_handle", AsyncMock(return_value=None)):
        assert await bot._normalize_note_mention("alice") == "@alice"


@pytest.mark.anyio
async def test_build_mentions_filters_self_and_dedupes(bot, make_note, make_user):
    author = make_user(id="u2", username="carol", host=None)
    note = make_note(
        user=author,
        mentions=[f"@{bot._config.bot_username}", "@alice", "@alice"],
    )
    with patch.object(bot, "_resolve_user_handle", AsyncMock(return_value=None)):
        mentions = await bot._build_mentions_from_note(note)
    # Self-mention stripped, duplicates collapsed, author appended.
    assert mentions == ["@alice", "@carol"]


@pytest.mark.anyio
async def test_build_mentions_returns_empty_when_no_note(bot):
    assert await bot._build_mentions_from_note(None) == []
