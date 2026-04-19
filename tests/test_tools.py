"""Tests for bot.tools."""

from unittest.mock import MagicMock

import pytest

from bot.tools import build_tools, current_datetime


def _tool_names(tools):
    return {t.__name__ for t in tools}


def test_current_datetime_returns_string():
    assert isinstance(current_datetime(), str)


def test_build_tools_minimal(config):
    tools = build_tools(config)
    names = _tool_names(tools)
    assert "current_datetime_tool" in names
    assert "search_users" in names
    assert "search_notes" in names
    # No searxng, no redis → these absent
    assert "search_web" not in names
    assert "get_social_credit" not in names


def test_build_tools_with_searxng(make_config):
    cfg = make_config(searxng_url="https://searx.example/")
    tools = build_tools(cfg)
    assert "search_web" in _tool_names(tools)


def test_build_tools_with_redis(config, fake_redis):
    tools = build_tools(config, redis_client=fake_redis)
    names = _tool_names(tools)
    assert {
        "get_social_credit",
        "adjust_social_credit",
        "get_social_credit_history",
        "get_social_credit_leaderboard",
    } <= names


def _find(tools, name):
    for t in tools:
        if t.__name__ == name:
            return t
    raise AssertionError(f"tool {name} not found")


@pytest.mark.anyio
async def test_get_social_credit_unknown_user(config, fake_redis):
    get_credit = _find(build_tools(config, redis_client=fake_redis), "get_social_credit")
    result = await get_credit("NewUser")
    assert "no social credit score yet" in result
    assert "@newuser" in result  # normalized to lowercase


@pytest.mark.anyio
async def test_adjust_social_credit_happy_path(config, fake_redis):
    tools = build_tools(config, redis_client=fake_redis)
    adjust = _find(tools, "adjust_social_credit")
    get_credit = _find(tools, "get_social_credit")
    history = _find(tools, "get_social_credit_history")
    leaderboard = _find(tools, "get_social_credit_leaderboard")

    ctx = MagicMock()
    ctx.deps.adjusted_credit_users = set()

    result = await adjust(ctx, "@Alice", 5, "solid post")
    assert "+5" in result
    assert "New score: 5" in result

    assert await get_credit("alice") == "User @alice has 5 social credit points."

    hist = await history("alice")
    assert "+5" in hist
    assert "solid post" in hist

    board = await leaderboard()
    assert "1. @alice: 5 points" in board


@pytest.mark.anyio
async def test_adjust_social_credit_requires_reason(config, fake_redis):
    adjust = _find(build_tools(config, redis_client=fake_redis), "adjust_social_credit")
    ctx = MagicMock()
    ctx.deps.adjusted_credit_users = set()
    result = await adjust(ctx, "alice", 1, "   ")
    assert "reason is required" in result


@pytest.mark.anyio
async def test_adjust_social_credit_blocks_double_adjustment(config, fake_redis):
    adjust = _find(build_tools(config, redis_client=fake_redis), "adjust_social_credit")
    ctx = MagicMock()
    ctx.deps.adjusted_credit_users = set()

    first = await adjust(ctx, "alice", 1, "good")
    assert "New score: 1" in first

    second = await adjust(ctx, "alice", 1, "good")
    assert "Already adjusted" in second


@pytest.mark.anyio
async def test_get_social_credit_history_empty(config, fake_redis):
    history = _find(build_tools(config, redis_client=fake_redis), "get_social_credit_history")
    result = await history("noone")
    assert "No social credit history" in result


@pytest.mark.anyio
async def test_get_social_credit_leaderboard_empty(config, fake_redis):
    leaderboard = _find(build_tools(config, redis_client=fake_redis), "get_social_credit_leaderboard")
    result = await leaderboard()
    assert "No social credit scores recorded yet" in result
