"""Tests for bot.tools."""

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from unittest.mock import patch

from bot.tools import build_tools, current_datetime


def _tool_names(tools: list[Any]) -> set[str]:
    return {t.__name__ for t in tools}


def test_current_datetime_returns_string():
    assert isinstance(current_datetime(), str)


def test_build_tools_minimal(config):
    tools = build_tools(config)
    names = _tool_names(tools)
    assert "current_datetime_tool" in names
    assert "create_note" not in names
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


def _find(tools: list[Any], name: str) -> Any:
    for t in tools:
        if t.__name__ == name:
            return t
    raise AssertionError(f"tool {name} not found")


def _mock_sync_client(response: Any = None, *, post_side_effect: Any = None) -> tuple[MagicMock, MagicMock]:
    client = MagicMock()
    if post_side_effect is not None:
        client.post.side_effect = post_side_effect
    else:
        client.post.return_value = response
    manager = MagicMock()
    manager.__enter__.return_value = client
    manager.__exit__.return_value = False
    return client, manager


def test_search_web_returns_top_results_and_uses_auth(make_config):
    cfg = make_config(
        searxng_url="https://searx.example/",
        searxng_user="searcher",
        searxng_password="secret",
    )
    search_web = _find(build_tools(cfg), "search_web")
    response = MagicMock()
    response.json.return_value = {
        "results": [{"content": f"result-{idx}"} for idx in range(1, 7)],
    }
    client, manager = _mock_sync_client(response)

    with (
        patch("bot.tools.httpx.BasicAuth", return_value="auth") as auth_mock,
        patch("bot.tools.httpx.Client", return_value=manager) as client_cls,
    ):
        result = search_web("fediverse")

    assert result == "result-1\n---\nresult-2\n---\nresult-3\n---\nresult-4\n---\nresult-5"
    auth_mock.assert_called_once_with("searcher", "secret")
    assert client_cls.call_args.kwargs["auth"] == "auth"
    client.post.assert_called_once_with(
        f"{cfg.searxng_url}search",
        params={"q": "fediverse", "format": "json"},
    )


def test_search_web_returns_none_on_http_error(make_config):
    cfg = make_config(searxng_url="https://searx.example/")
    search_web = _find(build_tools(cfg), "search_web")
    request = httpx.Request("POST", f"{cfg.searxng_url}search")
    error = httpx.RequestError("boom", request=request)
    _, manager = _mock_sync_client(post_side_effect=error)

    with patch("bot.tools.httpx.Client", return_value=manager):
        assert search_web("fediverse") is None


def test_search_users_formats_results_and_clamps_limit(config):
    search_users = _find(build_tools(config), "search_users")
    response = MagicMock()
    response.json.return_value = [
        {
            "username": "alice",
            "host": None,
            "name": "Alice",
            "description": "A" * 120,
        },
        {
            "username": "bob",
            "host": "remote.test",
            "name": None,
            "description": None,
        },
    ]
    client, manager = _mock_sync_client(response)

    with patch("bot.tools.httpx.Client", return_value=manager):
        result = search_users("ali", limit=999, offset=2)

    assert result.split("\n---\n") == [
        f"Alice (@alice): {'A' * 100}",
        "bob (@bob@remote.test): ",
    ]
    assert client.post.call_args.kwargs["json"] == {"query": "ali", "limit": 50, "offset": 2}


def test_search_users_returns_no_results(config):
    search_users = _find(build_tools(config), "search_users")
    response = MagicMock()
    response.json.return_value = []
    _, manager = _mock_sync_client(response)

    with patch("bot.tools.httpx.Client", return_value=manager):
        assert search_users("nobody") == "No users found."


def test_search_users_returns_none_on_http_error(config):
    search_users = _find(build_tools(config), "search_users")
    request = httpx.Request("POST", f"{config.url}api/users/search")
    error = httpx.RequestError("boom", request=request)
    _, manager = _mock_sync_client(post_side_effect=error)

    with patch("bot.tools.httpx.Client", return_value=manager):
        assert search_users("nobody") is None


def test_search_notes_formats_results_and_clamps_limit(config):
    search_notes = _find(build_tools(config), "search_notes")
    response = MagicMock()
    response.json.return_value = [
        {
            "user": {"username": "alice", "host": None},
            "text": "hello world",
        },
        {
            "user": {"username": "bob", "host": "remote.test"},
            "text": None,
        },
    ]
    client, manager = _mock_sync_client(response)

    with patch("bot.tools.httpx.Client", return_value=manager):
        result = search_notes("hello", limit=0, offset=3)

    assert result.split("\n---\n") == [
        "@alice: hello world",
        "@bob@remote.test: (no text)",
    ]
    assert client.post.call_args.kwargs["json"] == {"query": "hello", "limit": 1, "offset": 3}


def test_search_notes_returns_no_results(config):
    search_notes = _find(build_tools(config), "search_notes")
    response = MagicMock()
    response.json.return_value = []
    _, manager = _mock_sync_client(response)

    with patch("bot.tools.httpx.Client", return_value=manager):
        assert search_notes("nobody") == "No notes found."


def test_search_notes_returns_none_on_http_error(config):
    search_notes = _find(build_tools(config), "search_notes")
    request = httpx.Request("POST", f"{config.url}api/notes/search")
    error = httpx.RequestError("boom", request=request)
    _, manager = _mock_sync_client(post_side_effect=error)

    with patch("bot.tools.httpx.Client", return_value=manager):
        assert search_notes("nobody") is None


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
    ctx.deps.username = "alice"
    ctx.deps.social_credit_unrestricted = False

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
    ctx.deps.username = "alice"
    ctx.deps.social_credit_unrestricted = False
    result = await adjust(ctx, "alice", 1, "   ")
    assert "reason is required" in result


@pytest.mark.anyio
async def test_adjust_social_credit_blocks_double_adjustment(config, fake_redis):
    adjust = _find(build_tools(config, redis_client=fake_redis), "adjust_social_credit")
    ctx = MagicMock()
    ctx.deps.adjusted_credit_users = set()
    ctx.deps.username = "alice"
    ctx.deps.social_credit_unrestricted = False

    first = await adjust(ctx, "alice", 1, "good")
    assert "New score: 1" in first

    second = await adjust(ctx, "alice", 1, "good")
    assert "Already adjusted" in second


@pytest.mark.anyio
async def test_adjust_social_credit_blocks_non_author(config, fake_redis):
    """By default, only the author of the note being replied to can be adjusted."""
    adjust = _find(build_tools(config, redis_client=fake_redis), "adjust_social_credit")
    ctx = MagicMock()
    ctx.deps.adjusted_credit_users = set()
    ctx.deps.username = "alice"
    ctx.deps.social_credit_unrestricted = False

    result = await adjust(ctx, "bob", 5, "off-topic praise")
    assert "Refusing to adjust @bob" in result
    # The blocked target must not have been recorded or scored.
    assert ctx.deps.adjusted_credit_users == set()
    assert await fake_redis.get("score:bob") is None


@pytest.mark.anyio
async def test_adjust_social_credit_unrestricted_allows_non_author(config, fake_redis):
    """When the run is flagged unrestricted, any user can be adjusted."""
    adjust = _find(build_tools(config, redis_client=fake_redis), "adjust_social_credit")
    ctx = MagicMock()
    ctx.deps.adjusted_credit_users = set()
    ctx.deps.username = "alice"
    ctx.deps.social_credit_unrestricted = True

    result = await adjust(ctx, "bob", 5, "tribunal verdict")
    assert "New score: 5" in result
    assert await fake_redis.get("score:bob") == "5"


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
