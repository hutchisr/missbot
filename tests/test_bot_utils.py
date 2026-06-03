"""Tests for Bot helper methods that don't need a live WebSocket."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bot.bot import Bot
from bot.models import MiFile


@pytest.fixture
def bot(config):
    """Build a Bot without spinning up the websocket."""
    return Bot(config=config)


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


def test_strip_leading_mentions_lone_mention(bot):
    assert bot._strip_leading_mentions("@alice") == ""
    assert bot._strip_leading_mentions("@alice@remote.example") == ""


def test_note_has_prompt_content_false_without_text_or_images(bot, make_note):
    note = make_note(text=None, files=None)
    assert bot._note_has_prompt_content(note) is False


def test_reply_visibility_defaults_public(bot):
    assert bot._reply_visibility(None) == "public"


def test_reply_visible_user_ids_filters_bot_and_dedupes(bot, make_note, make_user):
    author = make_user(id="user-2", username="carol")
    note = make_note(user=author).model_copy(
        update={
            "visibility": "specified",
            "visibleUserIds": ["user-3", bot.user_id, "user-3", author.id],
        }
    )

    assert bot._reply_visible_user_ids(note) == ["user-3", author.id]


def test_compute_auto_reply_delay_no_jitter(make_config):
    cfg = make_config(auto_reply_interval=300, auto_reply_jitter=0)
    bot = Bot(cfg)
    assert bot._compute_auto_reply_delay() == 300


def test_compute_auto_reply_delay_with_jitter(make_config):
    cfg = make_config(auto_reply_interval=300, auto_reply_jitter=30)
    bot = Bot(cfg)
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
async def test_resolve_user_handle_formats_remote_user(bot):
    response = MagicMock()
    response.json.return_value = {"username": "alice", "host": "remote.test"}

    with patch("bot.bot.api_client.post", AsyncMock(return_value=response)) as post_mock:
        result = await bot._resolve_user_handle("user-123")

    assert result == "@alice@remote.test"
    post_mock.assert_awaited_once_with(
        "https://example.test/api/users/show",
        json={"userId": "user-123"},
    )


@pytest.mark.anyio
async def test_resolve_user_handle_returns_none_on_http_error(bot):
    error = httpx.RequestError(
        "boom",
        request=httpx.Request("POST", "https://example.test/api/users/show"),
    )

    with patch("bot.bot.api_client.post", AsyncMock(side_effect=error)):
        assert await bot._resolve_user_handle("user-123") is None


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


@pytest.mark.anyio
async def test_build_mentions_caps_amplification(make_config, make_note, make_user):
    """A note tagging many users yields at most max_reply_mentions in the reply."""
    bot = Bot(config=make_config(max_reply_mentions=3))
    author = make_user(id="u2", username="troll", host="evil.example")
    victims = [f"@victim{i}@target.example" for i in range(40)]
    note = make_note(user=author, mentions=victims)

    mentions = await bot._build_mentions_from_note(note)

    # Capped to 3 total, and the author is always present (reserved slot).
    assert len(mentions) == 3
    assert "@troll@evil.example" in mentions
    assert all(m in {*victims, "@troll@evil.example"} for m in mentions)


@pytest.mark.anyio
async def test_build_mentions_limit_one_keeps_only_author(make_config, make_note, make_user):
    bot = Bot(config=make_config(max_reply_mentions=1))
    note = make_note(user=make_user(username="carol"), mentions=["@alice", "@bob"])
    assert await bot._build_mentions_from_note(note) == ["@carol"]


@pytest.mark.anyio
async def test_send_note_preserves_reply_visibility(bot, make_note):
    note = make_note().model_copy(update={"visibility": "followers"})
    response = MagicMock()
    response.json.return_value = {"createdNote": {"id": "created-note"}}

    with (
        patch.object(bot, "_build_mentions_from_note", AsyncMock(return_value=["@alice"])),
        patch("bot.bot.api_client.post", AsyncMock(return_value=response)) as post_mock,
    ):
        await bot.send_note("@alice hello there", in_reply_to=note)

    assert post_mock.await_args is not None
    payload = post_mock.await_args.kwargs["json"]
    assert payload["visibility"] == "followers"
    assert payload["text"] == "@alice hello there"
    assert "visibleUserIds" not in payload


@pytest.mark.anyio
async def test_send_note_preserves_specified_recipients(bot, make_note, make_user):
    author = make_user(id="user-2", username="carol")
    note = make_note(user=author).model_copy(
        update={
            "visibility": "specified",
            "visibleUserIds": [author.id, bot.user_id, "user-3"],
            "localOnly": True,
        }
    )
    response = MagicMock()
    response.json.return_value = {"createdNote": {"id": "created-note"}}

    with (
        patch.object(bot, "_build_mentions_from_note", AsyncMock(return_value=["@carol"])),
        patch("bot.bot.api_client.post", AsyncMock(return_value=response)) as post_mock,
    ):
        await bot.send_note("@carol secret hello", in_reply_to=note)

    assert post_mock.await_args is not None
    payload = post_mock.await_args.kwargs["json"]
    assert payload["visibility"] == "specified"
    assert payload["visibleUserIds"] == [author.id, "user-3"]
    assert payload["localOnly"] is True


@pytest.mark.anyio
async def test_get_note_returns_parsed_note(bot, make_note):
    expected = make_note(id="context-note")
    response = MagicMock()
    response.json.return_value = expected.model_dump(mode="json")

    with patch("bot.bot.api_client.post", AsyncMock(return_value=response)) as post_mock:
        result = await bot.get_note("context-note")

    assert result.id == "context-note"
    post_mock.assert_awaited_once_with(
        "https://example.test/api/notes/show",
        json={"noteId": "context-note"},
    )


@pytest.mark.anyio
async def test_on_mention_processes_image_only_note(bot, make_note):
    note = make_note(
        text=None,
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
    )

    with (
        patch.object(bot._agent, "run", AsyncMock(return_value="vision reply")) as run_mock,
        patch.object(bot, "send_note", AsyncMock()) as send_note_mock,
    ):
        await bot.on_mention(note)

    run_mock.assert_awaited_once_with(note=note, context=[])
    send_note_mock.assert_awaited_once_with("vision reply", in_reply_to=note)


@pytest.mark.anyio
async def test_on_mention_ignores_direct_messages(bot, make_note):
    """DMs (Misskey 'specified' visibility) are skipped by default — no run, no reply."""
    note = make_note(text="psst, just between us").model_copy(update={"visibility": "specified"})

    with (
        patch.object(bot._agent, "run", AsyncMock()) as run_mock,
        patch.object(bot, "send_note", AsyncMock()) as send_note_mock,
    ):
        await bot.on_mention(note)

    run_mock.assert_not_awaited()
    send_note_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_on_mention_handles_dm_when_enabled(make_config, make_note):
    """With ignore_direct_messages=False the bot replies to DMs like any other note."""
    bot = Bot(config=make_config(ignore_direct_messages=False))
    note = make_note(text="hi privately").model_copy(update={"visibility": "specified"})

    with (
        patch.object(bot._agent, "run", AsyncMock(return_value="dm reply")) as run_mock,
        patch.object(bot, "send_note", AsyncMock()) as send_note_mock,
    ):
        await bot.on_mention(note)

    run_mock.assert_awaited_once_with(note=note, context=[])
    send_note_mock.assert_awaited_once_with("dm reply", in_reply_to=note)


@pytest.mark.anyio
async def test_on_mention_ignores_author_below_score_threshold(make_config, make_note):
    """An author scoring below social_credit_ignore_threshold is dropped — no run, no reply."""
    bot = Bot(config=make_config(social_credit_ignore_threshold=-50))
    note = make_note(text="i am a menace")

    with (
        patch.object(bot._agent, "get_author_score", AsyncMock(return_value=-60)),
        patch.object(bot._agent, "run", AsyncMock()) as run_mock,
        patch.object(bot, "send_note", AsyncMock()) as send_note_mock,
    ):
        await bot.on_mention(note)

    run_mock.assert_not_awaited()
    send_note_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_on_mention_replies_to_author_at_or_above_threshold(make_config, make_note):
    """Scores at/above the threshold are processed normally."""
    bot = Bot(config=make_config(social_credit_ignore_threshold=-50))
    note = make_note(text="hello")

    with (
        patch.object(bot._agent, "get_author_score", AsyncMock(return_value=-50)),
        patch.object(bot._agent, "run", AsyncMock(return_value="reply")) as run_mock,
        patch.object(bot, "send_note", AsyncMock()) as send_note_mock,
    ):
        await bot.on_mention(note)

    run_mock.assert_awaited_once_with(note=note, context=[])
    send_note_mock.assert_awaited_once_with("reply", in_reply_to=note)


@pytest.mark.anyio
async def test_on_mention_does_not_ignore_unscored_author(make_config, make_note):
    """An author with no score yet (None) is never ignored, even with a threshold set."""
    bot = Bot(config=make_config(social_credit_ignore_threshold=0))
    note = make_note(text="first time here")

    with (
        patch.object(bot._agent, "get_author_score", AsyncMock(return_value=None)),
        patch.object(bot._agent, "run", AsyncMock(return_value="reply")) as run_mock,
        patch.object(bot, "send_note", AsyncMock()) as send_note_mock,
    ):
        await bot.on_mention(note)

    run_mock.assert_awaited_once_with(note=note, context=[])
    send_note_mock.assert_awaited_once_with("reply", in_reply_to=note)


@pytest.mark.anyio
async def test_on_mention_skips_score_check_when_threshold_unset(make_config, make_note):
    """With no threshold configured the score is never fetched."""
    bot = Bot(config=make_config())
    note = make_note(text="hi")

    with (
        patch.object(bot._agent, "get_author_score", AsyncMock()) as score_mock,
        patch.object(bot._agent, "run", AsyncMock(return_value="reply")),
        patch.object(bot, "send_note", AsyncMock()),
    ):
        await bot.on_mention(note)

    score_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_on_auto_reply_updates_timestamp_and_triggers_reply(make_config, fake_redis, make_note):
    cfg = make_config(auto_reply_interval=5, auto_reply_jitter=0)
    bot = Bot(config=cfg, redis_client=fake_redis)
    note = make_note()
    bot._last_auto_reply_time = 0
    bot._next_auto_reply_delay = 5

    with (
        patch("bot.bot.time.time", return_value=10),
        patch.object(bot, "_compute_auto_reply_delay", return_value=12),
        patch.object(bot, "_save_last_auto_reply_time", AsyncMock()) as save_mock,
        patch.object(bot, "on_mention", AsyncMock()) as mention_mock,
    ):
        await bot.on_auto_reply(note)

    assert bot._last_auto_reply_time == 10
    assert bot._next_auto_reply_delay == 12
    save_mock.assert_awaited_once_with()
    mention_mock.assert_awaited_once_with(note)


@pytest.mark.anyio
async def test_send_note_raises_on_oversized_output(bot, make_note):
    """Over-cap output is refused, not truncated — the model ignored its length budget."""
    note = make_note()
    limit = bot._config.max_note_length
    long_output = "a" * (limit + 500)

    with (
        patch.object(bot, "_build_mentions_from_note", AsyncMock(return_value=[])),
        patch("bot.bot.api_client.post", AsyncMock(return_value=MagicMock())) as post_mock,
    ):
        with pytest.raises(ValueError):
            await bot.send_note(long_output, in_reply_to=note)

    post_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_send_note_raises_when_mentions_push_over_cap(bot, make_note):
    """Mentions count toward the cap, so a body that just fits can overflow once they're prepended."""
    note = make_note()
    limit = bot._config.max_note_length
    # Body alone fits, but body + mention header overflows.
    body = "a" * (limit - 5)
    long_mention = "@" + ("b" * 100)

    with (
        patch.object(bot, "_build_mentions_from_note", AsyncMock(return_value=[long_mention])),
        patch("bot.bot.api_client.post", AsyncMock(return_value=MagicMock())) as post_mock,
    ):
        with pytest.raises(ValueError):
            await bot.send_note(body, in_reply_to=note)

    post_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_post_autonomous_raises_on_oversized_output(bot):
    limit = bot._config.max_note_length
    long_output = "z" * (limit + 200)

    with (
        patch.object(bot._agent, "run_auto", AsyncMock(return_value=long_output)),
        patch("bot.bot.api_client.post", AsyncMock(return_value=MagicMock())) as post_mock,
    ):
        with pytest.raises(ValueError):
            await bot.post_autonomous()

    post_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_post_autonomous_posts_public_note(bot):
    response = MagicMock()
    response.json.return_value = {"createdNote": {"id": "auto-1"}}

    with (
        patch.object(bot._agent, "run_auto", AsyncMock(return_value="hello timeline")) as run_auto_mock,
        patch("bot.bot.api_client.post", AsyncMock(return_value=response)) as post_mock,
    ):
        await bot.post_autonomous()

    run_auto_mock.assert_awaited_once_with()
    post_mock.assert_awaited_once_with(
        "https://example.test/api/notes/create",
        json={"text": "hello timeline", "visibility": "public"},
    )


def test_task_done_callback_logs_failures(bot):
    task = MagicMock()
    task.cancelled.return_value = False
    task.result.side_effect = RuntimeError("boom")

    with patch("bot.bot.logfire.exception") as exception_mock:
        bot._task_done_callback(task)

    exception_mock.assert_called_once_with("Task failed with exception")
