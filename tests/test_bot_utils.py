"""Tests for Bot helper methods that don't need a live WebSocket."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError
from pydantic_ai import BinaryContent, ImageUrl

from bot.bot import Bot, _image_urls_for, _user_handle
from bot.core import AgentTurn, AutoPost
from bot.imagegen import GeneratedImage
from bot.models import MiFile


@pytest.fixture
def bot(config):
    """Build a Bot without spinning up the websocket."""
    return Bot(config=config)


def _awaited_turn(run_mock) -> AgentTurn:
    """Pull the AgentTurn the Bot handed to ChatAgent.run."""
    run_mock.assert_awaited_once()
    assert run_mock.await_args is not None
    turn = run_mock.await_args.args[0]
    assert isinstance(turn, AgentTurn)
    return turn


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


def test_reply_visibility_narrows_followers_to_author(bot, make_note):
    note = make_note().model_copy(update={"visibility": "followers"})

    assert bot._reply_visibility(note) == "specified"
    assert bot._reply_visible_user_ids(note) == [note.user.id]


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
async def test_send_note_narrows_followers_reply_to_source_author(bot, make_note):
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
    assert payload["visibility"] == "specified"
    assert payload["text"] == "@alice hello there"
    assert payload["visibleUserIds"] == [note.user.id]


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

    turn = _awaited_turn(run_mock)
    assert turn.source_id == note.id
    assert len(turn.images) == 1
    # url mode passes the media URL through rather than downloading it.
    image = turn.images[0]
    assert isinstance(image, ImageUrl)
    assert image.url == "https://media.example/1.png"
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

    turn = _awaited_turn(run_mock)
    assert turn.source_id == note.id
    # Replyable, but its content must stay out of the bot-global memory namespace.
    assert turn.memory_writes_allowed is False
    send_note_mock.assert_awaited_once_with("dm reply", in_reply_to=note)


@pytest.mark.anyio
async def test_on_mention_ignores_author_below_score_threshold(make_config, make_note):
    """An author scoring below social_credit_ignore_threshold is dropped — no run, no reply."""
    bot = Bot(config=make_config(social_credit_ignore_threshold=-50))
    note = make_note(text="i am a menace")

    with (
        patch.object(bot._agent, "get_score", AsyncMock(return_value=-60)),
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
        patch.object(bot._agent, "get_score", AsyncMock(return_value=-50)),
        patch.object(bot._agent, "run", AsyncMock(return_value="reply")) as run_mock,
        patch.object(bot, "send_note", AsyncMock()) as send_note_mock,
    ):
        await bot.on_mention(note)

    assert _awaited_turn(run_mock).source_id == note.id
    send_note_mock.assert_awaited_once_with("reply", in_reply_to=note)


@pytest.mark.anyio
async def test_on_mention_does_not_ignore_unscored_author(make_config, make_note):
    """An author with no score yet (None) is never ignored, even with a threshold set."""
    bot = Bot(config=make_config(social_credit_ignore_threshold=0))
    note = make_note(text="first time here")

    with (
        patch.object(bot._agent, "get_score", AsyncMock(return_value=None)),
        patch.object(bot._agent, "run", AsyncMock(return_value="reply")) as run_mock,
        patch.object(bot, "send_note", AsyncMock()) as send_note_mock,
    ):
        await bot.on_mention(note)

    assert _awaited_turn(run_mock).source_id == note.id
    send_note_mock.assert_awaited_once_with("reply", in_reply_to=note)


@pytest.mark.anyio
async def test_on_mention_skips_score_check_when_threshold_unset(make_config, make_note):
    """With no threshold configured the score is never fetched."""
    bot = Bot(config=make_config())
    note = make_note(text="hi")

    with (
        patch.object(bot._agent, "get_score", AsyncMock()) as score_mock,
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
        handled = await bot.on_auto_reply(note)

    assert handled is True
    assert bot._last_auto_reply_time == 10
    assert bot._next_auto_reply_delay == 12
    save_mock.assert_awaited_once_with()
    mention_mock.assert_awaited_once_with(note)


@pytest.mark.anyio
async def test_note_event_deduplicates_main_and_due_auto_reply(make_config, make_note):
    bot = Bot(config=make_config(auto_reply_interval=5, auto_reply_jitter=0))
    note = make_note()
    bot._last_auto_reply_time = 0
    bot._next_auto_reply_delay = 5
    mention_started = asyncio.Event()
    release_mention = asyncio.Event()

    async def mention_handler(_note):
        mention_started.set()
        await release_mention.wait()

    with (
        patch("bot.bot.time.time", return_value=10),
        patch.object(bot, "on_mention", AsyncMock(side_effect=mention_handler)) as mention_mock,
    ):
        auto_task = asyncio.create_task(bot._process_note_event(note, auto_reply=True))
        await mention_started.wait()
        main_task = asyncio.create_task(bot._process_note_event(note, auto_reply=False))
        await asyncio.sleep(0)
        release_mention.set()
        await asyncio.gather(auto_task, main_task)

    mention_mock.assert_awaited_once_with(note)


@pytest.mark.anyio
async def test_not_due_auto_reply_does_not_suppress_main_mention(bot, make_note):
    note = make_note()
    auto_started = asyncio.Event()
    release_auto = asyncio.Event()

    async def not_due(_note):
        auto_started.set()
        await release_auto.wait()
        return False

    with (
        patch.object(bot, "on_auto_reply", AsyncMock(side_effect=not_due)),
        patch.object(bot, "on_mention", AsyncMock()) as mention_mock,
    ):
        auto_task = asyncio.create_task(bot._process_note_event(note, auto_reply=True))
        await auto_started.wait()
        main_task = asyncio.create_task(bot._process_note_event(note, auto_reply=False))
        await asyncio.sleep(0)
        mention_mock.assert_not_awaited()
        release_auto.set()
        await asyncio.gather(auto_task, main_task)

    mention_mock.assert_awaited_once_with(note)


@pytest.mark.anyio
async def test_failed_note_processing_can_be_retried(bot, make_note):
    note = make_note()

    with patch.object(
        bot,
        "on_mention",
        AsyncMock(side_effect=[RuntimeError("temporary failure"), None]),
    ) as mention_mock:
        with pytest.raises(RuntimeError, match="temporary failure"):
            await bot._process_note_event(note, auto_reply=False)
        await bot._process_note_event(note, auto_reply=False)

    assert mention_mock.await_count == 2


def test_max_concurrent_handlers_must_be_positive(make_config):
    with pytest.raises(ValidationError):
        make_config(max_concurrent_handlers=0)


@pytest.mark.anyio
async def test_background_handler_capacity_drops_without_creating_coroutine(make_config):
    bot = Bot(config=make_config(max_concurrent_handlers=1))
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def first_handler():
        first_started.set()
        await release_first.wait()

    first_task = bot._spawn_background_task(first_handler, note_id="note-1")
    assert first_task is not None
    await first_started.wait()

    rejected_factory = MagicMock()
    with patch("bot.bot.logfire.warning") as warning_mock:
        rejected_task = bot._spawn_background_task(rejected_factory, note_id="note-2")

    assert rejected_task is None
    rejected_factory.assert_not_called()
    warning_mock.assert_called_once_with(
        "Dropping note because handler capacity is full",
        note_id="note-2",
        capacity=1,
    )

    release_first.set()
    await first_task


@pytest.mark.anyio
async def test_run_cancels_handlers_before_agent_context_exits(bot):
    events: list[str] = []
    handler_started = asyncio.Event()
    never_finishes = asyncio.Event()

    async def handler():
        try:
            handler_started.set()
            await never_finishes.wait()
        finally:
            events.append("handler-stopped")

    class TrackingAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            events.append("agent-exited")

    async def run_loop():
        task = bot._spawn_background_task(handler, note_id="note-1")
        assert task is not None
        await handler_started.wait()

    bot._agent = TrackingAgent()  # type: ignore[assignment]
    with patch.object(bot, "_run_loop", AsyncMock(side_effect=run_loop)):
        await bot.run()

    assert events == ["handler-stopped", "agent-exited"]
    assert not bot._background_tasks


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
        patch.object(bot._agent, "run_auto", AsyncMock(return_value=AutoPost(text=long_output))),
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
        patch.object(bot._agent, "run_auto", AsyncMock(return_value=AutoPost(text="hello timeline"))) as run_auto_mock,
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


# ---------------------------------------------------------------------------
# Misskey -> neutral turn adaptation
# ---------------------------------------------------------------------------


def test_user_handle_local(make_user):
    assert _user_handle(make_user(username="alice", host=None)) == "alice"


def test_user_handle_remote(make_user):
    assert _user_handle(make_user(username="alice", host="remote.host")) == "alice@remote.host"


def test_image_urls_for_vision_off(make_note):
    note = make_note(files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")])
    assert _image_urls_for(note, vision=False) == []


def test_image_urls_for_includes_image_and_video_thumbnails(make_note):
    files = [
        # Image: thumbnail used.
        MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png"),
        # Video: Misskey renders an image thumbnail — use it (NOT the raw video url).
        MiFile(
            id="f2",
            type="video/mp4",
            thumbnailUrl="https://media.example/2-thumb.jpg",
            url="https://media.example/2.mp4",
        ),
        # Video with no thumbnail: skipped (we must not feed the video file as an image).
        MiFile(id="f3", type="video/mp4", thumbnailUrl=None, url="https://media.example/3.mp4"),
        # Non-visual media: skipped.
        MiFile(id="f4", type="audio/mpeg", thumbnailUrl="https://media.example/4.png"),
        # Image with no thumbnail or url: skipped.
        MiFile(id="f5", type="image/jpeg", thumbnailUrl=None),
    ]
    note = make_note(files=files)
    urls = _image_urls_for(note, vision=True)
    assert all(isinstance(u, ImageUrl) for u in urls)
    assert [u.url for u in urls] == ["https://media.example/1.png", "https://media.example/2-thumb.jpg"]


def test_image_urls_for_no_files(make_note):
    note = make_note(files=None)
    assert _image_urls_for(note, vision=True) == []


def test_image_urls_for_drops_ssrf_urls(make_note):
    """Attacker-controlled internal URLs are dropped even with a spoofed image type."""
    files = [
        MiFile(id="f1", type="image/png", thumbnailUrl="http://169.254.169.254/latest/meta-data/"),
        MiFile(id="f2", type="image/png", url="http://missbot-redis.misskey.svc.cluster.local:6379/"),
        MiFile(id="f3", type="image/png", thumbnailUrl="https://media.example/ok.png"),
    ]
    note = make_note(files=files)
    urls = _image_urls_for(note, vision=True)
    # Only the public URL survives.
    assert [u.url for u in urls] == ["https://media.example/ok.png"]


@pytest.mark.anyio
async def test_note_to_turn_maps_author_and_text(bot, make_note, make_user):
    note = make_note(text="hi", user=make_user(username="alice", host="remote.host", location="Berlin"))
    turn = await bot._note_to_turn(note, [])

    assert turn.text == "hi"
    assert turn.author.handle == "alice@remote.host"
    assert turn.author.rendered == "alice@remote.host"
    assert turn.author.location == "Berlin"
    assert turn.author.privileged is False
    assert turn.source_id == note.id


@pytest.mark.anyio
async def test_note_to_turn_empty_text_becomes_empty_string(bot, make_note):
    assert (await bot._note_to_turn(make_note(text=None), [])).text == ""


@pytest.mark.anyio
async def test_note_to_turn_flags_privileged_author_by_user_id(make_config, make_note, make_user):
    bot = Bot(config=make_config(social_credit_unrestricted_user_ids=["admin-id"]))
    note = make_note(text="judge bob", user=make_user(id="admin-id", username="boss"))

    assert (await bot._note_to_turn(note, [])).author.privileged is True


@pytest.mark.anyio
async def test_note_to_turn_budgets_below_note_cap(bot, make_note):
    """The reply budget leaves headroom for the mention prefix send_note prepends."""
    turn = await bot._note_to_turn(make_note(text="hi"), [])
    assert turn.char_budget == bot._config.max_note_length - 280


@pytest.mark.parametrize("visibility", ["followers", "specified"])
@pytest.mark.anyio
async def test_note_to_turn_blocks_memory_writes_for_restricted_notes(bot, make_note, visibility):
    note = make_note(text="secret").model_copy(update={"visibility": visibility})
    assert (await bot._note_to_turn(note, [])).memory_writes_allowed is False


@pytest.mark.anyio
async def test_note_to_turn_allows_memory_writes_for_public_notes(bot, make_note):
    note = make_note(text="public thing").model_copy(update={"visibility": "public"})
    assert (await bot._note_to_turn(note, [])).memory_writes_allowed is True


@pytest.mark.anyio
async def test_note_to_turn_orders_history_oldest_first(bot, make_note, make_user):
    """`context` is nearest-parent first; the turn's history is chronological."""
    older = make_note(id="n1", text="oldest", user=make_user(username="bob"))
    newer = make_note(id="n2", text="middle", user=make_user(username="carol"))
    # Nearest parent first, as on_mention builds it.
    turn = await bot._note_to_turn(make_note(text="latest"), [newer, older])

    assert [(h.role, h.author, h.text) for h in turn.history] == [
        ("user", "bob", "oldest"),
        ("user", "carol", "middle"),
    ]


@pytest.mark.anyio
async def test_note_to_turn_marks_bot_notes_as_assistant_and_strips_mentions(bot, make_note, make_user):
    bot_note = make_note(id="n1", text="@alice my earlier reply", user=make_user(id=bot.user_id, username="grok"))
    turn = await bot._note_to_turn(make_note(text="follow-up"), [bot_note])

    assert [(h.role, h.text) for h in turn.history] == [("assistant", "my earlier reply")]
    # The repeat guard compares against the raw prior reply.
    assert turn.previous_reply == "@alice my earlier reply"


@pytest.mark.anyio
async def test_note_to_turn_previous_reply_none_without_bot_notes(bot, make_note, make_user):
    other = make_note(id="n1", text="someone else", user=make_user(username="bob"))
    assert (await bot._note_to_turn(make_note(text="hi"), [other])).previous_reply is None


@pytest.mark.anyio
async def test_note_to_turn_carries_context_images(bot, make_note, make_user):
    parent = make_note(
        id="n1",
        text="look at this",
        user=make_user(username="bob"),
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
    )
    turn = await bot._note_to_turn(make_note(text="what is it"), [parent])

    assert [i.url for i in turn.history[0].images] == ["https://media.example/1.png"]


# --- vision image modes -----------------------------------------------------


@pytest.mark.anyio
async def test_media_for_defaults_to_urls(bot, make_note):
    """Default mode hands the provider the URL — no fetching from this process."""
    note = make_note(files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")])

    with patch("bot.bot.fetch_image", AsyncMock()) as fetch_mock:
        media = await bot._media_for(note)

    assert [m.url for m in media] == ["https://media.example/1.png"]
    fetch_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_media_for_fetch_mode_sends_bytes_inline(make_config, make_note):
    """Providers that refuse URLs (Ollama Cloud) need BinaryContent instead."""
    bot = Bot(config=make_config(vision_image_mode="fetch"))
    note = make_note(files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")])

    with patch("bot.bot.fetch_image", AsyncMock(return_value=(b"\x89PNG-bytes", "image/png"))) as fetch_mock:
        media = await bot._media_for(note)

    assert len(media) == 1
    assert isinstance(media[0], BinaryContent)
    assert media[0].data == b"\x89PNG-bytes"
    assert media[0].media_type == "image/png"
    assert fetch_mock.await_args is not None
    assert fetch_mock.await_args.kwargs["max_bytes"] == bot._config.vision_max_image_bytes


@pytest.mark.anyio
async def test_media_for_fetch_mode_drops_unfetchable_images(make_config, make_note):
    """A broken attachment must not cost the user their reply."""
    bot = Bot(config=make_config(vision_image_mode="fetch"))
    note = make_note(
        files=[
            MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/broken.png"),
            MiFile(id="f2", type="image/png", thumbnailUrl="https://media.example/ok.png"),
        ]
    )

    with patch("bot.bot.fetch_image", AsyncMock(side_effect=[None, (b"ok-bytes", "image/png")])):
        media = await bot._media_for(note)

    assert len(media) == 1
    # The surviving image is the second one — the broken fetch was dropped, not substituted.
    assert isinstance(media[0], BinaryContent)
    assert media[0].data == b"ok-bytes"


@pytest.mark.anyio
async def test_media_for_fetch_mode_skips_fetch_when_no_images(make_config, make_note):
    bot = Bot(config=make_config(vision_image_mode="fetch"))

    with patch("bot.bot.fetch_image", AsyncMock()) as fetch_mock:
        assert await bot._media_for(make_note(files=None)) == []

    fetch_mock.assert_not_awaited()


@pytest.mark.anyio
async def test_note_to_turn_uses_fetch_mode_for_current_and_history(make_config, make_note, make_user):
    bot = Bot(config=make_config(vision_image_mode="fetch"))
    parent = make_note(
        id="n1",
        text="look",
        user=make_user(username="bob"),
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
    )
    note = make_note(
        text="and this",
        files=[MiFile(id="f2", type="image/png", thumbnailUrl="https://media.example/2.png")],
    )

    with patch("bot.bot.fetch_image", AsyncMock(return_value=(b"bytes", "image/png"))):
        turn = await bot._note_to_turn(note, [parent])

    assert all(isinstance(i, BinaryContent) for i in turn.images)
    assert all(isinstance(i, BinaryContent) for i in turn.history[0].images)


_GENERATED_IMAGE = GeneratedImage(
    data=b"\x89PNG\r\n\x1a\nbytes",
    media_type="image/png",
    prompt="a shrimp in a tiny hat",
    alt_text="a shrimp wearing a hat",
)


def _drive_then_note(drive_response, note_response):
    """api_client.post side effect routing by URL: drive upload, then note creation."""

    async def post(url, *args, **kwargs):
        if "drive/files/create" in url:
            return drive_response
        return note_response

    return AsyncMock(side_effect=post)


def _note_response(note_id: str = "auto-1"):
    response = MagicMock()
    response.json.return_value = {"createdNote": {"id": note_id}}
    return response


@pytest.mark.anyio
async def test_post_autonomous_attaches_generated_image(bot):
    drive = MagicMock()
    drive.json.return_value = {"id": "file-1"}
    post_mock = _drive_then_note(drive, _note_response())

    with (
        patch.object(bot._agent, "run_auto", AsyncMock(return_value=AutoPost(text="shrimp", image=_GENERATED_IMAGE))),
        patch("bot.bot.api_client.post", post_mock),
    ):
        await bot.post_autonomous()

    assert post_mock.await_count == 2
    upload_call, note_call = post_mock.await_args_list
    # Upload must happen before the note that references the file.
    assert "drive/files/create" in upload_call.args[0]
    assert note_call.args[0] == "https://example.test/api/notes/create"
    assert note_call.kwargs["json"] == {
        "text": "shrimp",
        "visibility": "public",
        "fileIds": ["file-1"],
    }


@pytest.mark.anyio
async def test_post_autonomous_upload_sends_alt_text_and_sensitivity(bot):
    drive = MagicMock()
    drive.json.return_value = {"id": "file-1"}
    post_mock = _drive_then_note(drive, _note_response())

    with (
        patch.object(bot._agent, "run_auto", AsyncMock(return_value=AutoPost(text="shrimp", image=_GENERATED_IMAGE))),
        patch("bot.bot.api_client.post", post_mock),
    ):
        await bot.post_autonomous()

    upload = post_mock.await_args_list[0]
    assert upload.kwargs["data"] == {"isSensitive": "false", "comment": "a shrimp wearing a hat"}
    filename, content, media_type = upload.kwargs["files"]["file"]
    assert filename.endswith(".png")
    assert content == _GENERATED_IMAGE.data
    assert media_type == "image/png"


@pytest.mark.anyio
async def test_post_autonomous_marks_sensitive_when_configured(bot):
    bot._config.image_gen_mark_sensitive = True
    drive = MagicMock()
    drive.json.return_value = {"id": "file-1"}
    post_mock = _drive_then_note(drive, _note_response())

    with (
        patch.object(bot._agent, "run_auto", AsyncMock(return_value=AutoPost(text="shrimp", image=_GENERATED_IMAGE))),
        patch("bot.bot.api_client.post", post_mock),
    ):
        await bot.post_autonomous()

    assert post_mock.await_args_list[0].kwargs["data"]["isSensitive"] == "true"


@pytest.mark.anyio
async def test_post_autonomous_posts_text_when_upload_fails(bot):
    """Losing the image beats losing the post."""
    drive = MagicMock()
    drive.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom", request=MagicMock(), response=MagicMock(status_code=500)
    )
    post_mock = _drive_then_note(drive, _note_response())

    with (
        patch.object(bot._agent, "run_auto", AsyncMock(return_value=AutoPost(text="shrimp", image=_GENERATED_IMAGE))),
        patch("bot.bot.api_client.post", post_mock),
    ):
        await bot.post_autonomous()

    note_call = post_mock.await_args_list[-1]
    assert note_call.kwargs["json"] == {"text": "shrimp", "visibility": "public"}


@pytest.mark.anyio
async def test_post_autonomous_posts_text_when_upload_returns_no_id(bot):
    drive = MagicMock()
    drive.json.return_value = {}
    post_mock = _drive_then_note(drive, _note_response())

    with (
        patch.object(bot._agent, "run_auto", AsyncMock(return_value=AutoPost(text="shrimp", image=_GENERATED_IMAGE))),
        patch("bot.bot.api_client.post", post_mock),
    ):
        await bot.post_autonomous()

    assert post_mock.await_args_list[-1].kwargs["json"] == {"text": "shrimp", "visibility": "public"}


@pytest.mark.anyio
async def test_post_autonomous_skips_upload_when_text_over_cap(bot):
    """The length check comes first, so an unusable post never spends an upload."""
    long_text = "z" * (bot._config.max_note_length + 200)
    post_mock = AsyncMock()

    with (
        patch.object(bot._agent, "run_auto", AsyncMock(return_value=AutoPost(text=long_text, image=_GENERATED_IMAGE))),
        patch("bot.bot.api_client.post", post_mock),
    ):
        with pytest.raises(ValueError):
            await bot.post_autonomous()

    post_mock.assert_not_awaited()
