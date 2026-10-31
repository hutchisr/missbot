"""Nostr adapter regressions; transport authentication is tested separately."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import BinaryContent, ImageUrl

from bot.core import AutoPost, Poll
from bot.imagegen import GeneratedImage
from bot.nostr import NostrBot

BOT_KEY = "b" * 64
AUTHOR_KEY = "a" * 64
ROOT_ID = "1" * 64
PARENT_ID = "2" * 64
EVENT_ID = "3" * 64


def event(event_id=EVENT_ID, *, content="hello", pubkey=AUTHOR_KEY, kind=1, tags=None):
    return {
        "id": event_id,
        "pubkey": pubkey,
        "created_at": 1_800_000_000,
        "kind": kind,
        "content": content,
        "tags": tags if tags is not None else [["p", BOT_KEY]],
        "sig": "0" * 128,
    }


@pytest.fixture
def frontend(make_config):
    def make(**overrides):
        config = make_config(**overrides)
        agent = MagicMock()
        agent.run = AsyncMock(return_value="reply")
        agent.get_score = AsyncMock(return_value=None)
        agent.run_auto = AsyncMock(return_value=AutoPost(text="autonomous"))
        transport = MagicMock()
        transport.public_key = BOT_KEY
        transport.fetch_event = AsyncMock(return_value=None)
        transport.fetch_profile = AsyncMock(return_value=None)
        transport.publish = AsyncMock(return_value="4" * 64)
        transport.run = AsyncMock()
        upload = AsyncMock(return_value="https://media.example/image.png")
        return NostrBot(config, agent, transport, upload), agent, transport, upload

    return make


async def drain(bot):
    await asyncio.gather(*tuple(bot._tasks))


@pytest.mark.anyio
async def test_duplicate_delivery_pending_and_completed_produces_one_reply(frontend):
    bot, agent, transport, _ = frontend()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(turn):
        entered.set()
        await release.wait()
        return "reply"

    agent.run.side_effect = respond
    note = event()
    await bot.on_event(note)
    await entered.wait()
    await bot.on_event(note)
    release.set()
    await drain(bot)
    await bot.on_event(note)
    await drain(bot)
    agent.run.assert_awaited_once()
    transport.publish.assert_awaited_once_with(
        "reply", [["e", EVENT_ID, "", "root", AUTHOR_KEY], ["p", AUTHOR_KEY]]
    )
    assert not bot._pending


@pytest.mark.anyio
async def test_verified_thread_namespaced_identity_and_no_mention_amplification(frontend):
    bot, agent, transport, _ = frontend(social_credit_unrestricted_user_ids=[AUTHOR_KEY])
    root = event(ROOT_ID, content="root", tags=[])
    parent = event(PARENT_ID, content="prior bot reply", pubkey=BOT_KEY, tags=[["e", ROOT_ID, "", "root"]])
    transport.fetch_event.side_effect = [parent, root]
    note = event(tags=[
        ["p", BOT_KEY], ["p", "c" * 64],
        ["e", ROOT_ID, "", "root"], ["e", PARENT_ID, "", "reply"],
    ])
    await bot.on_event(note)
    await drain(bot)
    turn = agent.run.await_args.args[0]
    assert turn.author.handle == turn.author.user_id == f"nostr:{AUTHOR_KEY}"
    assert not turn.author.privileged
    assert turn.source_id == f"nostr:{EVENT_ID}"
    assert turn.conversation_id == f"nostr:{ROOT_ID}"
    assert turn.source == "nostr_note"
    assert turn.memory_access_allowed
    assert turn.occurred_at.timestamp() == note["created_at"]
    assert [(past.role, past.text, past.author) for past in turn.history] == [
        ("user", "root", f"nostr:{AUTHOR_KEY}"), ("assistant", "prior bot reply", None),
    ]
    assert turn.previous_reply == "prior bot reply"
    transport.publish.assert_awaited_once_with("reply", [
        ["e", ROOT_ID, "", "root", AUTHOR_KEY],
        ["e", EVENT_ID, "", "reply", AUTHOR_KEY],
        ["p", AUTHOR_KEY],
    ])


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["missing", "truncated", "forged-root", "cycle", "private", "ambiguous"])
async def test_incomplete_or_conflicting_ancestry_cannot_invent_root(frontend, failure):
    bot, agent, transport, _ = frontend(max_context=1 if failure == "truncated" else 3)
    root = event(ROOT_ID, tags=[])
    parent = event(PARENT_ID, tags=[["e", ROOT_ID, "", "root"]])
    note = event(tags=[["p", BOT_KEY], ["e", ROOT_ID, "", "root"], ["e", PARENT_ID, "", "reply"]])
    if failure == "missing":
        transport.fetch_event.return_value = None
    elif failure == "truncated":
        transport.fetch_event.return_value = parent
    elif failure == "forged-root":
        note["tags"][1][1] = "f" * 64
        transport.fetch_event.side_effect = [parent, root]
    elif failure == "cycle":
        parent["tags"] = [["e", EVENT_ID, "", "reply"]]
        transport.fetch_event.return_value = parent
    elif failure == "private":
        parent["kind"] = 4
        transport.fetch_event.return_value = parent
    else:
        note["tags"].append(["e", "f" * 64, "", "reply"])
    await bot.on_event(note)
    await drain(bot)
    assert agent.run.await_args.args[0].conversation_id == f"nostr:{EVENT_ID}"
    assert transport.publish.await_args.args[1][0] == ["e", EVENT_ID, "", "root", AUTHOR_KEY]


@pytest.mark.anyio
async def test_legacy_and_textless_ancestors_establish_verified_root(frontend):
    bot, agent, transport, _ = frontend()
    transport.fetch_event.side_effect = [
        event(PARENT_ID, content="parent", tags=[["e", ROOT_ID]]),
        event(ROOT_ID, content="", tags=[]),
    ]
    await bot.on_event(event(tags=[["p", BOT_KEY], ["e", ROOT_ID], ["e", PARENT_ID]]))
    await drain(bot)
    turn = agent.run.await_args.args[0]
    assert turn.conversation_id == f"nostr:{ROOT_ID}"
    assert [past.text for past in turn.history] == ["parent"]


@pytest.mark.anyio
@pytest.mark.parametrize("kind", [4, 14, 1059, 1068])
async def test_private_and_non_note_events_never_reach_agent(frontend, kind):
    bot, agent, transport, _ = frontend()
    await bot.on_event(event(kind=kind))
    await drain(bot)
    agent.run.assert_not_awaited()
    agent.get_score.assert_not_awaited()
    transport.publish.assert_not_awaited()


@pytest.mark.anyio
async def test_no_reply_and_low_score_are_not_published(frontend):
    bot, agent, transport, _ = frontend(social_credit_ignore_threshold=-5)
    agent.run.return_value = " NO_REPLY\n"
    await bot.on_event(event())
    await drain(bot)
    agent.get_score.assert_awaited_once_with(f"nostr:{AUTHOR_KEY}")
    agent.run.assert_awaited_once()
    transport.publish.assert_not_awaited()
    agent.get_score.return_value = -6
    await bot.on_event(event(PARENT_ID))
    await drain(bot)
    assert agent.run.await_count == 1
    transport.publish.assert_not_awaited()


@pytest.mark.anyio
async def test_own_empty_unaddressed_and_oversized_events_are_skipped(frontend):
    bot, agent, transport, _ = frontend(nostr_max_event_chars=1024)
    for note in (event(pubkey=BOT_KEY), event(content="  "), event(tags=[]), event(content="x" * 1025)):
        await bot.on_event(note)
    await drain(bot)
    agent.run.assert_not_awaited()
    transport.publish.assert_not_awaited()


@pytest.mark.anyio
async def test_capacity_and_recent_dedup_are_bounded(frontend):
    bot, _, _, _ = frontend(max_concurrent_handlers=1)
    with patch("bot.nostr._RECENT_EVENT_LIMIT", 2):
        await bot.on_event(event(ROOT_ID))
        await bot.on_event(event(PARENT_ID))
        assert len(bot._tasks) == len(bot._pending) == 1
        await drain(bot)
        await bot.on_event(event(PARENT_ID))
        await drain(bot)
        await bot.on_event(event(EVENT_ID))
        await drain(bot)
    assert bot._recent == {PARENT_ID, EVENT_ID}
    assert len(bot._recent_order) == 2


@pytest.mark.anyio
async def test_cancellation_drains_handlers_without_owning_agent(frontend):
    bot, agent, transport, _ = frontend()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def respond(turn):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def receive(handler):
        await handler(event())
        await asyncio.Event().wait()

    agent.run.side_effect = respond
    transport.run.side_effect = receive
    task = asyncio.create_task(bot.run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert not bot._tasks
    assert not bot._pending
    agent.__aenter__.assert_not_called()
    agent.__aexit__.assert_not_called()


@pytest.mark.anyio
async def test_handler_failure_isolated_and_duplicate_not_retried(frontend):
    bot, agent, transport, _ = frontend()
    agent.run.side_effect = RuntimeError("provider failed")
    await bot.on_event(event())
    await drain(bot)
    await bot.on_event(event())
    await drain(bot)
    agent.run.assert_awaited_once()
    transport.publish.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["url", "fetch"])
async def test_imeta_images_use_safe_urls_and_configured_vision_mode(frontend, mode):
    bot, agent, _, _ = frontend(vision=True, vision_image_mode=mode)
    url = "https://media.example/photo.png"
    note = event(content="", tags=[
        ["p", BOT_KEY], ["imeta", "url http://127.0.0.1/private.png", "m image/png"],
        ["imeta", f"url {url}", "m image/png"],
    ])
    with patch("bot.nostr.fetch_image", AsyncMock(return_value=(b"image", "image/png"))) as fetch:
        await bot.on_event(note)
        await drain(bot)
    images = agent.run.await_args.args[0].images
    assert len(images) == 1
    if mode == "url":
        assert isinstance(images[0], ImageUrl)
        assert images[0].url == url
        fetch.assert_not_awaited()
    else:
        assert isinstance(images[0], BinaryContent)
        assert images[0].data == b"image"
        fetch.assert_awaited_once_with(
            url, timeout=bot._config.http_timeout_seconds, max_bytes=bot._config.vision_max_image_bytes,
        )


@pytest.mark.anyio
async def test_auto_image_and_nip88_poll_publish_real_attachments(frontend):
    bot, agent, transport, upload = frontend(nostr_relays=["wss://relay.example"])
    image = GeneratedImage(data=b"image", media_type="image/png", prompt="a cat", alt_text="A cat")
    agent.run_auto.return_value = AutoPost(
        text="Which cat?", image=image, poll=Poll(choices=("A", "B"), multiple=True, duration_minutes=5),
    )
    with patch("bot.nostr.time.time", return_value=1000):
        await bot.post_autonomous()
    upload.assert_awaited_once_with(image)
    args = transport.publish.await_args
    assert args.args[0] == "Which cat?\n\nhttps://media.example/image.png"
    assert args.kwargs == {"kind": 1068}
    assert ["option", "0", "A"] in args.args[1]
    assert ["option", "1", "B"] in args.args[1]
    assert ["polltype", "multiplechoice"] in args.args[1]
    assert ["endsAt", "1300"] in args.args[1]
    assert ["relay", str(bot._config.nostr_relays[0])] in args.args[1]
    assert ["imeta", "url https://media.example/image.png", "m image/png", "alt A cat"] in args.args[1]


@pytest.mark.anyio
async def test_failed_image_upload_never_silently_discards_attachment(frontend):
    bot, agent, transport, upload = frontend()
    agent.run_auto.return_value = AutoPost(
        text="caption", image=GeneratedImage(data=b"image", media_type="image/png", prompt="cat", alt_text="cat"),
    )
    upload.return_value = None
    with pytest.raises(ValueError, match="refusing to discard"):
        await bot.post_autonomous()
    transport.publish.assert_not_awaited()


@pytest.mark.anyio
async def test_unavailable_image_only_content_does_not_reach_agent(frontend):
    bot, agent, transport, _ = frontend(vision=True, vision_image_mode="fetch")
    note = event(content="", tags=[
        ["p", BOT_KEY], ["imeta", "url https://media.example/photo.png", "m image/png"],
    ])
    with patch("bot.nostr.fetch_image", AsyncMock(return_value=None)):
        await bot.on_event(note)
        await drain(bot)
    agent.run.assert_not_awaited()
    transport.publish.assert_not_awaited()


@pytest.mark.anyio
async def test_nip05_display_keeps_public_key_identity_and_expires(frontend):
    bot, agent, transport, _ = frontend()
    transport.fetch_profile.return_value = event(
        kind=0, content=json.dumps({"nip05": "alice@example.com"}), tags=[]
    )
    with patch("bot.nostr.verify_identifier", new=AsyncMock(side_effect=[True, False])) as verify:
        await bot.on_event(event())
        await drain(bot)
        author = agent.run.await_args.args[0].author
        assert author.display == f"alice@example.com (nostr:{AUTHOR_KEY})"
        assert author.handle == author.user_id == f"nostr:{AUTHOR_KEY}"
        assert not author.privileged
        assert await bot._display_for(AUTHOR_KEY) == author.display
        assert verify.await_count == 1
        bot._identifiers[AUTHOR_KEY] = (0, author.display)
        assert await bot._display_for(AUTHOR_KEY) is None
        assert verify.await_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["missing", "invalid-json", "wrong-key", "offline", "disabled"])
async def test_unverified_nip05_never_changes_author_identity(frontend, failure):
    bot, agent, transport, _ = frontend(nostr_resolve_nip05=failure != "disabled")
    if failure == "offline":
        transport.fetch_profile.side_effect = RuntimeError("relay offline")
    elif failure == "invalid-json":
        transport.fetch_profile.return_value = event(kind=0, content="not json")
    elif failure == "wrong-key":
        transport.fetch_profile.return_value = event(kind=0, content='{"nip05":"alice@example.com"}')
    with patch("bot.nostr.verify_identifier", new=AsyncMock(return_value=False)):
        await bot.on_event(event())
        await drain(bot)
    author = agent.run.await_args.args[0].author
    assert author.display is None
    assert author.handle == author.user_id == f"nostr:{AUTHOR_KEY}"
    if failure == "disabled":
        transport.fetch_profile.assert_not_awaited()


@pytest.mark.anyio
async def test_nip05_publication_preserves_profile_and_is_strictly_newer(frontend):
    bot, _, transport, _ = frontend(nostr_nip05="bot@example.com")
    metadata = {"name": "Original", "about": "Keep me", "picture": "https://example.com/me.png", "custom": {"x": 1}}
    profile = event(kind=0, content=json.dumps(metadata), tags=[["custom", "preserved"]], pubkey=BOT_KEY)
    transport.fetch_profile.return_value = profile
    with (
        patch("bot.nostr.verify_identifier", new=AsyncMock(return_value=True)),
        patch("bot.nostr.time.time", return_value=profile["created_at"]),
    ):
        await bot._publish_profile()
        args, kwargs = transport.publish.await_args
        assert json.loads(args[0]) == {**metadata, "nip05": "bot@example.com"}
        assert args[1] == profile["tags"]
        assert kwargs == {"kind": 0, "created_at": profile["created_at"] + 1}
        transport.publish.reset_mock()
        profile["content"] = args[0]
        await bot._publish_profile()
        transport.publish.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["wrong-key", "offline", "invalid-profile"])
async def test_nip05_publication_failure_cannot_overwrite_profile(frontend, failure):
    bot, _, transport, _ = frontend(nostr_nip05="bot@example.com")
    if failure == "offline":
        transport.fetch_profile.side_effect = RuntimeError("incomplete relay fetch")
    elif failure == "invalid-profile":
        transport.fetch_profile.return_value = event(kind=0, content="[]")
    with patch("bot.nostr.verify_identifier", new=AsyncMock(return_value=failure != "wrong-key")):
        await bot._publish_profile()
    transport.publish.assert_not_awaited()


@pytest.mark.anyio
async def test_nip05_can_create_confirmed_absent_profile(frontend):
    bot, _, transport, _ = frontend(nostr_nip05="bot@example.com")
    with patch("bot.nostr.verify_identifier", new=AsyncMock(return_value=True)):
        await bot._publish_profile()
    args, kwargs = transport.publish.await_args
    assert json.loads(args[0]) == {"name": bot._config.bot_username, "nip05": "bot@example.com"}
    assert kwargs["kind"] == 0
