"""Public Nostr frontend; the transport authenticates every received/fetched event."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import logfire
from pydantic_ai import BinaryContent, ImageUrl

from .core import AgentTurn, HistoryTurn, TurnAuthor, TurnImage
from .net import fetch_image, is_safe_media_url
from .nip05 import normalize_identifier, verify_identifier

if TYPE_CHECKING:
    from .ai import ChatAgent
    from .imagegen import GeneratedImage
    from .models import Config
    from .nostr_transport import NostrTransport

_RECENT_EVENT_LIMIT = 1024
_EVENT_ID = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _CommentRefs:
    root: str
    root_kind: int
    root_author: str
    parent: str
    parent_kind: int
    parent_author: str | None


def _comment_refs(event: dict[str, Any]) -> _CommentRefs | None:
    """Accept unambiguous NIP-22 scopes on public notes/polls, not private or external scopes."""
    fields = {}
    for name in ("E", "K", "P", "e", "k"):
        matches = [tag for tag in event["tags"] if tag and tag[0] == name]
        if len(matches) != 1 or len(matches[0]) < 2:
            return None
        fields[name] = matches[0]
    if any(tag and tag[0] in ("A", "I", "a", "i") for tag in event["tags"]):
        return None
    if (
        not all(_EVENT_ID.fullmatch(fields[name][1]) for name in ("E", "P", "e"))
        or fields["K"][1] not in ("1", "1068")
        or fields["k"][1] not in ("1", "1111", "1068")
    ):
        return None
    root_author = fields["P"][1]
    root_hint = fields["E"][3] if len(fields["E"]) > 3 else ""
    parent_author = fields["e"][3] if len(fields["e"]) > 3 else ""
    if (root_hint and root_hint != root_author) or (parent_author and not _EVENT_ID.fullmatch(parent_author)):
        return None
    return _CommentRefs(
        fields["E"][1],
        int(fields["K"][1]),
        root_author,
        fields["e"][1],
        int(fields["k"][1]),
        parent_author or None,
    )


def _thread_refs(event: dict[str, Any]) -> tuple[str | None, str | None, bool]:
    """Read NIP-10 marked tags, or legacy first-root/last-parent positional tags."""
    if event["kind"] == 1111:
        comment = _comment_refs(event)
        return (comment.parent, comment.root, True) if comment else (None, None, False)
    roots: set[str] = set()
    replies: set[str] = set()
    positional: list[str] = []
    for tag in event["tags"]:
        if not tag or tag[0] != "e":
            continue
        marker = tag[3] if len(tag) > 3 else ""
        if marker not in ("", "root", "reply"):
            continue  # Explicit mentions are not ancestors.
        if len(tag) < 2 or not _EVENT_ID.fullmatch(tag[1]):
            return None, None, False
        if marker == "root":
            roots.add(tag[1])
        elif marker == "reply":
            replies.add(tag[1])
        else:
            positional.append(tag[1])
    if len(roots) > 1 or len(replies) > 1:
        return None, None, False
    if roots or replies:
        root = next(iter(roots), None)
        return next(iter(replies), root), root, True
    if positional:
        return positional[-1], positional[0] if len(positional) > 1 else None, True
    return None, None, True


class NostrBot:
    def __init__(
        self,
        config: Config,
        agent: ChatAgent,
        transport: NostrTransport,
        upload_image: Callable[[GeneratedImage], Awaitable[str | None]],
    ) -> None:
        self._config = config
        self._agent = agent
        self._transport = transport
        self._upload_image = upload_image
        self._tasks: set[asyncio.Task[None]] = set()
        self._pending: set[str] = set()
        self._recent: set[str] = set()
        self._recent_order: deque[str] = deque()
        self._identifiers: dict[str, tuple[float, str | None]] = {}

    async def on_event(self, event: dict[str, Any]) -> None:
        """Admit only addressed public notes; reserve deduplication before yielding."""
        if (
            event["kind"] not in (1, 1111)
            or event["pubkey"] == self._transport.public_key
            or len(event["content"]) > self._config.nostr_max_event_chars
            or not any(tag[:2] == ["p", self._transport.public_key] for tag in event["tags"])
            or (not event["content"].strip() and not self._image_urls(event))
            or (event["kind"] == 1111 and _comment_refs(event) is None)
        ):
            return
        event_id = event["id"]
        if event_id in self._pending or event_id in self._recent:
            return
        if len(self._tasks) >= self._config.max_concurrent_handlers:
            logfire.warning("Dropping Nostr mention because handler capacity is full", event_id=event_id)
            return
        # NIP-01 has no authenticated isBot flag: Misskey's ignore_bots cannot apply here.
        self._pending.add(event_id)
        task = asyncio.create_task(self._handle_event(event))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle_event(self, event: dict[str, Any]) -> None:
        try:
            identity = f"nostr:{event['pubkey']}"
            threshold = self._config.social_credit_ignore_threshold
            if threshold is not None:
                score = await self._agent.get_score(identity)
                if score is not None and score < threshold:
                    return
            current_images = await self._media_for(event)
            if not event["content"].strip() and not current_images:
                return
            history, root = await self._context(event)
            comment = _comment_refs(event) if event["kind"] == 1111 else None
            scope = None
            if comment:
                scope = root if root["id"] == comment.root else await self._transport.fetch_event(comment.root)
                if (
                    scope is None
                    or scope["id"] != comment.root
                    or scope["kind"] != comment.root_kind
                    or scope["pubkey"] != comment.root_author
                ):
                    logfire.warning("Cannot verify Nostr comment scope", event_id=event["id"])
                    return
                parent = next((item for item in history if item["id"] == comment.parent), None)
                if parent is None:
                    parent = (
                        scope if comment.parent == comment.root else await self._transport.fetch_event(comment.parent)
                    )
                if (
                    parent is None
                    or parent["id"] != comment.parent
                    or parent["kind"] != comment.parent_kind
                    or (comment.parent_author is not None and parent["pubkey"] != comment.parent_author)
                    or not any(tag[:2] == ["p", parent["pubkey"]] for tag in event["tags"])
                ):
                    logfire.warning("Cannot verify Nostr comment parent", event_id=event["id"])
                    return
                for ancestor in history or [parent]:
                    if ancestor["kind"] == 1111:
                        refs = _comment_refs(ancestor)
                        if refs is None or (refs.root, refs.root_kind, refs.root_author) != (
                            comment.root,
                            comment.root_kind,
                            comment.root_author,
                        ):
                            logfire.warning("Conflicting Nostr comment scope", event_id=event["id"])
                            return
            rendered: list[HistoryTurn] = []
            for past in reversed(history):
                images = await self._media_for(past)
                if not past["content"].strip() and not images:
                    continue
                own = past["pubkey"] == self._transport.public_key
                rendered.append(
                    HistoryTurn(
                        role="assistant" if own else "user",
                        text=past["content"],
                        author=None if own else f"nostr:{past['pubkey']}",
                        images=images,
                    )
                )
            turn = AgentTurn(
                text=event["content"],
                author=TurnAuthor(handle=identity, user_id=identity, display=await self._display_for(event["pubkey"])),
                source_id=f"nostr:{event['id']}",
                conversation_id=f"nostr:{root['id']}",
                occurred_at=datetime.fromtimestamp(event["created_at"], tz=UTC),
                source="nostr_comment" if comment else "nostr_note",
                images=current_images,
                history=rendered,
                char_budget=self._config.nostr_max_event_chars,
                previous_reply=next(
                    (
                        past["content"]
                        for past in history
                        if past["pubkey"] == self._transport.public_key and past["content"].strip()
                    ),
                    None,
                ),
            )
            result = await self._agent.run(turn)
            if not result.strip() or result.strip() == "NO_REPLY":
                return
            if len(result) > self._config.nostr_max_event_chars:
                raise ValueError("Nostr reply exceeds the configured content limit")
            if comment:
                assert scope is not None
                tags = [
                    ["E", scope["id"], "", scope["pubkey"]],
                    ["K", str(scope["kind"])],
                    ["P", scope["pubkey"]],
                    ["e", event["id"], "", event["pubkey"]],
                    ["k", "1111"],
                    ["p", event["pubkey"]],
                ]
                await self._transport.publish(result, tags, kind=1111)
                return
            tags = [["e", root["id"], "", "root", root["pubkey"]]]
            if root["id"] != event["id"]:
                tags.append(["e", event["id"], "", "reply", event["pubkey"]])
            # Deliberately do not copy input p tags: attackers cannot amplify mentions.
            tags.append(["p", event["pubkey"]])
            await self._transport.publish(result, tags)
        except Exception:
            logfire.exception("Error processing Nostr mention", event_id=event["id"])
        finally:
            self._pending.discard(event["id"])
            # Also remember failures: ambiguous publication must not generate a second reply.
            if len(self._recent_order) >= _RECENT_EVENT_LIMIT:
                self._recent.discard(self._recent_order.popleft())
            self._recent_order.append(event["id"])
            self._recent.add(event["id"])

    async def _display_for(self, public_key: str) -> str | None:
        if not self._config.nostr_resolve_nip05:
            return None
        cached = self._identifiers.get(public_key)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        display = None
        try:
            profile = await self._transport.fetch_profile(public_key)
            metadata = json.loads(profile["content"]) if profile is not None else {}
            claimed = metadata.get("nip05") if isinstance(metadata, dict) else None
            if isinstance(claimed, str):
                identifier = normalize_identifier(claimed)
                if await verify_identifier(identifier, public_key):
                    display = f"{identifier} (nostr:{public_key})"
        except Exception:
            logfire.debug("Nostr identifier lookup failed; using public key")
        # Both successes and misses expire, so domain reassignment never changes identity.
        if public_key not in self._identifiers and len(self._identifiers) >= _RECENT_EVENT_LIMIT:
            del self._identifiers[next(iter(self._identifiers))]
        self._identifiers[public_key] = (time.monotonic() + 300, display)
        return display

    async def _publish_profile(self) -> None:
        identifier = self._config.nostr_nip05
        if identifier is None:
            return
        try:
            public_key = self._transport.public_key
            if not await verify_identifier(identifier, public_key):
                logfire.warning("Nostr NIP-05 domain does not identify this key; leaving profile unchanged")
                return
            profile = await self._transport.fetch_profile(public_key)
            metadata = json.loads(profile["content"]) if profile is not None else {"name": self._config.bot_username}
            if not isinstance(metadata, dict):
                raise TypeError("Existing Nostr profile is not a JSON object")
            if metadata.get("nip05") == identifier:
                return
            metadata["nip05"] = identifier
            created_at = max(int(time.time()), profile["created_at"] + 1) if profile is not None else int(time.time())
            await self._transport.publish(
                json.dumps(metadata, ensure_ascii=False, allow_nan=False),
                profile["tags"] if profile is not None else [],
                kind=0,
                created_at=created_at,
            )
            logfire.info("Published Nostr NIP-05 profile")
        except Exception:
            logfire.exception("Nostr NIP-05 profile update failed")

    async def _context(self, event: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Only a complete verified ancestry establishes a shared conversation root."""
        history: list[dict[str, Any]] = []
        seen = {event["id"]}
        claimed_roots: set[str] = set()
        claimed_scopes: set[tuple[str, int, str]] = set()
        current = event
        while True:
            comment = _comment_refs(current) if current["kind"] == 1111 else None
            if comment:
                claimed_scopes.add((comment.root, comment.root_kind, comment.root_author))
            parent_id, claimed_root, valid = _thread_refs(current)
            if not valid:
                return history, event
            if claimed_root:
                claimed_roots.add(claimed_root)
            if parent_id is None:
                root_matches = not claimed_roots or claimed_roots == {current["id"]}
                scope_matches = not claimed_scopes or claimed_scopes == {
                    (current["id"], current["kind"], current["pubkey"])
                }
                return history, current if root_matches and scope_matches else event
            if len(history) >= self._config.max_context or parent_id in seen:
                return history, event
            try:
                parent = await self._transport.fetch_event(parent_id)
            except Exception:
                logfire.exception("Error fetching Nostr context", event_id=parent_id)
                return history, event
            if (
                parent is None
                or parent["id"] != parent_id
                or parent["kind"] != (comment.parent_kind if comment else 1)
                or len(parent["content"]) > self._config.nostr_max_event_chars
            ):
                return history, event
            if comment and (
                (comment.parent_author is not None and parent["pubkey"] != comment.parent_author)
                or not any(tag[:2] == ["p", parent["pubkey"]] for tag in current["tags"])
            ):
                return history, event
            seen.add(parent_id)
            history.append(parent)
            current = parent

    def _image_urls(self, event: dict[str, Any]) -> list[str]:
        if not self._config.vision:
            return []
        urls: list[str] = []
        for tag in event["tags"]:
            if not tag or tag[0] != "imeta":
                continue
            fields = dict(item.split(" ", 1) for item in tag[1:] if " " in item)
            url = fields.get("url", "")
            mime = fields.get("m", "")
            if not url or not is_safe_media_url(url):
                continue
            if not mime.startswith("image/") and (
                mime or not urlsplit(url).path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
            ):
                continue
            if url not in urls:
                urls.append(url)
            # ponytail: four image attachments per event; raise only for demonstrated need.
            if len(urls) == 4:
                break
        return urls

    async def _media_for(self, event: dict[str, Any]) -> list[TurnImage]:
        images: list[TurnImage] = []
        for url in self._image_urls(event):
            if self._config.vision_image_mode != "fetch":
                images.append(ImageUrl(url=url))
                continue
            fetched = await fetch_image(
                url,
                timeout=self._config.http_timeout_seconds,
                max_bytes=self._config.vision_max_image_bytes,
            )
            if fetched is not None:
                data, media_type = fetched
                images.append(BinaryContent(data=data, media_type=media_type))
        return images

    async def post_autonomous(self) -> None:
        post = await self._agent.run_auto()
        if post.text.strip() == "NO_REPLY":
            return
        content = post.text
        tags: list[list[str]] = []
        kind = 1
        if post.poll is not None:
            if not content.strip() or len(post.poll.choices) < 2:
                raise ValueError("Nostr poll needs a question and at least two options")
            kind = 1068
            tags.extend(["option", str(index), choice] for index, choice in enumerate(post.poll.choices))
            tags.append(["polltype", "multiplechoice" if post.poll.multiple else "singlechoice"])
            tags.extend(["relay", str(relay)] for relay in self._config.nostr_relays)
            if post.poll.duration_minutes is not None:
                tags.append(["endsAt", str(int(time.time()) + post.poll.duration_minutes * 60)])
        if post.image is not None:
            url = await self._upload_image(post.image)
            if not url or not is_safe_media_url(url):
                raise ValueError("Nostr image upload failed; refusing to discard the attachment")
            content = f"{content.rstrip()}\n\n{url}".lstrip()
            tags.append(["imeta", f"url {url}", f"m {post.image.media_type}", f"alt {post.image.alt_text}"])
            if self._config.image_gen_mark_sensitive:
                tags.append(["content-warning", ""])
        if not content.strip() or len(content) > self._config.nostr_max_event_chars:
            raise ValueError("Nostr autonomous post is empty or exceeds the configured content limit")
        await self._transport.publish(content, tags, kind=kind)

    async def _auto_post_loop(self) -> None:
        interval = self._config.auto_post_interval
        assert interval is not None
        jitter = self._config.auto_post_jitter
        while True:
            delay = interval + random.randint(-jitter, jitter) if jitter else interval
            await asyncio.sleep(max(delay, 1))
            try:
                await self.post_autonomous()
            except Exception:
                logfire.exception("Error during autonomous Nostr post")

    async def run(self) -> None:
        """Run independently inside the already-entered shared ChatAgent lifetime."""
        auto_task = None
        profile_task = asyncio.create_task(self._publish_profile()) if self._config.nostr_nip05 else None
        if self._config.auto_post_interval:
            auto_task = asyncio.create_task(self._auto_post_loop())
        try:
            await self._transport.run(self.on_event)
        except Exception:
            logfire.exception("Nostr frontend stopped")
        finally:
            tasks = list(self._tasks)
            if profile_task is not None:
                tasks.append(profile_task)
            if auto_task is not None:
                tasks.append(auto_task)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.clear()
            self._pending.clear()
