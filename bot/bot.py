import json
import random
import re
import time

import asyncio
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any, Optional
from pydantic import ValidationError
from pydantic_ai import BinaryContent, ImageUrl
from redis.asyncio import Redis
from websockets import ClientConnection, ConnectionClosed
from websockets.asyncio.client import connect
import httpx
import logfire

from .ai import ChatAgent
from .core import AgentTurn, HistoryTurn, TurnAuthor, TurnImage
from .memory import MemoryStore
from .models import (
    Config,
    MiChannelConnect,
    MiChannelConnectBody,
    MiChannelConnectParams,
    MiWebsocketMessage,
    Note,
    User,
)
from .net import fetch_image, is_safe_media_url
from .api import api_client


_REDIS_AUTO_REPLY_KEY = "global:last_auto_reply_time"
_RECENT_NOTE_ID_LIMIT = 1024

# Misskey visibilities whose content must never reach the bot-global memory namespace.
_RESTRICTED_MEMORY_VISIBILITIES = frozenset({"followers", "specified"})

# Chars reserved for the mention prefix send_note prepends (up to ``max_reply_mentions``
# handles). Budgeting the reply below the raw note cap keeps the final note within the
# platform limit; the agent's budget validator enforces it (no truncation backstop —
# over-cap notes are refused).
_MENTION_HEADROOM = 280


def _user_handle(user: User) -> str:
    """Get full handle: username for local, username@host for remote."""
    if user.host:
        return f"{user.username}@{user.host}"
    return user.username


def _image_urls_for(note: Note, vision: bool) -> list[ImageUrl]:
    """Extract ImageUrl objects for a note's visual attachments.

    Images use their thumbnail (falling back to the full image). Videos have no
    image body, but Misskey renders an image thumbnail for them — use that so the
    vision model can still see a frame. Never fall back to a video's raw ``url``
    (that's the video file, not an image). Other media (audio, etc.) is skipped.
    """
    if not vision or not note.files:
        return []

    images: list[ImageUrl] = []
    for file in note.files:
        if file.type.startswith("image/"):
            image_url = file.thumbnailUrl or file.url
        elif file.type.startswith("video/"):
            image_url = file.thumbnailUrl
        else:
            continue
        if not image_url:
            continue
        # SSRF guard: the URL is attacker-controlled on federated notes.
        if not is_safe_media_url(image_url):
            logfire.warning("Dropping image with unsafe URL", url=image_url, file_id=file.id)
            continue
        images.append(ImageUrl(url=image_url))
    return images


async def _fetch_inline(images: list[ImageUrl], *, timeout: float, max_bytes: int) -> list[TurnImage]:
    """Download each image so it can be sent inline as base64.

    For providers that refuse image URLs (Ollama Cloud: "image URLs are not currently
    supported, please use base64 encoded data instead"). An image that can't be fetched
    is dropped rather than failing the note — a broken attachment shouldn't cost the
    user their reply. Fetches run concurrently since a note can carry several.
    """
    results = await asyncio.gather(*(fetch_image(image.url, timeout=timeout, max_bytes=max_bytes) for image in images))
    inline: list[TurnImage] = []
    for image, fetched in zip(images, results):
        if fetched is None:
            logfire.warning("Dropping image that could not be fetched inline", url=image.url)
            continue
        data, media_type = fetched
        inline.append(BinaryContent(data=data, media_type=media_type))
    return inline


class Bot:
    def __init__(
        self,
        config: Config,
        redis_client: Optional[Redis] = None,
        memory: Optional[MemoryStore] = None,
    ):
        self.url = config.url
        self.ws_url = config.ws_url
        self.api_key = config.token
        self.username = config.bot_username
        self.user_id = config.bot_user_id
        self.ws: Optional[ClientConnection] = None

        self._config = config
        self._redis = redis_client
        self._memory = memory
        self._agent = ChatAgent(config, redis_client=redis_client, memory=memory)
        self._shutdown_event = asyncio.Event()
        self._last_auto_reply_time: float = time.time()
        self._next_auto_reply_delay: float = self._compute_auto_reply_delay()
        # Strong refs for fire-and-forget handler tasks; asyncio holds only weakrefs,
        # so without this the tasks can be GC'd mid-run.
        self._background_tasks: set[asyncio.Task[Any]] = set()
        # A public mention can arrive on both the main and global-timeline channels.
        # Coordinate those paths by note id so only one successful handler runs. A
        # timeline event that is not due does not count as handled, allowing the main
        # mention to proceed.
        self._note_processing_lock = asyncio.Lock()
        self._note_processing: dict[str, asyncio.Future[bool]] = {}
        self._recent_note_ids: deque[str] = deque()
        self._recent_note_id_set: set[str] = set()

    @logfire.instrument(extract_args=["note"])
    async def on_mention(self, note: Note):
        if note.user.id == self.user_id:
            logfire.debug("Ignoring own mention")
            return
        # The bot is designed for public-timeline threads; don't engage with direct
        # messages (Misskey 'specified' visibility) unless explicitly configured to.
        if self._config.ignore_direct_messages and note.visibility == "specified":
            logfire.info("Ignoring direct message", note_id=note.id)
            return
        # Don't engage with other bots by default — bot-to-bot exchanges loop endlessly.
        if self._config.ignore_bots and note.user.isBot:
            logfire.info("Ignoring bot account", note_id=note.id, user_id=note.user.id)
            return
        if not self._note_has_prompt_content(note):
            logfire.info("Skipping note without text or supported images", note_id=note.id)
            return
        # Ignore authors below the configured social credit floor entirely: the note never
        # reaches the LLM, no reply is sent, and the author isn't scored or ingested.
        threshold = self._config.social_credit_ignore_threshold
        if threshold is not None:
            score = await self._agent.get_score(_user_handle(note.user))
            if score is not None and score < threshold:
                logfire.info(
                    "Ignoring low-social-credit author",
                    note_id=note.id,
                    user_id=note.user.id,
                    score=score,
                    threshold=threshold,
                )
                return
        context: list[Note] = []
        if note.replyId:
            reply_id = note.replyId
            for _ in range(self._config.max_context):
                try:
                    reply = await self.get_note(reply_id)
                except httpx.HTTPError:
                    logfire.exception("Error fetching context")
                    break
                if reply.text or reply.files:
                    context.append(reply)
                if reply.replyId:
                    reply_id = reply.replyId
                else:
                    break
        if note.renote and (note.renote.text or note.renote.files):
            context.append(note.renote)
        result = await self._agent.run(await self._note_to_turn(note, context))
        if result.strip() == "NO_REPLY":
            logfire.info(f"Skipping reply to note {note.id} (NO_REPLY)")
            return
        await self.send_note(result, in_reply_to=note)

    async def _media_for(self, note: Note) -> list[TurnImage]:
        """Extract a note's images in whichever form the configured provider accepts.

        ``url`` hands the provider the media URL (cheapest). ``fetch`` downloads it here
        and sends bytes inline, which providers like Ollama Cloud require.
        """
        images = _image_urls_for(note, self._config.vision)
        if not images or self._config.vision_image_mode != "fetch":
            return list(images)
        return await _fetch_inline(
            images,
            timeout=self._config.http_timeout_seconds,
            max_bytes=self._config.vision_max_image_bytes,
        )

    async def _note_to_turn(self, note: Note, context: list[Note]) -> AgentTurn:
        """Translate a Misskey note and its reply chain into a frontend-neutral turn.

        Everything Misskey-specific lives here: handle formatting, attachment
        extraction, the visibility rules that gate memory writes, privileged-author
        lookup by user id, and the note-length budget. ``context`` is nearest-parent
        first; ``AgentTurn.history`` is oldest-first.
        """
        author = TurnAuthor(
            handle=_user_handle(note.user),
            # Lift the author-only restriction when the note's author is a designated
            # privileged user (e.g. the operator), configured by user id.
            privileged=note.user.id in self._config.social_credit_unrestricted_user_ids,
            location=note.user.location,
        )

        history: list[HistoryTurn] = []
        for c in reversed(context):
            if c.userId == self.user_id:
                # Strip the leading @mention prefix send_note prepended, so the history
                # doesn't prime the model to re-open (and copy) its prior reply verbatim.
                history.append(HistoryTurn(role="assistant", text=self._strip_leading_mentions((c.text or "").strip())))
            else:
                history.append(
                    HistoryTurn(
                        role="user",
                        text=c.text or "",
                        author=_user_handle(c.user),
                        images=await self._media_for(c),
                    )
                )

        # The bot's most recent reply in this thread (context is nearest-parent first).
        # None when the bot hasn't spoken in the thread.
        previous_reply = next(
            (c.text for c in context if c.userId == self.user_id and (c.text or "").strip()),
            None,
        )
        return AgentTurn(
            text=note.text or "",
            author=author,
            images=await self._media_for(note),
            history=history,
            char_budget=max(1, self._config.max_note_length - _MENTION_HEADROOM),
            source_id=note.id,
            source="misskey_note",
            memory_writes_allowed=note.visibility not in _RESTRICTED_MEMORY_VISIBILITIES,
            previous_reply=previous_reply,
        )

    @logfire.instrument(extract_args=["output"])
    async def send_note(
        self,
        output: str,
        in_reply_to: Optional[Note] = None,
    ):
        mentions = await self._build_mentions_from_note(in_reply_to)
        text = self._strip_leading_mentions(output)
        if mentions:
            text = f"{' '.join(mentions)} {text}"
        # The reply model is told its character budget up front (bot/ai.py:_length_instruction),
        # so over-cap output means it ignored that budget. Fail fast rather than ship a note the
        # model didn't intend to end here (truncation chops mid-sentence and Misskey 400s anyway).
        limit = self._config.max_note_length
        if len(text) > limit:
            raise ValueError(
                f"Reply is {len(text)} chars, over the {limit}-char note cap "
                "(model ignored its length budget); refusing to send."
            )

        payload: dict[str, object] = {
            "text": text,
            "visibility": self._reply_visibility(in_reply_to),
        }
        if in_reply_to and in_reply_to.id:
            payload["replyId"] = in_reply_to.id
        if in_reply_to and in_reply_to.localOnly is not None:
            payload["localOnly"] = in_reply_to.localOnly
        visible_user_ids = self._reply_visible_user_ids(in_reply_to)
        if visible_user_ids:
            payload["visibleUserIds"] = visible_user_ids

        response = await api_client.post(f"{self.url}api/notes/create", json=payload)
        if response.is_error:
            logfire.error(
                "notes/create failed",
                status=response.status_code,
                response_body=response.text,
                request_payload=payload,
                reply_target=in_reply_to.id if in_reply_to else None,
            )
        response.raise_for_status()
        logfire.info("Sent note", id=response.json().get("createdNote").get("id"))

    def _note_has_prompt_content(self, note: Note) -> bool:
        if note.text:
            return True
        return bool(_image_urls_for(note, self._config.vision))

    def _reply_visibility(self, note: Optional[Note]) -> str:
        # "followers" is relative to the author. Reusing it on a bot-authored
        # reply could expose the response to the bot's followers while hiding it
        # from the source author, so narrow followers-only replies to that author.
        if note and note.visibility == "followers":
            return "specified"
        if note and note.visibility:
            return note.visibility
        return "public"

    def _reply_visible_user_ids(self, note: Optional[Note]) -> list[str]:
        if not note or note.visibility not in {"followers", "specified"}:
            return []

        if note.visibility == "followers":
            return [note.user.id] if note.user.id and note.user.id != self.user_id else []

        recipients: list[str] = []
        if note.visibleUserIds:
            recipients.extend(note.visibleUserIds)
        if note.user.id:
            recipients.append(note.user.id)

        # The bot is the author of the reply, not a recipient.
        return self._unique_ordered([user_id for user_id in recipients if user_id and user_id != self.user_id])

    async def _build_mentions_from_note(self, note: Optional[Note]) -> list[str]:
        if not note or not note.user:
            return []

        # Always mention the author (the reply target); cap the echoed mentions so
        # a note that tags many users can't turn the bot into a mass-notification /
        # harassment relay. The bound also limits handle-resolution API calls on
        # attacker-crafted notes (we stop once enough slots are filled).
        limit = self._config.max_reply_mentions
        mentions: list[str] = []
        if note.mentions:
            for mention in note.mentions:
                if len(mentions) >= limit - 1:  # reserve one slot for the author
                    break
                normalized = await self._normalize_note_mention(mention)
                if not normalized:
                    continue
                if re.match(
                    rf"^@?{self._config.bot_username}(@{self._config.domain})?$",
                    normalized.strip(),
                    re.IGNORECASE,
                ):
                    continue
                mentions.append(normalized)

        mentions.append(self._format_handle(note.user))
        return self._unique_ordered(mentions)

    async def _normalize_note_mention(self, mention: str) -> Optional[str]:
        raw = mention.strip()
        if not raw:
            return None

        raw = raw.lstrip("@")
        if not raw:
            return None

        if "@" in raw:
            username, host = raw.split("@", 1)
            if not username:
                return None
            return f"@{username}@{host}" if host else f"@{username}"

        resolved = await self._resolve_user_handle(raw)
        return resolved or f"@{raw}"

    async def _resolve_user_handle(self, user_id: str) -> Optional[str]:
        try:
            response = await api_client.post(
                f"{self.url}api/users/show",
                json={"userId": user_id},
            )
            response.raise_for_status()
            data = response.json()
            username = data.get("username")
            if not username:
                return None
            host = data.get("host")
            return f"@{username}@{host}" if host else f"@{username}"
        except httpx.HTTPError:
            return None

    @logfire.instrument(extract_args=["note_id"])
    async def get_note(self, note_id: str) -> Note:
        response = await api_client.post(
            f"{self.url}api/notes/show",
            json={"noteId": note_id},
        )
        response.raise_for_status()
        return Note(**response.json())

    async def _load_last_auto_reply_time(self):
        """Load last auto reply time from Redis."""
        assert self._redis
        val = await self._redis.get(_REDIS_AUTO_REPLY_KEY)
        if val is not None:
            self._last_auto_reply_time = float(val)
            logfire.info("Loaded last auto reply time from Redis", t=self._last_auto_reply_time)
        else:
            await self._save_last_auto_reply_time()
            logfire.info("Initialized last auto reply time in Redis", t=self._last_auto_reply_time)

    async def _save_last_auto_reply_time(self):
        """Save last auto reply time to Redis."""
        assert self._redis
        await self._redis.set(_REDIS_AUTO_REPLY_KEY, str(self._last_auto_reply_time))

    def _compute_auto_reply_delay(self) -> float:
        interval = self._config.auto_reply_interval
        jitter = self._config.auto_reply_jitter
        return interval + random.randint(-jitter, jitter) if jitter else interval

    async def on_auto_reply(self, note: Note) -> bool:
        """Automatically reply to a timeline note if enough time has passed."""
        if not self._note_has_prompt_content(note):
            return False

        now = time.time()
        elapsed = now - self._last_auto_reply_time
        if elapsed < self._next_auto_reply_delay:
            return False

        self._last_auto_reply_time = now
        self._next_auto_reply_delay = self._compute_auto_reply_delay()
        if self._redis:
            await self._save_last_auto_reply_time()

        logfire.info("Auto-reply triggered", note=note)
        await self.on_mention(note)
        return True

    @logfire.instrument(extract_args=False)
    async def post_autonomous(self):
        """Generate and post an autonomous note to the timeline."""
        result = await self._agent.run_auto()
        limit = self._config.max_note_length
        if len(result) > limit:
            raise ValueError(
                f"Autonomous post is {len(result)} chars, over the {limit}-char note cap "
                "(model ignored its length budget); refusing to send."
            )
        response = await api_client.post(
            f"{self.url}api/notes/create",
            json={"text": result, "visibility": "public"},
        )
        response.raise_for_status()
        note_id = response.json().get("createdNote", {}).get("id")
        logfire.info(f"Posted autonomous note: {note_id}")

    async def _auto_post_loop(self):
        """Periodically post autonomous notes at the configured interval."""
        interval = self._config.auto_post_interval
        jitter = self._config.auto_post_jitter
        assert interval is not None
        logfire.info(f"Starting autonomous post loop (interval: {interval}s, jitter: ±{jitter}s)")
        while True:
            delay = interval + random.randint(-jitter, jitter) if jitter else interval
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=float(max(delay, 1)),
                )
                break  # shutdown event fired
            except asyncio.TimeoutError:
                pass
            if self._shutdown_event.is_set():
                break
            try:
                await self.post_autonomous()
            except Exception:
                logfire.exception("Error during autonomous post")

    async def run(self):
        if self._redis:
            await self._load_last_auto_reply_time()

        async with self._agent:
            try:
                await self._run_loop()
            finally:
                # Handler tasks use the agent, Redis, and memory store. Stop them
                # before ChatAgent.__aexit__ closes those shared resources.
                await self._cancel_background_tasks()

    async def _run_loop(self):
        auto_post_task: Optional[asyncio.Task] = None
        if self._config.auto_post_interval:
            auto_post_task = asyncio.create_task(self._auto_post_loop())

        try:
            async for websocket in connect(f"{self.ws_url}/streaming?i={self.api_key}"):
                shutdown_task = asyncio.create_task(self._shutdown_event.wait())
                message_task = asyncio.create_task(self._handle_messages(websocket))
                try:
                    await websocket.send(
                        MiChannelConnect(body=MiChannelConnectBody(channel="main", id="1")).model_dump_json(
                            exclude_none=True
                        )
                    )
                    if self._config.auto_reply_enabled:
                        await websocket.send(
                            MiChannelConnect(
                                body=MiChannelConnectBody(
                                    channel="globalTimeline",
                                    id="2",
                                    params=MiChannelConnectParams(),
                                )
                            ).model_dump_json(exclude_none=True)
                        )
                        logfire.info("Connected to websocket (main + globalTimeline)")
                    else:
                        logfire.info("Connected to websocket (main)")

                    done, _ = await asyncio.wait(
                        [shutdown_task, message_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if shutdown_task in done:
                        logfire.info("Shutdown requested, closing connection")
                        await websocket.close()
                        return

                    # Message task finished first — surface non-reconnect errors.
                    exc = message_task.exception()
                    if exc is not None and not isinstance(exc, ConnectionClosed):
                        raise exc
                    logfire.warning("WebSocket connection closed, reconnecting...")

                except ConnectionClosed:
                    if self._shutdown_event.is_set():
                        return
                    logfire.warning("WebSocket connection closed, reconnecting...")
                finally:
                    await self._cancel_and_wait(shutdown_task, message_task)
        finally:
            if auto_post_task:
                await self._cancel_and_wait(auto_post_task)

    @staticmethod
    async def _cancel_and_wait(*tasks: asyncio.Task[Any]):
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, ConnectionClosed):
                pass
            except Exception:
                logfire.exception("Background task finished with exception")

    async def _handle_messages(self, websocket: ClientConnection):
        async for message in websocket:
            try:
                msg = MiWebsocketMessage(**json.loads(message))
                if msg.type == "channel" and msg.body and msg.body.body:
                    if msg.body.type == "mention":
                        note = msg.body.body
                        self._spawn_background_task(
                            lambda: self._process_note_event(note, auto_reply=False),
                            note_id=note.id,
                        )
                    elif msg.body.type == "note" and self._config.auto_reply_enabled:
                        note = msg.body.body
                        self._spawn_background_task(
                            lambda: self._process_note_event(note, auto_reply=True),
                            note_id=note.id,
                        )
            except ValidationError as e:
                logfire.debug(f"Validation error: {e}. Message doesn't match expected format, ignoring.")
                pass
            except asyncio.CancelledError:
                logfire.info("Message handler cancelled")
                raise
            except Exception:
                logfire.exception("Error processing message")

    async def _process_note_event(self, note: Note, *, auto_reply: bool) -> None:
        """Deduplicate main/timeline delivery while preserving main mentions."""
        while True:
            async with self._note_processing_lock:
                if note.id in self._recent_note_id_set:
                    return

                pending = self._note_processing.get(note.id)
                if pending is None:
                    pending = asyncio.get_running_loop().create_future()
                    self._note_processing[note.id] = pending
                    owns_processing = True
                else:
                    owns_processing = False

            if not owns_processing:
                if await asyncio.shield(pending):
                    return
                # The prior timeline delivery was not due, or the handler failed.
                # Compete for ownership again so a main mention can still run.
                continue

            handled = False
            try:
                if auto_reply:
                    handled = await self.on_auto_reply(note)
                else:
                    await self.on_mention(note)
                    handled = True
            finally:
                async with self._note_processing_lock:
                    if handled:
                        self._remember_note_id(note.id)
                    if self._note_processing.get(note.id) is pending:
                        del self._note_processing[note.id]
                    if not pending.done():
                        pending.set_result(handled)
            return

    def _remember_note_id(self, note_id: str) -> None:
        if note_id in self._recent_note_id_set:
            return
        if len(self._recent_note_ids) >= _RECENT_NOTE_ID_LIMIT:
            self._recent_note_id_set.discard(self._recent_note_ids.popleft())
        self._recent_note_ids.append(note_id)
        self._recent_note_id_set.add(note_id)

    def _spawn_background_task(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        *,
        note_id: str,
    ) -> Optional[asyncio.Task[Any]]:
        """Create a tracked handler task, dropping work above the configured bound."""
        capacity = self._config.max_concurrent_handlers
        if len(self._background_tasks) >= capacity:
            logfire.warning(
                "Dropping note because handler capacity is full",
                note_id=note_id,
                capacity=capacity,
            )
            return None

        # Invoke the factory only after the capacity check so overload never
        # leaves an unawaited coroutine object behind.
        task = asyncio.create_task(coro_factory())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._task_done_callback)
        return task

    async def _cancel_background_tasks(self) -> None:
        tasks = tuple(self._background_tasks)
        if tasks:
            await self._cancel_and_wait(*tasks)
        self._background_tasks.difference_update(tasks)

    def _task_done_callback(self, task: asyncio.Task[Any]):
        """Handle completed tasks - log exceptions and discard."""
        if task.cancelled():
            return

        try:
            task.result()  # This will raise any exception that occurred
        except Exception:
            logfire.exception("Task failed with exception")

    def shutdown(self):
        self._shutdown_event.set()

    def _format_handle(self, user: User) -> str:
        handle = f"@{user.username}"
        if user.host:
            handle += f"@{user.host}"
        return handle

    def _unique_ordered(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _strip_leading_mentions(self, text: str) -> str:
        return re.sub(r"^(?:@[\w\-]+(?:@[\w\-\.]+)?(?:\s+|$))+", "", text)
