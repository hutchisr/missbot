import json
import random
import re
import time

import asyncio
from typing import Optional
from pydantic import ValidationError
from redis.asyncio import Redis
from websockets import ClientConnection, ConnectionClosed
from websockets.asyncio.client import connect
import httpx
import logfire

from .ai import ChatAgent, _image_urls_for
from .models import (
    Config,
    MiChannelConnect,
    MiChannelConnectBody,
    MiChannelConnectParams,
    MiWebsocketMessage,
    Note,
    User,
)
from .api import api_client


_REDIS_AUTO_REPLY_KEY = "global:last_auto_reply_time"

# Misskey enforces a 3000-character cap on note text.
MAX_NOTE_LENGTH = 3000
_TRUNCATION_SUFFIX = "…"


def _truncate_to_limit(text: str, limit: int = MAX_NOTE_LENGTH) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX


class Bot:
    def __init__(
        self,
        config: Config,
        redis_client: Optional[Redis] = None,
    ):
        self.url = config.url
        self.ws_url = config.ws_url
        self.api_key = config.token
        self.username = config.bot_username
        self.user_id = config.bot_user_id
        self.ws: Optional[ClientConnection] = None

        self._config = config
        self._redis = redis_client
        self._agent = ChatAgent(config, redis_client=redis_client)
        self._shutdown_event = asyncio.Event()
        self._last_auto_reply_time: float = time.time()
        self._next_auto_reply_delay: float = self._compute_auto_reply_delay()
        # Strong refs for fire-and-forget handler tasks; asyncio holds only weakrefs,
        # so without this the tasks can be GC'd mid-run.
        self._background_tasks: set[asyncio.Task] = set()

    @logfire.instrument(extract_args=["note"])
    async def on_mention(self, note: Note):
        if note.user.id == self.user_id:
            logfire.debug("Ignoring own mention")
            return
        if not self._note_has_prompt_content(note):
            logfire.info("Skipping note without text or supported images", note_id=note.id)
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
        result = await self._agent.run(note=note, context=context)
        if result.strip() == "NO_REPLY":
            logfire.info(f"Skipping reply to note {note.id} (NO_REPLY)")
            return
        await self.send_note(result, in_reply_to=note)

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
        if len(text) > MAX_NOTE_LENGTH:
            logfire.warning(
                "Reply exceeds Misskey note length cap; truncating",
                original_length=len(text),
                limit=MAX_NOTE_LENGTH,
            )
            text = _truncate_to_limit(text)

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
        if note and note.visibility:
            return note.visibility
        return "public"

    def _reply_visible_user_ids(self, note: Optional[Note]) -> list[str]:
        if not note or note.visibility != "specified":
            return []

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

        mentions: list[str] = []
        if note.mentions:
            for mention in note.mentions:
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

    async def on_auto_reply(self, note: Note):
        """Automatically reply to a timeline note if enough time has passed."""
        if not self._note_has_prompt_content(note):
            return

        now = time.time()
        elapsed = now - self._last_auto_reply_time
        if elapsed < self._next_auto_reply_delay:
            return

        self._last_auto_reply_time = now
        self._next_auto_reply_delay = self._compute_auto_reply_delay()
        if self._redis:
            await self._save_last_auto_reply_time()

        logfire.info("Auto-reply triggered", note=note)
        await self.on_mention(note)

    @logfire.instrument(extract_args=False)
    async def post_autonomous(self):
        """Generate and post an autonomous note to the timeline."""
        result = await self._agent.run_auto()
        if len(result) > MAX_NOTE_LENGTH:
            logfire.warning(
                "Autonomous post exceeds Misskey note length cap; truncating",
                original_length=len(result),
                limit=MAX_NOTE_LENGTH,
            )
            result = _truncate_to_limit(result)
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
            await self._run_loop()

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
    async def _cancel_and_wait(*tasks: asyncio.Task):
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
                        self._spawn_background_task(self.on_mention(msg.body.body))
                    elif msg.body.type == "note" and self._config.auto_reply_enabled:
                        self._spawn_background_task(self.on_auto_reply(msg.body.body))
            except ValidationError as e:
                logfire.debug(f"Validation error: {e}. Message doesn't match expected format, ignoring.")
                pass
            except asyncio.CancelledError:
                logfire.info("Message handler cancelled")
                raise
            except Exception:
                logfire.exception("Error processing message")

    def _spawn_background_task(self, coro) -> asyncio.Task:
        """Create a tracked background task (holds a strong ref, logs failures)."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._task_done_callback)
        return task

    def _task_done_callback(self, task: asyncio.Task):
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
