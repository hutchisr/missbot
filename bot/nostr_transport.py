"""NIP-01 relay IO; nostr-sdk owns cryptography, deduplication and reconnects."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from datetime import timedelta
from typing import Any

import logfire
from nostr_sdk import (
    AckPolicy,
    Client,
    ClientBuilder,
    ClientNotification,
    Event,
    EventBuilder,
    EventId,
    Filter,
    Keys,
    Kind,
    RelayLimits,
    RelayUrl,
    ReqTarget,
    Tag,
    Timestamp,
)
from websockets.asyncio.client import connect

IO_TIMEOUT_SECONDS = 15
FETCH_TIMEOUT_SECONDS = 5
MAX_EVENT_BYTES = 512 * 1024
MAX_NOTIFICATIONS = 128
MAX_FUTURE_SECONDS = 300
MENTIONS_SUBSCRIPTION = "missbot-mentions"


class NostrTransport:
    def __init__(self, secret_key: str, relays: list[str]):
        try:
            self._keys = Keys.parse(secret_key.strip())
        except Exception:
            # SDK error strings aren't a safe place to expose private-key input.
            raise ValueError("Invalid Nostr private key; expected hex or nsec") from None
        self.public_key = self._keys.public_key().to_hex()
        if not relays or len(relays) > 16:
            raise ValueError("Configure between 1 and 16 Nostr relays")
        try:
            self._relays = [RelayUrl.parse(url) for url in dict.fromkeys(relays)]
        except Exception:
            raise ValueError("Invalid Nostr relay URL") from None
        self._client: Client | None = None
        self._ready = asyncio.Event()
        self._running = False
        self._profile_lock = asyncio.Lock()

    async def run(self, handler: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        """Receive fresh public mentions until cancelled; handler must return quickly."""
        if self._running:
            raise RuntimeError("Nostr transport is already running")
        self._running = True
        started_at = int(time.time())
        # The SDK's bounded broadcast channel drops oldest notifications on overload.
        # Awaiting the handler here never creates an unbounded Python task backlog.
        client = (
            ClientBuilder()
            .notification_channel_size(MAX_NOTIFICATIONS)
            .max_relays(16)
            .connect_timeout(timedelta(seconds=IO_TIMEOUT_SECONDS))
            .relay_limits(
                RelayLimits()
                .message_max_size(MAX_EVENT_BYTES + 4096)
                .event_max_size(MAX_EVENT_BYTES)
                .event_max_num_tags(256)
            )
            .verify_subscriptions(True)
            .build()
        )
        self._client = client
        try:
            for relay in self._relays:
                await client.add_relay(relay)
            notifications = client.notifications()
            await client.connect()
            mention_filter = (
                Filter()
                .kinds([Kind(1), Kind(1111)])
                .pubkey(self._keys.public_key())
                .since(Timestamp.from_secs(started_at))
                .limit(MAX_NOTIFICATIONS)
            )
            await client.subscribe(ReqTarget.auto([mention_filter]), id=MENTIONS_SUBSCRIPTION)
            self._ready.set()
            while (notification := await notifications.next()) is not None:
                if not isinstance(notification, ClientNotification.NEW_EVENT):
                    continue
                if notification.subscription_id != MENTIONS_SUBSCRIPTION:
                    continue
                event = self._verified_event(notification.event)
                if event is None or event["pubkey"] == self.public_key:
                    continue
                if event["kind"] not in (1, 1111) or event["created_at"] < started_at:
                    continue
                if not any(tag[:2] == ["p", self.public_key] for tag in event["tags"]):
                    continue
                try:
                    await handler(event)
                except Exception:
                    logfire.exception("Nostr mention handler failed")
        finally:
            self._ready.clear()
            self._client = None
            try:
                await client.shutdown()
            finally:
                self._running = False

    @staticmethod
    def _verified_event(event: Event) -> dict[str, Any] | None:
        """Check even SDK-returned events before passing them across the boundary."""
        if not event.verify() or event.created_at().as_secs() > time.time() + MAX_FUTURE_SECONDS:
            return None
        raw = event.as_json()
        if len(raw.encode("utf-8")) > MAX_EVENT_BYTES:
            return None
        return json.loads(raw)

    async def _connected_client(self) -> Client:
        await self._ready.wait()
        if self._client is None:
            raise RuntimeError("Nostr transport is not running")
        return self._client

    async def publish(
        self, content: str, tags: list[list[str]], kind: int = 1, *, created_at: int | None = None
    ) -> str:
        """Return the signed event ID only after at least one positive relay OK."""
        builder = EventBuilder(Kind(kind), content).tags([Tag.parse(tag) for tag in tags])
        if created_at is not None:
            if type(created_at) is not int or created_at < 0 or created_at > time.time() + MAX_FUTURE_SECONDS:
                raise ValueError("Invalid Nostr publication timestamp")
            builder = builder.custom_created_at(Timestamp.from_secs(created_at))
        event = builder.finalize(self._keys)
        if len(event.as_json().encode("utf-8")) > MAX_EVENT_BYTES:
            raise ValueError("Nostr event exceeds the transport size limit")
        try:
            async with asyncio.timeout(IO_TIMEOUT_SECONDS):
                client = await self._connected_client()
                result = await client.send_event(
                    event,
                    ack_policy=AckPolicy.all(),
                    ok_timeout=timedelta(seconds=IO_TIMEOUT_SECONDS - 1),
                )
                if not result.success:
                    raise RuntimeError("No Nostr relay accepted the event")
                return event.id().to_hex()
        except Exception:
            # Don't reflect relay-controlled ACK/error messages into application logs.
            raise RuntimeError("No Nostr relay accepted the event before the publication deadline") from None

    async def fetch_event(self, event_id: str) -> dict[str, Any] | None:
        """Fetch a verified ancestor by exact ID, with bounded time and results."""
        if len(event_id) != 64 or any(char not in "0123456789abcdef" for char in event_id):
            return None
        try:
            async with asyncio.timeout(FETCH_TIMEOUT_SECONDS):
                client = await self._connected_client()
                events = await client.fetch_events(
                    ReqTarget.auto(
                        [Filter().id(EventId.parse(event_id)).kinds([Kind(1), Kind(1111), Kind(1068)]).limit(1)]
                    ),
                    timeout=timedelta(seconds=FETCH_TIMEOUT_SECONDS - 1),
                    max_events=1,
                )
                for event in events:
                    verified = self._verified_event(event)
                    if verified is not None and verified["id"] == event_id and verified["kind"] in (1, 1111, 1068):
                        return verified
        except Exception:
            logfire.warning("Could not fetch Nostr thread ancestor")
        return None

    async def fetch_profile(self, public_key: str) -> dict[str, Any] | None:
        """Fetch the latest verified kind 0, conservatively requiring EOSE from every relay.

        None means all configured relays confirmed absence; incomplete or malformed
        replies raise instead of permitting an update that could destroy metadata.
        """
        if len(public_key) != 64 or any(char not in "0123456789abcdef" for char in public_key):
            raise ValueError("Invalid Nostr profile public key")
        # SDK fetch_events also ends on timeout, and its notification channel can drop
        # events before EOSE. Dedicated sockets distinguish absence from incomplete IO.
        subscription = "missbot-profile"
        request = json.dumps(["REQ", subscription, {"authors": [public_key], "kinds": [0], "limit": 1}])
        latest: dict[str, Any] | None = None
        try:
            async with asyncio.timeout(FETCH_TIMEOUT_SECONDS), self._profile_lock:
                for relay in self._relays:
                    async with connect(
                        str(relay),
                        open_timeout=FETCH_TIMEOUT_SECONDS,
                        close_timeout=1,
                        max_size=MAX_EVENT_BYTES + 4096,
                        max_queue=1,
                        compression=None,
                        proxy=None,
                    ) as websocket:
                        try:
                            await websocket.send(request)
                            for _ in range(MAX_NOTIFICATIONS):
                                message = json.loads(await websocket.recv())
                                if not isinstance(message, list) or len(message) < 2:
                                    raise ValueError("Malformed Nostr profile response")
                                if message[0] == "NOTICE" and len(message) == 2 and isinstance(message[1], str):
                                    continue
                                if message == ["EOSE", subscription]:
                                    break
                                if len(message) != 3 or message[:2] != ["EVENT", subscription]:
                                    raise ValueError("Incomplete Nostr profile response")
                                event = self._verified_event(Event.from_json(json.dumps(message[2])))
                                if (
                                    event is None
                                    or event["pubkey"] != public_key
                                    or event["kind"] != 0
                                    or len(event["tags"]) > 256
                                ):
                                    raise ValueError("Invalid Nostr profile event")
                                metadata = json.loads(event["content"])
                                if not isinstance(metadata, dict):
                                    raise TypeError("Malformed Nostr profile metadata")
                                # Python accepts NaN/Infinity on input; Nostr metadata is JSON.
                                json.dumps(metadata, allow_nan=False)
                                if latest is None or (-event["created_at"], event["id"]) < (
                                    -latest["created_at"],
                                    latest["id"],
                                ):
                                    latest = event
                            else:
                                raise ValueError("Nostr profile response exceeds the message limit")
                        finally:
                            with suppress(Exception):
                                async with asyncio.timeout(1):
                                    await websocket.send(json.dumps(["CLOSE", subscription]))
        except Exception:
            # Never let an incomplete lookup become permission to overwrite metadata.
            raise RuntimeError("Could not completely fetch Nostr profile from configured relays") from None
        return latest
