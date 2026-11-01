"""Bounded, authenticated Buzz relay IO; nostr-sdk owns event cryptography."""

import asyncio
import base64
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import logfire
from nostr_sdk import Event, EventBuilder, Filter, Keys, Kind, PublicKey, Tag, Timestamp
from websockets.asyncio.client import ClientConnection, connect

IO_TIMEOUT_SECONDS = 15
QUERY_TIMEOUT_SECONDS = 10
MAX_EVENT_BYTES = 512 * 1024
MAX_QUERY_BYTES = 2 * 1024 * 1024
MAX_QUERY_EVENTS = 512
MAX_NOTIFICATIONS = 128
MAX_FUTURE_SECONDS = 300
RECONNECT_MIN_SECONDS = 1
RECONNECT_MAX_SECONDS = 60
LIVE_SUBSCRIPTION = "buzz-messages"
NOTICE_SUBSCRIPTION = "buzz-membership"
MAX_LIVE_CHANNELS = 128
MEMBERSHIP_REFRESH_SECONDS = 60


class _DirectConnect(connect):
    def process_redirect(self, exc: Exception) -> Exception:
        # Never move an owner attestation to another endpoint on a redirect.
        return exc


@dataclass
class _Query:
    subscription: str
    filters: list[Filter]
    done: asyncio.Future[list[dict[str, Any]]]
    events: dict[str, dict[str, Any]] = field(default_factory=dict)
    rows: int = 0
    bytes: int = 0


class BuzzRelay:
    def __init__(self, keys: Keys, community: str, auth: list[str]):
        try:
            url = urlsplit(community)
            loopback = url.hostname in {"localhost", "127.0.0.1", "::1"}
            if (
                url.scheme not in {"https", "wss", "http", "ws"}
                or (url.scheme in {"http", "ws"} and not loopback)
                or not url.hostname
                or url.username is not None
                or url.password is not None
                or url.path not in {"", "/"}
                or url.query
                or url.fragment
                or url.port == 0
            ):
                raise ValueError
        except Exception:
            raise ValueError("Invalid Buzz community origin") from None
        if (
            not isinstance(auth, list)
            or len(auth) != 4
            or not all(isinstance(value, str) for value in auth)
            or auth[0] != "auth"
            or len(auth[1]) != 64
            or len(auth[2]) > 4096
            or len(auth[3]) != 128
        ):
            raise ValueError("Invalid Buzz owner authorization")
        secure = url.scheme in {"https", "wss"}
        self._http_url = urlunsplit(("https" if secure else "http", url.netloc, "/", "", ""))
        self._relay_url = urlunsplit(("wss" if secure else "ws", url.netloc, "/", "", ""))
        self._keys = keys
        self._auth = list(auth)
        self.public_key = keys.public_key().to_hex()
        self.relay_public_key = ""
        self.connected = False
        self.error: str | None = None
        self._ready = asyncio.Event()
        self._io_ready = asyncio.Event()
        self._live_ready = asyncio.Event()
        self._channels_changed = asyncio.Event()
        self._live_channels: set[str] = set()
        self._task: asyncio.Task[Any] | None = None
        self._closed = False
        self._socket: ClientConnection | None = None
        # One outbound operation at a time bounds pending relay state without a queue.
        self._operation = asyncio.Lock()
        self._query: _Query | None = None
        self._ack: tuple[str, asyncio.Future[bool]] | None = None

    def sign(self, content: str, tags: list[list[str]], kind: int = 9, created_at: int | None = None) -> dict[str, Any]:
        """Sign once; persist this result and reuse it verbatim for publication retries."""
        try:
            if type(kind) is not int or not 0 <= kind <= 65535 or not isinstance(content, str):
                raise ValueError
            if not isinstance(tags, list) or len(tags) > 255:
                raise ValueError
            if any(not isinstance(tag, list) or not tag or not all(isinstance(v, str) for v in tag) for tag in tags):
                raise ValueError
            # The operator-provided owner proof is authoritative, never a caller tag.
            signed_tags = [tag for tag in tags if tag[0] != "auth"] + [self._auth]
            builder = EventBuilder(Kind(kind), content).tags([Tag.parse(tag) for tag in signed_tags])
            if created_at is not None:
                if type(created_at) is not int or not 0 <= created_at <= time.time() + MAX_FUTURE_SECONDS:
                    raise ValueError
                builder = builder.custom_created_at(Timestamp.from_secs(created_at))
            raw = builder.finalize(self._keys).as_json()
            if len(raw.encode()) > MAX_EVENT_BYTES:
                raise ValueError
            return json.loads(raw)
        except Exception:
            raise ValueError("Invalid Buzz publication event") from None

    @staticmethod
    def _verified(raw: Any) -> tuple[Event, dict[str, Any]]:
        try:
            if not isinstance(raw, dict) or len(raw.get("tags", [])) > 256:
                raise ValueError
            encoded = json.dumps(raw, separators=(",", ":"), allow_nan=False)
            if len(encoded.encode()) > MAX_EVENT_BYTES:
                raise ValueError
            event = Event.from_json(encoded)
            if not event.verify() or event.created_at().as_secs() > time.time() + MAX_FUTURE_SECONDS:
                raise ValueError
            return event, json.loads(event.as_json())
        except Exception:
            raise RuntimeError("Invalid Buzz relay event") from None

    async def _discover(self) -> None:
        proof = self.sign("", [["u", self._http_url], ["method", "GET"], ["nonce", uuid.uuid4().hex]], 27235)
        authorization = base64.b64encode(json.dumps(proof, separators=(",", ":")).encode()).decode()
        try:
            async with (
                asyncio.timeout(IO_TIMEOUT_SECONDS),
                httpx.AsyncClient(timeout=IO_TIMEOUT_SECONDS, follow_redirects=False, trust_env=False) as client,
            ):
                async with client.stream(
                    "GET",
                    self._http_url,
                    headers={"Accept": "application/nostr+json", "Authorization": f"Nostr {authorization}"},
                ) as response:
                    if response.status_code != 200:
                        raise ValueError
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > 64 * 1024:
                            raise ValueError
                info = json.loads(body)
                authority = info["self"]
                if not isinstance(authority, str) or len(authority) != 64:
                    raise ValueError
                self.relay_public_key = PublicKey.parse(authority).to_hex()
        except Exception:
            raise RuntimeError("Buzz relay discovery failed") from None

    @staticmethod
    async def _send(socket: ClientConnection, message: list[Any]) -> None:
        raw = json.dumps(message, separators=(",", ":"), allow_nan=False)
        if len(raw.encode()) > MAX_EVENT_BYTES + 4096:
            raise RuntimeError("Buzz relay request exceeds size limit")
        async with asyncio.timeout(IO_TIMEOUT_SECONDS):
            await socket.send(raw)

    @staticmethod
    async def _receive(socket: ClientConnection) -> list[Any]:
        try:
            raw = await socket.recv()
            if not isinstance(raw, str):
                raise TypeError
            message = json.loads(raw)
            if not isinstance(message, list) or len(message) < 2 or not isinstance(message[1], str):
                raise ValueError
            return message
        except Exception:
            raise RuntimeError("Buzz relay connection failed") from None

    async def _authenticate(self, socket: ClientConnection) -> None:
        try:
            async with asyncio.timeout(IO_TIMEOUT_SECONDS):
                auth_id = None
                for _ in range(MAX_NOTIFICATIONS):
                    message = await self._receive(socket)
                    if message[0] == "AUTH" and len(message) == 2 and auth_id is None:
                        if not message[1] or len(message[1]) > 4096:
                            raise ValueError
                        event = self.sign("", [["relay", self._relay_url], ["challenge", message[1]]], 22242)
                        auth_id = event["id"]
                        await self._send(socket, ["AUTH", event])
                    elif message[0] == "OK" and len(message) == 4 and message[1] == auth_id:
                        if message[2] is not True:
                            raise ValueError
                        return
                    elif message[0] != "NOTICE":
                        raise ValueError
                raise ValueError
        except Exception:
            raise RuntimeError("Buzz relay authentication failed") from None

    def _disconnected(self) -> None:
        self.connected = False
        self._ready.clear()
        self._io_ready.clear()
        self._live_ready.clear()
        self._live_channels.clear()
        self._socket = None
        if self._query is not None and not self._query.done.done():
            self._query.done.set_exception(RuntimeError("Buzz relay query interrupted"))
        if self._ack is not None and not self._ack[1].done():
            self._ack[1].set_exception(RuntimeError("Buzz relay publication interrupted"))

    async def run(self, on_event: Callable[[dict[str, Any]], Awaitable[None]], since: int) -> None:
        """Reconnect until cancelled; callbacks must only enqueue bounded work."""
        if self._task is not None or self._closed:
            raise RuntimeError("Buzz relay is already running or closed")
        if type(since) is not int or since < 0:
            raise ValueError("Invalid Buzz subscription timestamp")
        self._task = asyncio.current_task()
        delay = RECONNECT_MIN_SECONDS
        try:
            while not self._closed:
                started = time.monotonic()
                try:
                    await self._discover()
                    async with _DirectConnect(
                        self._relay_url,
                        open_timeout=IO_TIMEOUT_SECONDS,
                        close_timeout=1,
                        max_size=MAX_EVENT_BYTES + 4096,
                        max_queue=MAX_NOTIFICATIONS,
                        compression=None,
                        proxy=None,
                    ) as socket:
                        await self._authenticate(socket)
                        self._socket = socket
                        self._io_ready.set()
                        await self._send(
                            socket,
                            [
                                "REQ",
                                NOTICE_SUBSCRIPTION,
                                {"kinds": [44100, 44101], "#p": [self.public_key], "limit": 0},
                                {"kinds": [44100, 44101], "#p": [self.public_key], "#t": ["system"], "limit": 0},
                            ],
                        )
                        reader = asyncio.create_task(self._read_messages(socket, on_event, since))
                        membership = asyncio.create_task(self._refresh_channels(socket, since))
                        try:
                            done, _ = await asyncio.wait({reader, membership}, return_when=asyncio.FIRST_COMPLETED)
                            for task in done:
                                task.result()
                            raise RuntimeError("Buzz relay connection closed")
                        finally:
                            reader.cancel()
                            membership.cancel()
                            await asyncio.gather(reader, membership, return_exceptions=True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Peer messages, exception strings and auth tags never enter logs.
                    self.error = "Buzz relay connection unavailable; retrying"
                    logfire.warning("Buzz relay connection unavailable; retrying")
                finally:
                    self._disconnected()
                if time.monotonic() - started >= RECONNECT_MAX_SECONDS:
                    delay = RECONNECT_MIN_SECONDS
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_SECONDS)
        finally:
            self._disconnected()
            self._task = None

    async def _read_messages(self, socket: ClientConnection, on_event, since: int) -> None:
        while True:
            await self._dispatch(await self._receive(socket), on_event, since)

    async def _refresh_channels(self, socket: ClientConnection, since: int) -> None:
        """Follow current membership, rather than treating an unscoped snapshot as a live feed."""
        while True:
            self._channels_changed.clear()
            rosters = await self.query([{"kinds": [39002], "#p": [self.public_key], "limit": MAX_QUERY_EVENTS}])
            channels = set()
            for roster in rosters:
                ids = [tag[1] for tag in roster["tags"] if len(tag) >= 2 and tag[0] == "d"]
                if roster["pubkey"] == self.relay_public_key and len(ids) == 1:
                    channels.add(ids[0])
            if len(channels) > MAX_LIVE_CHANNELS:
                raise RuntimeError("Buzz live channel capacity exceeded")
            if channels != self._live_channels:
                self.connected = False
                self._ready.clear()
                self._live_ready.clear()
                if self._live_channels:
                    await self._send(socket, ["CLOSE", LIVE_SUBSCRIPTION])
                self._live_channels = channels
                if channels:
                    await self._send(
                        socket,
                        [
                            "REQ",
                            LIVE_SUBSCRIPTION,
                            {"kinds": [9], "#h": sorted(channels), "since": since, "limit": MAX_QUERY_EVENTS},
                        ],
                    )
                    async with asyncio.timeout(QUERY_TIMEOUT_SECONDS):
                        await self._live_ready.wait()
            self.connected = True
            self.error = None
            self._ready.set()
            try:
                async with asyncio.timeout(MEMBERSHIP_REFRESH_SECONDS):
                    await self._channels_changed.wait()
            except TimeoutError:
                pass

    async def _dispatch(
        self, message: list[Any], on_event: Callable[[dict[str, Any]], Awaitable[None]], since: int
    ) -> None:
        command, subscription = message[:2]
        query = self._query
        if command == "AUTH":
            # A new challenge invalidates all pending work; reconnect and replay auth.
            raise RuntimeError("Buzz relay requested reauthentication")
        if command == "OK":
            if self._ack is not None and subscription == self._ack[0] and not self._ack[1].done():
                if len(message) != 4 or type(message[2]) is not bool:
                    raise RuntimeError("Invalid Buzz relay acknowledgement")
                self._ack[1].set_result(message[2])
            return
        if query is not None and subscription == query.subscription and not query.done.done():
            if command == "EOSE" and len(message) == 2:
                query.done.set_result(list(query.events.values()))
            elif command == "EVENT" and len(message) == 3:
                event, raw = self._verified(message[2])
                if not any(f.match_event(event) for f in query.filters):
                    raise RuntimeError("Buzz relay query returned an unrelated event")
                query.rows += 1
                query.bytes += len(json.dumps(raw).encode())
                if query.rows > MAX_QUERY_EVENTS or query.bytes > MAX_QUERY_BYTES:
                    raise RuntimeError("Buzz relay query exceeds result limits")
                query.events[raw["id"]] = raw
            else:
                raise RuntimeError("Buzz relay query did not complete")
        elif subscription == NOTICE_SUBSCRIPTION:
            if command == "EVENT" and len(message) == 3:
                try:
                    _, notice = self._verified(message[2])
                except RuntimeError:
                    return
                if (
                    notice["kind"] in (44100, 44101)
                    and notice["pubkey"] == self.relay_public_key
                    and any(tag[:2] == ["p", self.public_key] for tag in notice["tags"])
                ):
                    self._channels_changed.set()
            elif command != "EOSE" or len(message) != 2:
                raise RuntimeError("Buzz membership subscription closed")
        elif subscription == LIVE_SUBSCRIPTION:
            if command == "EVENT" and len(message) == 3:
                try:
                    _, raw = self._verified(message[2])
                except RuntimeError:
                    logfire.warning("Discarded invalid Buzz relay event")
                    return
                channels = [tag[1] for tag in raw["tags"] if len(tag) >= 2 and tag[0] == "h"]
                if (
                    raw["kind"] == 9
                    and raw["created_at"] >= since
                    and len(channels) == 1
                    and channels[0] in self._live_channels
                ):
                    async with asyncio.timeout(IO_TIMEOUT_SECONDS):
                        await on_event(raw)
            elif command == "EOSE" and len(message) == 2:
                self._live_ready.set()
            else:
                raise RuntimeError("Buzz relay subscription closed")

    async def wait_ready(self, timeout: float = 15) -> None:
        """Wait for authenticated discovery and subscription, with a bounded deadline."""
        if self._closed:
            raise RuntimeError("Buzz relay is closed")
        try:
            async with asyncio.timeout(timeout):
                await self._ready.wait()
                if not self.connected:
                    raise RuntimeError
        except Exception:
            raise RuntimeError("Buzz relay did not become ready") from None

    async def _connected_socket(self) -> ClientConnection:
        if self._closed or self._task is None:
            raise RuntimeError("Buzz relay is not running")
        await self._io_ready.wait()
        if self._socket is None:
            raise RuntimeError("Buzz relay is not connected")
        return self._socket

    async def query(self, filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return only a complete bounded EOSE result; failures never mean absence."""
        try:
            if not isinstance(filters, list) or not 1 <= len(filters) <= 8:
                raise ValueError
            for value in filters:
                if not isinstance(value, dict) or any(
                    key not in {"ids", "authors", "kinds", "since", "until", "limit"}
                    and not (isinstance(key, str) and len(key) == 2 and key.startswith("#"))
                    for key in value
                ):
                    raise ValueError
                if "limit" in value and (
                    type(value["limit"]) is not int or not 1 <= value["limit"] <= MAX_QUERY_EVENTS
                ):
                    raise ValueError
            filters = [dict(value, limit=value.get("limit", MAX_QUERY_EVENTS)) for value in filters]
            parsed = [Filter.from_json(json.dumps(value, allow_nan=False)) for value in filters]
        except Exception:
            raise ValueError("Invalid bounded Buzz query") from None
        try:
            async with asyncio.timeout(QUERY_TIMEOUT_SECONDS), self._operation:
                socket = await self._connected_socket()
                query = _Query(uuid.uuid4().hex, parsed, asyncio.get_running_loop().create_future())
                self._query = query
                try:
                    await self._send(socket, ["REQ", query.subscription, *filters])
                    return await query.done
                finally:
                    self._query = None
                    if not query.done.done():
                        query.done.cancel()
                    else:
                        # Consume a disconnect error even if send failed before awaiting it.
                        with suppress(asyncio.CancelledError, Exception):
                            query.done.exception()
                    with suppress(Exception):
                        async with asyncio.timeout(1):
                            await self._send(socket, ["CLOSE", query.subscription])
        except Exception:
            raise RuntimeError("Could not completely query Buzz relay") from None

    async def publish(
        self, content: str, tags: list[list[str]], kind: int = 9, *, created_at: int | None = None
    ) -> str:
        return await self.publish_event(self.sign(content, tags, kind, created_at))

    async def publish_event(self, event: dict[str, Any]) -> str:
        """Require positive OK, never rebuild a signed event during a retry."""
        _, raw = self._verified(event)
        if raw["pubkey"] != self.public_key or [tag for tag in raw["tags"] if tag[0] == "auth"] != [self._auth]:
            raise ValueError("Buzz publication identity or authorization does not match")
        if raw["kind"] == 0:
            return await self._publish_profile(raw)
        try:
            async with asyncio.timeout(IO_TIMEOUT_SECONDS), self._operation:
                socket = await self._connected_socket()
                ack = asyncio.get_running_loop().create_future()
                self._ack = (raw["id"], ack)
                try:
                    await self._send(socket, ["EVENT", raw])
                    if not await ack:
                        raise RuntimeError("Buzz relay rejected publication")
                    return raw["id"]
                finally:
                    self._ack = None
                    if not ack.done():
                        ack.cancel()
                    else:
                        with suppress(asyncio.CancelledError, Exception):
                            ack.exception()
        except Exception:
            raise RuntimeError("Buzz relay did not acknowledge publication") from None

    async def _publish_profile(self, event: dict[str, Any]) -> str:
        """Buzz setup profiles use the native signed HTTP /events contract, not WS EVENT."""
        url = self._http_url + "events"
        body = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode()
        tags = [
            ["u", url],
            ["method", "POST"],
            ["payload", hashlib.sha256(body).hexdigest()],
            ["nonce", uuid.uuid4().hex],
        ]
        proof = EventBuilder(Kind(27235), "").tags([Tag.parse(tag) for tag in tags]).finalize(self._keys).as_json()
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Nostr " + base64.b64encode(proof.encode()).decode(),
            "x-auth-tag": json.dumps(self._auth, separators=(",", ":")),
        }
        try:
            async with (
                asyncio.timeout(IO_TIMEOUT_SECONDS),
                self._operation,
                httpx.AsyncClient(timeout=IO_TIMEOUT_SECONDS, follow_redirects=False, trust_env=False) as client,
                client.stream("POST", url, headers=headers, content=body) as response,
            ):
                payload = bytearray()
                async for chunk in response.aiter_bytes():
                    payload.extend(chunk)
                    if len(payload) > 65536:
                        raise ValueError("Oversized profile receipt")
                receipt = json.loads(payload)
                if (
                    not response.is_success
                    or not isinstance(receipt, dict)
                    or receipt.get("accepted") is not True
                    or receipt.get("event_id") != event["id"]
                ):
                    raise ValueError("Profile was not accepted")
            return event["id"]
        except Exception:
            raise RuntimeError("Buzz community did not accept the signed HTTP profile publication") from None

    async def aclose(self) -> None:
        self._closed = True
        task = self._task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._disconnected()
