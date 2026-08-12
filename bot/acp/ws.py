"""Serve the ACP agent over WebSocket, wire-compatible with `acpremote mirror`.

ACP's own HTTP transport is still a draft RFD and the Python SDK ships stdio only, so
remote clients reach a hosted agent through a bridge. [acpremote](https://github.com/vcoderun/acpkit)
is that bridge: on the consumer side, ``acpremote mirror ws://host/acp/ws`` turns a
remote WebSocket endpoint back into a local stdio ACP command — exactly what
``BUZZ_ACP_AGENT_COMMAND`` (or a Zed agent server entry) needs.

This module implements the *server* half of that contract directly, so missbot has no
acpremote dependency — which also keeps `websockets` at the version the Misskey client
uses, rather than the `<16.0` pin acpremote carries.

The contract, taken from acpremote's ``stream.py``/``client.py``:

- **One WebSocket text frame carries exactly one ACP JSON-RPC message**, with no
  trailing newline. The SDK's `Connection` speaks newline-delimited JSON over asyncio
  streams, so the bridge here strips the newline on send and re-adds it on receive.
- **Binary frames are an error.**
- Optional bearer auth: ``Authorization: Bearer <token>``.
- ``GET <mount>`` returns transport metadata; ``GET /healthz`` is a health probe.
  A client treats a non-200 metadata response as "no metadata" rather than failing,
  but serving it advertises auth mode and message limits.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, NoReturn

import logfire
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from ..provider import PROJECT_VERSION

# acpremote's defaults; matching them means `acpremote mirror ws://host:8080/acp/ws`
# works with no extra flags.
DEFAULT_MOUNT_PATH = "/acp"
DEFAULT_HEALTH_PATH = "/healthz"
# acpremote's DEFAULT_MAX_MESSAGE_SIZE / DEFAULT_MAX_QUEUE.
DEFAULT_MAX_SIZE = 16 * 1024 * 1024
DEFAULT_MAX_QUEUE = 32
DEFAULT_OUTPUT_MAX_SIZE = 1024 * 1024
DEFAULT_OUTPUT_MAX_QUEUE = 8
DEFAULT_MAX_CONNECTIONS = 8

# Transport contract version we implement (acpremote's TransportMetadata).
_TRANSPORT_KIND = "websocket"
_TRANSPORT_VERSION = 1


@dataclass(frozen=True)
class ServerPaths:
    """Routes derived from the mount path, mirroring acpremote's `build_server_paths`."""

    metadata_path: str
    websocket_path: str
    health_path: str = DEFAULT_HEALTH_PATH

    @classmethod
    def from_mount(cls, mount_path: str) -> ServerPaths:
        normalized = mount_path.strip()
        if not normalized:
            raise ValueError("mount_path must not be empty")
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        normalized = normalized.rstrip("/")
        if normalized == DEFAULT_HEALTH_PATH:
            raise ValueError(f"mount_path must not overlap the health endpoint `{DEFAULT_HEALTH_PATH}`")
        websocket_path = f"{normalized}/ws" if normalized else "/ws"
        return cls(metadata_path=normalized or "/", websocket_path=websocket_path)


def build_metadata(*, paths: ServerPaths, auth_required: bool) -> dict[str, Any]:
    """The JSON body served at the metadata route.

    Field names and types are dictated by acpremote's `ServerMetadata`; a client that
    cannot parse them simply proceeds without metadata.
    """
    return {
        "transport_kind": _TRANSPORT_KIND,
        "transport_version": _TRANSPORT_VERSION,
        "package_version": PROJECT_VERSION,
        "auth_required": auth_required,
        "supported_auth_modes": ["bearer"] if auth_required else [],
        "max_size": DEFAULT_MAX_SIZE,
        "max_queue": DEFAULT_MAX_QUEUE,
        "compression": None,
        "health_path": paths.health_path,
        "metadata_path": paths.metadata_path,
        "websocket_path": paths.websocket_path,
        "supported_agent_families": ["missbot"],
        "remote_cwd": None,
    }


def is_authorized(headers: Any, token: str | None) -> bool:
    """Check bearer authorization, failing closed for invalid configured tokens."""
    if token is None:
        return True
    normalized = token.strip()
    if not normalized:
        return False
    return headers.get("Authorization") == f"Bearer {normalized}"


class _FrameWriterTransport(asyncio.Transport):
    """Turns newline-delimited SDK writes into bounded WebSocket frames."""

    def __init__(self, websocket: Any, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._websocket = websocket
        self._loop = loop
        self._buffer = bytearray()
        # One reserved slot ensures close() can always enqueue its sentinel.
        self._pending: asyncio.Queue[str | None] = asyncio.Queue(maxsize=DEFAULT_OUTPUT_MAX_QUEUE + 1)
        self._writable = asyncio.Event()
        self._writable.set()
        self._closed = False
        self._error: ConnectionResetError | None = None
        self._protocol: asyncio.BaseProtocol | None = None
        self._websocket_close: asyncio.Task[Any] | None = None
        self._sender = loop.create_task(self._sender_loop())

    def set_protocol(self, protocol: asyncio.BaseProtocol) -> None:
        self._protocol = protocol

    def get_protocol(self) -> asyncio.BaseProtocol:
        if self._protocol is None:
            raise RuntimeError("writer protocol has not been set")
        return self._protocol

    def write(self, data: Any) -> None:
        if self._closed:
            raise self._error or ConnectionResetError("transport is closing")
        self._buffer.extend(bytes(data))
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                if len(self._buffer) > DEFAULT_OUTPUT_MAX_SIZE:
                    self._fail("ACP output frame exceeds the size limit")
                return
            if newline > DEFAULT_OUTPUT_MAX_SIZE:
                self._fail("ACP output frame exceeds the size limit")
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if self._pending.qsize() >= DEFAULT_OUTPUT_MAX_QUEUE:
                self._fail("ACP output backlog exceeded")
            # Strip the framing newline: the frame boundary *is* the message boundary.
            self._pending.put_nowait(line.decode("utf-8"))
            if self._pending.qsize() >= DEFAULT_OUTPUT_MAX_QUEUE:
                self._writable.clear()

    def writelines(self, list_of_data: Iterable[Any]) -> None:
        for line in list_of_data:
            self.write(line)

    def can_write_eof(self) -> bool:
        return False

    def write_eof(self) -> None:
        self.close()

    def is_closing(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writable.set()
        self._pending.put_nowait(None)
        if self._protocol is not None:
            self._protocol.connection_lost(self._error)

    def abort(self) -> None:
        with suppress(ConnectionResetError):
            self._fail("ACP output transport aborted")

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default

    async def wait_writable(self) -> None:
        await self._writable.wait()
        if self._error is not None:
            raise self._error
        if self._closed:
            raise ConnectionResetError("transport is closing")

    def _fail(self, reason: str) -> NoReturn:
        self._error = ConnectionResetError(reason)
        self._buffer.clear()
        self.close()
        if self._websocket_close is None:
            self._websocket_close = self._loop.create_task(self._websocket.close(code=1013, reason=reason))
        raise self._error

    async def _sender_loop(self) -> None:
        while True:
            payload = await self._pending.get()
            if self._pending.qsize() < DEFAULT_OUTPUT_MAX_QUEUE:
                self._writable.set()
            if payload is None:
                return
            try:
                await self._websocket.send(payload)
            except (ConnectionClosed, RuntimeError):
                return

    async def aclose(self) -> None:
        self.close()
        try:
            await asyncio.wait_for(self._sender, timeout=5)
        except TimeoutError:
            self._sender.cancel()
            with suppress(asyncio.CancelledError):
                await self._sender
        if self._websocket_close is not None:
            with suppress(ConnectionClosed, RuntimeError):
                await self._websocket_close


class _WriterProtocol(asyncio.Protocol):
    """Minimal protocol providing real drain backpressure over the frame transport."""

    def __init__(self, loop: asyncio.AbstractEventLoop, transport: _FrameWriterTransport):
        self._closed: asyncio.Future[None] = loop.create_future()
        self._transport = transport

    async def _drain_helper(self) -> None:
        await self._transport.wait_writable()

    def _get_close_waiter(self, stream: Any) -> Any:
        return self._closed

    def connection_lost(self, exc: Exception | None) -> None:
        if not self._closed.done():
            self._closed.set_result(None)


async def _pump_frames(websocket: Any, reader: asyncio.StreamReader) -> None:
    """Feed each inbound text frame to the reader as one newline-terminated message."""
    try:
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                raise TypeError("binary WebSocket frames are not supported")
            reader.feed_data(message.encode("utf-8") + b"\n")
    except ConnectionClosed:
        reader.feed_eof()
    except asyncio.CancelledError:
        reader.feed_eof()
        raise
    except Exception as exc:
        reader.set_exception(exc)
        reader.feed_eof()
        raise


def open_stream_bridge(
    websocket: Any,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, asyncio.Task, _FrameWriterTransport]:
    """Adapt a WebSocket into the reader/writer pair the ACP SDK's `Connection` expects."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=DEFAULT_MAX_SIZE)
    transport = _FrameWriterTransport(websocket, loop)
    protocol = _WriterProtocol(loop, transport)
    transport.set_protocol(protocol)
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    pump = loop.create_task(_pump_frames(websocket, reader))
    return reader, writer, pump, transport


class _ConnectionLimiter:
    """Single-event-loop admission counter for live WebSocket connections."""

    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError("max_connections must be positive")
        self._limit = limit
        self._active = 0

    def try_acquire(self) -> bool:
        if self._active >= self._limit:
            return False
        self._active += 1
        return True

    def release(self) -> None:
        if self._active <= 0:
            raise RuntimeError("ACP connection limit released without an active connection")
        self._active -= 1


async def serve_acp_websocket(
    agent_factory: Callable[[], Any],
    *,
    host: str,
    port: int,
    mount_path: str = DEFAULT_MOUNT_PATH,
    bearer_token: str | None = None,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
) -> None:
    """Serve ACP over WebSocket until cancelled.

    ``agent_factory`` is called once per connection: `MissbotAgent` holds the client
    handle it pushes `session/update` through, so connections must not share one.
    """
    # Imported here so this module stays importable without the SDK's agent connection
    # machinery being pulled in at import time by the stdio path.
    from acp.agent.connection import AgentSideConnection

    paths = ServerPaths.from_mount(mount_path)
    if bearer_token is not None:
        bearer_token = bearer_token.strip()
        if not bearer_token:
            raise ValueError("bearer_token must be nonempty when configured")
    auth_required = bearer_token is not None
    metadata_text = json.dumps(build_metadata(paths=paths, auth_required=auth_required))
    connection_limiter = _ConnectionLimiter(max_connections)

    def process_request(connection: ServerConnection, request: Any):
        """Answer the HTTP routes; bearer auth gates the WebSocket upgrade only.

        Health and metadata stay open, matching acpremote's server. That is not an
        oversight there: `auth_required` in the metadata body is how a client
        *discovers* it needs a token, so gating it behind that token is circular — and
        an authenticated health route breaks ordinary Kubernetes probes.
        """
        path = request.path.split("?", 1)[0]
        if path == paths.health_path:
            return connection.respond(HTTPStatus.OK, "ok\n")
        if path == paths.metadata_path:
            # Body must go through respond() so Content-Length matches; only the
            # content type needs correcting (Headers appends, so clear it first).
            response = connection.respond(HTTPStatus.OK, metadata_text)
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json"
            return response
        if path == paths.websocket_path:
            if is_authorized(request.headers, bearer_token):
                return None  # continue with the WebSocket handshake
            logfire.warning("ACP WebSocket upgrade rejected (bad bearer token)", peer=str(connection.remote_address))
            return connection.respond(HTTPStatus.UNAUTHORIZED, "missing or invalid bearer token\n")
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

    async def handler(websocket: ServerConnection) -> None:
        peer = websocket.remote_address
        if not connection_limiter.try_acquire():
            logfire.warning("ACP WebSocket connection rejected (connection limit)", peer=str(peer))
            await websocket.close(code=1013, reason="ACP connection limit reached")
            return

        logfire.info("ACP WebSocket connected", peer=str(peer))
        try:
            reader, writer, pump, transport = open_stream_bridge(websocket)
            agent = agent_factory()
            async with agent:
                # `listening=False` then `listen()` — matching acp.run_agent. Letting
                # the constructor start its own receive loop as well would make two
                # coroutines read the same stream and fail every request.
                connection = AgentSideConnection(lambda _conn: agent, writer, reader, listening=False)
                try:
                    await connection.listen()
                except ConnectionClosed:
                    pass
                finally:
                    pump.cancel()
                    try:
                        await asyncio.shield(connection.close())
                    finally:
                        await transport.aclose()
        finally:
            connection_limiter.release()
            logfire.info("ACP WebSocket disconnected", peer=str(peer))

    async with serve(
        handler,
        host,
        port,
        process_request=process_request,
        max_size=DEFAULT_MAX_SIZE,
        max_queue=DEFAULT_MAX_QUEUE,
    ):
        logfire.info(
            "ACP WebSocket server listening",
            url=f"ws://{host}:{port}{paths.websocket_path}",
            metadata=f"http://{host}:{port}{paths.metadata_path}",
            health=f"http://{host}:{port}{paths.health_path}",
            auth_required=auth_required,
        )
        await asyncio.get_running_loop().create_future()  # serve until cancelled
