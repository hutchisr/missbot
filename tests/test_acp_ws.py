"""Tests for the ACP WebSocket transport.

The wire contract is dictated by `acpremote`: one text frame carries exactly one ACP
JSON-RPC message, with no trailing newline. Getting the framing wrong desynchronizes
the stream, so the round-trip is pinned here.
"""

import asyncio

import pytest

from bot.acp.ws import (
    DEFAULT_HEALTH_PATH,
    ServerPaths,
    build_metadata,
    is_authorized,
    open_stream_bridge,
)


class FakeWebSocket:
    """Minimal stand-in exposing the two methods the bridge uses."""

    def __init__(self, incoming=None):
        self.sent: list[str] = []
        self._incoming = asyncio.Queue()
        for message in incoming or []:
            self._incoming.put_nowait(message)

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self):
        return await self._incoming.get()

    def push(self, message) -> None:
        self._incoming.put_nowait(message)


# --- routes -----------------------------------------------------------------


def test_paths_from_default_mount():
    paths = ServerPaths.from_mount("/acp")
    assert paths.metadata_path == "/acp"
    assert paths.websocket_path == "/acp/ws"
    assert paths.health_path == DEFAULT_HEALTH_PATH


@pytest.mark.parametrize("mount", ["acp", "/acp/", "/acp"])
def test_paths_normalize_equivalent_mounts(mount):
    assert ServerPaths.from_mount(mount).websocket_path == "/acp/ws"


def test_paths_custom_mount():
    paths = ServerPaths.from_mount("/agents/missbot")
    assert paths.metadata_path == "/agents/missbot"
    assert paths.websocket_path == "/agents/missbot/ws"


def test_paths_root_mount():
    paths = ServerPaths.from_mount("/")
    assert paths.metadata_path == "/"
    assert paths.websocket_path == "/ws"


def test_paths_reject_empty_mount():
    with pytest.raises(ValueError):
        ServerPaths.from_mount("   ")


def test_paths_reject_health_overlap():
    """A mount at /healthz would shadow the health probe."""
    with pytest.raises(ValueError):
        ServerPaths.from_mount(DEFAULT_HEALTH_PATH)


# --- metadata ---------------------------------------------------------------


def test_metadata_reports_paths_and_no_auth():
    meta = build_metadata(paths=ServerPaths.from_mount("/acp"), auth_required=False)
    assert meta["transport_kind"] == "websocket"
    assert meta["transport_version"] == 1
    assert meta["auth_required"] is False
    assert meta["supported_auth_modes"] == []
    assert meta["websocket_path"] == "/acp/ws"
    assert meta["metadata_path"] == "/acp"
    assert meta["health_path"] == "/healthz"


def test_metadata_advertises_bearer_when_required():
    meta = build_metadata(paths=ServerPaths.from_mount("/acp"), auth_required=True)
    assert meta["auth_required"] is True
    assert meta["supported_auth_modes"] == ["bearer"]


def test_metadata_has_every_field_the_client_reads():
    """acpremote's ServerMetadata indexes these directly; a missing key drops metadata."""
    meta = build_metadata(paths=ServerPaths.from_mount("/acp"), auth_required=True)
    required = {
        "transport_kind",
        "transport_version",
        "package_version",
        "auth_required",
        "supported_auth_modes",
        "max_size",
        "max_queue",
        "compression",
        "health_path",
        "metadata_path",
        "websocket_path",
    }
    assert required <= set(meta)


# --- auth -------------------------------------------------------------------


def test_authorized_when_no_token_configured():
    assert is_authorized({}, None) is True
    assert is_authorized({}, "   ") is True


def test_authorized_with_matching_bearer():
    assert is_authorized({"Authorization": "Bearer s3cret"}, "s3cret") is True


def test_rejects_wrong_or_missing_bearer():
    assert is_authorized({"Authorization": "Bearer wrong"}, "s3cret") is False
    assert is_authorized({}, "s3cret") is False
    assert is_authorized({"Authorization": "s3cret"}, "s3cret") is False


def test_configured_token_is_stripped_before_comparison():
    assert is_authorized({"Authorization": "Bearer s3cret"}, "  s3cret  ") is True


# --- framing ----------------------------------------------------------------


@pytest.mark.anyio
async def test_writer_emits_one_frame_per_message_without_newline():
    ws = FakeWebSocket()
    _reader, writer, pump, transport = open_stream_bridge(ws)
    try:
        writer.write(b'{"jsonrpc":"2.0","id":1}\n{"jsonrpc":"2.0","id":2}\n')
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert ws.sent == ['{"jsonrpc":"2.0","id":1}', '{"jsonrpc":"2.0","id":2}']
    finally:
        pump.cancel()
        await transport.aclose()


@pytest.mark.anyio
async def test_writer_buffers_partial_lines_until_complete():
    """A message split across writes must not be sent as two frames."""
    ws = FakeWebSocket()
    _reader, writer, pump, transport = open_stream_bridge(ws)
    try:
        writer.write(b'{"jsonrpc":"2.0",')
        await asyncio.sleep(0)
        assert ws.sent == []
        writer.write(b'"id":1}\n')
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert ws.sent == ['{"jsonrpc":"2.0","id":1}']
    finally:
        pump.cancel()
        await transport.aclose()


@pytest.mark.anyio
async def test_reader_terminates_each_frame_with_a_newline():
    ws = FakeWebSocket(incoming=['{"jsonrpc":"2.0","id":1}', '{"jsonrpc":"2.0","id":2}'])
    reader, _writer, pump, transport = open_stream_bridge(ws)
    try:
        assert await reader.readline() == b'{"jsonrpc":"2.0","id":1}\n'
        assert await reader.readline() == b'{"jsonrpc":"2.0","id":2}\n'
    finally:
        pump.cancel()
        await transport.aclose()


@pytest.mark.anyio
async def test_reader_rejects_binary_frames():
    ws = FakeWebSocket(incoming=[b"\x00\x01"])
    reader, _writer, pump, transport = open_stream_bridge(ws)
    try:
        with pytest.raises(TypeError):
            await reader.readline()
    finally:
        pump.cancel()
        await transport.aclose()


@pytest.mark.anyio
async def test_round_trip_preserves_message_boundaries():
    """What the writer emits is exactly what a peer's reader would parse back."""
    messages = ['{"a":1}', '{"b":[1,2,3]}', '{"c":"text with spaces"}']
    ws_out = FakeWebSocket()
    _r1, writer, pump1, transport1 = open_stream_bridge(ws_out)
    try:
        for message in messages:
            writer.write(message.encode() + b"\n")
        for _ in range(4):
            await asyncio.sleep(0)
        assert ws_out.sent == messages

        ws_in = FakeWebSocket(incoming=ws_out.sent)
        reader, _w2, pump2, transport2 = open_stream_bridge(ws_in)
        try:
            for message in messages:
                assert await reader.readline() == message.encode() + b"\n"
        finally:
            pump2.cancel()
            await transport2.aclose()
    finally:
        pump1.cancel()
        await transport1.aclose()
