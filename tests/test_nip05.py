"""NIP-05 syntax, identity matching, and network trust-boundary tests."""

import asyncio
import json
import socket
import ssl

import httpx
import pytest
from httpcore._backends.auto import AutoBackend

from bot import nip05

PUBLIC_KEY = "ab" * 32


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("alice@EXAMPLE.COM", "alice@example.com"),
        ("_@example.com", "_@example.com"),
        ("a.b-c_123@example.com", "a.b-c_123@example.com"),
        ("alice@bücher.de", "alice@xn--bcher-kva.de"),
        ("alice@xn--bcher-kva.de", "alice@xn--bcher-kva.de"),
    ],
)
def test_normalize_identifier(value, expected):
    assert nip05.normalize_identifier(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "alice",
        "@example.com",
        "Alice@example.com",
        "a+tag@example.com",
        "a/b@example.com",
        "álîce@example.com",
        "alice@example.com@other.com",
        "alice@example.com:443",
        "alice@example.com/path",
        "alice@example.com?name=other",
        "alice@example.com#fragment",
        "alice@user:password@example.com",
        " alice@example.com",
        "alice@example.com\n",
        "alice@exam ple.com",
        "alice@localhost",
        "alice@db.internal",
        "alice@service.local",
        "alice@service.svc.cluster.local",
        "alice@router.home.arpa",
        "alice@router.lan",
        "alice@127.0.0.1",
        "alice@8.8.8.8",
        "alice@[2606:4700:4700::1111]",
        "alice@0177.0.0.1",
        "alice@0x7f.0x0.0x0.0x1",
        "alice@2130706433",
        "alice@example.com.",
        "alice@-example.com",
        "alice@example-.com",
        "alice@ex_ample.com",
        "alice@example..com",
        "alice@xn--.com",
        f"alice@{'a' * 64}.com",
    ],
)
def test_reject_invalid_identifier(value):
    with pytest.raises(ValueError):
        nip05.normalize_identifier(value)


def _addresses(*ips):
    return [
        (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443))
        for ip in ips
    ]


@pytest.fixture
async def public_dns(monkeypatch):
    calls = []

    async def resolve(host, port, **kwargs):
        calls.append((host, port, kwargs))
        return _addresses("8.8.8.8")

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    return calls


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"names": {"alice": PUBLIC_KEY}}, True),
        ({"names": {"alice": PUBLIC_KEY}, "relays": {PUBLIC_KEY: ["ws://127.0.0.1/"]}}, True),
        ({"names": {"alice": "cd" * 32}}, False),
        ({"names": {"Alice": PUBLIC_KEY}}, False),
        ({"names": {"alice": PUBLIC_KEY.upper()}}, False),
        ({"names": {"alice": "npub1nothex"}}, False),
        ({"names": {"alice": [PUBLIC_KEY]}}, False),
        ({"names": []}, False),
        ({}, False),
        ([], False),
        (None, False),
    ],
)
async def test_matches_only_exact_name_and_hex_key(public_dns, payload, expected):
    def handler(request):
        assert str(request.url) == "https://8.8.8.8/.well-known/nostr.json?name=alice"
        assert request.headers["host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        return httpx.Response(200, json=payload)

    result = await nip05.verify_identifier("alice@EXAMPLE.COM", PUBLIC_KEY, transport=httpx.MockTransport(handler))
    assert result is expected
    assert public_dns == [("example.com", 443, {"type": socket.SOCK_STREAM})]


@pytest.mark.anyio
@pytest.mark.parametrize("public_key", ["", "ab" * 31, PUBLIC_KEY.upper(), "npub1invalid", None])
async def test_invalid_public_key_never_resolves(public_dns, public_key):
    assert not await nip05.verify_identifier("alice@example.com", public_key)
    assert public_dns == []


@pytest.mark.anyio
@pytest.mark.parametrize("status", [301, 302, 307, 308, 404, 500])
async def test_redirects_and_http_errors_are_not_followed(public_dns, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "https://127.0.0.1/private"})

    assert not await nip05.verify_identifier("alice@example.com", PUBLIC_KEY, transport=httpx.MockTransport(handler))
    assert len(calls) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("body", [b"not json", b"\xff", b"[" * 2000, b"x" * (64 * 1024 + 1)])
async def test_rejects_malformed_and_oversized_responses(public_dns, body):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    assert not await nip05.verify_identifier("alice@example.com", PUBLIC_KEY, transport=transport)


@pytest.mark.anyio
async def test_stream_limit_stops_before_reading_rest(public_dns):
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b" " * (64 * 1024)
            yield b" "
            pytest.fail("Response was read after reaching the size limit")

    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=Body()))
    assert not await nip05.verify_identifier("alice@example.com", PUBLIC_KEY, transport=transport)


@pytest.mark.anyio
async def test_exact_size_limit_is_allowed(public_dns):
    body = json.dumps({"names": {"alice": PUBLIC_KEY}}).encode()
    body += b" " * (64 * 1024 - len(body))
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    assert await nip05.verify_identifier("alice@example.com", PUBLIC_KEY, transport=transport)


@pytest.mark.anyio
async def test_refuses_unsolicited_compression(public_dns):
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("Compressed response must not be decoded")
            yield b""

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-encoding": "gzip"}, stream=Body())
    )
    assert not await nip05.verify_identifier("alice@example.com", PUBLIC_KEY, transport=transport)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "ips",
    [
        (),
        ("127.0.0.1",),
        ("10.0.0.1",),
        ("169.254.169.254",),
        ("::1",),
        ("::ffff:127.0.0.1",),
        ("100.64.0.1",),
        ("224.0.0.1",),
        ("8.8.8.8", "192.168.1.1"),
        ("8.8.8.8", "fd00::1"),
    ],
)
async def test_rejects_any_nonpublic_dns_answer(monkeypatch, ips):
    async def resolve(*args, **kwargs):
        return _addresses(*ips)

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    transport = httpx.MockTransport(lambda request: pytest.fail("Unsafe DNS must not connect"))
    assert not await nip05.verify_identifier("alice@example.com", PUBLIC_KEY, transport=transport)


@pytest.mark.anyio
async def test_dns_and_http_fail_soft(monkeypatch):
    async def resolve(*args, **kwargs):
        raise socket.gaierror("unavailable")

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    assert not await nip05.verify_identifier("alice@example.com", PUBLIC_KEY)

    async def public(*args, **kwargs):
        return _addresses("8.8.8.8")

    def handler(request):
        raise httpx.ConnectError("unavailable")

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", public)
    assert not await nip05.verify_identifier("alice@example.com", PUBLIC_KEY, transport=httpx.MockTransport(handler))


@pytest.mark.anyio
async def test_overall_timeout_includes_dns(monkeypatch):
    async def resolve(*args, **kwargs):
        await asyncio.sleep(60)

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    monkeypatch.setattr(nip05, "_TIMEOUT_SECONDS", 0.01)
    assert not await nip05.verify_identifier("alice@example.com", PUBLIC_KEY)


@pytest.mark.anyio
@pytest.mark.parametrize("ip", ["8.8.8.8", "2606:4700:4700::1111"])
async def test_real_transport_pins_ip_and_preserves_tls_hostname(monkeypatch, ip):
    dns_calls = []
    connections = []
    tls_names = []
    writes = bytearray()
    body = json.dumps({"names": {"alice": PUBLIC_KEY}}).encode()

    async def resolve(host, port, **kwargs):
        dns_calls.append(host)
        # A second lookup would rebind to loopback; the transport must use the IP.
        return _addresses(ip if len(dns_calls) == 1 else "127.0.0.1")

    class Stream:
        async def start_tls(self, ssl_context, server_hostname, timeout):
            assert ssl_context.check_hostname
            assert ssl_context.verify_mode == ssl.CERT_REQUIRED
            tls_names.append(server_hostname)
            return self

        async def write(self, buffer, timeout):
            writes.extend(buffer)

        async def read(self, max_bytes, timeout):
            return b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body

        async def aclose(self):
            pass

        def get_extra_info(self, name):
            return None

    async def connect(self, host, port, **kwargs):
        connections.append((host, port))
        return Stream()

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    monkeypatch.setattr(AutoBackend, "connect_tcp", connect)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/environment-ca.pem")
    assert await nip05.verify_identifier("alice@example.com", PUBLIC_KEY)
    assert dns_calls == ["example.com"]
    assert connections == [(ip, 443)]
    assert tls_names == ["example.com"]
    assert b"GET /.well-known/nostr.json?name=alice HTTP/1.1\r\n" in writes
    assert b"Host: example.com\r\n" in writes
    assert b"Accept-Encoding: identity\r\n" in writes


@pytest.mark.anyio
async def test_invalid_identifier_never_resolves(public_dns):
    assert not await nip05.verify_identifier("Alice@example.com", PUBLIC_KEY)
    assert public_dns == []


@pytest.mark.anyio
async def test_overall_timeout_also_bounds_response_stream(public_dns, monkeypatch):
    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b" "
                await asyncio.sleep(0.001)

    monkeypatch.setattr(nip05, "_TIMEOUT_SECONDS", 0.01)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=Body()))
    assert not await nip05.verify_identifier("alice@example.com", PUBLIC_KEY, transport=transport)
