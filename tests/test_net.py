"""Tests for the SSRF media-URL guard and media fetching."""

import httpx
import pytest

from bot.net import fetch_image, is_safe_media_url


@pytest.mark.parametrize(
    "url",
    [
        "https://media.example.com/a.png",
        "http://cdn.example.net/path/to/image.jpg?x=1",
        "https://files.misskey.example/thumb.webp",
        "https://8.8.8.8/x.png",  # public IP literal is fine
        "https://[2606:4700:4700::1111]/x.png",  # public IPv6
    ],
)
def test_allows_public_http_urls(url):
    assert is_safe_media_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        # Cloud metadata + loopback + private + link-local IP literals
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1/",
        "http://10.0.0.5/x.png",
        "http://192.168.1.10/x.png",
        "http://172.16.0.9/x.png",
        "http://[::1]/x.png",
        "http://[::ffff:169.254.169.254]/x",  # IPv4-mapped link-local
        # Internal hostnames
        "http://localhost/x",
        "http://redis/x",  # bare single label
        "http://missbot-redis.misskey.svc.cluster.local:6379/",
        "http://db.internal/x",
        "http://service.local/x",
        "http://metadata.google.internal/computeMetadata/v1/",
        # Obfuscated IPs that don't parse as ip_address
        "http://2130706433/",
        "http://0177.0.0.1/",
        "http://0x7f.0x0.0x0.0x1/",  # dotted hex loopback
        "http://0xa9.0xfe.0xa9.0xfe/latest/meta-data/",  # dotted hex link-local
        # Non-http schemes
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/x",
        # Garbage
        "not a url",
        "http://",
    ],
)
def test_blocks_unsafe_urls(url):
    assert is_safe_media_url(url) is False


# --- media fetching (base64 vision mode) ------------------------------------


def _transport(handler):
    return httpx.MockTransport(handler)


def _ok_image(request):
    return httpx.Response(200, content=b"\x89PNG\r\n\x1a\n" + b"x" * 100, headers={"content-type": "image/png"})


@pytest.mark.anyio
async def test_fetch_image_returns_bytes_and_media_type():
    result = await fetch_image(
        "https://media.example/pic.png", timeout=5.0, max_bytes=1_000_000, transport=_transport(_ok_image)
    )

    # None is the "unusable image" signal, not a fetch of zero bytes.
    assert result is not None
    data, media_type = result
    assert data.startswith(b"\x89PNG")
    assert media_type == "image/png"


@pytest.mark.anyio
async def test_fetch_image_refuses_unsafe_url_without_requesting():
    """The SSRF guard is re-checked here, not just at extraction time."""
    called = False

    def handler(request):
        nonlocal called
        called = True
        return _ok_image(request)

    assert (
        await fetch_image(
            "http://169.254.169.254/latest/meta-data/", timeout=5.0, max_bytes=1_000_000, transport=_transport(handler)
        )
        is None
    )
    assert called is False


@pytest.mark.anyio
async def test_fetch_image_rejects_non_image_content_type():
    def handler(request):
        return httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})

    assert (
        await fetch_image(
            "https://media.example/pic.png", timeout=5.0, max_bytes=1_000_000, transport=_transport(handler)
        )
        is None
    )


@pytest.mark.anyio
async def test_fetch_image_rejects_oversized_body():
    """A huge attacker-supplied file must not be buffered into the pod's memory."""

    def handler(request):
        return httpx.Response(200, content=b"x" * 5000, headers={"content-type": "image/png"})

    assert (
        await fetch_image("https://media.example/big.png", timeout=5.0, max_bytes=1000, transport=_transport(handler))
        is None
    )


@pytest.mark.anyio
async def test_fetch_image_returns_none_on_http_error():
    def handler(request):
        return httpx.Response(404)

    assert (
        await fetch_image(
            "https://media.example/gone.png", timeout=5.0, max_bytes=1_000_000, transport=_transport(handler)
        )
        is None
    )


@pytest.mark.anyio
async def test_fetch_image_returns_none_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    assert (
        await fetch_image(
            "https://media.example/pic.png", timeout=5.0, max_bytes=1_000_000, transport=_transport(handler)
        )
        is None
    )
