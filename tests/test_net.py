"""Tests for the SSRF media-URL guard."""

import pytest

from bot.net import is_safe_media_url


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
