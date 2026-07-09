"""URL safety checks for attacker-supplied media (SSRF guard).

Note files (`url` / `thumbnailUrl`) on a federated note are fully controlled by a
remote user. Those URLs get handed to the vision model — and depending on the
deployment they may be fetched from inside your network (a self-hosted
OpenAI-compatible vision endpoint, or pydantic-ai inlining media from the bot
process). `is_safe_media_url` rejects the URLs that make SSRF useful: non-HTTP
schemes, IP literals in private/reserved ranges, and internal hostnames.

This is a synchronous, no-DNS check by design (resolving in the hot path would
block the event loop and add a slow-DNS DoS vector). The residual gap is a public
hostname whose A record points at a private address; close that with a media-domain
allowlist if your instance proxies all media.
"""

import ipaddress
import socket
from urllib.parse import urlsplit

# Internal-by-convention names that should never be reachable media hosts.
_BLOCKED_HOSTS = frozenset({"localhost", "metadata", "metadata.google.internal"})
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".svc", ".cluster.local")


def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only for globally-routable unicast addresses."""
    # IPv4-mapped IPv6 (e.g. ::ffff:169.254.169.254) — judge by the embedded v4.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def is_safe_media_url(url: str) -> bool:
    """Whether ``url`` is safe to hand to the vision model / a media fetcher.

    Conservative: returns False for anything that isn't an http(s) URL to a host
    that is plausibly public.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False

    if parts.scheme not in ("http", "https"):
        return False

    host = parts.hostname
    if not host:
        return False
    host = host.lower().rstrip(".")

    # IP literal: decide purely on the address range.
    try:
        return _ip_is_public(ipaddress.ip_address(host))
    except ValueError:
        pass  # not an IP literal — treat as a hostname

    # ``ipaddress`` deliberately accepts only canonical text, but system resolvers also
    # understand legacy dotted-hex/octal forms (for example ``0xa9.0xfe.0xa9.0xfe`` ->
    # 169.254.169.254). Reject anything inet_aton recognizes here: canonical IPv4 literals
    # already returned above, so a match at this point is necessarily an obfuscated address.
    try:
        socket.inet_aton(host)
    except OSError:
        pass
    else:
        return False

    if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_HOST_SUFFIXES):
        return False
    # Bare single-label hosts ("redis", "db") resolve to internal services.
    if "." not in host:
        return False
    # Numeric-only hosts that neither parser accepted are still suspicious obfuscated IPs.
    if all(ch in "0123456789." for ch in host):
        return False
    return True
