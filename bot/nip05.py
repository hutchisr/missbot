"""Bounded, public-network-only NIP-05 identifier lookup (display metadata only)."""

import asyncio
import ipaddress
import json
import re
import socket

import httpx

from .net import _ip_is_public, is_safe_media_url

_TIMEOUT_SECONDS = 5.0
_MAX_RESPONSE_BYTES = 64 * 1024
_LOCAL_PART = re.compile(r"[a-z0-9._-]+")
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_PUBLIC_KEY = re.compile(r"[0-9a-f]{64}")


def normalize_identifier(value: str) -> str:
    """Validate the strict NIP-05 local part and canonicalize a public DNS domain."""
    if not isinstance(value, str) or value.count("@") != 1 or any(char.isspace() for char in value):
        raise ValueError("Invalid NIP-05 identifier")
    name, domain = value.split("@")
    if not _LOCAL_PART.fullmatch(name):
        raise ValueError("Invalid NIP-05 local part")
    try:
        domain = domain.encode("idna").decode("ascii").lower()
        # Validate existing xn-- labels too, rather than accepting malformed punycode.
        domain.encode("ascii").decode("idna")
    except UnicodeError as exc:
        raise ValueError("Invalid NIP-05 domain") from exc
    if (
        len(domain) > 253
        or not all(_DNS_LABEL.fullmatch(label) for label in domain.split("."))
        or domain.endswith((".home.arpa", ".lan", ".home", ".corp", ".onion"))
        or not is_safe_media_url(f"https://{domain}/")
    ):
        raise ValueError("Invalid NIP-05 domain")
    try:
        ipaddress.ip_address(domain)
    except ValueError:
        return f"{name}@{domain}"
    raise ValueError("NIP-05 requires a DNS domain, not an IP address")


async def verify_identifier(
    identifier: str,
    public_key: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> bool:
    """Match an identifier to its event key without trusting relays or redirects.

    Resolve once, reject mixed public/private answers, and connect to the selected
    numeric address. Host and TLS certificate verification still use the domain,
    so a second DNS lookup cannot rebind the connection into a private network.
    """
    try:
        identifier = normalize_identifier(identifier)
        if not isinstance(public_key, str) or not _PUBLIC_KEY.fullmatch(public_key):
            return False
        name, domain = identifier.split("@")
        async with asyncio.timeout(_TIMEOUT_SECONDS):
            addresses = await asyncio.get_running_loop().getaddrinfo(domain, 443, type=socket.SOCK_STREAM)
            if not addresses:
                return False
            for family, _, _, _, address in addresses:
                if family not in (socket.AF_INET, socket.AF_INET6):
                    return False
                ip = ipaddress.ip_address(address[0])
                if not ip.is_global or not _ip_is_public(ip):
                    return False
            address = str(ipaddress.ip_address(addresses[0][4][0]))
            url = httpx.URL(f"https://{domain}/.well-known/nostr.json").copy_with(host=address)
            # No shared Misskey client, environment proxies, cookies, or credentials.
            async with (
                httpx.AsyncClient(
                    timeout=_TIMEOUT_SECONDS,
                    follow_redirects=False,
                    trust_env=False,
                    transport=transport,
                ) as client,
                client.stream(
                    "GET",
                    url,
                    params={"name": name},
                    headers={"Host": domain, "Accept-Encoding": "identity"},
                    extensions={"sni_hostname": domain},
                ) as response,
            ):
                if response.status_code != 200:
                    return False
                # Refuse unsolicited compression to keep decoding memory bounded too.
                if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                    return False
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                        return False
                    body.extend(chunk)
                payload = json.loads(body)
                names = payload.get("names") if isinstance(payload, dict) else None
                return isinstance(names, dict) and names.get(name) == public_key
    except (httpx.HTTPError, OSError, ValueError, TimeoutError, RecursionError):
        return False
