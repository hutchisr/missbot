"""Parse optional identity and event metadata from an ACP harness's text header.

ACP has no structured sender or source-event fields. Harnesses such as buzz-acp
render a Nostr event into prompt text before the user-controlled ``Content:``::

    Event ID: <64 hex>
    From: alice (npub: npub1..., hex: <64 hex>)
    Time: 2026-07-24T12:00:00+00:00
    Content: <the user's message>

This parser provides structural separation, not authentication. Reading only the
region before the first ``Content:`` blocks a user message body from injecting a
second ``From:`` line, but an ACP client can fabricate the entire header. Therefore
``Config.acp_parse_sender_header`` is disabled by default and must be enabled only
when the operator trusts the client/harness which constructs prompts.

When explicitly enabled, the parser uses only a syntactically valid pubkey as the
identity key, never the display label, and accepts only a 64-hex Nostr event id as
source provenance. Malformed fields fail closed to adapter fallbacks. A batched
prompt still attributes only its first structurally delimited event block; the
format remains a buzz-acp implementation detail which may change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

# Start of the body region. Everything from here on is untrusted user content.
_CONTENT_BOUNDARY_RE = re.compile(r"^Content:", re.MULTILINE)

# The harness's sender line. Kept deliberately loose on the label and strict on the key.
_FROM_LINE_RE = re.compile(r"^From:[ \t]*(?P<body>.*)$", re.MULTILINE)
_EVENT_ID_RE = re.compile(r"^Event ID:[ \t]*(?P<event_id>[0-9a-f]{64})[ \t]*$", re.IGNORECASE | re.MULTILINE)
_TIME_RE = re.compile(r"^Time:[ \t]*(?P<timestamp>\S+)[ \t]*$", re.MULTILINE)

# Nostr pubkey forms, most authoritative first.
_HEX_KEY_RE = re.compile(r"\bhex:[ \t]*(?P<key>[0-9a-f]{64})\b", re.IGNORECASE)
_NPUB_RE = re.compile(r"\b(?P<key>npub1[02-9ac-hj-np-z]{58})\b")

# Label shown to the model — the leading text before the parenthesised key material.
_LABEL_RE = re.compile(r"^(?P<label>.*?)\s*\(")


@dataclass(frozen=True)
class AcpIdentity:
    """Identity structurally parsed from an operator-trusted ACP harness."""

    key: str
    """Namespaced identity used for social credit and memory authorship."""
    label: str | None = None
    """Human-readable name for the prompt only. Never used as a key."""
    parsed: bool = False
    """True when parsed from the optional text header; this does not prove authentication."""


def _header_region(text: str) -> str:
    """Return the candidate header region before the first ``Content:`` line.

    An absent boundary is ambiguous, so return nothing. The region becomes trusted
    only when the operator explicitly enables parsing for a trusted ACP harness.
    """
    match = _CONTENT_BOUNDARY_RE.search(text)
    if match is None:
        return ""
    return text[: match.start()]


def _extract_key(body: str) -> str | None:
    """Pull a pubkey out of a ``From:`` line body, preferring the explicit hex form."""
    hex_match = _HEX_KEY_RE.search(body)
    if hex_match:
        return hex_match.group("key").lower()
    npub_match = _NPUB_RE.search(body)
    if npub_match:
        return npub_match.group("key")
    return None


def _extract_label(body: str) -> str | None:
    """Pull the display label preceding the parenthesised key material, if any."""
    label_match = _LABEL_RE.match(body)
    if not label_match:
        return None
    label = label_match.group("label").strip()
    return label or None


def parse_event_id(text: str) -> str | None:
    """Return the first syntactically valid 64-hex event id from the header region."""
    match = _EVENT_ID_RE.search(_header_region(text))
    return match.group("event_id").lower() if match is not None else None


def parse_event_time(text: str) -> datetime | None:
    """Return the first valid header timestamp normalized to UTC."""
    match = _TIME_RE.search(_header_region(text))
    if match is None:
        return None
    try:
        timestamp = datetime.fromisoformat(match.group("timestamp"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def parse_sender(text: str, *, default_identity: str, enabled: bool = True) -> AcpIdentity:
    """Derive the caller identity from an explicitly trusted text-header source.

    Falls back to ``acp:<default_identity>`` whenever parsing is disabled or the
    candidate header is absent, malformed, or carries no usable pubkey.
    """
    fallback = AcpIdentity(key=f"acp:{default_identity}", parsed=False)
    if not enabled:
        return fallback

    region = _header_region(text)
    if not region:
        return fallback

    from_match = _FROM_LINE_RE.search(region)
    if from_match is None:
        return fallback

    body = from_match.group("body").strip()
    key = _extract_key(body)
    if key is None:
        return fallback

    return AcpIdentity(key=f"acp:{key}", label=_extract_label(body), parsed=True)
