"""Derive a caller identity from an ACP harness's message header.

ACP carries no per-sender identity field: a harness like buzz-acp renders the Nostr
event into the prompt *text* as a header block::

    Event ID: <hex>
    Channel: general (#<uuid>)
    Kind: 9
    From: alice (npub: npub1..., hex: <64 hex>)
    Time: 2026-07-24T12:00:00+00:00
    Content: <the user's message>
    Tags: [...]

Attribution therefore has to come out of that text, which is only safe because of
where the user's own words sit. Two rules make it defensible:

1. **Only the region before the first ``Content:`` line is read.** Everything a user
   typed lands in ``Content:`` or after it, so no message body can reach the region we
   parse. A user who types their own fake ``From:`` line is writing it *below* the
   boundary, where it is ignored.
2. **Only the pubkey is used as the key** — never the display label. Pubkeys are
   relay-issued; display names are user-settable.

Known limits, deliberately not papered over: a batched prompt concatenates several
event blocks, so only the *first* block's header is structurally protected — a first
sender's content can forge a second block's header. And the format is buzz-acp's
internal detail, which can change without notice. `Config.acp_parse_sender_header`
turns the whole mechanism off if either becomes a problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Start of the user-controlled payload. Everything from here on is untrusted.
_CONTENT_BOUNDARY_RE = re.compile(r"^Content:", re.MULTILINE)

# The harness's sender line. Kept deliberately loose on the label and strict on the key.
_FROM_LINE_RE = re.compile(r"^From:[ \t]*(?P<body>.*)$", re.MULTILINE)

# Nostr pubkey forms, most authoritative first.
_HEX_KEY_RE = re.compile(r"\bhex:[ \t]*(?P<key>[0-9a-f]{64})\b", re.IGNORECASE)
_NPUB_RE = re.compile(r"\b(?P<key>npub1[02-9ac-hj-np-z]{58})\b")

# Label shown to the model — the leading text before the parenthesised key material.
_LABEL_RE = re.compile(r"^(?P<label>.*?)\s*\(")


@dataclass(frozen=True)
class AcpIdentity:
    """Who an ACP turn is attributed to."""

    key: str
    """Namespaced identity used for social credit and memory authorship."""
    label: Optional[str] = None
    """Human-readable name for the prompt only. Never used as a key."""
    parsed: bool = False
    """True when derived from a sender header; False when it is the configured fallback."""


def _trusted_region(text: str) -> str:
    """Return the harness-generated header region: everything before the first ``Content:``.

    Returns an empty string when there is no ``Content:`` boundary — without it we cannot
    tell harness output from user text, so nothing is trusted (fail closed).
    """
    match = _CONTENT_BOUNDARY_RE.search(text)
    if match is None:
        return ""
    return text[: match.start()]


def _extract_key(body: str) -> Optional[str]:
    """Pull a pubkey out of a ``From:`` line body, preferring the explicit hex form."""
    hex_match = _HEX_KEY_RE.search(body)
    if hex_match:
        return hex_match.group("key").lower()
    npub_match = _NPUB_RE.search(body)
    if npub_match:
        return npub_match.group("key")
    return None


def _extract_label(body: str) -> Optional[str]:
    """Pull the display label preceding the parenthesised key material, if any."""
    label_match = _LABEL_RE.match(body)
    if not label_match:
        return None
    label = label_match.group("label").strip()
    return label or None


def parse_sender(text: str, *, default_identity: str, enabled: bool = True) -> AcpIdentity:
    """Derive the caller identity for one ACP prompt.

    Falls back to ``acp:<default_identity>`` whenever the header is absent, malformed,
    carries no usable pubkey, or parsing is disabled — never to an unattributed write.
    """
    fallback = AcpIdentity(key=f"acp:{default_identity}", parsed=False)
    if not enabled:
        return fallback

    region = _trusted_region(text)
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
