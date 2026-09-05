"""Transport-neutral turn types shared by every Missbot frontend.

`ChatAgent` speaks these instead of Misskey `Note`s, so a frontend only has to
translate its own wire format into an `AgentTurn`. Everything platform-specific
— handle formatting, attachment extraction, visibility rules, length caps —
belongs in the frontend adapter (`bot/bot.py` for Misskey), never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic_ai import BinaryContent, ImageUrl

from .imagegen import GeneratedImage


@dataclass(frozen=True)
class TurnAuthor:
    """Who sent the turn, as the agent should account for them."""

    handle: str
    """Normalized identity key — Misskey ``alice@remote.host``, ACP ``acp:<pubkey>``.

    Social credit and Hindsight provenance derive from this, so an adapter must
    build it from something the author cannot freely assert. Namespace non-Misskey
    identities (``acp:``) so they can never collide with a fediverse handle.
    """
    display: str | None = None
    """Handle as rendered into the prompt. Falls back to ``handle``."""
    user_id: str | None = None
    """Stable platform identity recorded as Hindsight provenance.

    Misskey supplies its user id; ACP supplies its namespaced pubkey identity.
    Display names must never be substituted for this value.
    """
    privileged: bool = False
    """Author may manually adjust anyone's social credit (see `AgentDeps`)."""
    location: str | None = None
    """Free-text location prepended to the prompt when the platform exposes one."""

    @property
    def rendered(self) -> str:
        return self.display or self.handle


TurnImage = ImageUrl | BinaryContent
"""An image as either a URL the provider fetches, or bytes sent inline as base64.
Which one an adapter produces depends on `Config.vision_image_mode`."""


@dataclass
class HistoryTurn:
    """One earlier message in the same conversation."""

    role: Literal["user", "assistant"]
    text: str
    author: str | None = None
    """Handle prefixed to user turns. ``None`` on assistant turns (the bot itself)."""
    images: list[TurnImage] = field(default_factory=list)


@dataclass
class AgentTurn:
    """One inbound message for `ChatAgent.run`, in frontend-neutral form."""

    text: str
    author: TurnAuthor
    source_id: str
    """Platform event id used for Hindsight provenance and idempotency.

    Stable only when the adapter receives authenticated metadata or the operator
    explicitly trusts its harness; otherwise use a collision-resistant best-effort id.
    """
    conversation_id: str
    """Stable thread/session grouping used as the Hindsight document id.

    When a frontend cannot prove a stable root, it must fail closed to an
    event-specific id rather than silently merge unrelated or shifting groups.
    """
    occurred_at: datetime | None = None
    """When the source event occurred, when the frontend exposes it."""
    source: str = "unknown"
    """Provenance label retained with the exchange (``misskey_note``, ``acp_prompt``)."""
    images: list[TurnImage] = field(default_factory=list)
    history: list[HistoryTurn] = field(default_factory=list)
    """Prior conversation, **oldest first**."""
    char_budget: int | None = None
    """Hard character cap for the reply, or ``None`` for no cap. Misskey passes its
    note limit less the mention prefix; frontends without a cap pass ``None``."""
    memory_access_allowed: bool = True
    """False for private/restricted interactions, so neither recall nor retention can
    cross the bot-global memory boundary. The adapter owns that judgement."""
    previous_reply: str | None = None
    """The bot's most recent reply in this conversation, for the repeat guard."""


@dataclass(frozen=True)
class Poll:
    """Transport-neutral poll attached to an autonomous post."""

    choices: tuple[str, ...]
    multiple: bool = False
    duration_minutes: int | None = None
    """Minutes until the poll closes, or ``None`` to leave it open indefinitely."""


@dataclass(frozen=True)
class AutoPost:
    """Result of one autonomous-post run; ``text`` may be empty when ``image`` is set.

    Neutral like the turn types above: attachments carry no Misskey wire fields. The
    adapter uploads an image and translates a poll duration into the platform payload.
    """

    text: str
    image: GeneratedImage | None = None
    poll: Poll | None = None
