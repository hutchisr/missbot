"""Transport-neutral turn types shared by every Missbot frontend.

`ChatAgent` speaks these instead of Misskey `Note`s, so a frontend only has to
translate its own wire format into an `AgentTurn`. Everything platform-specific
— handle formatting, attachment extraction, visibility rules, length caps —
belongs in the frontend adapter (`bot/bot.py` for Misskey), never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from pydantic_ai import BinaryContent, ImageUrl

from .imagegen import GeneratedImage


@dataclass(frozen=True)
class TurnAuthor:
    """Who sent the turn, as the agent should account for them."""

    handle: str
    """Normalized identity key — Misskey ``alice@remote.host``, ACP ``acp:<pubkey>``.

    Social credit keys and memory authorship derive from this, so an adapter must
    build it from something the author cannot freely assert. Namespace non-Misskey
    identities (``acp:``) so they can never collide with a fediverse handle.
    """
    display: Optional[str] = None
    """Handle as rendered into the prompt. Falls back to ``handle``."""
    privileged: bool = False
    """Author may manually adjust anyone's social credit (see `AgentDeps`)."""
    location: Optional[str] = None
    """Free-text location prepended to the prompt when the platform exposes one."""

    @property
    def rendered(self) -> str:
        return self.display or self.handle


TurnImage = Union[ImageUrl, BinaryContent]
"""An image as either a URL the provider fetches, or bytes sent inline as base64.
Which one an adapter produces depends on `Config.vision_image_mode`."""


@dataclass
class HistoryTurn:
    """One earlier message in the same conversation."""

    role: Literal["user", "assistant"]
    text: str
    author: Optional[str] = None
    """Handle prefixed to user turns. ``None`` on assistant turns (the bot itself)."""
    images: list[TurnImage] = field(default_factory=list)


@dataclass
class AgentTurn:
    """One inbound message for `ChatAgent.run`, in frontend-neutral form."""

    text: str
    author: TurnAuthor
    images: list[TurnImage] = field(default_factory=list)
    history: list[HistoryTurn] = field(default_factory=list)
    """Prior conversation, **oldest first**."""
    char_budget: Optional[int] = None
    """Hard character cap for the reply, or ``None`` for no cap. Misskey passes its
    note limit less the mention prefix; frontends without a cap pass ``None``."""
    source_id: Optional[str] = None
    """Platform id of the source message, recorded as memory provenance."""
    source: str = "unknown"
    """Provenance label stored on memories inferred from this turn (``misskey_note``,
    ``acp_prompt``, ...). Every adapter sets its own; maintenance treats anything that
    is not an explicit ``add_memory`` write as inferred, so a new frontend's label is
    covered by retention and per-author caps without further changes."""
    memory_writes_allowed: bool = True
    """False for private/restricted interactions, so their content stays out of the
    bot-global memory namespace. The adapter owns that judgement."""
    previous_reply: Optional[str] = None
    """The bot's most recent reply in this conversation, for the repeat guard."""


@dataclass(frozen=True)
class AutoPost:
    """Result of one autonomous-post run; ``text`` may be empty when ``image`` is set.

    Neutral like the turn types above: an image is bytes plus a media type, nothing
    platform-specific. The adapter decides how to attach it (Misskey uploads to drive and
    passes `fileIds`).
    """

    text: str
    image: Optional[GeneratedImage] = None
