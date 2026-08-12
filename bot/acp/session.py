"""Per-session conversation state for the ACP frontend.

Unlike Misskey — where each mention rebuilds its context by walking the reply chain —
an ACP session is a long-lived conversation the agent is expected to remember. This
module keeps that history in memory, bounded, and tracks the in-flight task so
``session/cancel`` can interrupt it.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from uuid import uuid4

from ..core import HistoryTurn


@dataclass
class AcpSession:
    """One ACP conversation."""

    id: str
    max_history_turns: int
    history: deque[HistoryTurn] = field(default_factory=deque)
    task: asyncio.Task | None = None
    """The in-flight prompt task, so ``session/cancel`` can interrupt this turn."""
    cancelled: bool = False
    """Set by ``session/cancel`` so the prompt can report ``stop_reason='cancelled'``."""

    def __post_init__(self) -> None:
        # A "turn" is a user message plus the bot's reply, so the deque holds twice as many.
        self.history = deque(self.history, maxlen=self.max_history_turns * 2)

    def record(self, user_text: str, author: str, reply: str | None) -> None:
        """Append this exchange, evicting the oldest once the bound is reached."""
        self.history.append(HistoryTurn(role="user", text=user_text, author=author))
        if reply:
            self.history.append(HistoryTurn(role="assistant", text=reply))

    def previous_reply(self) -> str | None:
        """The bot's most recent reply in this session, for the repeat guard."""
        for past in reversed(self.history):
            if past.role == "assistant" and past.text.strip():
                return past.text
        return None


class SessionRegistry:
    """Bounded collection of live ACP sessions."""

    def __init__(self, *, max_sessions: int, max_history_turns: int):
        self._sessions: dict[str, AcpSession] = {}
        self._max_sessions = max_sessions
        self._max_history_turns = max_history_turns

    def __len__(self) -> int:
        return len(self._sessions)

    @property
    def at_capacity(self) -> bool:
        return len(self._sessions) >= self._max_sessions

    def create(self) -> AcpSession:
        """Register a new session. Raises when the configured cap is reached."""
        if self.at_capacity:
            raise RuntimeError(f"ACP session limit reached ({self._max_sessions} concurrent sessions)")
        session = AcpSession(id=str(uuid4()), max_history_turns=self._max_history_turns)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> AcpSession | None:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> AcpSession | None:
        return self._sessions.pop(session_id, None)
