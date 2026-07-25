"""ACP (Agent Client Protocol) frontend for Missbot.

Run with ``python -m bot.acp -c config.yaml``. See `bot/acp/agent.py`.
"""

from .agent import MissbotAgent
from .identity import AcpIdentity, parse_sender
from .session import AcpSession, SessionRegistry

__all__ = ["AcpIdentity", "AcpSession", "MissbotAgent", "SessionRegistry", "parse_sender"]
