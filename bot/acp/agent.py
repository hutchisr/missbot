"""ACP frontend: serves the same persona, memory, and tools as the Misskey bot.

`ChatAgent` is transport-neutral (see `bot/core.py`), so this module is an adapter in
the same sense `bot/bot.py` is — it translates ACP prompts into `AgentTurn`s and pushes
replies back as session updates. State lives in Postgres (mem0) and Redis, so an ACP
process pointed at the same backends is genuinely the same bot, not a copy of it.

Deliberately not implemented: `session/load` (no persistence across process restarts),
and the client's file-system/terminal capabilities. Missbot is a conversational agent,
not a coding agent, so it never asks a client to read files or run commands.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, NoReturn, Optional

import acp
import logfire
from acp import schema
from redis.asyncio import Redis

from ..ai import ChatAgent
from ..core import AgentTurn, TurnAuthor
from ..memory import MemoryStore
from ..models import Config
from ..provider import PROJECT_VERSION
from .identity import parse_sender
from .session import SessionRegistry

_AGENT_NAME = "missbot"


class MissbotAgent(acp.Agent):
    """Exposes `ChatAgent` over the Agent Client Protocol."""

    def __init__(
        self,
        config: Config,
        redis_client: Optional[Redis] = None,
        memory: Optional[MemoryStore] = None,
        chat_agent: Optional[ChatAgent] = None,
        *,
        session_registry: Optional[SessionRegistry] = None,
        prompt_semaphore: Optional[asyncio.Semaphore] = None,
    ):
        self._config = config
        # One protocol adapter per connection, but the expensive parts (model chain,
        # MCP sessions, Redis, memory) are shareable: the WebSocket server builds one
        # ChatAgent and passes it to every connection's adapter. Only an adapter that
        # created its own ChatAgent is responsible for opening and closing it.
        self._owns_chat_agent = chat_agent is None
        self._agent = chat_agent or ChatAgent(config, redis_client=redis_client, memory=memory)
        self._sessions = (
            session_registry
            if session_registry is not None
            else SessionRegistry(
                max_sessions=config.acp_max_sessions,
                max_history_turns=config.acp_max_history_turns,
            )
        )
        self._prompt_slots = prompt_semaphore or asyncio.Semaphore(config.acp_max_sessions)
        self._session_ids: set[str] = set()
        self._conn: Optional[acp.Client] = None

    async def __aenter__(self) -> "MissbotAgent":
        if self._owns_chat_agent:
            await self._agent.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
        if self._owns_chat_agent:
            await self._agent.__aexit__(exc_type, exc, tb)

    def on_connect(self, conn: acp.Client) -> None:
        """Capture the connection so prompts can push `session/update` notifications."""
        self._conn = conn

    # --- protocol surface -------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Optional[schema.ClientCapabilities] = None,
        client_info: Optional[schema.Implementation] = None,
        **kwargs: Any,
    ) -> acp.InitializeResponse:
        logfire.info(
            "ACP initialize",
            client_protocol_version=protocol_version,
            client=client_info.name if client_info else None,
        )
        return acp.InitializeResponse(
            # Negotiate down if the client speaks an older version than we do.
            protocol_version=min(protocol_version, acp.PROTOCOL_VERSION),
            agent_capabilities=schema.AgentCapabilities(
                # No cross-restart persistence, so a client must not try to resume.
                load_session=False,
                # Text only for now: image/audio blocks are dropped in `prompt`.
                prompt_capabilities=schema.PromptCapabilities(image=False, audio=False, embedded_context=False),
            ),
            # Over stdio the trust boundary is process spawn — whoever launched this
            # process already holds the config, token, and database credentials.
            auth_methods=[],
            agent_info=schema.Implementation(name=_AGENT_NAME, version=PROJECT_VERSION),
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> Optional[acp.AuthenticateResponse]:
        """No-op: `initialize` advertises no auth methods (see the note there)."""
        return None

    async def new_session(
        self,
        cwd: str,
        additional_directories: Optional[list[str]] = None,
        mcp_servers: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> acp.NewSessionResponse:
        """Create a session, ignoring the client's `cwd` and MCP servers.

        Missbot brings its own toolset from config and does not act on a working
        directory, so accepting client-supplied MCP servers would only widen its
        attack surface for no gain.
        """
        if self._sessions.at_capacity:
            raise acp.RequestError.internal_error(
                data={"reason": f"session limit reached ({self._config.acp_max_sessions})"}
            )
        session = self._sessions.create()
        self._session_ids.add(session.id)
        logfire.info("ACP session created", session_id=session.id, live_sessions=len(self._sessions))
        return acp.NewSessionResponse(session_id=session.id)

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        **kwargs: Any,
    ) -> acp.PromptResponse:
        """Run one bounded turn and push its reply as a session update."""
        session = self._sessions.get(session_id)
        if session is None or session_id not in self._session_ids:
            raise acp.RequestError.invalid_params(data={"reason": f"unknown session {session_id}"})

        try:
            text = _text_from_blocks(prompt, max_chars=self._config.acp_max_prompt_chars)
        except ValueError as exc:
            raise acp.RequestError.invalid_params(data={"reason": str(exc)}) from exc
        if not text.strip():
            logfire.info("ACP prompt had no usable text blocks", session_id=session_id)
            return acp.PromptResponse(stop_reason="end_turn")

        if session.task is not None and not session.task.done():
            raise acp.RequestError.internal_error(
                data={"reason": f"session {session_id} already has a prompt in flight"}
            )

        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("ACP prompt is not running in an asyncio task")
        session.cancelled = False
        session.task = current_task

        try:
            async with self._prompt_slots:
                identity = parse_sender(
                    text,
                    default_identity=self._config.acp_default_identity,
                    enabled=self._config.acp_parse_sender_header,
                )

                # Same social credit floor as the Misskey path: a caller below it never
                # reaches the model, is not scored, and nothing is ingested.
                threshold = self._config.social_credit_ignore_threshold
                if threshold is not None:
                    score = await self._agent.get_score(identity.key)
                    if score is not None and score < threshold:
                        logfire.info(
                            "ACP prompt refused (low social credit)",
                            session_id=session_id,
                            identity=identity.key,
                            score=score,
                            threshold=threshold,
                        )
                        return acp.PromptResponse(stop_reason="refusal")

                turn = AgentTurn(
                    text=text,
                    # parse_sender keys on the harness-supplied pubkey and namespaces it
                    # as acp:<key>, making it stable enough for provenance trust checks.
                    author=TurnAuthor(handle=identity.key, display=identity.label, user_id=identity.key),
                    history=list(session.history),
                    # The reply has no Misskey note budget; the input still has a
                    # separate safety limit before provider and memory calls.
                    char_budget=None,
                    source_id=f"acp:{session_id}",
                    source="acp_prompt",
                    previous_reply=session.previous_reply(),
                )
                reply = await self._agent.run(turn)
        except asyncio.CancelledError:
            # `session/cancel` arrived mid-turn. ACP requires the pending prompt to
            # answer with `cancelled` rather than raising.
            if session.cancelled:
                logfire.info("ACP prompt cancelled", session_id=session_id)
                return acp.PromptResponse(stop_reason="cancelled")
            raise
        finally:
            if session.task is current_task:
                session.task = None

        # Mirrors bot/bot.py:on_mention — the sentinel means "say nothing".
        if reply.strip() == "NO_REPLY":
            logfire.info("ACP prompt produced NO_REPLY", session_id=session_id)
            session.record(text, identity.key, None)
            return acp.PromptResponse(stop_reason="end_turn")

        session.record(text, identity.key, reply)
        if self._conn is not None:
            await self._conn.session_update(session_id, acp.update_agent_message_text(reply))
        return acp.PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Interrupt the session's in-flight turn, if any."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.cancelled = True
        if session.task is not None and not session.task.done():
            session.task.cancel()

    async def close_session(self, session_id: str, **kwargs: Any) -> None:
        if session_id not in self._session_ids:
            return None
        self._session_ids.remove(session_id)
        session = self._sessions.remove(session_id)
        if session is not None and session.task is not None and not session.task.done():
            session.cancelled = True
            session.task.cancel()
            with suppress(asyncio.CancelledError):
                await session.task
        logfire.info("ACP session closed", session_id=session_id, live_sessions=len(self._sessions))
        return None

    async def aclose(self) -> None:
        """Close every session owned by this protocol connection."""
        for session_id in tuple(self._session_ids):
            await self.close_session(session_id)

    # --- deliberately unsupported ----------------------------------------
    #
    # `acp.Agent` is a Protocol, so every method left out here is still *inherited* as
    # a stub with an `...` body. The SDK's router resolves handlers with `getattr`
    # (acp/router.py:_resolve_handler), so those stubs get routed anyway and return
    # None, which the connection then reports to the client as a success. Declining
    # explicitly is the difference between "no such method" and a client concluding
    # its `session/load` restored a session that never existed.

    def _unsupported(self, method: str) -> NoReturn:
        logfire.info("ACP method declined", method=method)
        raise acp.RequestError.method_not_found(method)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: Optional[list[Any]] = None,
        additional_directories: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> NoReturn:
        """History is in-process and does not survive a restart (`initialize` says so)."""
        self._unsupported(acp.AGENT_METHODS["session_load"])

    async def list_sessions(self, cwd: str | None = None, cursor: str | None = None, **kwargs: Any) -> NoReturn:
        """Sessions are per-connection and not enumerable across clients."""
        self._unsupported(acp.AGENT_METHODS["session_list"])

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> NoReturn:
        """One persona, one mode — `initialize` advertises no modes to select."""
        self._unsupported(acp.AGENT_METHODS["session_set_mode"])

    async def set_config_option(self, config_id: str, session_id: str, value: str | bool, **kwargs: Any) -> NoReturn:
        """Configuration is the operator's, from config.yaml — not the client's to set."""
        self._unsupported(acp.AGENT_METHODS["session_set_config_option"])

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: Optional[list[str]] = None,
        mcp_servers: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> NoReturn:
        self._unsupported(acp.AGENT_METHODS["session_fork"])

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: Optional[list[str]] = None,
        mcp_servers: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> NoReturn:
        self._unsupported(acp.AGENT_METHODS["session_resume"])

    async def ext_method(self, method: str, params: dict[str, Any]) -> NoReturn:
        """No `_`-prefixed extensions: the toolset comes from config, not the wire."""
        self._unsupported(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        """Notifications have no response channel, so an unknown one is dropped, not refused."""
        logfire.info("ACP extension notification ignored", method=method)


def _text_from_blocks(blocks: list[Any], *, max_chars: Optional[int] = None) -> str:
    """Concatenate text blocks, rejecting oversized aggregate prompt text."""
    parts: list[str] = []
    total_chars = 0
    dropped = 0
    for block in blocks:
        if getattr(block, "type", None) == "text":
            text = block.text
            total_chars += len(text) + bool(parts)
            if max_chars is not None and total_chars > max_chars:
                raise ValueError(f"ACP prompt exceeds the {max_chars}-character limit")
            parts.append(text)
        else:
            dropped += 1
    if dropped:
        logfire.warning("Dropped non-text ACP content blocks", count=dropped)
    return "\n".join(parts)
