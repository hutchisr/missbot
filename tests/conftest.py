"""Shared fixtures for the test suite."""

from __future__ import annotations

from typing import Any

import pytest
from fakeredis import FakeAsyncRedis

from pydantic_ai import ImageUrl

from bot.core import AgentTurn, HistoryTurn, TurnAuthor
from bot.models import Config, MiFile, Note, User


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _base_config_kwargs(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "domain": "example.test",
        "url": "https://example.test/",
        "ws_url": "wss://example.test/",
        "token": "secret-token",
        "llm_models": ["openrouter:test/model"],
        "max_tokens": 1024,
        "bot_user_id": "bot-id",
        "bot_username": "grok",
        "system_prompt": "You are a test bot.",
        "max_retries": 2,
    }
    data.update(overrides)
    return data


@pytest.fixture
def make_config():
    """Factory that builds a valid Config, with optional overrides."""

    def _factory(**overrides: Any) -> Config:
        return Config(**_base_config_kwargs(**overrides))

    return _factory


@pytest.fixture
def config(make_config) -> Config:
    return make_config()


@pytest.fixture
def make_user():
    def _factory(
        *,
        id: str = "user-1",
        username: str = "alice",
        host: str | None = None,
        name: str | None = None,
        location: str | None = None,
    ) -> User:
        return User(id=id, username=username, host=host, name=name, location=location)

    return _factory


@pytest.fixture
def make_note(make_user):
    def _factory(
        *,
        id: str = "note-1",
        text: str | None = "hello",
        user: User | None = None,
        files: list[MiFile] | None = None,
        reply_id: str | None = None,
        mentions: list[str] | None = None,
    ) -> Note:
        u = user or make_user()
        return Note(
            id=id,
            text=text,
            userId=u.id,
            user=u,
            replyId=reply_id,
            mentions=mentions,
            files=files,
        )

    return _factory


@pytest.fixture
def make_turn():
    """Factory for a frontend-neutral AgentTurn, as an adapter would build it."""

    def _factory(
        *,
        text: str = "hello",
        handle: str = "alice",
        display: str | None = None,
        privileged: bool = False,
        location: str | None = None,
        images: list[ImageUrl] | None = None,
        history: list[HistoryTurn] | None = None,
        char_budget: int | None = None,
        source_id: str | None = "note-1",
        source: str = "unknown",
        memory_writes_allowed: bool = True,
        previous_reply: str | None = None,
    ) -> AgentTurn:
        return AgentTurn(
            text=text,
            author=TurnAuthor(handle=handle, display=display, privileged=privileged, location=location),
            images=images or [],
            history=history or [],
            char_budget=char_budget,
            source_id=source_id,
            source=source,
            memory_writes_allowed=memory_writes_allowed,
            previous_reply=previous_reply,
        )

    return _factory


@pytest.fixture
def fake_redis() -> FakeAsyncRedis:
    """An in-memory async Redis client. No teardown needed — purely in-memory."""
    return FakeAsyncRedis(decode_responses=True)
