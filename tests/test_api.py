"""Tests for bot.api."""

from typing import Any, cast
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from tenacity import RetryCallState

from bot.api import ApiClient, RetryTransport, _RetryableStatus, _log_retry


class DummyTransport(httpx.AsyncBaseTransport):
    def __init__(self, events):
        self._events = list(events)
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        event = self._events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_retry_transport_returns_retryable_response_after_exhaustion():
    request = httpx.Request("GET", "https://example.test/api")
    response = httpx.Response(503, request=request)
    wrapped = DummyTransport([response])
    transport = RetryTransport(wrapped, max_retries=0)

    result = await transport.handle_async_request(request)
    await transport.aclose()

    assert result is response
    assert wrapped.closed is True


def test_log_retry_reports_retryable_status():
    response = httpx.Response(503, request=httpx.Request("GET", "https://example.test/api"))
    retry_state = cast(
        RetryCallState,
        SimpleNamespace(
            outcome=SimpleNamespace(exception=lambda: _RetryableStatus(response)),
            next_action=SimpleNamespace(sleep=1.5),
            attempt_number=2,
        ),
    )

    with patch("bot.api.logfire.warning") as warning_mock:
        _log_retry(retry_state)

    warning_mock.assert_called_once_with(
        "Retrying request",
        attempt=2,
        sleep=1.5,
        reason="HTTP 503",
    )


def test_log_retry_reports_generic_exception_name():
    retry_state = cast(
        RetryCallState,
        SimpleNamespace(
            outcome=SimpleNamespace(exception=lambda: RuntimeError("boom")),
            next_action=None,
            attempt_number=1,
        ),
    )

    with patch("bot.api.logfire.warning") as warning_mock:
        _log_retry(retry_state)

    warning_mock.assert_called_once_with(
        "Retrying request",
        attempt=1,
        sleep=0.0,
        reason="RuntimeError",
    )


def test_api_client_warns_when_accessed_before_configuration():
    client = ApiClient()
    async_client = MagicMock(spec=httpx.AsyncClient)

    with (
        patch("bot.api.httpx.AsyncClient", return_value=async_client) as client_cls,
        patch("bot.api.logfire.warning") as warning_mock,
    ):
        assert client.get_client() is async_client

    client_cls.assert_called_once_with()
    warning_mock.assert_called_once_with("API client accessed before configuration")


def test_api_client_uses_configured_retry_transport_and_auth_header(config):
    client = ApiClient()
    async_client = MagicMock(spec=httpx.AsyncClient)
    base_transport = MagicMock()

    with (
        patch("bot.api.httpx.AsyncHTTPTransport", return_value=base_transport) as transport_cls,
        patch("bot.api.httpx.AsyncClient", return_value=async_client) as client_cls,
    ):
        client.configure(config)
        assert client.get_client() is async_client

    transport_cls.assert_called_once_with(retries=config.max_retries)
    assert isinstance(client_cls.call_args.kwargs["transport"], RetryTransport)
    assert client_cls.call_args.kwargs["transport"]._wrapped is base_transport
    assert client_cls.call_args.kwargs["headers"] == {"Authorization": f"Bearer {config.token}"}


def test_configure_closes_existing_client_with_running_loop(config):
    client = ApiClient()
    old_client = MagicMock(spec=httpx.AsyncClient)
    old_client.is_closed = False
    old_client.aclose = MagicMock(return_value="close-task")
    loop = MagicMock()
    setattr(client, "_ApiClient__async_client", old_client)

    with patch("bot.api.asyncio.get_running_loop", return_value=loop):
        client.configure(config)

    assert getattr(client, "_ApiClient__async_client") is None
    loop.create_task.assert_called_once_with("close-task")


def test_configure_closes_existing_client_without_running_loop(config):
    client = ApiClient()
    old_client = MagicMock(spec=httpx.AsyncClient)
    old_client.is_closed = False
    old_client.aclose = MagicMock(return_value="close-task")
    setattr(client, "_ApiClient__async_client", old_client)

    with (
        patch("bot.api.asyncio.get_running_loop", side_effect=RuntimeError),
        patch("bot.api.asyncio.run") as run_mock,
    ):
        client.configure(config)

    run_mock.assert_called_once_with("close-task")


@pytest.mark.anyio
async def test_api_client_delegates_context_manager_and_close():
    client = ApiClient()
    underlying = MagicMock(spec=httpx.AsyncClient)
    underlying.is_closed = False
    underlying.aclose = AsyncMock()
    underlying.__aenter__ = AsyncMock(return_value="entered")
    underlying.__aexit__ = AsyncMock(return_value=False)
    underlying.post = MagicMock()
    setattr(client, "_ApiClient__async_client", underlying)

    assert cast(Any, client).post is underlying.post
    assert await client.__aenter__() == "entered"
    assert await client.__aexit__(None, None, None) is False

    await client.close()

    underlying.aclose.assert_awaited_once()
