"""Tests for bot.api."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from tenacity import RetryCallState, wait_none

from bot.api import ApiClient, RetryTransport, _log_retry, _RetryableStatus


class DummyTransport(httpx.AsyncBaseTransport):
    def __init__(self, events):
        self._events = list(events)
        self.closed = False
        self.attempts = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.attempts += 1
        event = self._events.pop(0)
        if isinstance(event, Exception):
            raise event
        return event

    async def aclose(self) -> None:
        self.closed = True


class TrackingStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes):
        self.content = content
        self.closed = False

    async def __aiter__(self):
        yield self.content

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_retry_transport_returns_retryable_response_after_exhaustion():
    request = httpx.Request("GET", "https://example.test/api")
    streams = [TrackingStream(b"retry-1"), TrackingStream(b"retry-2"), TrackingStream(b"final")]
    responses = [httpx.Response(503, request=request, stream=stream) for stream in streams]
    wrapped = DummyTransport(responses)
    transport = RetryTransport(wrapped, max_retries=2)

    with patch("bot.api.wait_random_exponential", return_value=wait_none()):
        result = await transport.handle_async_request(request)

    assert result is responses[-1]
    assert wrapped.attempts == 3
    assert streams[0].closed is True
    assert streams[1].closed is True
    assert streams[2].closed is False
    assert await result.aread() == b"final"

    await transport.aclose()

    assert wrapped.closed is True


@pytest.mark.anyio
async def test_retry_transport_retries_safe_get_and_closes_discarded_response():
    request = httpx.Request("GET", "https://example.test/api")
    retry_stream = TrackingStream(b"try later")
    success_stream = TrackingStream(b"ok")
    retry_response = httpx.Response(503, request=request, stream=retry_stream)
    success_response = httpx.Response(200, request=request, stream=success_stream)
    wrapped = DummyTransport([retry_response, success_response])
    transport = RetryTransport(wrapped, max_retries=2)

    with patch("bot.api.wait_random_exponential", return_value=wait_none()):
        result = await transport.handle_async_request(request)

    assert result is success_response
    assert wrapped.attempts == 2
    assert retry_stream.closed is True
    assert success_stream.closed is False


@pytest.mark.anyio
async def test_retry_transport_never_retries_post_after_5xx():
    request = httpx.Request("POST", "https://example.test/api/notes/create")
    response = httpx.Response(503, request=request, content=b"ambiguous failure")
    wrapped = DummyTransport([response, httpx.Response(200, request=request)])
    transport = RetryTransport(wrapped, max_retries=2)

    result = await transport.handle_async_request(request)

    assert result is response
    assert wrapped.attempts == 1
    assert result.content == b"ambiguous failure"


@pytest.mark.anyio
async def test_retry_transport_never_retries_post_after_transport_error():
    request = httpx.Request("POST", "https://example.test/api/notes/create")
    error = httpx.ReadError("response lost", request=request)
    wrapped = DummyTransport([error, httpx.Response(200, request=request)])
    transport = RetryTransport(wrapped, max_retries=2)

    with pytest.raises(httpx.ReadError, match="response lost"):
        await transport.handle_async_request(request)

    assert wrapped.attempts == 1


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


def test_api_client_uses_configured_retry_transport_auth_header_and_timeout(config):
    client = ApiClient()
    async_client = MagicMock(spec=httpx.AsyncClient)
    base_transport = MagicMock()

    with (
        patch("bot.api.httpx.AsyncHTTPTransport", return_value=base_transport) as transport_cls,
        patch("bot.api.httpx.AsyncClient", return_value=async_client) as client_cls,
    ):
        client.configure(config)
        assert client.get_client() is async_client

    transport_cls.assert_called_once_with(retries=0)
    assert isinstance(client_cls.call_args.kwargs["transport"], RetryTransport)
    assert client_cls.call_args.kwargs["transport"]._wrapped is base_transport
    assert client_cls.call_args.kwargs["headers"] == {"Authorization": f"Bearer {config.token}"}
    timeout = client_cls.call_args.kwargs["timeout"]
    assert timeout.connect == config.http_timeout_seconds
    assert timeout.read == config.http_timeout_seconds
    assert timeout.write == config.http_timeout_seconds
    assert timeout.pool == config.http_timeout_seconds


def test_configure_closes_existing_client_with_running_loop(config):
    client = ApiClient()
    old_client = MagicMock(spec=httpx.AsyncClient)
    old_client.is_closed = False
    old_client.aclose = MagicMock(return_value="close-task")
    loop = MagicMock()
    cast(Any, client)._ApiClient__async_client = old_client

    with patch("bot.api.asyncio.get_running_loop", return_value=loop):
        client.configure(config)

    assert client._ApiClient__async_client is None
    loop.create_task.assert_called_once_with("close-task")


def test_configure_closes_existing_client_without_running_loop(config):
    client = ApiClient()
    old_client = MagicMock(spec=httpx.AsyncClient)
    old_client.is_closed = False
    old_client.aclose = MagicMock(return_value="close-task")
    cast(Any, client)._ApiClient__async_client = old_client

    with (
        patch("bot.api.asyncio.get_running_loop", side_effect=RuntimeError),
        patch("bot.api.asyncio.run") as run_mock,
    ):
        client.configure(config)

    run_mock.assert_called_once_with("close-task")


@pytest.mark.anyio
async def test_api_client_delegates_attrs_and_close():
    client = ApiClient()
    underlying = MagicMock(spec=httpx.AsyncClient)
    underlying.is_closed = False
    underlying.aclose = AsyncMock()
    underlying.post = MagicMock()
    cast(Any, client)._ApiClient__async_client = underlying

    assert cast(Any, client).post is underlying.post

    await client.close()

    underlying.aclose.assert_awaited_once()


@pytest.mark.anyio
async def test_api_client_context_manager_returns_self_without_touching_underlying():
    client = ApiClient()
    underlying = MagicMock(spec=httpx.AsyncClient)
    underlying.__aenter__ = AsyncMock()
    underlying.__aexit__ = AsyncMock()
    cast(Any, client)._ApiClient__async_client = underlying

    async with client as c:
        assert c is client

    # Lifecycle is owned by configure()/close(); the context manager is a no-op.
    underlying.__aenter__.assert_not_awaited()
    underlying.__aexit__.assert_not_awaited()
