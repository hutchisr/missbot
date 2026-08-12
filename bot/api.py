import asyncio
import threading
from typing import TYPE_CHECKING, cast

import httpx
import logfire
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .models import Config

_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_RETRY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class _RetryableStatus(Exception):
    """Raised internally to signal that a response's status code warrants a retry."""

    def __init__(self, response: httpx.Response):
        super().__init__(f"Retryable status {response.status_code}")
        self.response = response


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, _RetryableStatus):
        reason = f"HTTP {exc.response.status_code}"
    else:
        reason = type(exc).__name__ if exc else "unknown"
    sleep = retry_state.next_action.sleep if retry_state.next_action else 0.0
    logfire.warning(
        "Retrying request",
        attempt=retry_state.attempt_number,
        sleep=sleep,
        reason=reason,
    )


async def _close_retryable_response(retry_state: RetryCallState) -> None:
    """Close a response that will be discarded before the next attempt."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, _RetryableStatus):
        await exc.response.aclose()
    _log_retry(retry_state)


class RetryTransport(httpx.AsyncBaseTransport):
    """Retry transient failures for requests that are safe to replay."""

    def __init__(self, wrapped: httpx.AsyncBaseTransport, max_retries: int):
        self._wrapped = wrapped
        self._max_retries = max_retries

    def _retrying(self) -> AsyncRetrying:
        # AsyncRetrying keeps mutable per-call state, so do not share one across
        # concurrent requests.
        return AsyncRetrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_random_exponential(multiplier=0.5, max=30),
            retry=retry_if_exception_type((httpx.TransportError, _RetryableStatus)),
            before_sleep=_close_retryable_response,
            reraise=True,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Misskey uses POST for reads as well as writes. Retrying any POST after
        # an ambiguous transport failure could duplicate a note or other action.
        if request.method not in _RETRY_METHODS:
            return await self._wrapped.handle_async_request(request)

        try:
            return await self._retrying()(self._send_once, request)
        except _RetryableStatus as exc:
            # Tenacity only invokes before_sleep when another attempt will run,
            # so the exhausted response remains open for the caller to inspect.
            return exc.response

    async def _send_once(self, request: httpx.Request) -> httpx.Response:
        response = await self._wrapped.handle_async_request(request)
        if response.status_code in _RETRY_STATUS_CODES:
            raise _RetryableStatus(response)
        return response

    async def aclose(self) -> None:
        await self._wrapped.aclose()


class ApiClient:
    def __init__(self):
        self.__async_client: httpx.AsyncClient | None = None
        self.__config: Config | None = None
        self.__lock = threading.Lock()

    @property
    def __client(self) -> httpx.AsyncClient:
        with self.__lock:
            if self.__async_client is None:
                if self.__config:
                    # RetryTransport is the sole retry layer. httpx transport
                    # retries here would compound attempts and bypass method
                    # safety decisions made by the wrapper.
                    base_transport = httpx.AsyncHTTPTransport(retries=0)
                    self.__async_client = httpx.AsyncClient(
                        transport=RetryTransport(base_transport, max_retries=self.__config.max_retries),
                        headers={"Authorization": f"Bearer {self.__config.token}"},
                        timeout=httpx.Timeout(self.__config.http_timeout_seconds),
                    )
                else:
                    logfire.warning("API client accessed before configuration")
                    self.__async_client = httpx.AsyncClient()
        return self.__async_client

    def configure(self, config: Config) -> None:
        with self.__lock:
            self.__config = config
            old_client = self.__async_client if self.__async_client and not self.__async_client.is_closed else None
            if old_client:
                self.__async_client = None

        if old_client:
            try:
                asyncio.get_running_loop().create_task(old_client.aclose())
            except RuntimeError:
                asyncio.run(old_client.aclose())

    def get_client(self) -> httpx.AsyncClient:
        """Gets the underlying AsyncClient"""
        return self.__client

    async def close(self) -> None:
        """Close current client."""
        if self.__async_client and not self.__async_client.is_closed:
            await self.__async_client.aclose()

    def __getattr__(self, name):
        return getattr(self.__client, name)

    # Dunder methods bypass __getattr__, so define explicitly to keep the
    # wrapper usable as a drop-in for httpx.AsyncClient. Lifecycle is managed
    # via configure()/close(), so these are no-ops that preserve `self`.
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


# Create the instance
api_client = ApiClient()

if TYPE_CHECKING:

    class ApiAsyncClient(ApiClient, httpx.AsyncClient): ...

    api_client = cast(ApiAsyncClient, api_client)
