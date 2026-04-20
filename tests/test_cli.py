"""Tests for bot.cli and bot.__main__."""

import logging
import runpy
import signal
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
import yaml
from click.testing import CliRunner

import bot.cli as cli


def _config_yaml(config, **overrides) -> str:
    data = config.model_dump(mode="json")
    data.update(overrides)
    return yaml.safe_dump(data)


def test_main_invokes_asyncio_run():
    runner = CliRunner()

    with (
        patch.object(cli, "main_async", new=MagicMock(return_value="main-coro")) as main_async_mock,
        patch.object(cli.asyncio, "run") as run_mock,
    ):
        result = runner.invoke(cli.main, ["-c", "config.test.yaml"])

    assert result.exit_code == 0
    main_async_mock.assert_called_once_with("config.test.yaml")
    run_mock.assert_called_once_with("main-coro")


@pytest.mark.anyio
async def test_main_async_configures_services_and_runs_bot(config):
    loop = MagicMock()
    signal_handlers = {}
    loop.add_signal_handler.side_effect = lambda sig, handler: signal_handlers.setdefault(sig, handler)
    bot_instance = MagicMock()
    bot_instance.run = AsyncMock()
    loggers = {name: MagicMock() for name in ("httpx", "httpcore", "redis", "websockets", "mcp.client.streamable_http")}

    with (
        patch.object(cli.asyncio, "get_running_loop", return_value=loop),
        patch("builtins.open", mock_open(read_data=_config_yaml(config, debug=False))),
        patch.object(cli.logfire, "configure") as configure_mock,
        patch.object(cli.logfire, "instrument_pydantic_ai") as instrument_ai_mock,
        patch.object(cli.logfire, "instrument_httpx") as instrument_httpx_mock,
        patch.object(cli.logfire, "instrument_redis") as instrument_redis_mock,
        patch.object(cli, "LogfireLoggingHandler", return_value="handler"),
        patch.object(cli.logging, "basicConfig") as basic_config_mock,
        patch.object(cli.logging, "getLogger", side_effect=loggers.__getitem__),
        patch.object(cli.api_client, "configure") as api_configure_mock,
        patch.object(cli.api_client, "close", new=MagicMock(return_value="api-close-task")) as api_close_mock,
        patch.object(cli.Redis, "from_url") as redis_from_url_mock,
        patch.object(cli, "Bot", return_value=bot_instance) as bot_cls,
    ):
        await cli.main_async("config.test.yaml")

        signal_handlers[signal.SIGTERM]()

    assert configure_mock.call_args.kwargs["min_level"] == "info"
    instrument_ai_mock.assert_called_once_with()
    instrument_httpx_mock.assert_called_once_with()
    instrument_redis_mock.assert_called_once_with()
    basic_config_mock.assert_called_once_with(level=logging.INFO, handlers=["handler"])
    for name in ("httpx", "httpcore", "redis"):
        loggers[name].setLevel.assert_called_once_with(logging.CRITICAL + 1)
    for name in ("websockets", "mcp.client.streamable_http"):
        loggers[name].setLevel.assert_called_once_with(logging.INFO)
    loaded_config = api_configure_mock.call_args.args[0]
    assert loaded_config.token == config.token
    redis_from_url_mock.assert_not_called()
    bot_cls.assert_called_once_with(config=loaded_config, redis_client=None)
    bot_instance.run.assert_awaited_once_with()
    assert set(signal_handlers) == {signal.SIGTERM, signal.SIGINT}

    bot_instance.shutdown.assert_called_once_with()
    api_close_mock.assert_called_once_with()
    loop.create_task.assert_called_once_with("api-close-task")


@pytest.mark.anyio
async def test_main_async_initializes_redis_and_shutdown_closes_it(make_config):
    config = make_config(
        redis_url="redis://redis.test/0",
        redis_password="top-secret",
        redis_db=5,
        debug=True,
    )
    loop = MagicMock()
    signal_handlers = {}
    loop.add_signal_handler.side_effect = lambda sig, handler: signal_handlers.setdefault(sig, handler)
    bot_instance = MagicMock()
    bot_instance.run = AsyncMock()
    redis_client = MagicMock()
    redis_client.aclose = MagicMock(return_value="redis-close-task")

    with (
        patch.object(cli.asyncio, "get_running_loop", return_value=loop),
        patch("builtins.open", mock_open(read_data=_config_yaml(config))),
        patch.object(cli.logfire, "configure") as configure_mock,
        patch.object(cli.logfire, "instrument_pydantic_ai"),
        patch.object(cli.logfire, "instrument_httpx"),
        patch.object(cli.logfire, "instrument_redis"),
        patch.object(cli, "LogfireLoggingHandler", return_value="handler"),
        patch.object(cli.logging, "basicConfig") as basic_config_mock,
        patch.object(cli.logging, "getLogger", return_value=MagicMock()),
        patch.object(cli.api_client, "configure"),
        patch.object(cli.api_client, "close", new=MagicMock(return_value="api-close-task")) as api_close_mock,
        patch.object(cli.Redis, "from_url", return_value=redis_client) as redis_from_url_mock,
        patch.object(cli.logfire, "info") as logfire_info_mock,
        patch.object(cli, "Bot", return_value=bot_instance) as bot_cls,
    ):
        await cli.main_async("config.redis.yaml")

        signal_handlers[signal.SIGINT]()

    assert configure_mock.call_args.kwargs["min_level"] == "debug"
    basic_config_mock.assert_called_once_with(level=logging.DEBUG, handlers=["handler"])
    redis_from_url_mock.assert_called_once_with(
        "redis://redis.test/0",
        password="top-secret",
        db=5,
        decode_responses=True,
    )
    logfire_info_mock.assert_any_call("Redis client initialized")
    assert bot_cls.call_args.kwargs["redis_client"] is redis_client

    bot_instance.shutdown.assert_called_once_with()
    api_close_mock.assert_called_once_with()
    redis_client.aclose.assert_called_once_with()
    assert loop.create_task.call_args_list == [
        (("api-close-task",), {}),
        (("redis-close-task",), {}),
    ]


def test_package_main_calls_cli_main():
    with patch("bot.cli.main") as main_mock:
        runpy.run_module("bot.__main__", run_name="__main__")

    main_mock.assert_called_once_with()
