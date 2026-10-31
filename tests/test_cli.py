"""Tests for bot.cli and bot.__main__."""

import runpy
import signal
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
import yaml
from click.testing import CliRunner

from bot import cli


def _config_yaml(config, **overrides) -> str:
    data = config.model_dump(mode="json")
    data.update(overrides)
    return yaml.safe_dump(data)


def test_main_loads_config_and_invokes_asyncio_run(config):
    runner = CliRunner()

    with (
        patch("builtins.open", mock_open(read_data=_config_yaml(config))) as open_mock,
        patch.object(cli, "main_async", new=MagicMock(return_value="main-coro")) as main_async_mock,
        patch.object(cli.asyncio, "run") as run_mock,
    ):
        result = runner.invoke(cli.main, ["-c", "config.test.yaml"])

    assert result.exit_code == 0
    open_mock.assert_called_once_with("config.test.yaml", encoding="utf-8")
    loaded_config = main_async_mock.call_args.args[0]
    assert loaded_config.token == config.token
    run_mock.assert_called_once_with("main-coro")


@pytest.mark.anyio
@pytest.mark.parametrize("fail", [False, True])
async def test_backends_close_only_after_frontends_stop(make_config, fail):
    config = make_config(redis_url="redis://redis.test/0", memory_enabled=True)
    events: list[str] = []
    handlers = {}
    loop = cli.asyncio.get_running_loop()
    bot = MagicMock()
    redis = MagicMock()
    memory = MagicMock()
    redis.aclose = AsyncMock(side_effect=lambda: events.append("redis-closed"))
    memory.close = AsyncMock(side_effect=lambda: events.append("memory-closed"))

    async def run():
        handlers[signal.SIGTERM]()
        assert events == []  # Signal requests shutdown; resources remain usable while draining.
        bot.shutdown.assert_called_once()
        events.append("frontends-stopped")
        if fail:
            raise RuntimeError("frontend failure")

    bot.run = run
    with (
        patch.object(loop, "add_signal_handler", side_effect=lambda sig, handler: handlers.setdefault(sig, handler)),
        patch.object(loop, "remove_signal_handler"),
        patch.object(cli.logfire, "configure"),
        patch.object(cli.logfire, "instrument_pydantic_ai"),
        patch.object(cli.logfire, "instrument_httpx"),
        patch.object(cli.logfire, "instrument_redis"),
        patch.object(cli.logging, "basicConfig"),
        patch.object(cli.api_client, "configure"),
        patch.object(cli.api_client, "close", new=AsyncMock(side_effect=lambda: events.append("http-closed"))),
        patch.object(cli.Redis, "from_url", return_value=redis),
        patch.object(cli.MemoryStore, "create", new=AsyncMock(return_value=memory)),
        patch.object(cli, "Bot", return_value=bot),
    ):
        if fail:
            with pytest.raises(RuntimeError, match="frontend failure"):
                await cli.main_async(config)
        else:
            await cli.main_async(config)
    assert events == ["frontends-stopped", "http-closed", "redis-closed", "memory-closed"]


def test_package_main_calls_cli_main():
    with patch("bot.cli.main") as main_mock:
        runpy.run_module("bot.__main__", run_name="__main__")

    main_mock.assert_called_once_with()
