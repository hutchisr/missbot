"""Run Missbot as an ACP agent.

Two modes:

- ``python -m bot.acp stdio`` — JSON-RPC over stdin/stdout. Clients that spawn agents
  as subprocesses (Zed, JetBrains) use this.
- ``python -m bot.acp serve`` — the same agent over WebSocket, wire-compatible with
  ``acpremote mirror``. Deploy this and let remote consumers connect:

  ```
  Buzz Relay ──WS──→ buzz-acp ──stdio──→ acpremote mirror ──WS──→ [here]
  ```

**stdout is the protocol channel in stdio mode.** A single stray print or log line
corrupts the JSON-RPC stream, so every diagnostic is routed to stderr — unlike
`bot/cli.py`, where logfire's console exporter defaults to stdout. `serve` mode keeps
the same routing for consistency.
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

import acp
import click
import logfire
import yaml
from logfire.integrations.logging import LogfireLoggingHandler
from redis.asyncio import Redis

from ..ai import ChatAgent
from ..api import api_client
from ..memory import MemoryStore
from ..models import Config
from .agent import MissbotAgent
from .ws import DEFAULT_MOUNT_PATH, serve_acp_websocket

_CONFIG_OPTION = click.option("--config", "-c", default="config.local.yaml", help="Path to the config file.")


@click.group()
def main():
    """Missbot's ACP frontend."""


def _configure_logging(debug_enabled: bool) -> None:
    """Send all diagnostics to stderr, leaving stdout clear for JSON-RPC frames."""
    min_level = "debug" if debug_enabled else "info"
    logfire.configure(
        min_level=min_level,
        console=logfire.ConsoleOptions(
            min_log_level=min_level,
            verbose=debug_enabled,
            include_timestamps=True,
            output=sys.stderr,
        ),
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()
    logfire.instrument_redis(capture_statement=True)
    logging.basicConfig(
        level=logging.DEBUG if debug_enabled else logging.INFO,
        handlers=[LogfireLoggingHandler(), logging.StreamHandler(sys.stderr)],
    )
    # Same muting as bot/cli.py: these are covered by logfire spans, or too noisy.
    for _name in ("httpx", "httpcore", "redis"):
        logging.getLogger(_name).setLevel(logging.CRITICAL + 1)
    for _name in ("websockets", "mcp.client.streamable_http"):
        logging.getLogger(_name).setLevel(logging.INFO)


def _load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        return Config(**yaml.safe_load(f))


@asynccontextmanager
async def _runtime(config: Config):
    """Open the shared backends and tear them down on exit."""
    api_client.configure(config)

    redis_client: Optional[Redis] = None
    if config.redis_url:
        redis_client = Redis.from_url(
            config.redis_url,
            password=config.redis_password,
            db=config.redis_db or 0,
            decode_responses=True,
        )
        logfire.info("Redis client initialized")

    memory: Optional[MemoryStore] = None
    if config.memory_enabled:
        memory = await MemoryStore.create(config)

    try:
        yield redis_client, memory
    finally:
        await api_client.close()
        if redis_client:
            await redis_client.aclose()
        if memory:
            await memory.close()


def _install_shutdown(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)


@main.command()
@_CONFIG_OPTION
def stdio(config):
    """Serve ACP over stdin/stdout (clients spawn this as a subprocess)."""
    asyncio.run(_stdio_async(config))


async def _stdio_async(config_path):
    config = _load_config(config_path)
    _configure_logging(bool(config.debug))

    async with _runtime(config) as (redis_client, memory):
        agent = MissbotAgent(config=config, redis_client=redis_client, memory=memory)
        logfire.info(
            "Starting ACP agent on stdio",
            bot_username=config.bot_username,
            max_sessions=config.acp_max_sessions,
            parse_sender_header=config.acp_parse_sender_header,
        )
        stop = asyncio.Event()
        _install_shutdown(stop)
        async with agent:
            serve_task = asyncio.create_task(acp.run_agent(agent))
            shutdown = asyncio.create_task(stop.wait())
            done, pending = await asyncio.wait({serve_task, shutdown}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            # Surface a transport failure; a clean stdin EOF just ends the process.
            if serve_task in done:
                serve_task.result()


@main.command()
@_CONFIG_OPTION
@click.option("--host", default="127.0.0.1", help="Bind address. Use 0.0.0.0 in a container.")
@click.option("--port", default=8080, type=int, help="Bind port.")
@click.option("--mount-path", default=DEFAULT_MOUNT_PATH, help="Route prefix; the socket is <mount>/ws.")
@click.option("--token-env", default=None, help="Env var holding the bearer token clients must send.")
def serve(config, host, port, mount_path, token_env):
    """Serve ACP over WebSocket for `acpremote mirror` and other remote clients."""
    asyncio.run(_serve_async(config, host, port, mount_path, token_env))


async def _serve_async(config_path, host, port, mount_path, token_env):
    config = _load_config(config_path)
    _configure_logging(bool(config.debug))

    bearer_token = os.environ.get(token_env) if token_env else None
    if token_env and not bearer_token:
        raise click.ClickException(f"--token-env {token_env} is set but that variable is empty or missing")

    async with _runtime(config) as (redis_client, memory):
        # One ChatAgent backs every connection — it owns the model chain, MCP sessions,
        # Redis, and memory. Each connection gets its own thin protocol adapter, because
        # MissbotAgent holds the client handle it pushes session updates through.
        chat_agent = ChatAgent(config, redis_client=redis_client, memory=memory)
        stop = asyncio.Event()
        _install_shutdown(stop)
        async with chat_agent:
            serve_task = asyncio.create_task(
                serve_acp_websocket(
                    lambda: MissbotAgent(config=config, chat_agent=chat_agent),
                    host=host,
                    port=port,
                    mount_path=mount_path,
                    bearer_token=bearer_token,
                )
            )
            shutdown = asyncio.create_task(stop.wait())
            done, pending = await asyncio.wait({serve_task, shutdown}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if serve_task in done:
                serve_task.result()


if __name__ == "__main__":
    main()
