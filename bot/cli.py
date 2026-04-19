import asyncio
import logging
import signal
from typing import Optional

import click
import logfire
import yaml
from logfire.integrations.logging import LogfireLoggingHandler
from redis.asyncio import Redis

from .bot import Bot
from .models import Config
from .api import api_client


@click.command()
@click.option("--config", "-c", default="config.local.yaml", help="Path to the config file.")
def main(config):
    asyncio.run(main_async(config))


async def main_async(config):
    loop = asyncio.get_running_loop()

    with open(config, "r") as f:
        config = Config(**yaml.safe_load(f))

    debug_enabled = bool(config.debug)
    min_level = "debug" if debug_enabled else "info"
    logfire.configure(
        min_level=min_level,
        console=logfire.ConsoleOptions(
            min_log_level=min_level,
            verbose=debug_enabled,
            include_timestamps=True,
        ),
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx()
    logfire.instrument_redis()
    logging.basicConfig(
        level=logging.DEBUG if debug_enabled else logging.INFO,
        handlers=[LogfireLoggingHandler()],
    )
    # httpx/httpcore/redis are covered by logfire.instrument_* spans, so their
    # stdlib logs (at any level) duplicate span data — mute them entirely.
    for _name in ("httpx", "httpcore", "redis"):
        logging.getLogger(_name).setLevel(logging.CRITICAL + 1)
    # websockets and mcp have no logfire instrumentation; keep INFO so
    # connect/negotiate events are visible, but drop DEBUG (websockets.protocol
    # ships bytearray Frame data that crashes logfire's console exporter).
    for _name in ("websockets", "mcp.client.streamable_http"):
        logging.getLogger(_name).setLevel(logging.INFO)

    # Configure global HTTP client with config settings
    api_client.configure(config)

    # Initialize Redis client if configured
    redis_client: Optional[Redis] = None
    if config.redis_url:
        redis_client = Redis.from_url(
            config.redis_url,
            password=config.redis_password,
            db=config.redis_db or 0,
            decode_responses=True,  # Get strings instead of bytes
        )
        logfire.info("Redis client initialized")

    bot = Bot(
        config=config,
        redis_client=redis_client,
    )

    def shutdown_handler():
        logfire.info("Shutting down")
        bot.shutdown()
        loop.create_task(api_client.close())
        if redis_client:
            loop.create_task(redis_client.aclose())

    loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
    loop.add_signal_handler(signal.SIGINT, shutdown_handler)

    await bot.run()
