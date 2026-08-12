import asyncio
import logging
import signal

import click
import logfire
import yaml
from logfire.integrations.logging import LogfireLoggingHandler
from redis.asyncio import Redis

from .api import api_client
from .bot import Bot
from .memory import MemoryStore
from .models import Config


@click.command()
@click.option("--config", "-c", default="config.local.yaml", help="Path to the config file.")
def main(config: str) -> None:
    with open(config, encoding="utf-8") as f:
        loaded_config = Config(**yaml.safe_load(f))
    asyncio.run(main_async(loaded_config))


async def main_async(config: Config) -> None:
    loop = asyncio.get_running_loop()

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
    logfire.instrument_redis(capture_statement=True)
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
    redis_client: Redis | None = None
    if config.redis_url:
        redis_client = Redis.from_url(
            config.redis_url,
            password=config.redis_password,
            db=config.redis_db or 0,
            decode_responses=True,  # Get strings instead of bytes
        )
        logfire.info("Redis client initialized")

    # Initialize the long-term memory store (Postgres + pgvector) if enabled
    memory: MemoryStore | None = None
    if config.memory_enabled:
        memory = await MemoryStore.create(config)

    bot = Bot(
        config=config,
        redis_client=redis_client,
        memory=memory,
    )

    def shutdown_handler():
        logfire.info("Shutting down")
        bot.shutdown()
        loop.create_task(api_client.close())
        if redis_client:
            loop.create_task(redis_client.aclose())
        if memory:
            loop.create_task(memory.close())

    loop.add_signal_handler(signal.SIGTERM, shutdown_handler)
    loop.add_signal_handler(signal.SIGINT, shutdown_handler)

    await bot.run()
