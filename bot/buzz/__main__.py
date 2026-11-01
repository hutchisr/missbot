"""Run the hosted Buzz relay agent and its private management endpoint."""

import asyncio
import os
from contextlib import suppress

import click
from nostr_sdk import Keys

from ..acp.__main__ import _bearer_token_from_env, _configure_logging, _install_shutdown, _load_config, _runtime
from ..ai import ChatAgent
from .server import serve_hosted
from .service import HostedBuzz
from .store import BuzzStore


@click.command()
@click.option("--config", "-c", default="config.local.yaml", help="Server configuration, never sent to Buzz.")
@click.option("--state", default="buzz.sqlite3", help="Persistent SQLite state file.")
@click.option("--community", multiple=True, required=True, help="Allowed Buzz community origin; repeat to allow more.")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8080, type=int)
@click.option("--token-env", default="BUZZ_TOKEN", help="Environment variable containing the management token.")
def main(config, state, community, host, port, token_env):
    """Run Grok in Buzz without a desktop runner; reuse its configured Nostr key."""
    asyncio.run(run(config, state, list(community), host, port, token_env))


async def run(config_path: str, state: str, communities: list[str], host: str, port: int, token_env: str) -> None:
    config = _load_config(config_path)
    _configure_logging(bool(config.debug))
    token = _bearer_token_from_env(token_env)
    assert token is not None
    try:
        keys = Keys.parse(os.environ[config.nostr_private_key_env].strip())
    except Exception:
        raise click.ClickException("The configured Nostr private key is missing or invalid") from None
    store = BuzzStore(state)
    try:
        async with (
            _runtime(config) as (redis, memory),
            ChatAgent(config, redis_client=redis, memory=memory) as agent,
        ):
            service = HostedBuzz(config, agent, store, keys, communities)
            stop = asyncio.Event()
            _install_shutdown(stop)
            await service.start()
            server = asyncio.create_task(serve_hosted(service, host=host, port=port, token=token))
            stopping = asyncio.create_task(stop.wait())
            try:
                done, _ = await asyncio.wait({server, stopping}, return_when=asyncio.FIRST_COMPLETED)
                if server in done:
                    server.result()
            finally:
                server.cancel()
                stopping.cancel()
                with suppress(asyncio.CancelledError):
                    await server
                with suppress(asyncio.CancelledError):
                    await stopping
                await service.close()
    finally:
        store.close()


if __name__ == "__main__":
    main()
