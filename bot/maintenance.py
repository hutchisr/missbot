"""Maintenance & admin CLI for the world-knowledge store.

Runnable out-of-process (so it can be a k8s CronJob, separate from the long-running
bot) via ``python -m bot.maintenance <command> -c /config.yaml``:

    consolidate         merge duplicate entities (name, then embedding, then optional LLM pass)
    stats               print store counts (entities, claims, distinct subjects/authors)
    calibrate-entities  print the most-similar entity pairs to help tune entity_match_threshold

Each command builds a one-shot :class:`MemoryStore`, runs, and closes it. Requires
``memory_enabled: true`` in the config.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import click
import logfire
import yaml

from .ai import build_entity_linker
from .memory import MemoryStore
from .models import Config


def load_config(path: str) -> Config:
    """Load and validate the YAML config (same shape the bot uses)."""
    with open(path, "r") as f:
        return Config(**yaml.safe_load(f))


def _run(config_path: str, action: Callable[[MemoryStore], Awaitable[Any]]) -> Any:
    """Build a MemoryStore from config, run ``action`` against it, then close it.

    The entity linker is wired in so the consolidation LLM merge pass works here in the
    headless CronJob (which has no ChatAgent).
    """

    async def _main() -> Any:
        config = load_config(config_path)
        if not config.memory_enabled:
            raise click.ClickException("memory_enabled must be true in the config to run maintenance")
        store = await MemoryStore.create(config)
        store.entity_linker = build_entity_linker(config)
        try:
            return await action(store)
        finally:
            await store.close()

    return asyncio.run(_main())


_config_option = click.option(
    "-c",
    "--config",
    "config_path",
    default="config.local.yaml",
    show_default=True,
    help="Path to the config file.",
)


@click.group()
def cli() -> None:
    """World-knowledge store maintenance."""
    # Trace maintenance decisions to Logfire (and the console). Sends only when a token
    # is present in the environment, so local runs stay quiet/offline.
    logfire.configure(console=logfire.ConsoleOptions(min_log_level="info"))


@cli.command()
@_config_option
def consolidate(config_path: str) -> None:
    """Merge duplicate entities (name, then embedding, then optional LLM pass)."""
    summary = _run(config_path, lambda store: store.consolidate())
    click.echo(json.dumps(summary))


@cli.command()
@_config_option
def stats(config_path: str) -> None:
    """Print store counts (entities, claims, distinct subjects/authors)."""
    click.echo(json.dumps(_run(config_path, lambda store: store.stats()), indent=2))


@cli.command("calibrate-entities")
@_config_option
@click.option("--limit", default=50, show_default=True, help="Max entity pairs to print.")
@click.option(
    "--min-similarity",
    default=0.5,
    show_default=True,
    type=float,
    help="Only show pairs at or above this cosine similarity.",
)
def calibrate_entities(config_path: str, limit: int, min_similarity: float) -> None:
    """Print the most-similar entity pairs to help tune `entity_match_threshold`.

    Lists each entity's nearest neighbour and their cosine similarity, most-similar
    first. Set the threshold just ABOVE the highest-similarity pair that should stay
    separate (pairs at/above it are linked on write and merged by `consolidate`).
    """
    pairs = _run(config_path, lambda store: store.entity_neighbors(limit, min_similarity))
    current = load_config(config_path).entity_match_threshold
    click.echo(f"current entity_match_threshold = {current}\n")
    if not pairs:
        click.echo(f"No entity pairs at or above similarity {min_similarity}.")
        return
    click.echo("similarity  entity A  <->  entity B")
    for p in pairs:
        marker = " <- below threshold" if p.similarity < current else ""
        click.echo(f"{p.similarity:.4f}      {p.name_a}  <->  {p.name_b}{marker}")


if __name__ == "__main__":
    cli()
