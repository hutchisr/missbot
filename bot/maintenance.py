"""Maintenance & admin CLI for the world-knowledge store.

Runnable out-of-process (so it can be a k8s CronJob, separate from the long-running
bot) via ``python -m bot.maintenance <command> -c /config.yaml``:

    consolidate         merge duplicate entities (name/embedding/LLM), then backfill relationship links
    stats               print store counts (entities, claims, distinct sources/authors)
    reembed             regenerate ALL embeddings with the configured model (model swap / seed cleanup)
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


def _run(config_path: str, action: Callable[[MemoryStore], Awaitable[Any]], *, skip_dim_check: bool = False) -> Any:
    """Build a MemoryStore from config, run ``action`` against it, then close it.

    The entity linker is wired in so the consolidation LLM merge pass works here in the
    headless CronJob (which has no ChatAgent). ``skip_dim_check`` is for the ``reembed`` command,
    which is about to re-dimension and repopulate the vectors, so the startup dimension guard
    (which would otherwise abort on a model/dim change) must be suppressed.
    """

    async def _main() -> Any:
        config = load_config(config_path)
        if not config.memory_enabled:
            raise click.ClickException("memory_enabled must be true in the config to run maintenance")
        store = await MemoryStore.create(config, skip_dim_check=skip_dim_check)
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
    """Merge duplicate entities (name/embedding/LLM), then backfill stale relationship links."""
    summary = _run(config_path, lambda store: store.consolidate())
    click.echo(json.dumps(summary))


@cli.command()
@_config_option
def stats(config_path: str) -> None:
    """Print store counts (entities, claims, distinct sources/authors)."""
    click.echo(json.dumps(_run(config_path, lambda store: store.stats()), indent=2))


@cli.command()
@_config_option
@click.option("--batch-size", default=64, show_default=True, help="Embeddings requested per API call.")
def reembed(config_path: str, batch_size: int) -> None:
    """Regenerate ALL embeddings (entity names + relation questions) with the configured model.

    Use after changing `embedding_model`, or to upgrade migration-seeded relation vectors to clean
    question vectors. Re-embeds in place and is resumable. If the model's vector dimension changed,
    it re-dimensions the columns first — note the bot itself fails to start on a dimension mismatch,
    so for a dim change run this while the bot is down (it can start once embeddings are rebuilt).
    """
    summary = _run(config_path, lambda store: store.reembed_all(batch_size=batch_size), skip_dim_check=True)
    click.echo(json.dumps(summary, indent=2))


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
