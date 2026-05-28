"""Maintenance & admin CLI for the world-knowledge store (M4/M5).

Runnable out-of-process (so it can be a k8s CronJob, separate from the long-running
bot) via ``python -m bot.maintenance <command> -c /config.yaml``:

    consolidate            merge duplicate entities + recompute corroboration
    detect-contradictions  flag same-subject+predicate disagreements as `disputed`
    resolve-disputes       resolve disputes past the grace period (supersede losing values)
    decay                  soft-retract stale, never-recalled, low-trust claims
    run-all                consolidate -> detect -> resolve disputes -> decay
    retract-source         tombstone every claim from a source (and recompute)
    unmerge                reverse a soft entity-merge (reactivate + restore claims)
    stats                  print store counts (entities/sources/claims by status & tier)

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
    """Merge duplicate entities and recompute corroboration across the store."""
    summary = _run(config_path, lambda store: store.consolidate())
    click.echo(json.dumps(summary))


@cli.command("detect-contradictions")
@_config_option
def detect_contradictions(config_path: str) -> None:
    """Flag same-subject+predicate disagreements as `disputed` (and clear resolved ones)."""
    summary = _run(config_path, lambda store: store.detect_contradictions())
    click.echo(json.dumps(summary))


@cli.command("resolve-disputes")
@_config_option
def resolve_disputes(config_path: str) -> None:
    """Autonomously resolve contradictions older than the grace period (supersede losers)."""
    summary = _run(config_path, lambda store: store.resolve_disputes())
    click.echo(json.dumps(summary))


@cli.command()
@_config_option
def decay(config_path: str) -> None:
    """Soft-retract stale, never-recalled, low-trust claims (autonomous pruning)."""
    summary = _run(config_path, lambda store: store.decay())
    click.echo(json.dumps(summary))


@cli.command("run-all")
@_config_option
def run_all(config_path: str) -> None:
    """Run the full scheduled pass: consolidate, detect, resolve disputes, then decay."""

    async def _action(store: MemoryStore) -> dict[str, Any]:
        return {
            "consolidate": await store.consolidate(),
            "detect_contradictions": await store.detect_contradictions(),
            "resolve_disputes": await store.resolve_disputes(),
            "decay": await store.decay(),
        }

    click.echo(json.dumps(_run(config_path, _action)))


@cli.command("retract-source")
@_config_option
@click.option("--name", required=True, help="Source name to retract (e.g. an author handle or web domain).")
@click.option("--kind", default=None, help="Optional source kind (web|doc|user|model) to disambiguate.")
def retract_source(config_path: str, name: str, kind: str | None) -> None:
    """Tombstone every claim from a source and recompute corroboration everywhere."""
    count = _run(config_path, lambda store: store.retract_source(name, kind))
    click.echo(f"Retracted {count} claim(s) from source {name!r}" + (f" (kind={kind})" if kind else ""))


@cli.command()
@_config_option
@click.option("--id", "merged_id", required=True, type=int, help="Entity id to un-merge (reactivate).")
def unmerge(config_path: str, merged_id: int) -> None:
    """Reverse a soft entity-merge: reactivate the entity and move its claims back."""
    ok = _run(config_path, lambda store: store.unmerge_entity(merged_id))
    click.echo(f"Un-merged entity {merged_id}" if ok else f"Nothing to un-merge for entity {merged_id}")


@cli.command()
@_config_option
def stats(config_path: str) -> None:
    """Print store counts (entities, sources, live claims by status and tier)."""
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
