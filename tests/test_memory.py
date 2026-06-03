"""Tests for bot.memory pure-logic helpers (no live Postgres required)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.memory import (
    ConflictingClaim,
    MemoryStore,
    RecalledClaim,
    _vector_literal,
    merge_aliases,
    normalize_entity_name,
    normalize_predicate,
    normalize_value,
    render_relation,
    resolve_conflict,
    value_group_key,
)


def _store_with_fake_http(config) -> tuple[MemoryStore, MagicMock]:
    """A MemoryStore whose embeddings endpoint is a mock (embed() touches http only, not the pool)."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": [{"embedding": [0.1, 0.2], "index": 0}]})
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)
    return MemoryStore(pool=MagicMock(), http=http, config=config), http


def test_vector_literal_formats_floats():
    assert _vector_literal([1, 2.5, -0.3]) == "[1.0,2.5,-0.3]"


def test_vector_literal_empty():
    assert _vector_literal([]) == "[]"


def test_normalize_predicate_canonicalizes():
    assert normalize_predicate("Latest Version!!") == "latest version"
    assert normalize_predicate("  capital-of ") == "capital of"
    assert normalize_predicate("latest_version") == "latest version"  # legacy snake_case folds in
    assert normalize_predicate("") == "fact"
    assert normalize_predicate("!!!") == "fact"


def test_render_relation_is_subject_predicate_question():
    assert render_relation("Python", "latest version") == "Python — latest version"


def _value(value_text: str, agreed_by: int, recency: datetime) -> dict:
    return {"value_text": value_text, "agreed_by": agreed_by, "recency": recency}


def test_resolve_conflict_prefers_more_agreement():
    now = datetime.now(timezone.utc)
    older = now - timedelta(days=1)
    # More distinct users assert "A", so it wins even though "B" is more recent.
    assert resolve_conflict([_value("A", 3, older), _value("B", 1, now)])["value_text"] == "A"


def test_resolve_conflict_breaks_ties_by_recency():
    now = datetime.now(timezone.utc)
    older = now - timedelta(days=1)
    # Equal agreement -> the most recently asserted value wins.
    assert resolve_conflict([_value("old", 2, older), _value("new", 2, now)])["value_text"] == "new"


def test_merge_aliases_unions_dedupes_and_drops_keeper_name():
    out = merge_aliases("Python", ["py"], "python", ["CPython", "py", "Python3"])
    # dup canonical "python" == keeper's name (case-insensitive) -> dropped; "py" deduped.
    assert out == ["py", "CPython", "Python3"]


def test_merge_aliases_from_empty_keeper():
    assert merge_aliases("X", [], "Y", []) == ["Y"]


def test_normalize_entity_name():
    assert normalize_entity_name("@anemone") == "anemone"
    assert normalize_entity_name("Hòabìnhian") == "hoabinhian"  # accents stripped
    assert normalize_entity_name("PGG.Han") == "pgg han"  # punctuation -> space
    assert normalize_entity_name("Cordillerans") == "cordilleran"  # plural fold
    assert normalize_entity_name("class") == "class"  # 'ss' is not folded


def test_normalize_entity_name_collapses_only_formatting_differences():
    # Same name, different formatting/accent/punctuation -> same key (safe to merge).
    assert normalize_entity_name("@anemone") == normalize_entity_name("anemone")
    assert normalize_entity_name("Hòabìnhian") == normalize_entity_name("Hoabinhian")
    assert normalize_entity_name("PGG.Han") == normalize_entity_name("PGG Han")
    # Granularity differences are NOT collapsed (no generic-word stripping).
    assert normalize_entity_name("Tausug") != normalize_entity_name("Tausug people")
    assert normalize_entity_name("Hoabinhian") != normalize_entity_name("Hoabinhian culture")


def test_normalize_entity_name_keeps_distinct_entities_apart():
    # The false-merge traps the rule must avoid (validated on real store data).
    assert normalize_entity_name("Philippines") != normalize_entity_name("Filipinos")
    assert normalize_entity_name("Native Americans") != normalize_entity_name("Amazonian Native Americans")
    assert normalize_entity_name("early Austronesians") != normalize_entity_name("early Austronesians (Taiwan)")
    assert normalize_entity_name("Larena et al.") != normalize_entity_name("Maximilian Larena")  # paper vs author


def test_normalize_value_collapses_formatting_variants():
    # Same value, different case/accents/whitespace -> one key (so they agree).
    assert normalize_value("Arch Linux") == normalize_value("arch linux")
    assert normalize_value("Arch  Linux") == normalize_value("Arch Linux")  # collapsed whitespace
    assert normalize_value("  Arch Linux\n") == normalize_value("Arch Linux")  # surrounding whitespace
    assert normalize_value("São Paulo") == normalize_value("Sao Paulo")  # accents stripped
    assert normalize_value("Arch Linux") == "arch linux"


def test_normalize_value_keeps_distinct_values_apart():
    # Deliberately conservative: NO plural fold and NO internal-punctuation stripping, because
    # claim values are heterogeneous and a wrong merge miscounts agreement.
    assert normalize_value("3.13") != normalize_value("3 13")  # version punctuation preserved
    assert normalize_value("Windows") != normalize_value("Window")  # plurals NOT folded
    assert normalize_value("v3.13") != normalize_value("3.13")
    assert normalize_value("tribes") != normalize_value("tribe")


def test_normalize_value_handles_empty():
    assert normalize_value("") == ""
    assert normalize_value("   ") == ""


def test_value_group_key_uses_entity_then_text():
    # Relationship claims group by destination entity id (namespaced 'e'); attribute claims fall
    # back to the normalized literal key ('k').
    assert value_group_key(5, "anything") == "e5"
    assert value_group_key(None, "arch linux") == "karch linux"
    # An entity id and a numeric-looking literal can't collide thanks to the prefixes.
    assert value_group_key(5, "5") != value_group_key(None, "5")
    # Two surface forms linked to the same entity share a key; an unlinked variant does not.
    assert value_group_key(5, "arch linux") == value_group_key(5, "arch")
    assert value_group_key(5, "arch linux") != value_group_key(None, "arch linux")


@pytest.mark.anyio
async def test_embed_sends_dimensions_when_configured(make_config):
    cfg = make_config(
        memory_enabled=True,
        postgres_url="postgres://u:p@db/x",
        embedding_model="perplexity/pplx-embed-v1-4b",
        embedding_dim=1024,
        embedding_dimensions=1024,
    )
    store, http = _store_with_fake_http(cfg)
    await store.embed("hi")
    body = http.post.call_args.kwargs["json"]
    assert body["dimensions"] == 1024
    assert body["input"] == "hi"


@pytest.mark.anyio
async def test_embed_omits_dimensions_when_unset(make_config):
    store, http = _store_with_fake_http(make_config())  # embedding_dimensions defaults to None
    await store.embed("hi")
    assert "dimensions" not in http.post.call_args.kwargs["json"]


@pytest.mark.anyio
async def test_embed_batch_sends_dimensions_and_list_input(make_config):
    cfg = make_config(
        memory_enabled=True,
        postgres_url="postgres://u:p@db/x",
        embedding_model="perplexity/pplx-embed-v1-4b",
        embedding_dim=1024,
        embedding_dimensions=1024,
    )
    store, http = _store_with_fake_http(cfg)
    http.post.return_value.json.return_value = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}
    await store.embed_batch(["one"])
    body = http.post.call_args.kwargs["json"]
    assert body["dimensions"] == 1024
    assert body["input"] == ["one"]


def test_recalled_claim_shape():
    c = RecalledClaim(
        subject="Python",
        predicate="latest version",
        value_text="3.13",
        agreed_by=3,
        similarity=0.88,
        conflicts=[ConflictingClaim(value_text="3.12", agreed_by=1)],
    )
    assert c.agreed_by == 3
    assert c.conflicts[0].value_text == "3.12"
    assert c.conflicts[0].agreed_by == 1
