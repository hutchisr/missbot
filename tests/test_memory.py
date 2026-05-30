"""Tests for bot.memory pure-logic helpers (no live Postgres required)."""

from datetime import datetime, timedelta, timezone

from bot.memory import (
    ConflictingClaim,
    RecalledClaim,
    _vector_literal,
    merge_aliases,
    normalize_entity_name,
    normalize_object,
    normalize_predicate,
    object_group_key,
    render_claim,
    resolve_conflict,
)


def test_vector_literal_formats_floats():
    assert _vector_literal([1, 2.5, -0.3]) == "[1.0,2.5,-0.3]"


def test_vector_literal_empty():
    assert _vector_literal([]) == "[]"


def test_normalize_predicate_canonicalizes():
    assert normalize_predicate("Latest Version!!") == "latest_version"
    assert normalize_predicate("  capital-of ") == "capital_of"
    assert normalize_predicate("") == "fact"
    assert normalize_predicate("!!!") == "fact"


def test_render_claim_humanizes_predicate():
    assert render_claim("Python", "latest_version", "3.13") == "Python — latest version: 3.13"


def _value(object_text: str, agreed_by: int, recency: datetime) -> dict:
    return {"object_text": object_text, "agreed_by": agreed_by, "recency": recency}


def test_resolve_conflict_prefers_more_agreement():
    now = datetime.now(timezone.utc)
    older = now - timedelta(days=1)
    # More distinct users assert "A", so it wins even though "B" is more recent.
    assert resolve_conflict([_value("A", 3, older), _value("B", 1, now)])["object_text"] == "A"


def test_resolve_conflict_breaks_ties_by_recency():
    now = datetime.now(timezone.utc)
    older = now - timedelta(days=1)
    # Equal agreement -> the most recently asserted value wins.
    assert resolve_conflict([_value("old", 2, older), _value("new", 2, now)])["object_text"] == "new"


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


def test_normalize_object_collapses_formatting_variants():
    # Same value, different case/accents/whitespace -> one key (so they agree).
    assert normalize_object("Arch Linux") == normalize_object("arch linux")
    assert normalize_object("Arch  Linux") == normalize_object("Arch Linux")  # collapsed whitespace
    assert normalize_object("  Arch Linux\n") == normalize_object("Arch Linux")  # surrounding whitespace
    assert normalize_object("São Paulo") == normalize_object("Sao Paulo")  # accents stripped
    assert normalize_object("Arch Linux") == "arch linux"


def test_normalize_object_keeps_distinct_values_apart():
    # Deliberately conservative: NO plural fold and NO internal-punctuation stripping, because
    # object values are heterogeneous and a wrong merge miscounts agreement.
    assert normalize_object("3.13") != normalize_object("3 13")  # version punctuation preserved
    assert normalize_object("Windows") != normalize_object("Window")  # plurals NOT folded
    assert normalize_object("v3.13") != normalize_object("3.13")
    assert normalize_object("tribes") != normalize_object("tribe")


def test_normalize_object_handles_empty():
    assert normalize_object("") == ""
    assert normalize_object("   ") == ""


def test_object_group_key_uses_entity_then_text():
    # Linked objects group by entity id (namespaced 'e'); unlinked fall back to the text key ('k').
    assert object_group_key(5, "anything") == "e5"
    assert object_group_key(None, "arch linux") == "karch linux"
    # An entity id and a numeric-looking literal can't collide thanks to the prefixes.
    assert object_group_key(5, "5") != object_group_key(None, "5")
    # Two surface forms linked to the same entity share a key; an unlinked variant does not.
    assert object_group_key(5, "arch linux") == object_group_key(5, "arch")
    assert object_group_key(5, "arch linux") != object_group_key(None, "arch linux")


def test_recalled_claim_shape():
    c = RecalledClaim(
        subject="Python",
        predicate="latest_version",
        object_text="3.13",
        agreed_by=3,
        similarity=0.88,
        conflicts=[ConflictingClaim(object_text="3.12", agreed_by=1)],
    )
    assert c.agreed_by == 3
    assert c.conflicts[0].object_text == "3.12"
    assert c.conflicts[0].agreed_by == 1
