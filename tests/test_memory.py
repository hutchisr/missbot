"""Tests for bot.memory pure-logic helpers (no live Postgres required)."""

from datetime import datetime, timedelta, timezone

from bot.memory import (
    PROMOTABLE_TIERS,
    ConflictingClaim,
    RecalledClaim,
    _vector_literal,
    is_stale,
    normalize_predicate,
    render_claim,
    resolve_conflict,
    tier_rank,
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


def test_quarantine_and_user_tiers_are_not_promotable():
    # The safety core: LLM (model_quarantine) and plain user claims can never be
    # corroborated into 'believed' — only secondary/primary tiers count.
    assert "model_quarantine" not in PROMOTABLE_TIERS
    assert "user" not in PROMOTABLE_TIERS
    assert set(PROMOTABLE_TIERS) == {"secondary", "primary"}
    assert tier_rank("primary") > tier_rank("secondary") > tier_rank("user") > tier_rank("model_quarantine")
    assert tier_rank("nonsense") == 0


def test_is_stale_only_for_volatile_past_ttl():
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=3)
    fresh = now - timedelta(minutes=1)
    assert is_stale("volatile", old, now, 86400) is True
    assert is_stale("volatile", fresh, now, 86400) is False
    assert is_stale("stable", old, now, 86400) is False
    assert is_stale("slow", old, now, 86400) is False
    assert is_stale("volatile", None, now, 86400) is False


def _claim(status: str, tier: str, recorded_at: datetime, object_text: str, valid_from=None) -> dict:
    return {
        "status": status,
        "trust_tier": tier,
        "recorded_at": recorded_at,
        "valid_from": valid_from,
        "object_text": object_text,
    }


def test_resolve_conflict_prefers_believed_over_asserted():
    now = datetime.now(timezone.utc)
    asserted = _claim("asserted", "primary", now, "A")
    believed = _claim("believed", "user", now, "B")  # weaker tier but believed
    assert resolve_conflict([asserted, believed])["object_text"] == "B"


def test_resolve_conflict_breaks_ties_by_trust_then_recency():
    now = datetime.now(timezone.utc)
    older = now - timedelta(days=1)
    # Same status -> higher trust tier wins.
    assert (
        resolve_conflict([_claim("asserted", "user", now, "D"), _claim("asserted", "primary", older, "C")])[
            "object_text"
        ]
        == "C"
    )
    # Same status and tier -> most recent valid_from/recorded_at wins.
    assert (
        resolve_conflict([_claim("asserted", "secondary", older, "E"), _claim("asserted", "secondary", now, "F")])[
            "object_text"
        ]
        == "F"
    )


def test_recalled_claim_carries_provenance():
    now = datetime.now(timezone.utc)
    c = RecalledClaim(
        subject="Python",
        predicate="latest_version",
        object_text="3.13",
        status="believed",
        trust_tier="secondary",
        source_name="python.org",
        source_kind="web",
        confidence=0.9,
        corroboration_count=2,
        volatility="volatile",
        recorded_at=now,
        valid_from=None,
        similarity=0.88,
        stale=False,
        conflicts=[ConflictingClaim("3.12", "blog", "web", "secondary", "asserted")],
    )
    assert c.source_name == "python.org"
    assert c.corroboration_count == 2
    assert c.conflicts[0].object_text == "3.12"
