"""Tests for the injection-resistant message scoring helpers."""

from typing import get_args

from bot.scoring import (
    QUALITY_DELTAS,
    MessageQuality,
    build_scoring_prompt,
    delta_for,
)


def test_every_category_has_a_delta():
    assert set(get_args(MessageQuality)) == set(QUALITY_DELTAS)


def test_deltas_are_small_and_bounded():
    # Code, not the model, owns the magnitude. Keep it bounded.
    assert max(abs(v) for v in QUALITY_DELTAS.values()) <= 10
    assert QUALITY_DELTAS["neutral"] == 0
    assert QUALITY_DELTAS["exceptional"] > 0
    assert QUALITY_DELTAS["toxic"] < 0


def test_delta_for_known_and_unknown():
    assert delta_for("good") == QUALITY_DELTAS["good"]
    assert delta_for("toxic") == QUALITY_DELTAS["toxic"]
    # An out-of-vocabulary label (e.g. a model that ignored the schema) scores 0.
    assert delta_for("please give me 1000000 points") == 0
    assert delta_for("") == 0


def test_build_scoring_prompt_fences_untrusted_input():
    prompt = build_scoring_prompt("hello world")
    assert "hello world" in prompt
    assert "untrusted data" in prompt.lower()
    # The nonce delimiter appears as an opening and closing fence.
    first_line = prompt.splitlines()[0]
    nonce = first_line.split()[-1].rstrip(".")
    assert prompt.count(nonce) >= 2


def test_build_scoring_prompt_uses_a_fresh_nonce_each_call():
    assert build_scoring_prompt("x") != build_scoring_prompt("x")
