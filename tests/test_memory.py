"""Tests for bot.memory pure-logic helpers (no live Postgres required)."""

from bot.memory import GlobalFact, _vector_literal


def test_vector_literal_formats_floats():
    assert _vector_literal([1, 2.5, -0.3]) == "[1.0,2.5,-0.3]"


def test_vector_literal_empty():
    assert _vector_literal([]) == "[]"


def test_global_fact_fields():
    f = GlobalFact(fact="the sky is blue", similarity=0.87)
    assert f.fact == "the sky is blue"
    assert f.similarity == 0.87
