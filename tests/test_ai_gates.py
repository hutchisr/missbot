"""Tests for gate meta-tool generation in bot.ai."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bot.ai import AgentDeps, _make_enable_gate_tool


def test_agent_deps_defaults():
    deps = AgentDeps(username="alice")
    assert deps.enabled_gates == set()
    assert deps.adjusted_credit_users == set()


def test_make_enable_gate_tool_sets_name_and_doc():
    tool = _make_enable_gate_tool("research", ["tavily", "exa"])
    assert tool.__name__ == "enable_research"
    assert "research" in tool.__doc__
    assert "tavily" in tool.__doc__
    assert "exa" in tool.__doc__


@pytest.mark.anyio
async def test_enable_gate_tool_mutates_deps():
    tool = _make_enable_gate_tool("research", ["tavily"])
    deps = AgentDeps(username="alice")
    ctx = SimpleNamespace(deps=deps)

    result = await tool(ctx)  # type: ignore[arg-type]

    assert "research" in deps.enabled_gates
    assert "research" in result
    assert "tavily" in result
