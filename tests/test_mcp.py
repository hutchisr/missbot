"""Tests for bot.mcp: config, filters, gate wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from bot.mcp import _make_allow_block_filter, _make_gate_filter, build_mcp_toolsets, gate_names
from bot.models import MCPServerConfig


def _tool_def(name: str):
    return SimpleNamespace(name=name)


def _ctx(**deps):
    return SimpleNamespace(deps=SimpleNamespace(**deps))


def test_mcp_server_config_defaults():
    cfg = MCPServerConfig(name="tavily", url="https://tavily.example/mcp")  # type: ignore[arg-type]
    assert cfg.enabled is True
    assert cfg.headers == {}
    assert cfg.blocked_tools == []
    assert cfg.allowed_tools is None
    assert cfg.gate is None
    assert cfg.tool_prefix is None


def test_mcp_server_config_requires_url():
    with pytest.raises(ValidationError):
        MCPServerConfig(name="x")  # type: ignore[call-arg]


def test_allow_block_filter_strips_prefix_before_matching():
    cfg = MCPServerConfig(
        name="tavily",
        url="https://tavily.example/mcp",  # type: ignore[arg-type]
        tool_prefix="tavily",
        allowed_tools=["search", "extract"],
        blocked_tools=["extract"],
    )
    f = _make_allow_block_filter(cfg)
    ctx = _ctx()

    assert f(ctx, _tool_def("tavily_search")) is True
    assert f(ctx, _tool_def("tavily_crawl")) is False  # not in allowed
    assert f(ctx, _tool_def("tavily_extract")) is False  # allowed but also blocked


def test_allow_block_filter_without_prefix():
    cfg = MCPServerConfig(
        name="x",
        url="https://x.example/mcp",  # type: ignore[arg-type]
        blocked_tools=["danger"],
    )
    f = _make_allow_block_filter(cfg)
    ctx = _ctx()
    assert f(ctx, _tool_def("safe")) is True
    assert f(ctx, _tool_def("danger")) is False


def test_gate_filter_hides_until_enabled():
    f = _make_gate_filter("research")
    td = _tool_def("tavily_search")

    locked = _ctx(enabled_gates=set())
    assert f(locked, td) is False

    unlocked = _ctx(enabled_gates={"research"})
    assert f(unlocked, td) is True


def test_gate_filter_without_deps_returns_false():
    f = _make_gate_filter("research")
    ctx_no_deps = SimpleNamespace(deps=object())
    assert f(ctx_no_deps, _tool_def("x")) is False


def test_gate_names_groups_by_gate(make_config):
    cfg = make_config(
        mcp_servers=[
            {"name": "tavily", "url": "https://a.example/mcp", "gate": "research"},
            {"name": "exa", "url": "https://b.example/mcp", "gate": "research"},
            {"name": "docs", "url": "https://c.example/mcp"},
            {"name": "off", "url": "https://d.example/mcp", "gate": "research", "enabled": False},
        ]
    )
    assert gate_names(cfg) == {"research": ["tavily", "exa"]}


def test_build_mcp_toolsets_skips_disabled(make_config):
    cfg = make_config(
        mcp_servers=[
            {"name": "on", "url": "https://on.example/mcp"},
            {"name": "off", "url": "https://off.example/mcp", "enabled": False},
        ]
    )
    toolsets = build_mcp_toolsets(cfg)
    assert len(toolsets) == 1


def test_build_mcp_toolsets_empty_when_no_servers(config):
    assert build_mcp_toolsets(config) == []
