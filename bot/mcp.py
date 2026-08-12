"""Streamable-HTTP MCP server integration.

Builds :class:`MCPToolset` toolsets from :class:`MCPServerConfig` entries (streamable
HTTP is the default transport for HTTP URLs), applying the tool-name prefix and
allow/block filtering and (optionally) wrapping them in a filter that hides the whole
server until a gate flag on ``ctx.deps`` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

from .models import Config, MCPServerConfig

if TYPE_CHECKING:
    from .ai import AgentDeps


def _strip_prefix(name: str, prefix: str | None) -> str:
    if prefix and name.startswith(f"{prefix}_"):
        return name[len(prefix) + 1 :]
    return name


def _make_allow_block_filter(cfg: MCPServerConfig):
    allowed = set(cfg.allowed_tools) if cfg.allowed_tools is not None else None
    blocked = set(cfg.blocked_tools)
    prefix = cfg.tool_prefix

    def _filter(ctx: RunContext[AgentDeps], tool_def: ToolDefinition) -> bool:
        name = _strip_prefix(tool_def.name, prefix)
        if allowed is not None and name not in allowed:
            return False
        return name not in blocked

    return _filter


def _make_gate_filter(gate: str):
    def _filter(ctx: RunContext[AgentDeps], tool_def: ToolDefinition) -> bool:
        enabled = getattr(ctx.deps, "enabled_gates", None)
        return enabled is not None and gate in enabled

    return _filter


def build_mcp_toolsets(config: Config) -> list[AbstractToolset[AgentDeps]]:
    """Build filtered MCP toolsets for every enabled server in ``config.mcp_servers``."""
    toolsets: list[AbstractToolset[AgentDeps]] = []
    for cfg in config.mcp_servers:
        if not cfg.enabled:
            continue
        # MCPToolset has no tool_prefix arg; apply the prefix via .prefixed() (same
        # ``{prefix}_`` scheme the old server used and that _strip_prefix expects). The
        # prefix must wrap before the filters so they see the prefixed tool names.
        server: AbstractToolset[AgentDeps] = MCPToolset(
            str(cfg.url),
            headers=cfg.headers or None,
            init_timeout=cfg.timeout,
            id=cfg.name,
        )
        if cfg.tool_prefix:
            server = server.prefixed(cfg.tool_prefix)
        if cfg.allowed_tools is not None or cfg.blocked_tools:
            server = server.filtered(_make_allow_block_filter(cfg))
        if cfg.gate:
            server = server.filtered(_make_gate_filter(cfg.gate))
        toolsets.append(server)
    return toolsets


def gate_names(config: Config) -> dict[str, list[str]]:
    """Return ``{gate: [server_name, ...]}`` for every gate referenced by an enabled server."""
    out: dict[str, list[str]] = {}
    for cfg in config.mcp_servers:
        if not cfg.enabled or not cfg.gate:
            continue
        out.setdefault(cfg.gate, []).append(cfg.name)
    return out
