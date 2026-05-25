import os
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional, Union

import httpx

from pydantic_ai import Agent, ImageUrl, RunContext
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire
from redis.asyncio import Redis

from .mcp import build_mcp_toolsets, gate_names
from .models import Config, CustomOpenAIModel, Note, User
from .tools import build_tools, normalize_username


@dataclass
class _ProviderReportedPrice:
    input_price: Decimal
    output_price: Decimal
    total_price: Decimal


_original_cost = getattr(ModelResponse, "cost", None)


def _cost_prefer_provider(self: ModelResponse):
    # OpenRouter returns the actual routed price in provider_details['cost'];
    # prefer it over genai-prices' static lookup, which lacks many OR models
    # (e.g. qwen3-235b, claude-sonnet-4-5) and silently drops operation.cost.
    try:
        details = self.provider_details or {}
        reported = details.get("cost")
        if reported is not None:
            total = Decimal(str(reported))
            input_tokens = self.usage.input_tokens or 0
            output_tokens = self.usage.output_tokens or 0
            denom = input_tokens + output_tokens
            if denom > 0:
                input_price = total * Decimal(input_tokens) / Decimal(denom)
                output_price = total - input_price
            else:
                input_price = Decimal(0)
                output_price = total
            return _ProviderReportedPrice(input_price, output_price, total)
    except Exception:
        logfire.exception("Provider-cost override failed; falling back to upstream")
    if _original_cost is None:
        raise AttributeError("ModelResponse.cost is unavailable on this pydantic_ai version")
    return _original_cost(self)


if _original_cost is not None:
    ModelResponse.cost = _cost_prefer_provider  # type: ignore[method-assign]


def _resolve_model_spec(spec: Union[str, CustomOpenAIModel]) -> Union[str, Model]:
    """Convert a config llm_models entry into something Pydantic AI accepts.

    Strings pass through (Pydantic AI parses them as 'provider:model'). Dict
    entries with `base_url` build an OpenAIChatModel; without `base_url` the
    `model` field is treated as a pydantic-ai provider string.
    """
    if isinstance(spec, str):
        return spec
    if spec.base_url is None:
        return spec.model
    api_key = spec.api_key
    if api_key is None and spec.api_key_env:
        api_key = os.environ.get(spec.api_key_env)
    return OpenAIChatModel(
        spec.model,
        provider=OpenAIProvider(base_url=str(spec.base_url), api_key=api_key),
    )


def _spec_supports_vision(spec: Union[str, CustomOpenAIModel]) -> bool:
    """Whether a model entry should receive image input. Strings default True."""
    if isinstance(spec, str):
        return True
    return spec.vision


def _user_handle(user: User) -> str:
    """Get full handle: username for local, username@host for remote."""
    if user.host:
        return f"{user.username}@{user.host}"
    return user.username


def _image_urls_for(note: Note, vision: bool) -> list[ImageUrl]:
    """Extract ImageUrl objects for a note's image attachments."""
    if not vision or not note.files:
        return []

    images: list[ImageUrl] = []
    for file in note.files:
        image_url = file.thumbnailUrl or file.url
        if image_url and file.type.startswith("image/"):
            images.append(ImageUrl(url=image_url))
    return images


def _build_user_content(note: Note, vision: bool) -> str | list[str | ImageUrl]:
    """Build content for a user prompt part, with optional images."""
    text = f"{_user_handle(note.user)}: {note.text or ''}"
    images = _image_urls_for(note, vision)
    if images:
        return [text, *images]
    return text


@dataclass
class AgentDeps:
    """Runtime dependencies passed to the agent on each run."""

    username: str
    """The handle of the user who sent the message."""
    social_credit_score: Optional[int] = None
    """The user's current social credit score, or None if unavailable."""
    adjusted_credit_users: set[str] = field(default_factory=set)
    """Tracks users whose social credit was already adjusted in this run."""
    social_credit_unrestricted: bool = False
    """When True, social credit may be adjusted for any user, not just `username`."""
    enabled_gates: set[str] = field(default_factory=set)
    """Gates opened during this run by `enable_<gate>` meta-tools."""


def _make_enable_gate_tool(gate: str, servers: list[str]):
    """Build an `enable_<gate>` tool that opens gate for the rest of the run."""
    server_list = ", ".join(servers)

    async def enable_gate(ctx: RunContext[AgentDeps]) -> str:
        ctx.deps.enabled_gates.add(gate)
        return f"Enabled '{gate}' toolset (servers: {server_list}). Their tools are now available."

    enable_gate.__name__ = f"enable_{gate}"
    enable_gate.__doc__ = (
        f"Enable the '{gate}' toolset for the rest of this interaction. "
        f"Call this when you need tools from: {server_list}. "
        "The tools become visible on the next model turn."
    )
    return enable_gate


class ChatAgent:
    def __init__(self, config: Config, redis_client: Optional[Redis] = None):
        self._config = config
        self._redis = redis_client

        fallback_on = (ModelAPIError, httpx.TimeoutException)

        def _chain(specs: list[Union[str, CustomOpenAIModel]]) -> Optional[Union[str, Model]]:
            resolved = [_resolve_model_spec(s) for s in specs]
            if not resolved:
                return None
            if len(resolved) == 1:
                return resolved[0]
            return FallbackModel(*resolved, fallback_on=fallback_on)

        model = _chain(config.llm_models)
        assert model is not None, "llm_models must not be empty"

        vision_specs = [s for s in config.llm_models if _spec_supports_vision(s)]
        self._has_vision_model: bool = bool(vision_specs)
        # Build a separate chain only when the filter actually narrows the list.
        # If every model is vision-capable, the agent's default model is fine.
        if not vision_specs or len(vision_specs) == len(config.llm_models):
            self._vision_model: Optional[Union[str, Model]] = None
        else:
            self._vision_model = _chain(vision_specs)

        tools = build_tools(config, redis_client=redis_client)
        gates = gate_names(config)
        for gate, servers in sorted(gates.items()):
            tools.append(_make_enable_gate_tool(gate, servers))
        mcp_toolsets = build_mcp_toolsets(config)

        # Auto agent runs without AgentDeps, so skip tools that touch ctx.deps
        # (social-credit tools and enable_<gate> meta-tools).
        auto_tools = build_tools(config, redis_client=None)

        async def _inject_social_credit(ctx: RunContext[AgentDeps]) -> str:
            parts: list[str] = []
            parts.append(f"Current user: @{ctx.deps.username}")
            if ctx.deps.social_credit_score is not None:
                parts.append(f"Current user's social credit score: {ctx.deps.social_credit_score}")
            else:
                parts.append("Current user's social credit score: 0 (no score recorded yet)")
            if ctx.deps.social_credit_unrestricted:
                parts.append("You may adjust the social credit of any user.")
            else:
                parts.append(
                    f"You may only adjust the social credit of @{ctx.deps.username} "
                    "(the author of the note you're replying to). Adjustments targeting anyone else are refused."
                )
            return "\n".join(parts)

        self._agent: Agent[AgentDeps, str] = Agent(
            model,
            output_type=str,
            deps_type=AgentDeps,
            instructions=[config.system_prompt, _inject_social_credit],
            tools=tools,
            toolsets=mcp_toolsets or None,
            retries=3,
        )

        self._auto_agent: Optional[Agent[Any, str]] = None
        self._auto_history: deque[str] = deque(maxlen=10)
        if config.system_prompt_auto:
            self._auto_agent = Agent(
                model,
                output_type=str,
                instructions=[config.system_prompt_auto],
                tools=auto_tools,
                retries=3,
            )

    async def __aenter__(self) -> "ChatAgent":
        """Open persistent connections (MCP sessions) on the main agent."""
        await self._agent.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._agent.__aexit__(exc_type, exc, tb)

    @logfire.instrument(extract_args=["note"])
    async def run(self, note: Note, context: Optional[list[Note]] = None) -> str:
        """Process a note and generate a reply."""
        bot_user_id = self._config.bot_user_id
        vision = self._config.vision
        current_images = _image_urls_for(note, vision)
        if not note.text and not current_images:
            raise ValueError("Note has no text or supported images")

        # Pick the model chain. If the prompt has images but no model in the
        # main chain is vision-capable, drop the images and run text-only —
        # otherwise the whole fallback fails with "no endpoints support image
        # input" and the user gets nothing.
        run_model: Optional[Union[str, Model]] = None
        if current_images:
            if not self._has_vision_model:
                logfire.warning("No vision-capable models configured; dropping images")
                current_images = []
            elif self._vision_model is not None:
                run_model = self._vision_model

        # Mirror the same drop on context-note images so the history matches.
        effective_vision = vision and self._has_vision_model
        message_history: list[ModelMessage] = []
        if context:
            for c in reversed(context):
                if c.userId == bot_user_id:
                    # Bot's own previous messages become assistant responses
                    message_history.append(ModelResponse(parts=[TextPart(content=c.text or "")]))
                else:
                    # Other users' messages become user prompts (with any attached images)
                    message_history.append(
                        ModelRequest(parts=[UserPromptPart(content=_build_user_content(c, effective_vision))])
                    )

        # Build current user prompt
        current_parts: list[str | ImageUrl] = []
        if note.user.location:
            current_parts.append(f"User location: {note.user.location}")
        current_parts.append(f"{_user_handle(note.user)}: {note.text or ''}")
        if current_images:
            current_parts.extend(current_images)

        prompt: str | list[str | ImageUrl]
        if len(current_parts) == 1 and isinstance(current_parts[0], str):
            prompt = current_parts[0]
        else:
            prompt = current_parts

        # Pre-fetch social credit score for the current user
        handle = _user_handle(note.user)
        score = await self._get_social_credit_score(handle)
        # Lift the author-only restriction when the note's author is a designated
        # privileged user (e.g. the operator), configured by user id.
        unrestricted = note.user.id in self._config.social_credit_unrestricted_user_ids
        deps = AgentDeps(username=handle, social_credit_score=score, social_credit_unrestricted=unrestricted)

        run_kwargs: dict[str, Any] = {
            "deps": deps,
            "message_history": message_history,
            "model_settings": {"timeout": 300.0},
        }
        if run_model is not None:
            run_kwargs["model"] = run_model
        result = await self._agent.run(prompt, **run_kwargs)
        return result.output

    @logfire.instrument(extract_args=False, record_return=True)
    async def run_auto(self) -> str:
        """Generate an autonomous post with no user input."""
        if not self._auto_agent:
            raise ValueError("No system_prompt_auto configured")

        message_history: list[ModelMessage] = []
        for past_post in self._auto_history:
            message_history.append(ModelRequest(parts=[UserPromptPart(content="Generate a post for the timeline.")]))
            message_history.append(ModelResponse(parts=[TextPart(content=past_post)]))

        result = await self._auto_agent.run(
            "Generate a post for the timeline.",
            message_history=message_history,
            model_settings={"timeout": 300.0},
        )
        self._auto_history.append(result.output)
        return result.output

    async def _get_social_credit_score(self, username: str) -> Optional[int]:
        """Fetch the user's social credit score from Redis."""
        if not self._redis:
            return None
        key = normalize_username(username)
        try:
            raw = await self._redis.get(f"score:{key}")
            if raw is None:
                return None
            return int(raw)
        except Exception:
            logfire.exception("Error pre-fetching social credit score")
            return None
