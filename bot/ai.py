import asyncio
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
from .net import is_safe_media_url
from .scoring import MessageQuality, SCORING_INSTRUCTIONS, build_scoring_prompt, delta_for
from .tools import apply_social_credit, build_tools, normalize_username


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
        if not image_url or not file.type.startswith("image/"):
            continue
        # SSRF guard: the URL is attacker-controlled on federated notes.
        if not is_safe_media_url(image_url):
            logfire.warning("Dropping image with unsafe URL", url=image_url, file_id=file.id)
            continue
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
                parts.append("You may manually adjust any user's social credit with the adjust_social_credit tool.")
            else:
                parts.append(
                    "Regular users' social credit is adjusted automatically based on their "
                    "messages; you cannot adjust scores yourself, and the adjust_social_credit "
                    "tool will refuse. Do not promise, threaten, or claim to change anyone's score."
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

        # Isolated, tool-less classifier for automatic message scoring. It treats
        # the message as untrusted data and can only emit a fixed category, which
        # is mapped to a bounded delta in code — see bot/scoring.py. Classification
        # is a simple labeling task, so it can use a cheaper, separate model chain
        # (config.score_models); falls back to the main reply model when unset.
        self._score_agent: Optional[Agent[None, MessageQuality]] = None
        self._score_model: Optional[Union[str, Model]] = None
        if redis_client is not None and config.social_credit_auto_score:
            self._score_model = _chain(config.score_models) if config.score_models else model
            self._score_agent = Agent(
                self._score_model,
                output_type=MessageQuality,
                instructions=[SCORING_INSTRUCTIONS],
                retries=2,
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
        # Score the author's message in parallel with generating the reply so it
        # adds no user-facing latency. _maybe_score_message swallows its own
        # errors, so it can never fail the reply.
        result, _ = await asyncio.gather(
            self._agent.run(prompt, **run_kwargs),
            self._maybe_score_message(note, unrestricted),
        )
        return result.output

    async def _maybe_score_message(self, note: Note, unrestricted: bool) -> None:
        """Classify a non-privileged author's message and apply a bounded delta.

        Uses the isolated classifier (no tools, constrained output) so the score is
        decided by code, not by anything the user can put in their message. Rate
        limited per user. Never raises — scoring must not break the reply path.
        """
        if self._score_agent is None or self._redis is None or unrestricted:
            return
        text = (note.text or "").strip()
        if not text:
            return
        username = normalize_username(_user_handle(note.user))
        try:
            cooldown_key = f"score_cooldown:{username}"
            # Skip the classifier call entirely while the user is on cooldown.
            if await self._redis.exists(cooldown_key):
                logfire.debug("Auto-score skipped (cooldown)", username=username)
                return

            scored = await self._score_agent.run(
                build_scoring_prompt(text),
                model_settings={"timeout": 60.0},
            )
            delta = delta_for(scored.output)
            if delta == 0:
                return

            # Atomically claim the cooldown window; bail if a concurrent run won it.
            claimed = await self._redis.set(cooldown_key, "1", nx=True, ex=self._config.social_credit_score_cooldown)
            if not claimed:
                return
            new_score = await apply_social_credit(self._redis, username, delta, f"automatic: {scored.output} message")
            logfire.info(
                "Auto-scored message",
                username=username,
                quality=scored.output,
                delta=delta,
                new_score=new_score,
            )
        except Exception:
            logfire.exception("Error during automatic message scoring")

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
