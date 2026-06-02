import asyncio
import os
import re
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional, Union

import httpx

from pydantic_ai import Agent, ImageUrl, ModelRetry, RunContext
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
from pydantic_ai.settings import ModelSettings
import logfire
from redis.asyncio import Redis

from .extract import (
    ENTITY_LINK_INSTRUCTIONS,
    EXTRACTION_INSTRUCTIONS,
    ClaimExtraction,
    EntityMatch,
    ExtractedClaim,
    Skip,
    build_entity_link_prompt,
    build_extraction_prompt,
    looks_sensitive,
    pick_entity_match,
)
from .mcp import build_mcp_toolsets, gate_names
from .memory import EntityLinker, MemoryStore
from .models import Config, CustomOpenAIModel, Note, User
from .net import is_safe_media_url
from .scoring import ScoringSpec, build_scoring_prompt, build_scoring_spec
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
    # setattr (not a direct assignment) so the type checker doesn't reject monkeypatching
    # a method slot with our differently-typed override.
    setattr(ModelResponse, "cost", _cost_prefer_provider)


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


# Mirrors bot/bot.py:_strip_leading_mentions — the `@handle` prefix send_note prepends to
# every reply. Stripped when a prior bot note is reconstructed as assistant history so it
# doesn't prime the model to open with the same mention (and copy the rest verbatim).
_LEADING_MENTIONS_RE = re.compile(r"^(?:@[\w\-]+(?:@[\w\-.]+)?(?:\s+|$))+")


def _strip_leading_mentions(text: str) -> str:
    return _LEADING_MENTIONS_RE.sub("", text)


def _normalize_for_repeat(text: str) -> str:
    """Canonicalize a reply for verbatim-repeat detection: drop the leading mention
    prefix, collapse whitespace, lowercase. Two replies that differ only by who they
    mention or by spacing normalize equal."""
    return " ".join(_strip_leading_mentions(text.strip()).split()).lower()


def _image_urls_for(note: Note, vision: bool) -> list[ImageUrl]:
    """Extract ImageUrl objects for a note's visual attachments.

    Images use their thumbnail (falling back to the full image). Videos have no
    image body, but Misskey renders an image thumbnail for them — use that so the
    vision model can still see a frame. Never fall back to a video's raw ``url``
    (that's the video file, not an image). Other media (audio, etc.) is skipped.
    """
    if not vision or not note.files:
        return []

    images: list[ImageUrl] = []
    for file in note.files:
        if file.type.startswith("image/"):
            image_url = file.thumbnailUrl or file.url
        elif file.type.startswith("video/"):
            image_url = file.thumbnailUrl
        else:
            continue
        if not image_url:
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
    source_note_id: Optional[str] = None
    """Id of the note being replied to, recorded as provenance on memory writes."""
    social_credit_score: Optional[int] = None
    """The user's current social credit score, or None if unavailable."""
    adjusted_credit_users: set[str] = field(default_factory=set)
    """Tracks users whose social credit was already adjusted in this run."""
    social_credit_unrestricted: bool = False
    """When True, social credit may be adjusted for any user, not just `username`."""
    enabled_gates: set[str] = field(default_factory=set)
    """Gates opened during this run by `enable_<gate>` meta-tools."""
    previous_bot_reply: Optional[str] = None
    """The bot's most recent reply in this thread, if any. Used by the verbatim-repeat
    output validator to reject a reply that just parrots the prior turn (a failure mode of
    weaker fallback models when the new message is a thin same-topic follow-up)."""


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


async def _guarded(coro: Awaitable[object], label: str) -> None:
    """Await a best-effort side task, swallowing ANY error so it can't fail the reply.

    Belt-and-suspenders over each side task's own try/except: it also covers work that
    runs before that try (handle/context building), which would otherwise propagate
    through ``asyncio.gather`` and cancel the in-flight reply.
    """
    try:
        await coro
    except Exception:
        logfire.exception(f"{label} failed (reply unaffected)")


_FALLBACK_ON = (ModelAPIError, httpx.TimeoutException)


def _model_chain(specs: list[Union[str, CustomOpenAIModel]]) -> Optional[Union[str, Model]]:
    """Resolve a list of model specs into a single model or a FallbackModel chain (or None)."""
    resolved = [_resolve_model_spec(s) for s in specs]
    if not resolved:
        return None
    if len(resolved) == 1:
        return resolved[0]
    return FallbackModel(*resolved, fallback_on=_FALLBACK_ON)


def build_entity_linker(config: Config) -> Optional[EntityLinker]:
    """Build the write-time entity-link classifier as a standalone async callable.

    Returns None when memory is disabled. Used both by ChatAgent (the write path) and by
    the maintenance CLI (so consolidation's LLM merge pass works in the headless CronJob,
    which has no ChatAgent). The callable returns the chosen entity id or None ("new"); it
    only ever returns one of the offered candidate ids, and degrades to None on any error.
    """
    if not config.memory_enabled:
        return None
    chain = _model_chain(config.memory_extract_models or config.llm_models)
    if chain is None:
        return None
    agent = Agent[None, EntityMatch](chain, output_type=EntityMatch, instructions=[ENTITY_LINK_INSTRUCTIONS], retries=2)

    async def link(subject: str, candidates: list[tuple[int, str]]) -> Optional[int]:
        if not candidates:
            return None
        names = [name for _, name in candidates]
        try:
            result = await agent.run(build_entity_link_prompt(subject, names), model_settings={"timeout": 60.0})
        except Exception:
            logfire.exception("Entity linking failed — treating subject as a new entity")
            return None
        return pick_entity_match(result.output, candidates)

    return link


# Chars reserved for the mention prefix the bot prepends (up to ``max_reply_mentions``
# handles). Budgeting the reply below the raw cap keeps the final note within the platform
# limit; ``_enforce_length`` enforces it (no truncation backstop — over-cap notes are refused).
_MENTION_HEADROOM = 280


def _length_instruction(char_limit: int) -> str:
    """Instruction telling a reply/auto model its hard character budget.

    No post-hoc truncation: an over-cap note is refused, so the model must compose a
    complete reply within the budget (also enforced by ``_enforce_length``).
    """
    return (
        f"Hard length limit: your entire reply must be at most {char_limit} characters. "
        "This is a strict platform cap, not a target. A reply over the limit is cut off "
        "mid-sentence, so plan a complete, self-contained answer that fits — do not begin "
        "anything you cannot finish within the budget. Prefer concision; stop when done."
    )


def _enforce_length(char_limit: int) -> Callable[[str], str]:
    """Output validator hard-gating a reply/auto note to ``char_limit`` (pairs with
    ``_length_instruction``). An over-budget output raises ``ModelRetry``, so pydantic-ai
    re-runs the model (up to the agent's ``retries``) instead of shipping a note that
    ``send_note`` would refuse (bot/bot.py:send_note).
    """

    def validate(output: str) -> str:
        if len(output) > char_limit:
            raise ModelRetry(
                f"Your reply was {len(output)} characters, over the {char_limit}-character "
                f"limit. Rewrite a complete, self-contained reply within {char_limit} characters."
            )
        return output

    return validate


class ChatAgent:
    def __init__(
        self,
        config: Config,
        redis_client: Optional[Redis] = None,
        memory: Optional[MemoryStore] = None,
    ):
        self._config = config
        self._redis = redis_client
        self._memory = memory

        _chain = _model_chain
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

        # Claim-extraction agent (the admission gate): turns free-text into a typed claim or
        # rejects it — tool-less + constrained output, so untrusted text can only pick a branch
        # (bot/extract.py). Built only when memory is enabled; the entity linker is wired into
        # the store to prevent write-time fragmentation.
        self._extract_agent: Optional[Agent[None, ClaimExtraction]] = None
        self._entity_linker: Optional[EntityLinker] = None
        if memory is not None and config.memory_enabled:
            extract_model = _chain(config.memory_extract_models) if config.memory_extract_models else model
            # Subscript Agent[...] explicitly and pass the union members as a list (pydantic-ai's
            # multi-output spec): the type checker can't infer the OutputDataT type var back out
            # of a bare ``Union`` value passed to output_type.
            self._extract_agent = Agent[None, ClaimExtraction](
                extract_model,
                output_type=[ExtractedClaim, Skip],
                instructions=[EXTRACTION_INSTRUCTIONS],
                retries=2,
            )
            self._entity_linker = build_entity_linker(config)
            memory.entity_linker = self._entity_linker
            # The bot's own claims (via remember_fact) are stored but excluded from agreement.
            memory.bot_author = normalize_username(config.bot_username)

        # Pass the extractor only when memory + the extract agent exist, so remember_fact is
        # exposed only when it can structure a submitted fact into a typed claim.
        extractor = self._extract_claim if self._extract_agent is not None else None
        tools = build_tools(config, redis_client=redis_client, memory=memory, extractor=extractor)
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

        # Budget the reply below the raw note cap so the prepended mention prefix
        # still fits; the auto post (no mentions) gets the full cap.
        reply_char_budget = max(1, config.max_note_length - _MENTION_HEADROOM)
        self._agent: Agent[AgentDeps, str] = Agent(
            model,
            output_type=str,
            deps_type=AgentDeps,
            instructions=[config.system_prompt, _length_instruction(reply_char_budget), _inject_social_credit],
            tools=tools,
            toolsets=mcp_toolsets or None,
            retries=3,
        )
        # Hard-gate the reply to the budget the instruction states (see _enforce_length).
        self._agent.output_validator(_enforce_length(reply_char_budget))
        # Reject a reply that just parrots the bot's prior turn in the same thread.
        self._agent.output_validator(self._reject_verbatim_repeat)

        self._auto_agent: Optional[Agent[Any, str]] = None
        self._auto_history: deque[str] = deque(maxlen=10)
        if config.system_prompt_auto:
            self._auto_agent = Agent(
                model,
                output_type=str,
                instructions=[config.system_prompt_auto, _length_instruction(config.max_note_length)],
                tools=auto_tools,
                retries=3,
            )
            # No mention prefix on auto posts, so the budget is the full note cap.
            self._auto_agent.output_validator(_enforce_length(config.max_note_length))

        # Isolated, tool-less classifier for auto-scoring: treats the message as untrusted
        # data and emits only a fixed category, mapped to a bounded delta in code (bot/scoring.py).
        # Uses config.score_models when set (a cheaper model is fine), else the reply model.
        self._score_agent: Optional[Agent[None, str]] = None
        self._score_model: Optional[Union[str, Model]] = None
        self._score_spec: Optional[ScoringSpec] = None
        if redis_client is not None and config.social_credit_auto_score:
            self._score_model = _chain(config.score_models) if config.score_models else model
            # Categories, deltas, and instructions are derived from config here, so the
            # operator owns the buckets while code still owns the numbers.
            self._score_spec = build_scoring_spec(config.social_credit_categories)
            # pydantic-ai constrains output to the Literal's values at runtime, but pyright
            # can't match the Literal special form to the Agent() overloads (hence type: ignore).
            self._score_agent = Agent(  # type: ignore[reportCallIssue, reportAttributeAccessIssue]
                self._score_model,
                output_type=self._score_spec.output_type,  # type: ignore[reportArgumentType]
                instructions=[self._score_spec.instructions],
                retries=2,
            )

    async def __aenter__(self) -> "ChatAgent":
        """Open persistent connections (MCP sessions) on the main agent."""
        await self._agent.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._agent.__aexit__(exc_type, exc, tb)

    def _reject_verbatim_repeat(self, ctx: RunContext[AgentDeps], output: str) -> str:
        """Output validator: reject a reply identical to the bot's previous turn in this
        thread. Weaker fallback models sometimes re-emit their prior (often long) assistant
        message verbatim when the new user note is a thin same-topic follow-up; the ModelRetry
        pushes the model (within the agent's retry budget) to actually answer the latest note."""
        prev = ctx.deps.previous_bot_reply
        if prev and _normalize_for_repeat(output) == _normalize_for_repeat(prev):
            raise ModelRetry(
                "Your draft is identical to your previous reply in this thread, but the user "
                "has since said something new. Respond to their latest message instead of "
                "repeating your last reply."
            )
        return output

    def _generation_settings(self, timeout: float) -> ModelSettings:
        """Model settings for the reply/auto agents: the configured token cap
        (``max_tokens`` — otherwise unused), any configured sampling/anti-repetition
        knobs, and a per-call timeout.

        ``max_tokens`` and the sampling params are only included when set in config, so
        unset ones keep the provider default (and aren't sent to models that reject them).
        """
        settings: ModelSettings = {
            "timeout": timeout,
        }
        if self._config.max_tokens is not None:
            settings["max_tokens"] = self._config.max_tokens
        if self._config.temperature is not None:
            settings["temperature"] = self._config.temperature
        if self._config.top_p is not None:
            settings["top_p"] = self._config.top_p
        if self._config.frequency_penalty is not None:
            settings["frequency_penalty"] = self._config.frequency_penalty
        if self._config.presence_penalty is not None:
            settings["presence_penalty"] = self._config.presence_penalty
        return settings

    @logfire.instrument(extract_args=["note"])
    async def run(self, note: Note, context: Optional[list[Note]] = None) -> str:
        """Process a note and generate a reply."""
        bot_user_id = self._config.bot_user_id
        vision = self._config.vision
        current_images = _image_urls_for(note, vision)
        if not note.text and not current_images:
            raise ValueError("Note has no text or supported images")

        # Pick the model chain. If the prompt has images but no model is vision-capable,
        # drop the images and run text-only — else the whole fallback fails ("no endpoints
        # support image input") and the user gets nothing.
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
                    # Bot's own previous messages become assistant responses. Strip the
                    # leading @mention prefix send_note prepended, so the history doesn't
                    # prime the model to re-open (and copy) its prior reply verbatim.
                    message_history.append(
                        ModelResponse(parts=[TextPart(content=_strip_leading_mentions((c.text or "").strip()))])
                    )
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
        # The bot's most recent reply in this thread (context is nearest-parent first), used by
        # the verbatim-repeat output validator. None when the bot hasn't spoken in the thread.
        previous_bot_reply = next(
            (c.text for c in (context or []) if c.userId == bot_user_id and (c.text or "").strip()),
            None,
        )
        deps = AgentDeps(
            username=handle,
            source_note_id=note.id,
            social_credit_score=score,
            social_credit_unrestricted=unrestricted,
            previous_bot_reply=previous_bot_reply,
        )

        run_kwargs: dict[str, Any] = {
            "deps": deps,
            "message_history": message_history,
            "model_settings": self._generation_settings(300.0),
        }
        if run_model is not None:
            run_kwargs["model"] = run_model
        # Score the author's message and learn any world-fact it asserts, both in parallel
        # with the reply (no added latency). Each side task is _guarded, so its errors are
        # swallowed and can never cancel the reply.
        result, _, _ = await asyncio.gather(
            self._agent.run(prompt, **run_kwargs),
            _guarded(self._maybe_score_message(note, context), "Message scoring"),
            _guarded(self._maybe_ingest_note(note, context), "Note ingestion"),
        )
        return result.output

    def _render_context_lines(self, context: Optional[list[Note]]) -> list[str]:
        """Render the prior thread chronologically as reference-only "handle: text" lines.

        ``context`` is newest-first (as built for ``message_history``), so reverse it.
        Empty notes are dropped; the bot's own notes use the configured bot handle. Shared
        by the note-ingestion and scoring side tasks so both see the same thread rendering.
        """
        lines: list[str] = []
        for c in reversed(context or []):
            ctext = (c.text or "").strip()
            if not ctext:
                continue
            handle = self._config.bot_username if c.userId == self._config.bot_user_id else _user_handle(c.user)
            lines.append(f"{handle}: {ctext}")
        return lines

    async def _maybe_score_message(self, note: Note, context: Optional[list[Note]] = None) -> None:
        """Classify the author's message and apply a bounded score delta.

        Applies to every author (privileged users included — the privileged flag
        only gates the manual adjust tool, not automatic scoring). Uses the isolated
        classifier (no tools, constrained output) so the score is decided by code,
        not by anything the user can put in their message. The prior thread (``context``)
        is passed to the classifier as reference-only material so tone is judged in context
        (a curt reply read as hostile vs. friendly ribbing depending on what it answers).
        Rate limited per user. Never raises — scoring must not break the reply path.
        """
        if self._score_agent is None or self._redis is None:
            return
        text = (note.text or "").strip()
        if not text:
            return
        username = normalize_username(_user_handle(note.user))
        cooldown_key = f"score_cooldown:{username}"
        try:
            # Skip the classifier call entirely while the user is on cooldown.
            if await self._redis.exists(cooldown_key):
                logfire.debug("Auto-score skipped (cooldown)", username=username)
                return

            # Classify in isolation. Surface failures loudly: a model that can't
            # emit a valid category would otherwise silently disable all scoring.
            # We still never re-raise — the reply must not break.
            try:
                scored = await self._score_agent.run(
                    build_scoring_prompt(text, context=self._render_context_lines(context) or None),
                    model_settings={"timeout": 60.0},
                )
            except Exception:
                logfire.exception(
                    "Message scoring classifier FAILED — no score applied (reply unaffected). "
                    "The score model likely can't produce structured output; set `score_models` "
                    "to a model that can.",
                    username=username,
                    score_model=getattr(self._score_model, "model_name", None) or str(self._score_model),
                )
                return

            # _score_spec is always set whenever _score_agent is (guarded above).
            assert self._score_spec is not None
            delta = self._score_spec.deltas.get(scored.output, 0)
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
            logfire.exception("Unexpected error applying automatic score (reply unaffected)", username=username)

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
            model_settings=self._generation_settings(300.0),
        )
        self._auto_history.append(result.output)
        return result.output

    async def _extract_claim(
        self, fact: str, speaker: Optional[str] = None, context: Optional[list[str]] = None
    ) -> Optional[ClaimExtraction]:
        """Run the claim extractor over a submitted fact for the world-knowledge store.

        ``speaker`` lets the extractor resolve first-person references to that handle (used
        by the note-ingestion path so a user's self-statement is attributed to them).
        ``context`` is the prior thread (reference-only) so cross-note references resolve.
        Returns an ``ExtractedClaim``, a ``Skip`` (rejection with a reason), or None if the
        classifier itself failed. Never raises — a failed extraction just means the fact
        isn't stored; it must not break the reply path.
        """
        if self._extract_agent is None:
            return None
        try:
            result = await self._extract_agent.run(
                build_extraction_prompt(fact, speaker=speaker, context=context),
                model_settings={"timeout": 60.0},
            )
        except Exception:
            logfire.exception("Claim extraction failed — fact not stored (reply unaffected)")
            return None
        return result.output

    async def _maybe_ingest_note(self, note: Note, context: Optional[list[Note]] = None) -> None:
        """Learn a world-fact claim from the author's note, attributed to the author.

        The claim's ``author`` is the note's author (set by code, not by anything the model
        says), so distinct authors asserting the same value raise its agreement count. The
        author's handle is passed to the extractor so a self-statement ("I use Arch") resolves
        to a claim about them, and the prior thread (``context``) is supplied as reference-only
        material so cross-note references resolve (e.g. "her name is Olive" after "I have a pet
        lizard"). Durable personal facts are allowed, but the extractor skips sensitive info and
        a ``looks_sensitive`` backstop drops any obvious PII (email/phone/IDs) that slips through.
        Rate-limited per author by ``global_write_cooldown``. Never raises — must not break the reply.
        """
        if (
            self._memory is None
            or self._extract_agent is None
            or not self._config.memory_enabled
            or not self._config.memory_ingest_notes
        ):
            return
        text = (note.text or "").strip()
        if not text:
            return
        author = normalize_username(_user_handle(note.user))
        context_lines = self._render_context_lines(context)
        try:
            cooldown = self._config.global_write_cooldown
            if cooldown > 0:
                since = await self._memory.seconds_since_last_write(author)
                if since is not None and since < cooldown:
                    logfire.debug("Note ingestion skipped (write cooldown)", author=author)
                    return
            extracted = await self._extract_claim(text, speaker=author, context=context_lines or None)
            if not isinstance(extracted, ExtractedClaim):
                return
            # Backstop behind the extractor's sensitivity judgement: never store obvious
            # PII, even if the model judged it storable.
            if looks_sensitive(extracted.object) or looks_sensitive(extracted.subject):
                logfire.info("Note claim dropped (sensitive-PII backstop)", author=author)
                return
            await self._memory.add_claim(
                subject=extracted.subject,
                predicate=extracted.predicate,
                object_text=extracted.object,
                author=author,
            )
        except Exception:
            logfire.exception("Note claim ingestion failed (reply unaffected)", author=author)

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
