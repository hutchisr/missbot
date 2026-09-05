from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, WebsocketUrl, field_validator, model_validator

_ALLOW_EXTRA = ConfigDict(extra="allow")


class User(BaseModel):
    model_config = _ALLOW_EXTRA

    id: str
    name: str | None = None
    username: str
    host: str | None = None
    location: str | None = None
    isBot: bool | None = None


class MiFile(BaseModel):
    model_config = _ALLOW_EXTRA

    id: str
    type: str
    thumbnailUrl: str | None = None
    url: str | None = None


class Note(BaseModel):
    model_config = _ALLOW_EXTRA

    id: str
    text: str | None = None
    userId: str
    user: User
    createdAt: datetime | None = None
    replyId: str | None = None
    renoteId: str | None = None
    reply: Optional["Note"] = None
    renote: Optional["Note"] = None
    visibility: Literal["public", "home", "followers", "specified"] | None = None
    visibleUserIds: list[str] | None = None
    localOnly: bool | None = None
    mentions: list[str] | None = None
    files: list[MiFile] | None = None


class MiChannelConnectParams(BaseModel):
    model_config = _ALLOW_EXTRA

    withRenotes: bool = True


class MiChannelConnectBody(BaseModel):
    model_config = _ALLOW_EXTRA

    channel: str
    id: str
    params: MiChannelConnectParams | None = None


class MiChannelConnect(BaseModel):
    model_config = _ALLOW_EXTRA

    type: Literal["connect"] = "connect"
    body: MiChannelConnectBody


class MiWebsocketMessageBody(BaseModel):
    model_config = _ALLOW_EXTRA

    type: str | None = None  # usually `mention` or `note`
    body: Note | None = None
    channel: str | None = None
    id: str | None = None


class MiWebsocketMessage(BaseModel):
    model_config = _ALLOW_EXTRA

    type: str
    body: MiWebsocketMessageBody | None = None


ModelAPIType = Literal["openai-chat", "openai-responses", "anthropic"]


class ModelSpec(BaseModel):
    """Rich model entry for the `llm_models` and `score_models` lists.

    Omit `api_type` and `base_url` to use `model` as a Pydantic-AI
    `provider:model` string with extra metadata such as `vision`. Set
    `api_type` to select a concrete API family, optionally at a custom
    `base_url`. For backward compatibility, setting only `base_url` selects
    `openai-chat`.
    """

    model: str = Field(description="Model name, or a pydantic-ai 'provider:model' string when api_type is unset.")
    api_type: ModelAPIType | None = Field(
        default=None,
        description="API family to use. Omit to infer from a provider:model string, or to retain the "
        "openai-chat default when base_url is set.",
    )
    base_url: AnyHttpUrl | None = Field(
        default=None,
        description="Optional API base URL. Supports OpenAI Chat Completions, OpenAI Responses, and Anthropic APIs.",
    )
    api_key: str | None = Field(
        default=None, description="API key to send to the endpoint. Use api_key_env to load from env instead."
    )
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable name to read the API key from (preferred over hard-coding api_key).",
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional JSON request-body parameters sent only to this model. Use for provider-specific "
        "features that Pydantic AI does not expose as first-class settings.",
    )
    vision: bool = Field(
        default=True,
        description="Whether this model can handle image input. Set to false for text-only models so "
        "image-bearing prompts skip them in the fallback chain.",
    )


# Compatibility for code importing the old, overly specific name. Config files
# do not encode this Python class name, so no YAML migration is needed.
CustomOpenAIModel = ModelSpec


class MCPServerConfig(BaseModel):
    """Configuration for a single streamable-HTTP MCP server."""

    name: str = Field(description="Human-readable identifier (used in logs and gate descriptions)")
    url: AnyHttpUrl = Field(description="Streamable-HTTP MCP endpoint URL")
    headers: dict[str, str] = Field(default_factory=dict, description="Extra HTTP headers (e.g. auth tokens)")
    tool_prefix: str | None = Field(
        default=None,
        description="Prefix added to every tool name from this server (avoids collisions). "
        "Becomes `<prefix>_<original_name>` in the model-visible tool name.",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="If set, only these tools are exposed. Match against UNPREFIXED MCP tool names.",
    )
    blocked_tools: list[str] = Field(
        default_factory=list,
        description="Tools to hide. Match against UNPREFIXED MCP tool names. Applied after allowed_tools.",
    )
    timeout: float = Field(default=30.0, gt=0, description="HTTP connection timeout in seconds")
    enabled: bool = Field(default=True, description="Disable without deleting the entry")
    gate: str | None = Field(
        default=None,
        description="If set, tools from this server are hidden until the model calls enable_<gate>(). "
        "Multiple servers can share a gate.",
    )


class ScoreCategory(BaseModel):
    """One sentiment bucket the auto-scoring classifier may assign.

    The classifier only ever picks a category *name* (constrained output); the
    point `delta` is owned entirely by code here, so a prompt injection can at most
    nudge the category — it can never choose the number. Keep names short, simple
    tokens (they become the classifier's allowed output values).
    """

    name: str = Field(description="Category label the classifier emits (e.g. 'good'). Short, simple token.")
    delta: int = Field(
        description="Points added (positive) or subtracted (negative) when a message is classified here."
    )
    description: str = Field(description="What this category means; shown to the classifier to guide labeling.")

    @field_validator("name")
    @classmethod
    def _name_non_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("score category name must be non-empty")
        return v


# Built-in default categories — match the previously hard-coded toxic/rude/neutral/
# good/exceptional set, so deployments that don't configure categories are unchanged.
DEFAULT_SCORE_CATEGORIES: list[ScoreCategory] = [
    ScoreCategory(name="toxic", delta=-10, description="harassment, slurs, threats, hateful content."),
    ScoreCategory(name="rude", delta=-5, description="dismissive, hostile, or insulting but not extreme."),
    ScoreCategory(name="neutral", delta=0, description="ordinary message, nothing notable either way."),
    ScoreCategory(name="good", delta=5, description="kind, helpful, thoughtful, or funny in good faith."),
    ScoreCategory(name="exceptional", delta=10, description="outstandingly insightful, generous, or constructive."),
]


class Config(BaseModel):
    domain: str = Field(description="domain")
    url: AnyHttpUrl = Field(description="url")
    ws_url: WebsocketUrl = Field(description="ws_url")
    token: str = Field(description="token")
    channel: str | None = None
    llm_models: list[str | ModelSpec] = Field(
        description="LLM models. Strings use pydantic-ai 'provider:model' format "
        "(e.g. 'openrouter:anthropic/claude-3.5-sonnet'). Dicts add metadata or "
        "select an API family and optional custom endpoint (see ModelSpec)."
    )
    auto_models: list[str | ModelSpec] | None = Field(
        default=None,
        min_length=1,
        description="Optional model chain used only for autonomous posts. None reuses llm_models.",
    )
    vision: bool = Field(default=True, description="Enable vision (pass images directly to the main LLM)")
    vision_models: list[str] | None = Field(
        default=None, description="Vision model strings (legacy, unused when vision=True)"
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Hard cap on reply generation and, when auto_max_tokens is unset, autonomous "
        "generation. When unset, the corresponding models generate without an explicit token cap.",
    )
    auto_max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Autonomous-post-only generation token cap. None reuses max_tokens, including "
        "its unbounded behavior.",
    )
    max_note_length: int = Field(
        default=3000,
        gt=0,
        description="Hard character cap for a single note (matches the Misskey instance's "
        "maxNoteTextLength, default 3000). The reply/auto models are told this budget so "
        "they compose a complete in-bounds reply; over-cap output is truncated only as a "
        "last-resort safety net (Misskey rejects an over-cap note with HTTP 400).",
    )
    auto_max_chars: int | None = Field(
        default=None,
        gt=0,
        description="Autonomous-post-only visible character cap. None reuses max_note_length. "
        "Over-limit drafts are retried and never added to autonomous history or published.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the reply/auto models. None leaves the provider default "
        "unchanged. Higher = more varied (helps avoid the bot repeating its own phrasing).",
    )
    auto_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature override for autonomous posts. None reuses temperature, "
        "including its provider-default behavior.",
    )
    auto_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Per-request timeout in seconds for autonomous-post model generation.",
    )
    top_p: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus-sampling top_p for the reply/auto models. None leaves the provider default.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty for the reply/auto models (penalizes tokens by how often they've "
        "appeared). Positive values reduce verbatim self-repetition. None = not sent (provider default).",
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for the reply/auto models (penalizes tokens that appeared at all). "
        "Positive values push the model toward new topics/phrasing. None = not sent (provider default).",
    )
    bot_user_id: str = Field(description="bot_user_id")
    bot_username: str = Field(description="bot_username")
    system_prompt: str = Field(description="system_prompt")
    system_prompt_auto: str | None = Field(
        default=None,
        description="System prompt for autonomous (unprompted) posts",
    )
    auto_post_interval: int | None = Field(
        default=None,
        gt=0,
        description="Interval in seconds between autonomous posts (None = disabled)",
    )
    auto_post_jitter: int = Field(
        default=0,
        ge=0,
        description="Random jitter in seconds added/subtracted from auto_post_interval",
    )
    auto_reply_enabled: bool = Field(
        default=False,
        description="Enable automatic replies to timeline notes",
    )
    auto_reply_interval: int = Field(
        default=900,
        gt=0,
        description="Minimum seconds between automatic replies",
    )
    auto_reply_jitter: int = Field(
        default=0,
        ge=0,
        description="Random jitter in seconds added/subtracted from auto_reply_interval",
    )
    max_retries: int = Field(gt=0)
    http_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="HTTP client timeout in seconds",
    )
    searxng_url: AnyHttpUrl | None = None
    searxng_user: str | None = None
    searxng_password: str | None = None
    redis_url: str | None = Field(default=None, description="Redis connection URL (redis://host:port/db)")
    redis_password: str | None = Field(default=None, description="Redis password for authentication")
    redis_db: int | None = Field(default=0, ge=0, description="Redis database number (0-15)")
    social_credit_unrestricted_user_ids: list[str] = Field(
        default_factory=list,
        description="User ids that lift the author-only restriction on social credit adjustments. "
        "When the author of the note being replied to has one of these user ids, the bot may adjust "
        "any user's score; otherwise it can only adjust that author.",
    )
    social_credit_auto_score: bool = Field(
        default=True,
        description="Automatically score every author's message via an isolated, injection-resistant "
        "classifier (only effective when Redis is configured). The category the classifier returns is "
        "mapped to a small fixed delta in code, so users cannot dictate their own score. Applies to "
        "privileged users too (the privileged flag only gates the manual adjust tool). "
        "Set false to disable automatic scoring entirely.",
    )
    social_credit_score_cooldown: int = Field(
        default=10,
        gt=0,
        description="Minimum seconds between automatic social credit changes for a given user. "
        "Bounds how fast a user can farm score even if every message is rated positively.",
    )
    social_credit_ignore_threshold: int | None = Field(
        default=None,
        description="When set (and Redis is configured), authors whose social credit score is below "
        "this value are ignored entirely: the note is never passed to the LLM and no reply is sent "
        "(nor are they scored or given memory access). Leave unset to never ignore on score.",
    )
    score_models: list[str | ModelSpec] = Field(
        default_factory=list,
        description="Models for the social-credit message classifier (same forms as llm_models). "
        "Defaults to llm_models when empty. Classification is a simple labeling task, so a smaller / "
        "cheaper model is usually fine.",
    )
    social_credit_categories: list[ScoreCategory] = Field(
        default_factory=lambda: [c.model_copy() for c in DEFAULT_SCORE_CATEGORIES],
        description="Sentiment categories the auto-scoring classifier may assign, each with a fixed point "
        "delta applied in code (the model only picks a category, never the number). Defaults to the built-in "
        "toxic/rude/neutral/good/exceptional set. Names must be unique (case-insensitive).",
    )
    max_context: int = Field(gt=0, default=3, description="Number of context messages to include")
    ignore_direct_messages: bool = Field(
        default=True,
        description="Ignore direct/private messages (Misskey 'specified' visibility) instead of replying. "
        "The bot is designed for public-timeline threads; set false to also respond to DMs.",
    )
    ignore_bots: bool = Field(
        default=True,
        description="Ignore mentions from accounts flagged as bots (Misskey user `isBot`) instead of "
        "replying. Prevents bot-to-bot loops; set false to also respond to other bots.",
    )
    max_reply_mentions: int = Field(
        default=5,
        gt=0,
        description="Maximum total mentions (including the author being replied to) the bot puts in "
        "a reply. Caps mention-amplification / harassment relaying via notes that tag many users.",
    )
    max_concurrent_handlers: int = Field(
        default=20,
        gt=0,
        description="Maximum number of concurrent mention and auto-reply handlers. New events are "
        "dropped when this capacity is full to bound LLM, memory, Redis, and HTTP work.",
    )
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list,
        description="Streamable-HTTP MCP servers to expose as tools.",
    )
    vision_image_mode: Literal["url", "fetch"] = Field(
        default="url",
        description="How images reach the model. 'url' sends the media URL (cheapest; works with "
        "OpenRouter and most hosted providers). 'fetch' downloads the image and sends it inline as "
        "base64 — required by providers that refuse URLs, e.g. Ollama Cloud answers 'image URLs are "
        "not currently supported, please use base64 encoded data instead'. Fetching makes this "
        "process retrieve attacker-supplied media, so it is bounded by vision_max_image_bytes and "
        "http_timeout_seconds and re-checked against the SSRF guard.",
    )
    vision_max_image_bytes: int = Field(
        default=8 * 1024 * 1024,
        gt=0,
        description="Maximum bytes downloaded per image when vision_image_mode='fetch'. The body is "
        "streamed and abandoned once it exceeds this, so oversized media cannot exhaust memory.",
    )
    image_gen_enabled: bool = Field(
        default=False,
        description="Let the autonomous-post agent generate an image before choosing an optional caption via "
        "the generate_image tool. Requires image_gen_model. Auto posts only — the reply path and ACP never "
        "receive the tool.",
    )
    image_gen_model: str | None = Field(
        default=None,
        description="Image model id sent to the images endpoint (e.g. 'google/gemini-2.5-flash-image').",
    )
    image_gen_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://openrouter.ai/api/v1"),
        description="OpenAI-compatible base URL for image generation (POSTed to <base_url>/images/generations).",
    )
    image_gen_api_key: str | None = Field(
        default=None,
        description="API key for the image endpoint. Use image_gen_api_key_env to load from env instead.",
    )
    image_gen_api_key_env: str = Field(
        default="OPENROUTER_API_KEY",
        description="Environment variable holding the image endpoint's API key (used when image_gen_api_key "
        "is unset). When neither resolves, requests are sent unauthenticated — which is valid for a keyless "
        "self-hosted endpoint, so it warns rather than failing startup.",
    )
    image_gen_size: str | None = Field(
        default=None,
        description="Optional `size` request param (e.g. '1024x1024'). Sent only when set, so backends that "
        "reject the field are unaffected while it is omitted.",
    )
    image_gen_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description="HTTP timeout for one image generation request. Image models are much slower than chat.",
    )
    image_gen_max_bytes: int = Field(
        default=8 * 1024 * 1024,
        gt=0,
        description="Maximum decoded image size accepted from the provider. A larger image is dropped and the "
        "post goes out as text only.",
    )
    image_gen_mark_sensitive: bool = Field(
        default=False,
        description="Upload generated images with isSensitive set, so Misskey blurs them behind a click.",
    )
    acp_default_identity: str = Field(
        default="acp",
        description="Identity used for ACP callers when no sender header can be parsed from the "
        "prompt. Namespaced as 'acp:<value>' so it can never collide with a fediverse handle. "
        "Only used by the `python -m bot.acp` frontend.",
    )
    acp_parse_sender_header: bool = Field(
        default=False,
        description="Trust and parse per-sender identity from an ACP harness's textual message header "
        "(buzz-acp emits 'From: <label> (npub: ..., hex: ...)'). Disabled by default because ACP does "
        "not authenticate that text: a client can fabricate the entire pre-Content header. Enable only "
        "when the process/transport admits a trusted harness, not arbitrary ACP clients. The parser uses "
        "only the pubkey, never the display name.",
    )
    acp_max_history_turns: int = Field(
        default=20,
        gt=0,
        description="Maximum conversation turns retained per ACP session. ACP sessions are "
        "long-lived, so history is bounded to keep prompts from growing without limit.",
    )
    acp_max_sessions: int = Field(
        default=8,
        gt=0,
        description="Shared cap for ACP WebSocket connections, sessions, and concurrent prompts. "
        "Bounds provider, Redis, and memory load across all clients.",
    )
    acp_max_prompt_chars: int = Field(
        default=65536,
        gt=0,
        description="Maximum aggregate text characters accepted in one ACP prompt before "
        "provider, scoring, or memory work begins.",
    )
    memory_enabled: bool = Field(
        default=False,
        description="Enable Hindsight's automatic memory lifecycle: recall before each public turn, "
        "then asynchronously retain the completed exchange.",
    )
    hindsight_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("http://localhost:8888"),
        description="Base URL of the Hindsight API service.",
    )
    hindsight_api_key: str | None = Field(
        default=None,
        description="Optional Hindsight API key. Takes precedence over hindsight_api_key_env.",
    )
    hindsight_api_key_env: str | None = Field(
        default="HINDSIGHT_API_KEY",
        description="Environment variable holding the Hindsight API key when hindsight_api_key is unset. "
        "Set null to disable environment lookup.",
    )
    hindsight_bank_id: str | None = Field(
        default=None,
        description="Shared Hindsight bank id. Defaults to the normalized bot_username so Misskey and ACP "
        "processes use the same memory.",
    )
    hindsight_retain_mission: str | None = Field(
        default=None,
        description="Optional Hindsight retain mission controlling which durable facts are extracted. "
        "The built-in Missbot durability and prompt-injection policy is used when unset.",
    )
    hindsight_observations_mission: str | None = Field(
        default=None,
        description="Optional mission for consolidating recurring public-conversation patterns into "
        "Hindsight observations. The built-in Missbot mission is used when unset.",
    )
    hindsight_recall_budget: Literal["low", "mid", "high"] = Field(
        default="mid",
        description="Hindsight breadth/cost budget for automatic pre-turn recall.",
    )
    hindsight_recall_max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens Hindsight may return as automatic pre-turn memory context.",
    )
    hindsight_recall_query_max_chars: int = Field(
        default=800,
        gt=0,
        description="Maximum characters from the current author/message sent as the Hindsight recall query.",
    )
    debug: bool | None = None

    @model_validator(mode="after")
    def check_auto_post_config(self) -> "Config":
        if self.auto_post_interval and not self.system_prompt_auto:
            raise ValueError("system_prompt_auto is required when auto_post_interval is set")
        return self

    @model_validator(mode="after")
    def check_score_categories(self) -> "Config":
        cats = self.social_credit_categories
        if not cats:
            raise ValueError("social_credit_categories must not be empty")
        names = [c.name.lower() for c in cats]
        if len(set(names)) != len(names):
            raise ValueError("social_credit_categories names must be unique (case-insensitive)")
        return self

    @model_validator(mode="after")
    def check_memory_config(self) -> "Config":
        if self.hindsight_bank_id is not None and not self.hindsight_bank_id.strip():
            raise ValueError("hindsight_bank_id must not be blank")
        return self

    @model_validator(mode="after")
    def check_image_gen_config(self) -> "Config":
        if self.image_gen_enabled and not self.image_gen_model:
            raise ValueError("image_gen_model is required when image_gen_enabled is true")
        if self.image_gen_enabled and not self.system_prompt_auto:
            raise ValueError(
                "system_prompt_auto is required when image_gen_enabled is true (the tool is auto-post-only)"
            )
        return self
