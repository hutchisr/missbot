from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, WebsocketUrl, field_validator, model_validator
from typing import List, Literal, Optional, Union


_ALLOW_EXTRA = ConfigDict(extra="allow")


class User(BaseModel):
    model_config = _ALLOW_EXTRA

    id: str
    name: Optional[str] = None
    username: str
    host: Optional[str] = None
    location: Optional[str] = None


class MiFile(BaseModel):
    model_config = _ALLOW_EXTRA

    id: str
    type: str
    thumbnailUrl: Optional[str] = None
    url: Optional[str] = None


class Note(BaseModel):
    model_config = _ALLOW_EXTRA

    id: str
    text: Optional[str] = None
    userId: str
    user: User
    replyId: Optional[str] = None
    renoteId: Optional[str] = None
    reply: Optional["Note"] = None
    renote: Optional["Note"] = None
    visibility: Optional[Literal["public", "home", "followers", "specified"]] = None
    visibleUserIds: Optional[List[str]] = None
    localOnly: Optional[bool] = None
    mentions: Optional[List[str]] = None
    files: Optional[List[MiFile]] = None


class MiChannelConnectParams(BaseModel):
    model_config = _ALLOW_EXTRA

    withRenotes: bool = True


class MiChannelConnectBody(BaseModel):
    model_config = _ALLOW_EXTRA

    channel: str
    id: str
    params: Optional[MiChannelConnectParams] = None


class MiChannelConnect(BaseModel):
    model_config = _ALLOW_EXTRA

    type: Literal["connect"] = "connect"
    body: MiChannelConnectBody


class MiWebsocketMessageBody(BaseModel):
    model_config = _ALLOW_EXTRA

    type: Optional[str] = None  # usually `mention` or `note`
    body: Optional[Note] = None
    channel: Optional[str] = None
    id: Optional[str] = None


class MiWebsocketMessage(BaseModel):
    model_config = _ALLOW_EXTRA

    type: str
    body: Optional[MiWebsocketMessageBody] = None


class CustomOpenAIModel(BaseModel):
    """Rich entry for the `llm_models` list.

    Two forms:
    - Custom OpenAI-compatible endpoint: set `base_url` (e.g. self-hosted vLLM,
      Modal). `model` is the name sent in API requests.
    - Pydantic-AI provider string with extra metadata: omit `base_url` and put
      a string like `"openrouter:foo/bar"` in `model`. Useful when you need to
      attach `vision: false` to a string-form model.
    """

    model: str = Field(description="Model name (custom endpoint) or pydantic-ai 'provider:model' string.")
    base_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description="OpenAI-compatible base URL. Omit to use `model` as a pydantic-ai provider string.",
    )
    api_key: Optional[str] = Field(
        default=None, description="API key to send to the endpoint. Use api_key_env to load from env instead."
    )
    api_key_env: Optional[str] = Field(
        default=None,
        description="Environment variable name to read the API key from (preferred over hard-coding api_key).",
    )
    vision: bool = Field(
        default=True,
        description="Whether this model can handle image input. Set to false for text-only models so "
        "image-bearing prompts skip them in the fallback chain.",
    )


class MCPServerConfig(BaseModel):
    """Configuration for a single streamable-HTTP MCP server."""

    name: str = Field(description="Human-readable identifier (used in logs and gate descriptions)")
    url: AnyHttpUrl = Field(description="Streamable-HTTP MCP endpoint URL")
    headers: dict[str, str] = Field(default_factory=dict, description="Extra HTTP headers (e.g. auth tokens)")
    tool_prefix: Optional[str] = Field(
        default=None,
        description="Prefix added to every tool name from this server (avoids collisions). "
        "Becomes `<prefix>_<original_name>` in the model-visible tool name.",
    )
    allowed_tools: Optional[List[str]] = Field(
        default=None,
        description="If set, only these tools are exposed. Match against UNPREFIXED MCP tool names.",
    )
    blocked_tools: List[str] = Field(
        default_factory=list,
        description="Tools to hide. Match against UNPREFIXED MCP tool names. Applied after allowed_tools.",
    )
    timeout: float = Field(default=30.0, gt=0, description="HTTP connection timeout in seconds")
    enabled: bool = Field(default=True, description="Disable without deleting the entry")
    gate: Optional[str] = Field(
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
DEFAULT_SCORE_CATEGORIES: List[ScoreCategory] = [
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
    channel: Optional[str] = None
    llm_models: List[Union[str, CustomOpenAIModel]] = Field(
        description="LLM models. Strings use pydantic-ai 'provider:model' format "
        "(e.g. 'openrouter:anthropic/claude-3.5-sonnet'). Dicts configure custom "
        "OpenAI-compatible endpoints (see CustomOpenAIModel)."
    )
    vision: bool = Field(default=True, description="Enable vision (pass images directly to the main LLM)")
    vision_models: Optional[List[str]] = Field(
        default=None, description="Vision model strings (legacy, unused when vision=True)"
    )
    max_tokens: int = Field(gt=0)
    bot_user_id: str = Field(description="bot_user_id")
    bot_username: str = Field(description="bot_username")
    system_prompt: str = Field(description="system_prompt")
    system_prompt_auto: Optional[str] = Field(
        default=None,
        description="System prompt for autonomous (unprompted) posts",
    )
    auto_post_interval: Optional[int] = Field(
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
    searxng_url: Optional[AnyHttpUrl] = None
    searxng_user: Optional[str] = None
    searxng_password: Optional[str] = None
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL (redis://host:port/db)")
    redis_password: Optional[str] = Field(default=None, description="Redis password for authentication")
    redis_db: Optional[int] = Field(default=0, ge=0, description="Redis database number (0-15)")
    social_credit_unrestricted_user_ids: List[str] = Field(
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
    score_models: List[Union[str, CustomOpenAIModel]] = Field(
        default_factory=list,
        description="Models for the social-credit message classifier (same forms as llm_models). "
        "Defaults to llm_models when empty. Classification is a simple labeling task, so a smaller / "
        "cheaper model is usually fine.",
    )
    social_credit_categories: List[ScoreCategory] = Field(
        default_factory=lambda: [c.model_copy() for c in DEFAULT_SCORE_CATEGORIES],
        description="Sentiment categories the auto-scoring classifier may assign, each with a fixed point "
        "delta applied in code (the model only picks a category, never the number). Defaults to the built-in "
        "toxic/rude/neutral/good/exceptional set. Names must be unique (case-insensitive).",
    )
    max_context: int = Field(gt=0, default=1, description="Number of context messages to include")
    ignore_direct_messages: bool = Field(
        default=True,
        description="Ignore direct/private messages (Misskey 'specified' visibility) instead of replying. "
        "The bot is designed for public-timeline threads; set false to also respond to DMs.",
    )
    max_reply_mentions: int = Field(
        default=5,
        gt=0,
        description="Maximum total mentions (including the author being replied to) the bot puts in "
        "a reply. Caps mention-amplification / harassment relaying via notes that tag many users.",
    )
    mcp_servers: List[MCPServerConfig] = Field(
        default_factory=list,
        description="Streamable-HTTP MCP servers to expose as tools.",
    )
    memory_enabled: bool = Field(
        default=False,
        description="Enable the persistent world-knowledge store (Postgres + pgvector). Off by default; "
        "requires postgres_url and embedding_model. Adds the remember_fact / search_memory tools. The store "
        "keeps claims-with-provenance (source, trust tier, time, corroboration), never bare facts.",
    )
    postgres_url: Optional[str] = Field(
        default=None,
        description="Postgres DSN for long-term memory (e.g. postgres://user:pass@host:5432/db). "
        "The pgvector extension must be available on the server.",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model id sent to the embeddings endpoint (e.g. "
        "'perplexity/pplx-embed-v1-0.6b'). Required when memory_enabled.",
    )
    embedding_dim: int = Field(
        default=1024,
        gt=0,
        description="Embedding vector dimension; must match the embedding_model's output and the "
        "pgvector column. pplx-embed-v1-0.6b is 1024. Changing this requires re-embedding all rows.",
    )
    embedding_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://openrouter.ai/api/v1"),
        description="OpenAI-compatible base URL for the embeddings endpoint (POSTed to <base_url>/embeddings).",
    )
    embedding_api_key: Optional[str] = Field(
        default=None,
        description="API key for the embeddings endpoint. Use embedding_api_key_env to load from env instead.",
    )
    embedding_api_key_env: str = Field(
        default="OPENROUTER_API_KEY",
        description="Environment variable holding the embeddings API key (used when embedding_api_key is unset).",
    )
    global_recall_k: int = Field(
        default=5,
        gt=0,
        description="Max number of claims returned per search_memory call (after conflict resolution).",
    )
    global_recall_min_similarity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Cosine-similarity floor for search_memory results; weaker matches are dropped.",
    )
    global_write_cooldown: int = Field(
        default=60,
        ge=0,
        description="Minimum seconds between global memory writes per author. Bounds memory poisoning rate.",
    )
    global_dedup_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="If a new claim from the same source is at least this cosine-similar to an existing "
        "(non-retracted) claim with the same subject+predicate, the write is skipped as a near-duplicate.",
    )
    max_fact_length: int = Field(
        default=500,
        gt=0,
        description="Maximum character length of a single submitted fact/claim; longer writes are rejected.",
    )
    corroboration_threshold: int = Field(
        default=2,
        gt=0,
        description="Number of independent sources of tier >= secondary that must assert the same "
        "subject+predicate+object before a claim is promoted from 'asserted' to 'believed'. Model-generated "
        "(model_quarantine) claims never count toward this and are never promoted.",
    )
    entity_match_threshold: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        description="Cosine-similarity floor for linking a claim's subject to an existing entity. Below this "
        "(and with no exact name/alias match) a new entity is created instead.",
    )
    volatile_ttl_seconds: int = Field(
        default=86400,
        gt=0,
        description="A claim marked 'volatile' older than this (by valid_from, else recorded_at) is flagged "
        "stale on recall so the model re-verifies it live rather than trusting it.",
    )
    memory_extract_models: List[Union[str, CustomOpenAIModel]] = Field(
        default_factory=list,
        description="Model chain for the claim-extraction classifier that turns a submitted fact into a "
        "typed subject/predicate/object claim or rejects it (same forms as llm_models). Defaults to "
        "llm_models when empty; a smaller/cheaper model is usually fine.",
    )
    memory_ingest_web: bool = Field(
        default=True,
        description="When memory is enabled, auto-ingest web-search results as claims attributed to their "
        "source domain at the 'secondary' trust tier (deterministic provenance, and the only channel that "
        "can corroborate a claim into 'believed'). Each result is run through the extraction admission gate. "
        "Set false to skip the extra extraction calls per search.",
    )
    memory_ingest_notes: bool = Field(
        default=True,
        description="When memory is enabled, auto-ingest each incoming user note as a claim attributed to "
        "its author at the 'user' trust tier (deterministic provenance; never promotable to 'believed' on "
        "its own). The extractor's Skip branch drops chatter/opinions/personal details, and writes are "
        "rate-limited per author by global_write_cooldown. Set false to disable note learning.",
    )
    debug: Optional[bool] = None

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
        if self.memory_enabled:
            if not self.postgres_url:
                raise ValueError("postgres_url is required when memory_enabled is true")
            if not self.embedding_model:
                raise ValueError("embedding_model is required when memory_enabled is true")
        return self
