from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, WebsocketUrl, model_validator
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
    max_context: int = Field(gt=0, default=1, description="Number of context messages to include")
    mcp_servers: List[MCPServerConfig] = Field(
        default_factory=list,
        description="Streamable-HTTP MCP servers to expose as tools.",
    )
    debug: Optional[bool] = None

    @model_validator(mode="after")
    def check_auto_post_config(self) -> "Config":
        if self.auto_post_interval and not self.system_prompt_auto:
            raise ValueError("system_prompt_auto is required when auto_post_interval is set")
        return self
