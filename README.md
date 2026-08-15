# Missbot

> [!IMPORTANT]
> This GitHub repository is a mirror of the canonical
> [Radicle repository](https://radicle.network/nodes/index.radicle.garden/rad:zLseUdKik1qrsiTonrjSoPGYbC6g).
> Radicle is the source of truth for branches, issues, and patches. To work from
> the canonical repository, run:
>
> ```bash
> rad clone rad:zLseUdKik1qrsiTonrjSoPGYbC6g
> ```

Missbot is a chatbot built with Python and Pydantic AI. It listens to Misskey's
streaming API, builds conversation context, runs a configurable LLM tool loop,
and publishes the reply back into the thread — and it serves that same agent
over the Agent Client Protocol so ACP clients reach the same persona.

## Frontends

Missbot runs two frontends over one shared brain:

- **Misskey/Fediverse** — WebSocket mentions, timeline auto-replies, autonomous
  posts.
- **ACP** — [Agent Client Protocol](https://agentclientprotocol.com) over stdio,
  for clients like Zed, JetBrains, and [buzz-acp](https://github.com/block/buzz).

Each is a thin adapter that translates its wire format into a neutral
`AgentTurn`; the agent itself never sees a platform type. Persona, memories, and
social credit live in Postgres and Redis, so an ACP process pointed at the same
backends is the same bot, not a copy of it.

## Features

### Conversation and model routing

- Real-time WebSocket handling for mentions, with reply-chain context and
  automatic reconnection.
- Optional automatic replies to timeline notes and autonomous timeline posts,
  each with configurable intervals and jitter.
- Autonomous posts may attach a model-composed Misskey poll with optional
  multi-select and expiration.
- Multi-provider Pydantic AI model chains with automatic fallback on provider
  errors and timeouts.
- Custom OpenAI-compatible endpoints alongside Pydantic AI provider strings.
- Multimodal prompts from image and video thumbnails, with text-only models
  skipped automatically when a fallback chain receives images.
- Configurable token limits, sampling controls, and deterministic guards against
  over-length or verbatim-repeat replies.

### Tools, integrations, and state

- Built-in current-time, Misskey user search, and Misskey note search tools.
- Optional web search through an authenticated SearXNG instance.
- Streamable-HTTP MCP servers with tool prefixes, allow/block lists, and
  progressive `enable_<gate>` disclosure.
- Optional Redis-backed social credit with history and leaderboard tools,
  configurable categories, cooldowns, and an isolated classifier that maps
  constrained labels to code-owned score changes.
- Optional mem0 long-term memory backed by Postgres/pgvector, including explicit
  `add_memory`/`search_memory` tools, remote reranking, and automatic ingestion
  of public notes.
- Retention, deduplication, expiration, and per-author limits through a dry-run
  capable maintenance command and Kubernetes CronJob.

### Safeguards and operations

- Direct-message and bot-account filtering by default to reduce accidental
  private ingestion and bot-to-bot loops.
- SSRF checks for federated media URLs before images reach a model provider.
- Caps for reply mentions and concurrent handlers to bound notification,
  provider, Redis, memory, and HTTP load.
- Private interactions cannot write long-term memory; public-note memory and
  social scoring failures are isolated from reply generation.
- Logfire instrumentation for Pydantic AI, HTTPX, Redis, and application events.
- Docker and Kustomize deployment with generated Secrets and scheduled memory
  maintenance.

## Requirements

- Python 3.13+
- `uv` package manager
- A Misskey account and API token for the bot

Redis, SearXNG, Postgres/pgvector, MCP servers, and long-term memory are optional.

## Setup

```bash
uv sync
```

## Configuration

Copy and edit the example config:

```bash
cp config.example.yaml config.local.yaml
```

See [config.example.yaml](config.example.yaml) for required and optional fields.
At minimum, configure the Misskey connection, bot identity, model chain, and
system prompt. Provider credentials should be supplied through environment
variables or an ignored local secrets file rather than committed to the repo.

## Run

```bash
uv run python -m bot -c config.local.yaml
```

Or via Mise:

```bash
mise run bot
```

### As an ACP agent

For clients that spawn agents as subprocesses (Zed, JetBrains), run the stdio
mode and point the client at that command:

```bash
uv run python -m bot.acp stdio -c config.local.yaml
```

For remote consumers, serve over WebSocket instead:

```bash
uv run python -m bot.acp serve -c config.local.yaml \
    --host 0.0.0.0 --port 8080 --token-env ACP_TOKEN
```

Remote clients bridge back to stdio ACP with
[acpremote](https://github.com/vcoderun/acpkit), which is what a consumer such
as buzz-acp is configured with:

```bash
acpremote mirror ws://your-host:8080/acp/ws --bearer-token "$ACP_TOKEN"
```

The endpoint serves `/acp/ws` for the socket, `/acp` for transport metadata, and
`/healthz` as a probe. Because stdout carries the protocol, both modes send all
logging to stderr.

## Memory Maintenance

When long-term memory is enabled, inspect a cleanup pass before deleting
anything:

```bash
uv run python -m bot.maintenance cleanup --dry-run -c config.local.yaml
```

Run the same cleanup for real after reviewing the summary:

```bash
uv run python -m bot.maintenance cleanup -c config.local.yaml
```

Explicit `add_memory` entries are protected from retention and per-author cap
cleanup. The Kubernetes deployment runs the destructive form on a schedule.

## Development

```bash
OPENROUTER_API_KEY=sk-dummy uv run pytest -q
uv run ruff check bot/
uv run ruff format --check bot/
uv run ty check bot/
```

The dummy OpenRouter key is required during test collection; tests mock provider
network calls.

## Kubernetes

Copy the runtime configuration to `k8s/config.yaml` and put environment-style
provider secrets in `k8s/secrets.txt`; both files are ignored by Git. Kustomize
generates Kubernetes Secrets for them, including the config file because it
contains the Misskey token and may contain database credentials.

```bash
cp config.example.yaml k8s/config.yaml
mise run build
mise run deploy
```

`mise run build` builds and pushes the image. `mise run deploy` applies the
Kustomize manifests and restarts the deployment. Edit
[k8s/maintenance-settings.yaml](k8s/maintenance-settings.yaml) to change the
memory-cleanup schedule or timezone.

## Project Layout

- [bot/core.py](bot/core.py) — Frontend-neutral turn types shared by every adapter
- [bot/bot.py](bot/bot.py) — Misskey adapter: WebSocket client and message routing
- [bot/acp/](bot/acp/) — ACP adapter: protocol surface, sender attribution, sessions
- [bot/ai.py](bot/ai.py) — Model fallback, prompts, vision routing, and concurrent side work
- [bot/models.py](bot/models.py) — Runtime and configuration models
- [bot/tools.py](bot/tools.py) — Built-in, social-credit, and memory tools
- [bot/scoring.py](bot/scoring.py) — Constrained automatic score classification
- [bot/memory.py](bot/memory.py) — mem0 adapter for runtime and maintenance access
- [bot/maintenance.py](bot/maintenance.py) — Long-term-memory cleanup CLI
- [bot/mcp.py](bot/mcp.py) — MCP server filtering, prefixes, and gates
- [bot/net.py](bot/net.py) — Federated-media SSRF protection
- [config.example.yaml](config.example.yaml) — Configuration template
- [mise.toml](mise.toml) — Task automation
- [k8s/](k8s/) — Kubernetes manifests

## License

Dual-licensed under the **MIT License** or the **Grok Public License, Version 1**,
at your option — see [LICENSE.md](LICENSE.md).
