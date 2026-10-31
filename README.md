# Missbot

Missbot is a chatbot built with Python and Pydantic AI. It listens to Misskey's
streaming API, builds conversation context, runs a configurable LLM tool loop,
and publishes the reply back into the thread. An optional Nostr frontend handles
public mentions and autonomous posts directly through relays, while Agent Client
Protocol clients reach the same persona over ACP.

## Frontends

Missbot supports three frontends over one shared brain:

- **Misskey/Fediverse** — WebSocket mentions, timeline auto-replies, autonomous
  posts.
- **Nostr** — signed relay events, public mentions, threaded replies, autonomous
  posts with images and polls, and NIP-05 identification. Runs alongside Misskey.
- **ACP** — [Agent Client Protocol](https://agentclientprotocol.com) over stdio,
  for clients like Zed, JetBrains, and [buzz-acp](https://github.com/block/buzz).

Each is a thin adapter that translates its wire format into a neutral
`AgentTurn`; the agent itself never sees a platform type. Persona, memories, and
social credit use Hindsight and Redis. Misskey and Nostr share an agent instance;
an ACP process pointed at the same backends reaches the same bot. User identities
remain separate across networks; matching handles do not automatically link accounts.

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
- Sandboxed `run_python` calculations and data transformations through
  pydantic-monty, with no host filesystem, network, environment, or subprocess
  access and strict execution-time, memory, recursion, code, and output limits.
- Optional web search through an authenticated SearXNG instance.
- Streamable-HTTP MCP servers with tool prefixes, allow/block lists, and
  progressive `enable_<gate>` disclosure.
- Optional Redis-backed social credit with history and leaderboard tools,
  configurable categories, cooldowns, and an isolated classifier that maps
  constrained labels to code-owned score changes.
- Optional Hindsight long-term memory with automatic author-scoped pre-turn
  recall, lazy mental-model profiles for established users, post-turn structured
  retention, observation consolidation, provenance, and one shared bank.

### Safeguards and operations

- Direct-message and bot-account filtering by default to reduce accidental
  private memory access and bot-to-bot loops.
- SSRF checks for federated media URLs before images reach a model provider.
- Caps for reply mentions and concurrent handlers to bound notification,
  provider, Redis, memory, and HTTP load.
- Private interactions cannot access long-term memory. Hindsight recall,
  retention, and social-scoring failures are isolated from reply generation.
- Logfire instrumentation for Pydantic AI, HTTPX, Redis, and application events.
- Docker and Kustomize deployment with generated Secrets and in-cluster
  Hindsight connectivity.

## Requirements

- Python 3.13+
- `uv` package manager
- A Misskey account and API token for the bot

Redis, SearXNG, Hindsight, and MCP servers are optional.

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

### Enable Nostr alongside Misskey

Add the following to your config, then use the same `python -m bot` command:

```yaml
nostr_enabled: true
nostr_relays:
  - wss://relay.primal.net
  - wss://relay.ditto.pub
  - wss://relay.snort.social
nostr_private_key_env: NOSTR_PRIVATE_KEY
nostr_nip05: bot@example.com  # optional; requires a matching domain-side mapping
nostr_resolve_nip05: true
```

Supply `NOSTR_PRIVATE_KEY` through the environment or a Kubernetes Secret, as a
hex or `nsec` private key. Never commit it. For NIP-05, your domain must serve
`/.well-known/nostr.json` mapping the configured name to the bot's lowercase hex
public key. The bot checks that mapping before updating its signed profile,
preserving existing metadata. Incoming NIP-05 identifiers are display-only:
public keys still determine memory, scores, and permissions.

To address the bot, send a public kind-1 note with a proper `p` mention tag;
pasting an `@npub` as plain text is not enough. Only events created after startup
are handled; DMs and offline catch-up are not supported. When autonomous posting
is configured, Nostr generates separate posts rather than mirroring Misskey.
Generated images use Misskey drive as their public media host.

See [Nostr configuration and NIP-05 setup](AGENTS.md#nostr-frontend) for details,
limits, and the domain-side JSON example.

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
Kustomize manifests and restarts the deployment. The runtime configuration in
`k8s/config.yaml` points Missbot at the in-cluster Hindsight API service.

## Project Layout

- [bot/core.py](bot/core.py) — Frontend-neutral turn types shared by every adapter
- [bot/bot.py](bot/bot.py) — Misskey adapter: WebSocket client and message routing
- [bot/acp/](bot/acp/) — ACP adapter: protocol surface, sender attribution, sessions
- [bot/ai.py](bot/ai.py) — Model fallback, prompts, vision routing, and concurrent side work
- [bot/models.py](bot/models.py) — Runtime and configuration models
- [bot/tools.py](bot/tools.py) — Built-in and social-credit tools
- [bot/scoring.py](bot/scoring.py) — Constrained automatic score classification
- [bot/memory.py](bot/memory.py) — Hindsight recall/retain lifecycle and provenance fencing
- [bot/mcp.py](bot/mcp.py) — MCP server filtering, prefixes, and gates
- [bot/net.py](bot/net.py) — Federated-media SSRF protection
- [config.example.yaml](config.example.yaml) — Configuration template
- [mise.toml](mise.toml) — Task automation
- [k8s/](k8s/) — Kubernetes manifests

## License

Dual-licensed under the **MIT License** or the **Grok Public License, Version 1**,
at your option — see [LICENSE.md](LICENSE.md).
