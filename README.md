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
  for clients like Zed, JetBrains, and [Buzz](https://github.com/block/buzz-app)
  through the installable Missbot plugin below.

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
[acpremote](https://github.com/vcoderun/acpkit):

```bash
acpremote mirror ws://your-host:8080/acp/ws --bearer-token "$ACP_TOKEN"
```

The endpoint serves `/acp/ws` for the socket, `/acp` for transport metadata, and
`/healthz` as a probe. Because stdout carries the protocol, both modes send all
logging to stderr.

### In Buzz (`block/buzz-app`)

The plugin in [`integrations/buzz`](integrations/buzz) connects **directly to a
remote Missbot ACP server**. Buzz needs only the plugin, the server's WebSocket
URL, and its access token—no local Python, `uv`, checkout, config file,
`missbot-acp` launcher, or `acpremote` installation.

1. On the **remote server**, run the updated ACP service using its own config
   and provider credentials:

   ```bash
   uv run python -m bot.acp serve -c config.local.yaml \
       --host 0.0.0.0 --port 8080 --token-env ACP_TOKEN
   ```

   Supply a nonempty `ACP_TOKEN` through the server's secret environment.
   Expose it through TLS as `wss://your-host/acp/ws`; the reverse proxy must
   forward WebSocket upgrades and subprotocol negotiation. `serve` does not
   terminate TLS itself. The plugin permits unencrypted `ws://` only on
   loopback for development testing; packaged Buzz's content policy requires
   `wss://`. The Fortuna deployment below already provides private TLS at
   `wss://missbot-acp.taile6e57.ts.net/acp/ws`.
2. In Buzz, open **Settings → Plugins → Manage plugins**, install/enable the
   `integrations/buzz` from a folder or Buzz's repository importer, or
   **Update/Reload** an existing installation. A repository import needs no
   local checkout or build.
   The new manifest requests **no host process or file permissions**.
3. Create an agent with **Harness → Missbot**, or open an existing Missbot
   agent's **Settings**. Under **Remote Missbot**, enter the **WebSocket
   endpoint** and **Access token**, then click **Connect**. It verifies
   authentication and private-session support before saving the settings.
   Add the agent to a conversation and mention it or send a one-to-one DM.
4. For per-author scores, enable `acp_parse_sender_header` on the **server**
   only when its ACP endpoint admits trusted clients. Otherwise callers share
   `acp_default_identity`. The plugin constructs the existing event header from
   Buzz's verified event, using public keys rather than display names.

Connection settings apply to this device's Missbot agents and live in browser
`sessionStorage`, never shared agent configuration. Re-enter them when the Buzz
session ends. Tokens are sent in an authenticated first-frame exchange, not URL
parameters or subprotocol strings. Remote URLs must not contain credentials,
queries, or fragments. Upgrade the remote server with this plugin; older servers
cannot perform its browser authentication handshake.

Any launcher created for the earlier local integration is no longer used.
The plugin neither executes it nor installs anything when enabled.

Buzz keeps its signing keys and handles reply publication. Missbot supplies text
through ACP, using its operator-configured tools; the plugin exposes no client
filesystem, terminal, or MCP tools. Channel replies stay in their thread; DMs and
Solo chats reply at channel level. Refusals, cancelled/incomplete replies, and
`NO_REPLY` are not published. Stop closes the connection, cancelling the remote
turn and discarding that conversation's in-memory history.

**Privacy:** every Buzz session requests `_meta: {"missbot": {"private": true}}`
on `session/new` and requires the same acknowledgement before sending content.
Upgrade the ACP server along with the plugin. Private sessions never recall or
retain shared Hindsight memories, even when memory is enabled for other
frontends. Persona, model-provider processing, configured tools, social scoring,
and operator telemetry still apply; this is not an offline or logging-free mode.

The plugin keeps at most eight conversation connections, evicting the least
recently used when needed. Conversations are isolated by community/agent,
channel, and thread. A bounded queue serializes up to twenty turns with a
five-minute run deadline. Restart, eviction, removal, or plugin unload loses
that conversation history. Publication is never retried automatically after an
uncertain relay acknowledgement; duplicate suppression remembers the latest
1024 completed event/agent pairs only while the plugin is loaded.
Attachments, Attention/watches, autonomous posts, session restoration, and
Buzz protected-agent policies are not supported by this conversational harness.

Plugin regression checks: `node --test integrations/buzz/plugin.test.mjs`
using Node.js 22.19 or newer.

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

Copy runtime configuration to `k8s/config.yaml`; it is ignored by Git and
generated as the `missbot-config` Secret because it can contain credentials.
Provision `missbot-secrets` separately for provider credentials. Routine apply
does not recreate it from the ignored local `k8s/secrets.txt`.

ACP also requires the separate `missbot-acp-auth` Secret. For first-time
bootstrap only (this refuses to overwrite an existing Secret):

```bash
openssl rand -hex 32 | tr -d '\n' | \
  kubectl --context admin@fortuna -n misskey create secret generic missbot-acp-auth \
    --from-file=ACP_TOKEN=/dev/stdin
```

```bash
cp config.example.yaml k8s/config.yaml
mise run build
mise run deploy
```

`mise run build` builds and pushes the image. `mise run deploy` applies the
Kustomize manifests, restarts both `missbot` and `missbot-acp`, and waits for
readiness. Both use the existing production config and backend credentials;
neither credential Secret is generated or rotated by routine apply.

The ACP service is available on the tailnet at
**`wss://missbot-acp.taile6e57.ts.net/acp/ws`**. The Tailscale Ingress supplies TLS;
the application additionally requires its token. On macOS, copy that token
directly to the clipboard without displaying it:

```bash
kubectl --context admin@fortuna -n misskey get secret missbot-acp-auth \
  -o jsonpath='{.data.ACP_TOKEN}' | base64 --decode | pbcopy
```

Paste it into the Buzz plugin's **Access token** field. For an ACP-only update,
build the image, apply `k8s/acp.yaml`, then restart and wait for
`deployment/missbot-acp`; the social bot need not restart. The shared production
config still controls models, tools, scores, and optional sender-header
attribution. Buzz's private sessions never access the shared Hindsight bank.

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
