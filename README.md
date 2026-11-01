# Missbot

Missbot is a chatbot built with Python and Pydantic AI. It listens to Misskey's
streaming API, builds conversation context, runs a configurable LLM tool loop,
and publishes the reply back into the thread. An optional Nostr frontend handles
public mentions and autonomous posts directly through relays, while Agent Client
Protocol clients reach the same persona over ACP.

## Frontends

Missbot supports four frontends over one shared brain:

- **Misskey/Fediverse** — WebSocket mentions, timeline auto-replies, autonomous
  posts.
- **Nostr** — signed relay events, public mentions, threaded replies, autonomous
  posts with images and polls, and NIP-05 identification. Runs alongside Misskey.
- **ACP** — [Agent Client Protocol](https://agentclientprotocol.com) over stdio or
  WebSocket for clients such as Zed and JetBrains.
- **Hosted Buzz** — a server-owned agent for [Buzz](https://github.com/block/buzz-app)
  that receives and answers community messages even when every desktop is closed.

Each is a thin adapter that translates its wire format into a neutral
`AgentTurn`; the agent itself never sees a platform type. Persona, memories, and
social credit use Hindsight and Redis. Misskey and Nostr share an agent instance;
ACP and hosted Buzz processes pointed at the same backends reach the same bot. User identities
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

To address the bot, send a public kind-1 note or NIP-22 comment (kind 1111) with
a proper `p` mention tag; pasting an `@npub` as plain text is not enough.
Replies to notes, nested comments, and poll comments are supported. NIP-22
comments must reference verifiable public notes or polls, and receive replies
in the same comment format. Only events created after startup are handled;
DMs and offline catch-up are not supported. When autonomous posting is
configured, Nostr generates separate posts rather than mirroring Misskey.
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

The plugin in [`integrations/buzz`](integrations/buzz) registers a genuine hosted
agent using Buzz's `AgentType.remote` interface. It manages the service; it does
not receive prompts, run models, or publish replies. **Grok continues answering
when Buzz or the plugin is closed.** ACP remains available separately.

1. Update/install `integrations/buzz` through Buzz's plugin manager. No local
   Python, checkout, launcher, or process/file permissions are needed.
2. Create a **new hosted agent** with **Harness → Missbot**. In its setup, use
   **`wss://missbot-buzz.taile6e57.ts.net/buzz/ws`** and the hosted management token,
   then click **Connect**. This is a different endpoint and token from ACP.
   Copy the token without displaying it:

   ```bash
   kubectl --context admin@fortuna -n misskey get secret missbot-buzz-auth \
     -o jsonpath='{.data.BUZZ_TOKEN}' | base64 --decode | pbcopy
   ```

3. Finish creating Grok in the desired community. Buzz signs the owner's NIP-OA
   authorization; the server verifies it before connecting and publishing the
   profile. Add Grok to a channel and mention it, or send a one-to-one DM.
   Group DMs still require a mention. Fortuna currently permits the Shrimp and
   Feedback community origins; other origins must be explicitly allowlisted
   by the server operator.

Grok reuses its existing `NOSTR_PRIVATE_KEY` from the server Secret, so its public
identity matches Nostr. The plugin never receives that key. The shared identity
can be provisioned repeatedly without generating another key. Existing desktop
agents are **not** silently converted: Buzz refuses a hosted provider taking over
a local key. If the same public key is already listed locally, resolve that
existing agent before creating the hosted entry.

The management token is saved unencrypted in device-local `localStorage` under
`missbot.hosted.v1`, outside shared agent configuration. It survives updates and
restarts; protect your Buzz profile. Old ACP settings are deliberately not reused.
Use `wss://`; unencrypted loopback is for development only. Tokens travel in the
first frame, never URLs. The plugin exposes provision, activation, profile,
removal, and interruption operations—not a desktop message runner.

Removing the hosted entry deactivates Buzz and clears its private conversation
state, **not Grok's shared Nostr key**. Ownership remains bound to the verified
owner. Profile changes preserve other metadata, including NIP-05 and the existing
picture when no replacement was supplied. They set Buzz's `bot: true` marker and
use its NIP-98-signed HTTP `/events` endpoint. Foreign `imeta` tags are not portable
into Buzz's local `/media/` namespace and are omitted; local tags from an existing
Buzz profile are retained. Updates go only to the Buzz community, never the
general Nostr relay set. Incomplete profile reads fail closed instead of
replacing unseen data.

The server verifies relay events, NIP-42 challenges and owner authorization;
channel metadata and rosters must be signed by the relay's NIP-11 `self` key.
Live message feeds explicitly name the bot's verified member channels (`#h`),
rather than relying on an unscoped message snapshot. Signed membership notices
refresh those feeds, with a 60-second reconciliation for missed notices and a
128-channel cap per community. Subscription readiness waits for EOSE.
Every Buzz turn disables shared Hindsight recall/retention. Author identities are
`buzz:<pubkey>`, independent of display names. Model providers, configured tools,
social scores and telemetry still apply.

SQLite on a retained PVC stores activation state, a bounded inbox, signed outbox
events, and private history by community/channel/thread. A restart restores active
communities without a desktop. Crash recovery reuses an already-signed reply's
event ID rather than generating another reply; observed uncertain publication
errors are not automatically retried. Each community has a serial worker, at most
128 pending events, 2048 completed deduplication records, and history for 128
conversations using `acp_max_history_turns`. Replay/query windows are bounded;
overload can drop input. Replies have a five-minute generation deadline.
Attachments, Attention/watches, autonomous posting, and Buzz protected-agent
execution are not provided by this text-conversation frontend.

For a separate server, use its own config, management token and Nostr key:

```bash
uv run python -m bot.buzz -c config.local.yaml --state /data/buzz.sqlite3 \
  --host 0.0.0.0 --port 8080 --token-env BUZZ_TOKEN \
  --community wss://your-community.example
```

Serve `/buzz/ws` through TLS and forward WebSocket subprotocols. `--community`
is repeatable and restricts which relay origins management clients can select.
Back up the state volume and runtime Secret separately; the database contains
owner authorizations and private conversation data but not the Nostr private key.

Plugin regression checks: `node --test integrations/buzz/plugin.test.mjs`.

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
Kustomize manifests, restarts `missbot`, `missbot-acp`, and `missbot-buzz`, and
waits for readiness. All use the existing production config and backend
credentials; credential Secrets are never generated or rotated by routine apply.

The ACP service is available on the tailnet at
**`wss://missbot-acp.taile6e57.ts.net/acp/ws`**. The Tailscale Ingress supplies TLS;
the application additionally requires its token. On macOS, copy that token
directly to the clipboard without displaying it:

```bash
kubectl --context admin@fortuna -n misskey get secret missbot-acp-auth \
  -o jsonpath='{.data.ACP_TOKEN}' | base64 --decode | pbcopy
```

This token is for ACP clients, not the hosted Buzz plugin. For a single-service
update, build the image, apply its manifest, then restart and wait for that
deployment; the other frontends need not restart.

The ignored local manifest `k8s/buzz.local.yaml` deploys the hosted service separately, with its own `missbot-buzz-auth`
Secret (`BUZZ_TOKEN`), `missbot-buzz-state` PVC, and private Tailscale ingress.
Provision this management Secret separately, just like ACP's. The 2-GiB state
volume uses `proxmox-retain`; do not delete its backing volume during an update.
The service uses the existing Nostr key from `missbot-secrets` and does not
generate or export a replacement key.
Kustomize references this local manifest; provision your own copy on a fresh
checkout before deploying. It is deliberately not committed.

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
