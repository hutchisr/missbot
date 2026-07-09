# Missbot

Missbot is a Python-based Misskey/Fediverse chatbot that responds to mentions using LLMs and optional web search.

## Features

- WebSocket mention handling
- Context-aware replies
- Optional web search via SearXNG
- Image description via vision models
- Multi-endpoint fallback for LLMs

## Requirements

- Python 3.13+
- `uv` package manager

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

## Run

```bash
uv run python -m bot -c config.local.yaml
```

Or via Mise:

```bash
mise run bot
```

## Kubernetes

Copy the runtime configuration to `k8s/config.yaml` and put environment-style
provider secrets in `k8s/secrets.txt`; both files are ignored by Git. Kustomize
generates Kubernetes Secrets for them, including the config file because it
contains the Misskey token and may contain database credentials.

```bash
cp config.example.yaml k8s/config.yaml
mise run deploy
```

## Project Layout

- [bot/bot.py](bot/bot.py) — WebSocket client and message routing
- [bot/ai.py](bot/ai.py) — LLM orchestration
- [bot/models.py](bot/models.py) — Pydantic models
- [bot/tools.py](bot/tools.py) — Utility tools
- [config.example.yaml](config.example.yaml) — Configuration template
- [mise.toml](mise.toml) — Task automation
- [k8s/](k8s/) — Kubernetes manifests

## License

Dual-licensed under the **MIT License** or the **Grok Public License, Version 1**,
at your option — see [LICENSE.md](LICENSE.md).
