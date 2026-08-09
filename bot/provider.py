"""Shared identification headers for third-party model providers."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _load_project_version() -> str:
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as project_file:
        project_version = tomllib.load(project_file)["project"]["version"]
    if not isinstance(project_version, str):
        raise TypeError("project.version must be a string")
    return project_version


PROJECT_VERSION = _load_project_version()
APP_URL = "rad://zLseUdKik1qrsiTonrjSoPGYbC6g"


def provider_request_headers() -> dict[str, str]:
    """Return a fresh set of versioned headers for model-provider requests."""
    return {
        "User-Agent": f"Missbot/{PROJECT_VERSION}",
        "HTTP-Referer": APP_URL,
        "X-OpenRouter-Title": f"missbot-{PROJECT_VERSION}",
    }
