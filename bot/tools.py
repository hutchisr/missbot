"""Tool utilities for Missbot."""

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Optional

import httpx
import logfire
from pydantic_ai import RunContext
from redis.asyncio import Redis

from .models import Config


def current_datetime() -> str:
    """Gets current date and time."""
    return str(datetime.now())


def normalize_username(username: str) -> str:
    """Normalize a handle to lowercase and drop a leading '@'."""
    username = username.strip().lower()
    if username.startswith("@"):
        username = username[1:]
    return username


async def apply_social_credit(redis: Redis, username: str, amount: int, reason: str) -> int:
    """Apply a score delta for ``username`` and record history + leaderboard.

    ``username`` must already be normalized. Returns the new score. Shared by the
    manual tool and the automatic message scorer so both record identically.
    """
    new_score = await redis.incrby(f"score:{username}", amount)  # type: ignore[misc]

    history_entry = json.dumps(
        {
            "amount": amount,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    history_key = f"history:{username}"
    # Pipeline history + leaderboard updates so they can't partially apply.
    # The incrby above is separate since zadd needs its returned score.
    pipe = redis.pipeline(transaction=True)
    pipe.lpush(history_key, history_entry)
    pipe.expire(history_key, 30 * 86400)  # 30-day TTL
    pipe.zadd("global:leaderboard", {username: float(new_score)})
    await pipe.execute()
    return new_score


def build_tools(config: Config, redis_client: Optional[Redis] = None) -> list[Callable[..., object]]:
    """Create tool functions for the given config.

    Tools are returned as plain functions and can be passed to Agent(..., tools=...).
    """
    tools: list[Callable[..., object]] = []

    def current_datetime_tool() -> str:
        """Gets current date and time."""
        return current_datetime()

    tools.append(current_datetime_tool)

    if config.searxng_url:

        @logfire.instrument()
        def search_web(query: str) -> Optional[str]:
            """Search the web for information."""
            auth: Optional[httpx.BasicAuth] = None
            if config.searxng_user and config.searxng_password:
                auth = httpx.BasicAuth(config.searxng_user, config.searxng_password)
            transport = httpx.HTTPTransport(retries=config.max_retries)
            with httpx.Client(auth=auth, transport=transport) as client:
                try:
                    response = client.post(
                        f"{config.searxng_url}search",
                        params={"q": query, "format": "json"},
                    )
                    response.raise_for_status()
                    data = response.json()
                    contents = [r.get("content") for r in data.get("results", [])[:5] if r.get("content")]
                    return "\n---\n".join(contents)
                except httpx.HTTPError:
                    logfire.exception("HTTP Error during web search")
                    return None

        tools.append(search_web)

    @logfire.instrument()
    def search_users(query: str, limit: int = 10, offset: int = 0) -> Optional[str]:
        """Search for users on this Misskey instance by username or display name.

        Args:
            query: The search query.
            limit: Maximum number of results to return (1-50, default 10)
            offset: Number of results to skip for pagination (default 0)
        """
        limit = max(1, min(50, limit))  # Clamp to 1-50
        transport = httpx.HTTPTransport(retries=config.max_retries)
        with httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(config.http_timeout_seconds),
        ) as client:
            try:
                response = client.post(
                    f"{config.url}api/users/search",
                    json={"query": query, "limit": limit, "offset": offset},
                    headers={"Authorization": f"Bearer {config.token}"},
                )
                response.raise_for_status()
                users = response.json()
                if not users:
                    return "No users found."
                results = []
                for user in users:
                    username = user.get("username", "unknown")
                    host = user.get("host")
                    name = user.get("name") or username
                    bio = user.get("description") or ""
                    handle = f"@{username}" + (f"@{host}" if host else "")
                    results.append(f"{name} ({handle}): {bio[:100]}")
                return "\n---\n".join(results)
            except httpx.HTTPError:
                logfire.exception("HTTP Error during user search")
                return None

    @logfire.instrument()
    def search_notes(query: str, limit: int = 10, offset: int = 0) -> Optional[str]:
        """Search for notes/posts on this Misskey instance.

        Args:
            query: The search query. Simple text search on note content.
            limit: Maximum number of results to return (1-50, default 10)
            offset: Number of results to skip for pagination (default 0)
        """
        limit = max(1, min(50, limit))  # Clamp to 1-50
        transport = httpx.HTTPTransport(retries=config.max_retries)
        with httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(config.http_timeout_seconds),
        ) as client:
            try:
                response = client.post(
                    f"{config.url}api/notes/search",
                    json={"query": query, "limit": limit, "offset": offset},
                    headers={"Authorization": f"Bearer {config.token}"},
                )
                response.raise_for_status()
                notes = response.json()
                if not notes:
                    return "No notes found."
                results = []
                for note in notes:
                    user = note.get("user", {})
                    username = user.get("username", "unknown")
                    host = user.get("host")
                    handle = f"@{username}" + (f"@{host}" if host else "")
                    text = note.get("text") or "(no text)"
                    results.append(f"{handle}: {text[:200]}")
                return "\n---\n".join(results)
            except httpx.HTTPError:
                logfire.exception("HTTP Error during note search")
                return None

    tools.extend([search_users, search_notes])

    # Social credit score tools (Redis-based)
    if redis_client:
        # Capture redis_client in closure with type assertion
        _redis: Redis = redis_client

        @logfire.instrument()
        async def get_social_credit(username: str) -> str:
            """Get a user's social credit score.

            Args:
                username: The username to look up (e.g. 'alice' for local, 'bob@remote.host' for remote).
            """
            username = normalize_username(username)
            try:
                score = await _redis.get(f"score:{username}")  # type: ignore[misc]
                if score is None:
                    return f"User @{username} has no social credit score yet (defaults to 0)."
                return f"User @{username} has {int(score)} social credit points."
            except Exception:
                logfire.exception("Error getting social credit score")
                return "Error retrieving social credit score."

        @logfire.instrument(extract_args=["username", "amount", "reason"])
        async def adjust_social_credit(ctx: RunContext[object], username: str, amount: int, reason: str) -> str:
            """Manually adjust a user's social credit score. Authorized users only.

            Regular users' scores are adjusted automatically based on the content of
            their own messages, so this tool refuses for non-privileged interactions.
            It works only when the author of the note being replied to is a privileged
            user (Config.social_credit_unrestricted_user_ids); then any user may be
            adjusted by any amount.

            Args:
                ctx: The run context (injected automatically).
                username: The username to adjust (e.g. 'alice' for local, 'bob@remote.host' for remote).
                amount: The amount to add (positive) or subtract (negative).
                reason: A brief explanation for the adjustment (required).
            """
            # Only privileged authors may drive manual adjustments. Regular users
            # cannot self-adjust here — their score moves only via the automatic,
            # injection-resistant message scorer.
            if not getattr(ctx.deps, "social_credit_unrestricted", False):
                return (
                    "Manual social credit adjustment is limited to authorized users. "
                    "Regular users' scores change automatically based on their messages."
                )

            username = normalize_username(username)
            if not reason or not reason.strip():
                return "Error: reason is required for social credit adjustments."

            # Prevent multiple adjustments per user per run
            adjusted = getattr(ctx.deps, "adjusted_credit_users", None)
            if adjusted is not None:
                if username in adjusted:
                    return f"Already adjusted @{username}'s social credit in this interaction. Only one adjustment per user per message is allowed."
                adjusted.add(username)
            try:
                new_score = await apply_social_credit(_redis, username, amount, reason)
                sign = "+" if amount >= 0 else ""
                return (
                    f"Adjusted @{username}'s social credit by {sign}{amount}. New score: {new_score}. Reason: {reason}"
                )
            except Exception:
                logfire.exception("Error adjusting social credit score")
                return "Error adjusting social credit score."

        @logfire.instrument()
        async def get_social_credit_history(username: str, limit: int = 10) -> str:
            """Get the history of social credit score changes for a user.

            Args:
                username: The username to look up (e.g. 'alice' for local, 'bob@remote.host' for remote).
                limit: Maximum number of history entries to return (default 10).
            """
            username = normalize_username(username)
            try:
                limit = max(1, min(50, limit))  # Clamp to 1-50

                # Get recent history entries
                entries = await _redis.lrange(f"history:{username}", 0, limit - 1)  # type: ignore[misc]

                if not entries:
                    return f"No social credit history found for @{username}."

                results = []
                for entry in entries:
                    data = json.loads(entry)
                    amount = data.get("amount", 0)
                    reason = data.get("reason", "No reason")
                    timestamp = data.get("timestamp", "Unknown time")
                    sign = "+" if amount >= 0 else ""
                    results.append(f"{timestamp}: {sign}{amount} - {reason}")

                return f"Social credit history for @{username}:\n" + "\n".join(results)
            except Exception:
                logfire.exception("Error getting social credit history")
                return "Error retrieving social credit history."

        @logfire.instrument()
        async def get_social_credit_leaderboard(limit: int = 10) -> str:
            """Get the top users by social credit score.

            Args:
                limit: Number of top users to return (default 10, max 50).
            """
            try:
                limit = max(1, min(50, limit))

                top_users = await _redis.zrange(  # type: ignore[misc]
                    "global:leaderboard",
                    0,
                    limit - 1,
                    desc=True,
                    withscores=True,
                )

                if not top_users:
                    return "No social credit scores recorded yet."

                results = []
                for rank, (username, score) in enumerate(top_users, 1):
                    results.append(f"{rank}. @{username}: {int(score)} points")

                return "Social Credit Leaderboard:\n" + "\n".join(results)
            except Exception:
                logfire.exception("Error getting social credit leaderboard")
                return "Error retrieving leaderboard."

        tools.extend(
            [
                get_social_credit,
                adjust_social_credit,
                get_social_credit_history,
                get_social_credit_leaderboard,
            ]
        )

    return tools
