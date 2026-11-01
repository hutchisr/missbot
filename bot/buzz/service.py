"""Hosted lifecycle for the existing Grok Nostr identity; never exports or deletes its key."""

import asyncio
import hashlib
import json
import re
import time
from typing import Any
from urllib.parse import urlsplit

from nostr_sdk import Event, Keys

from ..ai import ChatAgent
from ..models import Config
from ..nostr_transport import NostrTransport
from .runner import BuzzRunner
from .store import BuzzStore
from .transport import BuzzRelay

_KEY = re.compile(r"[0-9a-f]{64}")
_SIG = re.compile(r"[0-9a-f]{128}")


def community_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ("https", "wss", "http", "ws") or not parsed.hostname:
        raise ValueError("Invalid Buzz community origin")
    if parsed.scheme in ("http", "ws") and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Buzz communities require TLS")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("Community must be an origin without credentials, paths, queries, or fragments")
    scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    return f"{scheme}://{parsed.netloc.lower()}"


def validate_owner(auth: Any, pubkey: str) -> str:
    if (
        not isinstance(auth, list)
        or len(auth) != 4
        or any(not isinstance(item, str) for item in auth)
        or auth[0] != "auth"
        or not _KEY.fullmatch(auth[1])
        or auth[1] == pubkey
        or auth[2] != ""
        or not _SIG.fullmatch(auth[3])
    ):
        raise ValueError("A valid unrestricted Buzz owner authorization is required")
    digest = hashlib.sha256(f"nostr:agent-auth:{pubkey}:".encode()).hexdigest()
    # SDK exposes generic Schnorr verification through Event.verify_signature():
    # verify only this supplied digest/signature, deliberately not the event-id hash.
    proof = Event.from_json(
        json.dumps(
            {
                "id": digest,
                "pubkey": auth[1],
                "sig": auth[3],
                "created_at": 0,
                "kind": 0,
                "tags": [],
                "content": "",
            }
        )
    )
    if not proof.verify_signature():
        raise ValueError("Invalid Buzz owner authorization signature")
    return auth[1]


def name_value(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 64 or any(ord(c) < 32 for c in value):
        raise ValueError("Agent name must contain 1-64 printable characters")
    return value.strip()


def picture_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError("Invalid profile picture URL")
    url = urlsplit(value)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("Profile picture must be an HTTPS URL without credentials")
    return value


class HostedBuzz:
    def __init__(self, config: Config, agent: ChatAgent, store: BuzzStore, keys: Keys, allowed: list[str]):
        self.config, self.agent, self.store, self.keys = config, agent, store, keys
        self.public_key = keys.public_key().to_hex()
        self.allowed = {community_url(value) for value in allowed}
        if not self.allowed or len(self.allowed) > 8:
            raise ValueError("Configure between one and eight allowed Buzz communities")
        self.runners: dict[str, BuzzRunner] = {}
        self.lock = asyncio.Lock()
        self.profiles = (
            NostrTransport(keys.secret_key().to_hex(), [str(url) for url in config.nostr_relays])
            if config.nostr_relays
            else None
        )

    async def start(self) -> None:
        for record in self.store.communities():
            if record["active"] and record["community"] in self.allowed:
                self._start(record)

    def _start(self, record: dict[str, Any]) -> BuzzRunner:
        community = record["community"]
        existing = self.runners.get(community)
        if existing is not None:
            return existing
        auth = json.loads(record["auth"])
        if validate_owner(auth, self.public_key) != record["owner"]:
            raise ValueError("Stored owner authorization does not match its owner")
        relay = BuzzRelay(self.keys, community, auth)
        runner = BuzzRunner(self.config, self.agent, self.store, relay, community)
        self.runners[community] = runner
        runner.start()
        return runner

    async def close(self) -> None:
        await asyncio.gather(*(runner.close() for runner in self.runners.values()))
        self.runners.clear()

    def status(self) -> dict[str, Any]:
        communities = []
        for record in self.store.communities():
            runner = self.runners.get(record["community"])
            communities.append(
                {
                    "community": record["community"],
                    "name": record["name"],
                    "picture": record["picture"],
                    "active": bool(record["active"]),
                    "connected": bool(runner and runner.relay.connected),
                    "error": getattr(runner.relay, "error", None) if runner else None,
                }
            )
        return {"pubkey": self.public_key, "communities": communities}

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "status":
            return self.status()
        if method not in ("provision", "activate", "profile", "remove", "interrupt"):
            raise ValueError("Unknown hosted Buzz operation")
        if not isinstance(params.get("community"), str):
            raise TypeError("Community is required")
        community = community_url(params["community"])
        if community not in self.allowed:
            raise ValueError("Community is not in this hosted service's operator allowlist")
        if method != "provision" and params.get("pubkey") != self.public_key:
            raise ValueError("This service manages only Grok's configured Nostr identity")
        if method == "interrupt":
            channel, root = params.get("channelId"), params.get("threadRootId")
            if not isinstance(channel, str) or not isinstance(root, str):
                raise ValueError("Channel and thread are required")
            runner = self.runners.get(community)
            return {"interrupted": bool(runner and await runner.interrupt(channel, root))}
        async with self.lock:
            if method == "provision":
                record = self.store.provision(
                    community, name_value(params.get("name")), picture_value(params.get("picture"))
                )
                return {"pubkey": self.public_key, **({"picture": record["picture"]} if record["picture"] else {})}
            record = self.store.community(community)
            if method == "remove":
                # Stop and drain before clearing private state. Never touch the shared key.
                runner = self.runners.pop(community, None)
                if runner:
                    await runner.close()
                self.store.deactivate(community)
                return {}
            if record is None:
                raise ValueError("Provision this hosted identity in the community first")
            if method == "activate":
                owner = validate_owner(params.get("auth"), self.public_key)
                if record["owner"] and record["owner"] != owner:
                    raise ValueError("This community's hosted identity belongs to another owner")
                name = name_value(params.get("name"))
                self.store.profile(community, name, record["picture"])
                was_active = bool(record["active"])
                self.store.activate(community, owner, params["auth"], active=was_active)
                record = self.store.community(community)
                assert record is not None
                runner = self._start(record)
                try:
                    await runner.relay.wait_ready()
                    await self._profile(runner, name, record["picture"])
                except BaseException:
                    if not was_active:
                        self.runners.pop(community, None)
                        await runner.close()
                    raise
                self.store.activate(community, owner, params["auth"])
                return {}
            if not record["active"]:
                raise ValueError("Activate the hosted identity before editing its profile")
            runner = self._start(record)
            name = name_value(params.get("name"))
            picture = picture_value(params["picture"]) if "picture" in params else record["picture"]
            await runner.relay.wait_ready()
            await self._profile(runner, name, picture, clear_picture="picture" in params and picture is None)
            self.store.profile(community, name, picture)
            return {}

    async def _profile(
        self, runner: BuzzRunner, name: str, picture: str | None, *, clear_picture: bool = False
    ) -> None:
        events = await runner.relay.query([{"authors": [self.public_key], "kinds": [0], "limit": 1}])
        community_ids = {event["id"] for event in events}
        if self.profiles is not None:
            public = await self.profiles.fetch_profile(self.public_key)
            if public is not None:
                events.append(public)
        previous = min(events, key=lambda item: (-item["created_at"], item["id"])) if events else None
        metadata = json.loads(previous["content"]) if previous else {}
        if not isinstance(metadata, dict):
            raise TypeError("Existing Grok profile is malformed; refusing to replace it")
        updated = {**metadata, "name": name, "bot": True}
        if picture:
            updated["picture"] = picture
        elif clear_picture:
            updated.pop("picture", None)
        original_tags = [tag for tag in previous["tags"] if tag and tag[0] != "auth"] if previous else []
        tags = []
        for tag in original_tags:
            if tag[0] == "imeta":
                urls = [value[4:] for value in tag[1:] if value.startswith("url ")]
                # These describe relay-local media, not portable profile fields.
                # Keep local Buzz tags, but never reinterpret foreign media in this community.
                if (
                    previous is None
                    or previous["id"] not in community_ids
                    or not urls
                    or any(not url.startswith("/media/") for url in urls)
                ):
                    continue
            tags.append(tag)
        record = self.store.community(runner.community)
        assert record is not None
        auth = json.loads(record["auth"])
        if (
            previous is not None
            and previous["id"] in community_ids
            and updated == metadata
            and tags == original_tags
            and [tag for tag in previous["tags"] if tag and tag[0] == "auth"] == [auth]
        ):
            return
        content = json.dumps(updated, ensure_ascii=False, allow_nan=False)
        # Only the Buzz relay receives this event, never the general Nostr relay set.
        stamp = max(int(time.time()), (previous["created_at"] + 1) if previous else 0)
        if stamp > time.time() + 300:
            raise ValueError("Existing profile timestamp is too far in the future")
        await runner.relay.publish(content, tags, kind=0, created_at=stamp)
