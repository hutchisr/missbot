"""Relay-driven private conversations; management clients are never on the reply path."""

import asyncio
import json
import re
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

import logfire
from nostr_sdk import PublicKey

from ..ai import ChatAgent
from ..core import AgentTurn, TurnAuthor
from ..models import Config
from .store import BuzzStore
from .transport import BuzzRelay

_HEX = re.compile(r"[0-9a-f]{64}")


def values(event: dict[str, Any], name: str) -> list[str]:
    return [tag[1] for tag in event["tags"] if len(tag) >= 2 and tag[0] == name]


class BuzzRunner:
    def __init__(self, config: Config, agent: ChatAgent, store: BuzzStore, relay: BuzzRelay, community: str):
        self.config, self.agent, self.store, self.relay, self.community = config, agent, store, relay, community
        self.wake = asyncio.Event()
        self.tasks: list[asyncio.Task] = []
        self.turn: asyncio.Task | None = None
        self.channel: str | None = None
        self.conversation: str | None = None
        self.stopping = False
        self._displays: dict[str, tuple[float, str]] = {}

    def start(self) -> None:
        record = self.store.community(self.community)
        assert record is not None
        self.tasks = [
            asyncio.create_task(self.relay.run(self.on_event, max(0, record["since"] - 300))),
            asyncio.create_task(self.work()),
        ]
        self.wake.set()

    async def on_event(self, event: dict[str, Any]) -> None:
        if (
            event["kind"] != 9
            or event["pubkey"] == self.relay.public_key
            or not event["content"].strip()
            or len(event["content"]) > self.config.acp_max_prompt_chars
            or len(values(event, "h")) != 1
            or event["created_at"] > time.time() + 300
        ):
            return
        record = self.store.community(self.community)
        if record is None or not record["active"]:
            return
        if self.store.enqueue(self.community, event):
            self.wake.set()

    async def work(self) -> None:
        # ponytail: one serial lane per community; split by conversation only if throughput requires it.
        while True:
            row = self.store.next_event(self.community)
            if row is None:
                self.wake.clear()
                await self.wake.wait()
                continue
            try:
                await self.relay.wait_ready()
            except RuntimeError:
                await asyncio.sleep(5)
                continue
            self.turn = asyncio.create_task(self.process(row))
            try:
                await self.turn
            except asyncio.CancelledError:
                if self.stopping:
                    raise
                self.store.abandon(self.community, row["id"])
            except Exception:
                logfire.exception("Hosted Buzz turn failed", event_id=row["id"], community=self.community)
                # An uncertain acknowledgement must not trigger automatic publication retry.
                # Crash recovery may resend a stored signed event, never regenerate its identity.
                self.store.abandon(self.community, row["id"])
            finally:
                self.turn = None
                self.channel = self.conversation = None

    async def _display_for(self, public_key: str) -> str:
        cached = self._displays.get(public_key)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]
        npub = PublicKey.parse(public_key).to_bech32()
        display = npub
        try:
            async with asyncio.timeout(5):
                profiles = await self.relay.query([{"kinds": [0], "authors": [public_key], "limit": 8}])
            profiles = [item for item in profiles if item["kind"] == 0 and item["pubkey"] == public_key]
            if profiles:
                latest = min(profiles, key=lambda item: (-item["created_at"], item["id"]))
                metadata = json.loads(latest["content"])
                if isinstance(metadata, dict):
                    for field in ("display_name", "name"):
                        value = metadata.get(field)
                        if not isinstance(value, str):
                            continue
                        name = " ".join("".join(c for c in value if c.isprintable() or c.isspace()).split())[:128]
                        if name:
                            display = f"{json.dumps(name, ensure_ascii=False)} (mention: @{npub}; use the name only without @)"
                            break
        except Exception:
            logfire.debug("Buzz profile lookup failed; using public key")
        # Cache misses too; names never determine identity or authorization.
        if public_key not in self._displays and len(self._displays) >= 1024:
            del self._displays[next(iter(self._displays))]
        self._displays[public_key] = (time.monotonic() + 300, display)
        return display

    async def process(self, row: dict[str, Any]) -> None:
        event = json.loads(row["event"])
        if row["outgoing"]:
            self.channel = values(event, "h")[0]
            self.conversation = json.loads(row["conversation"])[1] or ""
            outgoing = json.loads(row["outgoing"])
            if outgoing:
                await self.relay.publish_event(outgoing)
            self.store.finish(self.community, row["id"], history_turns=self.config.acp_max_history_turns)
            return
        channel = values(event, "h")[0]
        metadata = await self.relay.query([{"kinds": [39000, 39002], "#d": [channel], "limit": 8}])
        trusted = [
            item
            for item in metadata
            if item["pubkey"] == self.relay.relay_public_key and values(item, "d") == [channel]
        ]
        heads = {}
        for item in sorted(trusted, key=lambda item: (item["created_at"], -int(item["id"], 16))):
            heads[item["kind"]] = item
        if 39000 not in heads or 39002 not in heads:
            raise ValueError("Buzz channel metadata or roster unavailable")
        members = set(values(heads[39002], "p"))
        if self.relay.public_key not in members or event["pubkey"] not in members:
            self.store.finish(self.community, event["id"])
            return
        types = values(heads[39000], "t")
        dm = bool(types and types[0] == "dm")
        addressed = self.relay.public_key in values(event, "p")
        if not addressed and not (dm and members - {self.relay.public_key} == {event["pubkey"]}):
            self.store.finish(self.community, event["id"])
            return
        flat = dm or (
            types
            and types[0] == "stream"
            and "buzz.solo" in types[1:]
            and any(tag[0] == "private" for tag in heads[39000]["tags"] if tag)
        )
        root, reply = None, None
        for tag in event["tags"]:
            if len(tag) >= 4 and tag[0] == "e" and _HEX.fullmatch(tag[1]):
                if tag[3] == "root":
                    root = tag[1]
                elif tag[3] == "reply":
                    reply = tag[1]
        root = None if flat else (root or reply) if reply else event["id"]
        if root and root != event["id"]:
            ancestors = await self.relay.query([{"kinds": [9], "ids": [root], "limit": 1}])
            if not any(item["id"] == root and values(item, "h") == [channel] for item in ancestors):
                raise ValueError("Buzz thread root does not belong to this channel")
        conversation = json.dumps([channel, root])
        self.channel, self.conversation = channel, root or ""
        author = f"buzz:{event['pubkey']}"
        threshold = self.config.social_credit_ignore_threshold
        if threshold is not None:
            score = await self.agent.get_score(author)
            if score is not None and score < threshold:
                self.store.finish(self.community, event["id"])
                return
        history = self.store.history(self.community, conversation)
        turn = AgentTurn(
            text=event["content"],
            author=TurnAuthor(handle=author, user_id=author, display=await self._display_for(event["pubkey"])),
            source_id=f"buzz:{event['id']}",
            conversation_id=f"buzz:{self.community}:{conversation}",
            occurred_at=datetime.fromtimestamp(event["created_at"], tz=UTC),
            source="buzz_message",
            history=history,
            char_budget=65536,
            memory_access_allowed=False,
            previous_reply=next((item.text for item in reversed(history) if item.role == "assistant"), None),
        )
        async with asyncio.timeout(300):
            result = await self.agent.run(turn)
        if not result.strip() or result.strip() == "NO_REPLY":
            self.store.outbox(self.community, event["id"], conversation, author, "", {})
            self.store.finish(self.community, event["id"], history_turns=self.config.acp_max_history_turns)
            return
        if len(result) > 65536:
            raise ValueError("Hosted Buzz reply exceeds message size limit")
        tags = [["h", channel]]
        if root:
            tags.append(["e", root, "", "reply"])
        outgoing = self.relay.sign(result, tags)
        self.store.outbox(self.community, event["id"], conversation, author, result, outgoing)
        await self.relay.publish_event(outgoing)
        self.store.finish(self.community, event["id"], history_turns=self.config.acp_max_history_turns)

    async def interrupt(self, channel: str, root: str) -> bool:
        if self.turn is None or self.turn.done() or channel != self.channel or self.conversation not in ("", root):
            return False
        self.turn.cancel()
        with suppress(asyncio.CancelledError):
            await self.turn
        return True

    async def close(self) -> None:
        self.stopping = True
        for task in self.tasks:
            task.cancel()
        if self.turn is not None:
            self.turn.cancel()
        await asyncio.gather(*self.tasks, *([self.turn] if self.turn is not None else []), return_exceptions=True)
        await self.relay.aclose()
