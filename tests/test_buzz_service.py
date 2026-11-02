import asyncio
import hashlib
import json
from unittest.mock import AsyncMock

import pytest
from nostr_sdk import EventBuilder, Keys, Kind, Tag
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from bot.ai import ChatAgent
from bot.buzz.runner import BuzzRunner
from bot.buzz.service import HostedBuzz, validate_owner
from bot.buzz.store import BuzzStore
from bot.buzz.transport import BuzzRelay

COMMUNITY = "wss://community.example"


def signed(keys, kind, content, tags):
    return json.loads(EventBuilder(Kind(kind), content).tags([Tag.parse(tag) for tag in tags]).finalize(keys).as_json())


def proof(owner, target):
    return [
        "auth",
        owner.public_key().to_hex(),
        "",
        owner.sign_schnorr(hashlib.sha256(f"nostr:agent-auth:{target}:".encode()).digest()),
    ]


def test_owner_attestation_is_bound_to_key_and_shared_key_is_not_changed():
    owner, bot = Keys.generate(), Keys.generate()
    key = bot.public_key().to_hex()
    auth = proof(owner, key)
    assert validate_owner(auth, key) == owner.public_key().to_hex()
    with pytest.raises(ValueError):
        validate_owner(auth, Keys.generate().public_key().to_hex())
    with pytest.raises(ValueError):
        validate_owner([*auth[:2], "kind=9", auth[3]], key)
    with pytest.raises(ValueError):
        validate_owner([*auth[:3], "0" * 128], key)
    assert bot.public_key().to_hex() == key


class RelayFixture:
    def __init__(self, keys, community, auth):
        self.real = BuzzRelay(keys, community, auth)
        self.public_key = self.real.public_key
        self.authority = Keys.generate()
        self.relay_public_key = self.authority.public_key().to_hex()
        self.connected = False
        self.error = None
        self.ready = asyncio.Event()
        self.received = asyncio.Event()
        self.sent = []
        self.accepted = {}
        self.lose_ack = False
        self.events = []

    async def run(self, on_event, since):
        self.connected = True
        self.on_event = on_event
        self.ready.set()
        await asyncio.Future()

    async def wait_ready(self):
        await self.ready.wait()

    async def query(self, filters):
        kinds = filters[0]["kinds"]
        return [event for event in self.events if event["kind"] in kinds]

    def sign(self, content, tags, kind=9, created_at=None):
        return self.real.sign(content, tags, kind, created_at)

    async def publish(self, content, tags, kind=9, created_at=None):
        return await self.publish_event(self.sign(content, tags, kind, created_at))

    async def publish_event(self, event):
        self.sent.append(event)
        self.accepted[event["id"]] = event
        if event["kind"] == 0:
            self.events = [existing for existing in self.events if existing["kind"] != 0] + [event]
        self.received.set()
        if self.lose_ack:
            raise RuntimeError("Simulated acknowledgement lost after acceptance")
        return event["id"]

    async def aclose(self):
        self.connected = False


@pytest.mark.anyio
async def test_hosted_lifecycle_preserves_identity_profile_and_owner_on_removal(make_config, tmp_path, monkeypatch):
    keys, owner = Keys.generate(), Keys.generate()
    pubkey = keys.public_key().to_hex()
    auth = proof(owner, pubkey)
    config = make_config(nostr_relays=[])
    store = BuzzStore(str(tmp_path / "state.db"))
    created = []

    def relay_factory(*args):
        relay = RelayFixture(*args)
        relay.events = [
            signed(
                keys,
                0,
                json.dumps(
                    {
                        "name": "Grok",
                        "nip05": "grok@example.com",
                        "picture": "https://example.com/a.png",
                        "about": "Keep this",
                    }
                ),
                [
                    ["client", "existing"],
                    ["imeta", "url https://external.example/avatar.png", "m image/png"],
                    ["imeta", "url /media/local-avatar.png", "m image/png"],
                ],
            )
        ]
        created.append(relay)
        return relay

    monkeypatch.setattr("bot.buzz.service.BuzzRelay", relay_factory)
    service = HostedBuzz(config, ChatAgent(config), store, keys, [COMMUNITY])
    try:
        first = await service.call("provision", {"name": "Grok", "community": COMMUNITY})
        assert first == await service.call("provision", {"name": "Ignored replay", "community": COMMUNITY})
        assert first["pubkey"] == pubkey
        await service.call("activate", {"pubkey": pubkey, "name": "Hosted Grok", "community": COMMUNITY, "auth": auth})
        profile = created[0].sent[0]
        assert json.loads(profile["content"]) == {
            "name": "Hosted Grok",
            "bot": True,
            "nip05": "grok@example.com",
            "picture": "https://example.com/a.png",
            "about": "Keep this",
        }
        assert ["client", "existing"] in profile["tags"]
        assert ["imeta", "url /media/local-avatar.png", "m image/png"] in profile["tags"]
        assert not any("https://external.example/avatar.png" in value for tag in profile["tags"] for value in tag)
        assert auth in profile["tags"]
        await service.call("activate", {"pubkey": pubkey, "name": "Hosted Grok", "community": COMMUNITY, "auth": auth})
        assert created[0].sent == [profile]
        await service.call("remove", {"pubkey": pubkey, "community": COMMUNITY})
        await service.call("remove", {"pubkey": pubkey, "community": COMMUNITY})
        assert service.public_key == pubkey
        inactive = store.community(COMMUNITY)
        assert inactive is not None and not inactive["active"]
        assert not any(event["kind"] == 5 for event in created[0].sent)
        with pytest.raises(ValueError, match="another owner"):
            await service.call(
                "activate",
                {"pubkey": pubkey, "name": "Takeover", "community": COMMUNITY, "auth": proof(Keys.generate(), pubkey)},
            )
        with pytest.raises(ValueError, match="allowlist"):
            await service.call("provision", {"name": "Grok", "community": "wss://attacker.example"})
    finally:
        await service.close()
        store.close()


@pytest.mark.anyio
async def test_signed_outbox_survives_restart_without_regeneration_or_shared_memory(make_config, tmp_path):
    bot, owner, author = Keys.generate(), Keys.generate(), Keys.generate()
    pubkey = bot.public_key().to_hex()
    auth = proof(owner, pubkey)
    config = make_config(memory_enabled=True)
    memory = AsyncMock()
    agent = ChatAgent(config, memory=memory)
    calls = 0

    def answer(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("A private reply")])

    path = str(tmp_path / "state.db")
    store = BuzzStore(path)
    store.provision(COMMUNITY, "Grok", None)
    store.activate(COMMUNITY, owner.public_key().to_hex(), auth)
    relay = RelayFixture(bot, COMMUNITY, auth)
    channel = "private-chat"
    relay.events = [
        signed(relay.authority, 39000, "", [["d", channel], ["t", "dm"], ["private"]]),
        signed(relay.authority, 39002, "", [["d", channel], ["p", pubkey], ["p", author.public_key().to_hex()]]),
    ]
    event = signed(author, 9, "Private conversation", [["h", channel]])
    runner = BuzzRunner(config, agent, store, relay, COMMUNITY)
    await runner.on_event(event)
    row = store.next_event(COMMUNITY)
    assert row is not None
    relay.lose_ack = True
    with agent._agent.override(model=FunctionModel(answer)), pytest.raises(RuntimeError):
        await runner.process(row)
    pending = store.next_event(COMMUNITY)
    assert pending is not None
    outgoing = pending["outgoing"]
    store.close()
    # Simulate process loss before it can classify the uncertain result: recover the exact signed outbox.
    restored = BuzzStore(path)
    relay.lose_ack = False
    restarted = BuzzRunner(config, agent, restored, relay, COMMUNITY)
    try:
        await restarted.process(restored.next_event(COMMUNITY))
        assert calls == 1
        assert len(relay.accepted) == 1
        assert relay.sent[0]["id"] == relay.sent[1]["id"] == json.loads(outgoing)["id"]
        assert relay.sent[1]["tags"] == [["h", channel], auth]
        assert not restored.enqueue(COMMUNITY, event)
        history = restored.history(COMMUNITY, json.dumps([channel, None]))
        assert [turn.text for turn in history] == ["Private conversation", "A private reply"]
        memory.recall_for_turn.assert_not_awaited()
        memory.retain_turn.assert_not_awaited()
    finally:
        restored.close()


@pytest.mark.anyio
async def test_runner_continues_without_management_and_does_not_retry_uncertain_publication(make_config, tmp_path):
    config = make_config()
    bot, owner, author = Keys.generate(), Keys.generate(), Keys.generate()
    pubkey = bot.public_key().to_hex()
    store = BuzzStore(str(tmp_path / "state.db"))
    store.provision(COMMUNITY, "Grok", None)
    store.activate(COMMUNITY, owner.public_key().to_hex(), proof(owner, pubkey))
    relay = RelayFixture(bot, COMMUNITY, proof(owner, pubkey))
    relay.lose_ack = True
    relay.events = [
        signed(relay.authority, 39000, "", [["d", "chat"], ["t", "dm"]]),
        signed(relay.authority, 39002, "", [["d", "chat"], ["p", pubkey], ["p", author.public_key().to_hex()]]),
    ]
    agent = ChatAgent(config)
    runner = BuzzRunner(config, agent, store, relay, COMMUNITY)
    try:
        with agent._agent.override(
            model=FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("Independent reply")]))
        ):
            runner.start()
            await relay.wait_ready()
            await runner.on_event(signed(author, 9, "Wake without a desktop", [["h", "chat"]]))
            await asyncio.wait_for(relay.received.wait(), 5)
            async with asyncio.timeout(5):
                while store.next_event(COMMUNITY) is not None:
                    await asyncio.sleep(0)
            assert len(relay.sent) == 1
            assert store.history(COMMUNITY, json.dumps(["chat", None])) == []
    finally:
        await runner.close()
        store.close()


@pytest.mark.anyio
@pytest.mark.parametrize("profile,expected_name", [
    ('{"display_name":"Rachel","name":"rachel"}', "Rachel"),
    ('{"display_name":" ","name":"rachel"}', "rachel"),
    ("{invalid", None),
    ("[]", None),
    (None, None),
    ("offline", None),
])
async def test_buzz_name_is_display_only_and_lookup_failures_preserve_reply(
    make_config, tmp_path, profile, expected_name
):
    bot, owner, author = Keys.generate(), Keys.generate(), Keys.generate()
    key = author.public_key().to_hex()
    auth = proof(owner, bot.public_key().to_hex())
    store = BuzzStore(str(tmp_path / "names.db"))
    store.provision(COMMUNITY, "Grok", None)
    store.activate(COMMUNITY, owner.public_key().to_hex(), auth)
    relay = RelayFixture(bot, COMMUNITY, auth)
    relay.events = [
        signed(relay.authority, 39000, "", [["d", "chat"], ["t", "dm"]]),
        signed(relay.authority, 39002, "", [["d", "chat"], ["p", bot.public_key().to_hex()], ["p", key]]),
        signed(owner, 0, '{"display_name":"Wrong person"}', []),
    ]
    if profile is not None and profile != "offline":
        relay.events.append(signed(author, 0, profile, []))
    original_query = relay.query

    async def query(filters):
        if profile == "offline" and filters[0]["kinds"] == [0]:
            raise RuntimeError("Profile unavailable")
        return await original_query(filters)

    relay.query = query
    agent = AsyncMock()
    agent.run.return_value = "A private reply"
    runner = BuzzRunner(make_config(), agent, store, relay, COMMUNITY)
    try:
        await runner.on_event(signed(author, 9, "Hello", [["h", "chat"]]))
        await runner.process(store.next_event(COMMUNITY))
        turn = agent.run.await_args.args[0]
        npub = author.public_key().to_bech32()
        assert turn.author.display == (
            f'"{expected_name}" (mention: @{npub}; use the name only without @)' if expected_name else npub
        )
        assert turn.author.handle == turn.author.user_id == f"buzz:{key}"
        assert not turn.author.privileged and not turn.memory_access_allowed
        assert relay.sent[0]["content"] == "A private reply"
        # A cached name or miss must keep replies independent of subsequent profile outages.
        relay.query = AsyncMock(side_effect=RuntimeError("Relay offline"))
        assert await runner._display_for(key) == turn.author.display
        relay.query.assert_not_awaited()
    finally:
        store.close()
