"""Exercise signatures and NIP-01 ACKs against a real local WebSocket relay."""

import asyncio
import json
import time
from contextlib import suppress

import pytest
from nostr_sdk import Event, EventBuilder, Keys, Kind, Tag, Timestamp
from websockets.asyncio.server import serve

from bot import nostr_transport
from bot.nostr_transport import MENTIONS_SUBSCRIPTION, NostrTransport


def signed_event(keys, content="hello", tags=(), created_at=None, kind=1):
    builder = EventBuilder(Kind(kind), content).tags([Tag.parse(tag) for tag in tags])
    if created_at is not None:
        builder = builder.custom_created_at(Timestamp.from_secs(created_at))
    return builder.finalize(keys)


async def stop(task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def test_keys_and_signature_boundary():
    keys = Keys.generate()
    secret = keys.secret_key()
    relay = ["wss://relay.example.com"]
    assert NostrTransport(secret.to_hex(), relay).public_key == keys.public_key().to_hex()
    assert NostrTransport(secret.to_bech32(), relay).public_key == keys.public_key().to_hex()
    with pytest.raises(ValueError, match="Invalid Nostr private key") as error:
        NostrTransport("not-a-key-SECRET", relay)
    assert "SECRET" not in str(error.value)
    event = signed_event(keys)
    verified = NostrTransport._verified_event(event)
    assert verified is not None and verified["content"] == "hello"
    tampered = json.loads(event.as_json())
    tampered["content"] = "forged"
    assert NostrTransport._verified_event(Event.from_json(json.dumps(tampered))) is None
    future = signed_event(keys, created_at=int(time.time()) + 3600)
    assert NostrTransport._verified_event(future) is None


def test_real_relay_mentions_publish_fetch_and_cancellation():
    async def scenario():
        bot_keys, author = Keys.generate(), Keys.generate()
        public_key = bot_keys.public_key().to_hex()
        mentions = [["p", public_key]]
        ancestor = signed_event(author, "root", created_at=int(time.time()) - 600)
        received = asyncio.Queue()
        published = asyncio.Queue()
        disconnected = asyncio.Event()

        async def relay(websocket):
            try:
                async for raw in websocket:
                    message = json.loads(raw)
                    if message[0] == "REQ":
                        subscription, request = message[1:3]
                        if subscription == MENTIONS_SUBSCRIPTION:
                            assert request["kinds"] == [1]
                            assert request["#p"] == [public_key]
                            assert abs(request["since"] - time.time()) < 5
                            assert request["limit"] <= 128
                            valid = signed_event(author, tags=mentions, created_at=request["since"])
                            tampered = json.loads(valid.as_json())
                            tampered["sig"] = "00" * 64
                            await websocket.send(json.dumps(["EVENT", subscription, tampered]))
                            for invalid in (
                                signed_event(author, "old", mentions, request["since"] - 1),
                                signed_event(author, "future", mentions, int(time.time()) + 3600),
                                signed_event(bot_keys, "own", mentions),
                                signed_event(author, "not a mention"),
                                signed_event(author, "private", mentions, kind=4),
                            ):
                                await websocket.send(json.dumps(["EVENT", subscription, json.loads(invalid.as_json())]))
                            for _ in range(2):
                                await websocket.send(json.dumps(["EVENT", subscription, json.loads(valid.as_json())]))
                        else:
                            assert request["ids"] == [ancestor.id().to_hex()]
                            await websocket.send(json.dumps(["EVENT", subscription, json.loads(ancestor.as_json())]))
                        await websocket.send(json.dumps(["EOSE", subscription]))
                    elif message[0] == "EVENT":
                        event = message[1]
                        assert Event.from_json(json.dumps(event)).verify()
                        await published.put(event)
                        await websocket.send(json.dumps(["OK", event["id"], True, ""]))
            finally:
                disconnected.set()

        async with serve(relay, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            transport = NostrTransport(bot_keys.secret_key().to_hex(), [f"ws://127.0.0.1:{port}"])
            task = asyncio.create_task(transport.run(received.put))
            try:
                incoming = await asyncio.wait_for(received.get(), 10)
                assert incoming["content"] == "hello"
                assert incoming["pubkey"] == author.public_key().to_hex()
                result = await transport.publish("reply", [["p", incoming["pubkey"]]])
                outgoing = await asyncio.wait_for(published.get(), 5)
                assert result == outgoing["id"]
                assert outgoing["pubkey"] == public_key
                assert outgoing["content"] == "reply"
                created_at = int(time.time()) + 1
                profile_id = await transport.publish('{"name":"bot"}', [], kind=0, created_at=created_at)
                profile = await asyncio.wait_for(published.get(), 5)
                assert profile["id"] == profile_id
                assert profile["kind"] == 0
                assert profile["created_at"] == created_at
                assert await transport.fetch_event(ancestor.id().to_hex()) == json.loads(ancestor.as_json())
                assert await transport.fetch_event("not-an-id") is None
                # Completing a round trip gives duplicate notifications time to drain.
                assert received.empty()
            finally:
                await stop(task)
            await asyncio.wait_for(disconnected.wait(), 5)
            assert transport._client is None

    asyncio.run(scenario())


@pytest.mark.parametrize("ack", ["reject", "disconnect", "silent"])
def test_publication_requires_positive_ack(ack, monkeypatch):
    monkeypatch.setattr(nostr_transport, "IO_TIMEOUT_SECONDS", 2)

    async def scenario():
        subscribed = asyncio.Event()

        async def relay(websocket):
            async for raw in websocket:
                message = json.loads(raw)
                if message[0] == "REQ":
                    subscribed.set()
                    await websocket.send(json.dumps(["EOSE", message[1]]))
                elif message[0] == "EVENT":
                    if ack == "reject":
                        await websocket.send(json.dumps(["OK", message[1]["id"], False, "blocked: test"]))
                    elif ack == "disconnect":
                        await websocket.close()
                    # A silent relay must time out, never count a socket write as success.

        async def ignore(_event):
            pass

        async with serve(relay, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            transport = NostrTransport(Keys.generate().secret_key().to_hex(), [f"ws://127.0.0.1:{port}"])
            task = asyncio.create_task(transport.run(ignore))
            try:
                await asyncio.wait_for(subscribed.wait(), 5)
                with pytest.raises(RuntimeError, match="No Nostr relay accepted"):
                    await asyncio.wait_for(transport.publish("not accepted", []), 4)
            finally:
                await stop(task)

    asyncio.run(scenario())


@pytest.mark.parametrize("empty", [False, True])
def test_profile_latest_tie_and_confirmed_absence_across_relays(empty):
    async def scenario():
        keys = Keys.generate()
        public_key = keys.public_key().to_hex()
        now = int(time.time())
        old = signed_event(keys, '{"name":"old"}', created_at=now - 1, kind=0)
        tied = [
            signed_event(keys, json.dumps({"name": name, "about": "preserve me"}), [["x", name]], now, kind=0)
            for name in ("one", "two")
        ]
        winner, loser = sorted(tied, key=lambda event: event.id().to_hex())
        responses = {}
        closed = asyncio.Queue()

        async def relay(websocket):
            async for raw in websocket:
                message = json.loads(raw)
                if message[0] == "REQ":
                    subscription, request = message[1:]
                    assert request == {"authors": [public_key], "kinds": [0], "limit": 1}
                    for event in responses[websocket.local_address[1]]:
                        await websocket.send(json.dumps(["EVENT", subscription, json.loads(event.as_json())]))
                    await websocket.send(json.dumps(["EOSE", subscription]))
                elif message[0] == "CLOSE":
                    await closed.put(message[1])

        async with serve(relay, "127.0.0.1", 0) as first, serve(relay, "127.0.0.1", 0) as second:
            ports = [server.sockets[0].getsockname()[1] for server in (first, second)]
            responses[ports[0]] = [] if empty else [loser, old]
            responses[ports[1]] = [] if empty else [old, winner, loser]
            transport = NostrTransport(keys.secret_key().to_hex(), [f"ws://127.0.0.1:{port}" for port in ports])
            result = await transport.fetch_profile(public_key)
            assert result == (None if empty else json.loads(winner.as_json()))
            for _ in ports:
                assert await asyncio.wait_for(closed.get(), 1) == "missbot-profile"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure",
    [
        "disconnect",
        "timeout",
        "closed",
        "malformed_frame",
        "malformed_metadata",
        "nonobject_metadata",
        "non_json_number",
        "forged_author",
        "wrong_author",
        "wrong_kind",
        "invalid_signature",
        "future",
        "message_limit",
    ],
)
def test_profile_failure_never_reports_absence_or_partial_result(failure, monkeypatch):
    monkeypatch.setattr(nostr_transport, "FETCH_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(nostr_transport, "MAX_NOTIFICATIONS", 8)

    async def scenario():
        keys, other = Keys.generate(), Keys.generate()
        public_key = keys.public_key().to_hex()
        valid = signed_event(keys, '{"name":"existing"}', kind=0)

        async def healthy(websocket):
            async for raw in websocket:
                message = json.loads(raw)
                if message[0] == "REQ":
                    await websocket.send(json.dumps(["EVENT", message[1], json.loads(valid.as_json())]))
                    await websocket.send(json.dumps(["EOSE", message[1]]))

        async def failing(websocket):
            async for raw in websocket:
                message = json.loads(raw)
                if message[0] != "REQ":
                    continue
                subscription = message[1]
                if failure == "disconnect":
                    await websocket.close()
                    return
                if failure == "timeout":
                    continue
                if failure == "closed":
                    await websocket.send(json.dumps(["CLOSED", subscription, "blocked: test"]))
                elif failure == "malformed_frame":
                    await websocket.send("{")
                elif failure == "message_limit":
                    for _ in range(nostr_transport.MAX_NOTIFICATIONS):
                        await websocket.send(json.dumps(["NOTICE", "noise"]))
                else:
                    content = {
                        "malformed_metadata": "{",
                        "nonobject_metadata": "[]",
                        "non_json_number": '{"name":"existing","value":NaN}',
                    }.get(failure, '{"name":"untrusted"}')
                    event = signed_event(
                        other if failure in ("forged_author", "wrong_author") else keys,
                        content,
                        kind=1 if failure == "wrong_kind" else 0,
                        created_at=int(time.time()) + 3600 if failure == "future" else None,
                    )
                    raw_event = json.loads(event.as_json())
                    if failure == "forged_author":
                        raw_event["pubkey"] = public_key
                    elif failure == "invalid_signature":
                        raw_event["sig"] = "00" * 64
                    await websocket.send(json.dumps(["EVENT", subscription, raw_event]))
                await websocket.send(json.dumps(["EOSE", subscription]))

        async with serve(healthy, "127.0.0.1", 0) as first, serve(failing, "127.0.0.1", 0) as second:
            ports = [server.sockets[0].getsockname()[1] for server in (first, second)]
            transport = NostrTransport(keys.secret_key().to_hex(), [f"ws://127.0.0.1:{port}" for port in ports])
            with pytest.raises(RuntimeError, match="Could not completely fetch Nostr profile"):
                await asyncio.wait_for(transport.fetch_profile(public_key), 2)

    asyncio.run(scenario())


def test_profile_lock_wait_and_invalid_public_key_are_bounded(monkeypatch):
    monkeypatch.setattr(nostr_transport, "FETCH_TIMEOUT_SECONDS", 0.02)

    async def scenario():
        transport = NostrTransport(Keys.generate().secret_key().to_hex(), ["wss://relay.example.com"])
        with pytest.raises(ValueError, match="Invalid Nostr profile public key"):
            await transport.fetch_profile("not-a-public-key")
        async with transport._profile_lock:
            with pytest.raises(RuntimeError, match="Could not completely fetch Nostr profile"):
                await asyncio.wait_for(transport.fetch_profile(transport.public_key), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize("created_at", [-1, True, "0", int(time.time()) + 3600])
def test_publication_rejects_invalid_custom_timestamp(created_at):
    async def scenario():
        transport = NostrTransport(Keys.generate().secret_key().to_hex(), ["wss://relay.example.com"])
        with pytest.raises(ValueError, match="Invalid Nostr publication timestamp"):
            await transport.publish("{}", [], kind=0, created_at=created_at)

    asyncio.run(scenario())
