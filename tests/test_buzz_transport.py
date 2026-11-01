"""Real local HTTP/WebSocket relay checks for Buzz authentication and bounded IO."""

import asyncio
import base64
import hashlib
import json
import time
from contextlib import asynccontextmanager, suppress

import httpx
import pytest
from nostr_sdk import Event, EventBuilder, Keys, Kind, Tag, Timestamp
from websockets.asyncio.server import serve

from bot.buzz import transport
from bot.buzz.transport import LIVE_SUBSCRIPTION, BuzzRelay


def signed(keys, content="hello", tags=(), kind=9, created_at=None):
    builder = EventBuilder(Kind(kind), content).tags([Tag.parse(tag) for tag in tags])
    if created_at is not None:
        builder = builder.custom_created_at(Timestamp.from_secs(created_at))
    return json.loads(builder.finalize(keys).as_json())


def owner_proof(keys):
    owner = Keys.generate()
    message = f"nostr:agent-auth:{keys.public_key().to_hex()}:".encode()
    return ["auth", owner.public_key().to_hex(), "", owner.sign_schnorr(hashlib.sha256(message).digest())]


@asynccontextmanager
async def local_relay(keys, script, *, auth_result=True, info=None, discovery_status=200, channels=("channel",)):
    auth = owner_proof(keys)
    authority_key = Keys.generate()
    authority = authority_key.public_key().to_hex()
    state = {"connections": 0, "auth": [], "http_auth": [], "closed": 0, "requests": []}
    state["channels"] = list(channels)
    state["notice"] = lambda channel: signed(
        authority_key, "", [["p", keys.public_key().to_hex()], ["h", channel]], kind=44100
    )

    async def request(socket, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        assert request.headers["Accept"] == "application/nostr+json"
        prefix, encoded = request.headers["Authorization"].split(" ", 1)
        assert prefix == "Nostr"
        event = json.loads(base64.b64decode(encoded))
        assert Event.from_json(json.dumps(event)).verify()
        assert event["pubkey"] == keys.public_key().to_hex()
        assert event["kind"] == 27235
        assert ["method", "GET"] in event["tags"]
        assert ["u", f"http://127.0.0.1:{socket.local_address[1]}/"] in event["tags"]
        assert auth in event["tags"]
        state["http_auth"].append(event)
        return socket.respond(discovery_status, json.dumps({"self": authority} if info is None else info))

    async def relay(socket):
        state["connections"] += 1
        state["socket"] = socket
        try:
            await socket.send(json.dumps(["AUTH", "challenge"]))
            message = json.loads(await socket.recv())
            assert message[0] == "AUTH"
            event = message[1]
            assert Event.from_json(json.dumps(event)).verify()
            assert event["pubkey"] == keys.public_key().to_hex()
            assert event["kind"] == 22242 and event["content"] == ""
            assert event["tags"] == [
                ["relay", f"ws://127.0.0.1:{socket.local_address[1]}/"],
                ["challenge", "challenge"],
                auth,
            ]
            state["auth"].append(event)
            if auth_result is not None:
                await socket.send(json.dumps(["OK", event["id"], auth_result, "SECRET relay-controlled reason"]))
            async for raw in socket:
                message = json.loads(raw)
                state["requests"].append(message)
                if message[0] == "REQ" and message[1] == transport.NOTICE_SUBSCRIPTION:
                    await socket.send(json.dumps(["EOSE", message[1]]))
                    continue
                if message[0] == "REQ" and message[2].get("kinds") == [39002]:
                    for channel in state["channels"]:
                        roster = signed(
                            authority_key, "", [["d", channel], ["p", keys.public_key().to_hex()]], kind=39002
                        )
                        await socket.send(json.dumps(["EVENT", message[1], roster]))
                    await socket.send(json.dumps(["EOSE", message[1]]))
                    continue
                await script(socket, message, state)
        finally:
            state["closed"] += 1

    async with serve(relay, "127.0.0.1", 0, process_request=request) as server:
        port = server.sockets[0].getsockname()[1]
        client = BuzzRelay(keys, f"http://127.0.0.1:{port}", auth)
        try:
            yield client, state, authority
        finally:
            await client.aclose()


async def ignore(_event):
    pass


def test_real_auth_verified_subscription_complete_query_and_retry_identity():
    async def scenario():
        keys, author = Keys.generate(), Keys.generate()
        now = int(time.time())
        received = asyncio.Queue()
        publications = []
        profile = signed(author, '{"name":"existing"}', kind=0)

        async def script(socket, message, state):
            if message[0] == "REQ":
                subscription, request = message[1:3]
                if subscription == LIVE_SUBSCRIPTION:
                    assert request == {
                        "kinds": [9],
                        "#h": ["channel"],
                        "since": now,
                        "limit": transport.MAX_QUERY_EVENTS,
                    }
                    valid = signed(author, "boundary", [["h", "channel"]], created_at=now)
                    invalid = dict(valid, content="tampered")
                    for event in [
                        invalid,
                        signed(author, "old", created_at=now - 1),
                        signed(author, "future", created_at=now + 3600),
                        signed(author, "wrong kind", kind=4),
                        signed(author, "wrong channel", [["h", "unjoined"]], created_at=now),
                        valid,
                    ]:
                        await socket.send(json.dumps(["EVENT", subscription, event]))
                elif request.get("authors"):
                    assert request["authors"] == [author.public_key().to_hex()]
                    await socket.send(json.dumps(["EVENT", subscription, profile]))
                await socket.send(json.dumps(["EOSE", subscription]))
            elif message[0] == "EVENT":
                event = message[1]
                assert Event.from_json(json.dumps(event)).verify()
                publications.append(event)
                await socket.send(json.dumps(["OK", event["id"], len(publications) > 1, "SECRET"]))

        async with local_relay(keys, script) as (client, state, authority):
            task = asyncio.create_task(client.run(received.put, now))
            await client.wait_ready(2)
            assert client.relay_public_key == authority
            incoming = await asyncio.wait_for(received.get(), 2)
            assert incoming["content"] == "boundary"
            event = client.sign("reply", [["h", "channel"], ["auth", "discard caller proof"]], created_at=now)
            with pytest.raises(RuntimeError, match="did not acknowledge") as error:
                await client.publish_event(event)
            assert "SECRET" not in str(error.value)
            assert await client.publish_event(event) == event["id"]
            assert publications == [event, event]
            assert await client.query([{"authors": [author.public_key().to_hex()], "kinds": [0], "limit": 1}]) == [
                profile
            ]
            assert await client.query([{"ids": ["00" * 32]}]) == []
            assert received.empty()
            assert len(state["http_auth"]) == len(state["auth"]) == 1
            await client.aclose()
            assert task.cancelled()
            assert not client.connected and client._socket is None
            assert client._ack is None and client._query is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure", ["timeout", "closed", "disconnect", "invalid", "unrelated", "rows", "bytes", "frame"]
)
def test_incomplete_query_never_reports_absence(failure, monkeypatch):
    monkeypatch.setattr(transport, "QUERY_TIMEOUT_SECONDS", 0.15)
    monkeypatch.setattr(transport, "RECONNECT_MIN_SECONDS", 10)
    if failure == "rows":
        monkeypatch.setattr(transport, "MAX_QUERY_EVENTS", 2)

    async def scenario():
        keys, author = Keys.generate(), Keys.generate()
        query_started = asyncio.Event()

        async def script(socket, message, _state):
            if message[0] != "REQ":
                return
            subscription = message[1]
            if subscription == LIVE_SUBSCRIPTION:
                await socket.send(json.dumps(["EOSE", subscription]))
                return
            query_started.set()
            if failure == "timeout":
                return
            if failure == "closed":
                await socket.send(json.dumps(["CLOSED", subscription, "SECRET"]))
                return
            if failure == "disconnect":
                await socket.close()
                return
            if failure == "frame":
                await socket.send("x" * (transport.MAX_EVENT_BYTES + 4097))
                return
            event = signed(author, kind=4 if failure == "unrelated" else 0)
            if failure == "invalid":
                event["sig"] = "00" * 64
            for _ in range(3 if failure == "rows" else 1):
                await socket.send(json.dumps(["EVENT", subscription, event]))
            await socket.send(json.dumps(["EOSE", subscription]))

        async with local_relay(keys, script) as (client, _state, _authority):
            asyncio.create_task(client.run(ignore, 0))
            await client.wait_ready(2)
            if failure == "bytes":
                monkeypatch.setattr(transport, "MAX_QUERY_BYTES", 10)
            with pytest.raises(RuntimeError, match="completely query") as error:
                await client.query([{"kinds": [0], "limit": 1}])
            assert query_started.is_set()
            assert "SECRET" not in str(error.value)
            assert client._query is None

    asyncio.run(scenario())


@pytest.mark.parametrize("ack", [False, None, "true", "disconnect"])
def test_publication_requires_positive_boolean_ack(ack, monkeypatch):
    monkeypatch.setattr(transport, "IO_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(transport, "RECONNECT_MIN_SECONDS", 10)

    async def scenario():
        async def script(socket, message, _state):
            if message[0] == "REQ":
                await socket.send(json.dumps(["EOSE", message[1]]))
            elif message[0] == "EVENT":
                if ack == "disconnect":
                    await socket.close()
                elif ack is not None:
                    await socket.send(json.dumps(["OK", message[1]["id"], ack, "SECRET"]))

        async with local_relay(Keys.generate(), script) as (client, _state, _authority):
            asyncio.create_task(client.run(ignore, 0))
            await client.wait_ready(2)
            with pytest.raises(RuntimeError, match="did not acknowledge"):
                await client.publish("not accepted", [])
            assert client._ack is None

    asyncio.run(scenario())


@pytest.mark.parametrize("ack", [False, None, "true"])
def test_auth_requires_positive_ack_and_retries_are_bounded(ack, monkeypatch):
    monkeypatch.setattr(transport, "IO_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(transport, "RECONNECT_MIN_SECONDS", 10)

    async def scenario():
        async with local_relay(Keys.generate(), ignore_message, auth_result=ack) as (client, state, _authority):
            task = asyncio.create_task(client.run(ignore, 0))
            with pytest.raises(RuntimeError, match="did not become ready"):
                await client.wait_ready(0.4)
            assert not client.connected
            assert state["connections"] == 1 and state["requests"] == []
            assert client.error is not None and "SECRET" not in client.error
            await client.aclose()
            assert task.cancelled()

    asyncio.run(scenario())


async def ignore_message(_socket, _message, _state):
    pass


@pytest.mark.parametrize(
    "info,status", [({}, 200), ({"pubkey": "ab" * 32}, 200), ({"self": "invalid"}, 200), ({}, 302)]
)
def test_discovery_requires_origin_authority_without_redirects(info, status, monkeypatch):
    monkeypatch.setattr(transport, "RECONNECT_MIN_SECONDS", 10)

    async def scenario():
        async with local_relay(Keys.generate(), ignore_message, info=info, discovery_status=status) as (
            client,
            state,
            _,
        ):
            asyncio.create_task(client.run(ignore, 0))
            with pytest.raises(RuntimeError, match="did not become ready"):
                await client.wait_ready(0.3)
            assert state["connections"] == 0
            assert client.relay_public_key == ""

    asyncio.run(scenario())


def test_reauthentication_fails_pending_work_and_replays_since(monkeypatch):
    monkeypatch.setattr(transport, "RECONNECT_MIN_SECONDS", 0.01)
    monkeypatch.setattr(transport, "RECONNECT_MAX_SECONDS", 0.02)

    async def scenario():
        reconnected = asyncio.Event()
        now = int(time.time())

        async def script(socket, message, state):
            if message[0] != "REQ":
                return
            if message[1] == LIVE_SUBSCRIPTION:
                assert message[2]["since"] == now
                if state["connections"] == 2:
                    reconnected.set()
                await socket.send(json.dumps(["EOSE", message[1]]))
            elif state["connections"] == 1:
                await socket.send(json.dumps(["AUTH", "new challenge"]))
            else:
                await socket.send(json.dumps(["EOSE", message[1]]))

        async with local_relay(Keys.generate(), script) as (client, state, _authority):
            asyncio.create_task(client.run(ignore, now))
            await client.wait_ready(2)
            with pytest.raises(RuntimeError, match="completely query"):
                await client.query([{"kinds": [0]}])
            await asyncio.wait_for(reconnected.wait(), 2)
            await client.wait_ready(2)
            assert await client.query([{"kinds": [0]}]) == []
            assert len(state["auth"]) == 2

    asyncio.run(scenario())


def test_cancellation_clears_query_and_outgoing_waits_are_bounded(monkeypatch):
    monkeypatch.setattr(transport, "QUERY_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(transport, "IO_TIMEOUT_SECONDS", 0.1)

    async def scenario():
        query_started = asyncio.Event()

        async def script(socket, message, _state):
            if message[0] == "REQ":
                if message[1] == LIVE_SUBSCRIPTION:
                    await socket.send(json.dumps(["EOSE", message[1]]))
                else:
                    query_started.set()

        async with local_relay(Keys.generate(), script) as (client, _state, _authority):
            asyncio.create_task(client.run(ignore, 0))
            await client.wait_ready(2)
            async with client._operation:
                with pytest.raises(RuntimeError, match="completely query"):
                    await client.query([{"kinds": [0]}])
                with pytest.raises(RuntimeError, match="did not acknowledge"):
                    await client.publish("locked", [])
            pending = asyncio.create_task(client.query([{"kinds": [0]}]))
            await asyncio.wait_for(query_started.wait(), 1)
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
            assert client._query is None
            pending = asyncio.create_task(client.query([{"kinds": [0]}]))
            await asyncio.sleep(0)
            await client.aclose()
            with pytest.raises(RuntimeError, match="completely query"):
                await pending
            assert client._query is None and not client.connected

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "origin",
    [
        "http://example.com",
        "wss://user:password@example.com",
        "https://example.com/path",
        "https://example.com/?token=secret",
    ],
)
def test_rejects_credentialed_or_nonorigin_communities(origin):
    keys = Keys.generate()
    with pytest.raises(ValueError, match="community origin"):
        BuzzRelay(keys, origin, owner_proof(keys))


def test_signing_checks_identity_authorization_timestamp_and_size(monkeypatch):
    async def scenario():
        keys = Keys.generate()
        client = BuzzRelay(keys, "https://relay.example", owner_proof(keys))
        for created_at in [-1, True, int(time.time()) + 3600]:
            with pytest.raises(ValueError, match="publication event"):
                client.sign("reply", [], created_at=created_at)
        with pytest.raises(ValueError, match="identity or authorization"):
            await client.publish_event(signed(Keys.generate()))
        with pytest.raises(ValueError, match="identity or authorization"):
            await client.publish_event(signed(keys))
        monkeypatch.setattr(transport, "MAX_EVENT_BYTES", 100)
        with pytest.raises(ValueError, match="publication event"):
            client.sign("x" * 100, [])
        await client.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await client.wait_ready(0.1)

    asyncio.run(scenario())


@pytest.mark.anyio
@pytest.mark.parametrize("receipt", ["accepted", "rejected", "wrong-id", "redirect", "not-json", "oversized"])
async def test_profiles_use_bound_http_authorization_and_require_exact_receipt(monkeypatch, receipt):
    keys = Keys.generate()
    auth = owner_proof(keys)
    client = BuzzRelay(keys, "https://relay.example", auth)
    profile = client.sign('{"name":"Grok","bot":true}', [], kind=0)
    original = httpx.AsyncClient
    requests = []

    def handle(request):
        requests.append(request)
        assert request.method == "POST" and str(request.url) == "https://relay.example/events"
        assert json.loads(request.content) == profile
        proof = json.loads(base64.b64decode(request.headers["Authorization"].split(" ", 1)[1]))
        assert Event.from_json(json.dumps(proof)).verify()
        assert proof["kind"] == 27235 and proof["pubkey"] == profile["pubkey"]
        assert ["u", str(request.url)] in proof["tags"] and ["method", "POST"] in proof["tags"]
        assert ["payload", hashlib.sha256(request.content).hexdigest()] in proof["tags"]
        assert json.loads(request.headers["x-auth-tag"]) == auth
        if receipt == "redirect":
            return httpx.Response(307, headers={"Location": "https://attacker.example"}, json={})
        if receipt in ("not-json", "oversized"):
            return httpx.Response(200, content="x" * (65537 if receipt == "oversized" else 1))
        return httpx.Response(
            200,
            json={
                "accepted": receipt != "rejected",
                "event_id": "wrong" if receipt == "wrong-id" else profile["id"],
            },
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs)
    )
    if receipt == "accepted":
        assert await client.publish_event(profile) == profile["id"]
    else:
        with pytest.raises(RuntimeError, match="signed HTTP profile"):
            await client.publish_event(profile)
    assert len(requests) == 1  # no fallback, redirect following, or uncertain-write retry


@pytest.mark.anyio
async def test_new_membership_installs_scoped_live_feed_without_reconnecting():
    keys, author = Keys.generate(), Keys.generate()
    received = asyncio.Queue()
    closed = asyncio.Event()

    async def script(socket, message, state):
        if message[0] == "REQ" and message[1] == LIVE_SUBSCRIPTION:
            # An unscoped REQ may yield EOSE, but is not a live channel route.
            if message[2].get("#h") == ["joined"]:
                event = signed(
                    author, "message sent while joining", [["h", "joined"], ["p", keys.public_key().to_hex()]]
                )
                await socket.send(json.dumps(["EVENT", LIVE_SUBSCRIPTION, event]))
            await socket.send(json.dumps(["EOSE", message[1]]))
        elif message == ["CLOSE", LIVE_SUBSCRIPTION]:
            closed.set()

    async with local_relay(keys, script, channels=()) as (client, state, _authority):
        asyncio.create_task(client.run(received.put, 0))
        await client.wait_ready(2)
        assert received.empty()
        state["channels"].append("joined")
        await state["socket"].send(json.dumps(["EVENT", transport.NOTICE_SUBSCRIPTION, state["notice"]("joined")]))
        assert (await asyncio.wait_for(received.get(), 2))["content"] == "message sent while joining"
        assert state["connections"] == 1
        state["channels"].clear()
        await state["socket"].send(json.dumps(["EVENT", transport.NOTICE_SUBSCRIPTION, state["notice"]("joined")]))
        await asyncio.wait_for(closed.wait(), 2)
        assert client._live_channels == set()
