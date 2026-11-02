"""A real management client can disconnect while the hosted relay runner keeps answering."""

import asyncio
import base64
import hashlib
import json
from contextlib import asynccontextmanager, suppress

import httpx
import pytest
from nostr_sdk import Event, EventBuilder, Keys, Kind, Tag
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from bot.ai import ChatAgent
from bot.buzz import server as management
from bot.buzz.service import HostedBuzz
from bot.buzz.store import BuzzStore


@pytest.mark.anyio
async def test_hosted_replies_after_management_disconnect_and_state_restart(make_config, tmp_path, monkeypatch):
    bot, owner, authority = Keys.generate(), Keys.generate(), Keys.generate()
    pubkey = bot.public_key().to_hex()
    owner_key = owner.public_key().to_hex()
    auth = ["auth", owner_key, "", owner.sign_schnorr(hashlib.sha256(f"nostr:agent-auth:{pubkey}:".encode()).digest())]
    published = asyncio.Queue()
    subscriptions = {}
    subscribed = asyncio.Event()
    management_port = 0
    management_ready = asyncio.Event()

    def sign(keys, kind, content, tags):
        return json.loads(
            EventBuilder(Kind(kind), content).tags([Tag.parse(tag) for tag in tags]).finalize(keys).as_json()
        )

    metadata = [
        sign(authority, 39000, "", [["d", "chat"], ["t", "dm"], ["private"]]),
        sign(authority, 39002, "", [["d", "chat"], ["p", pubkey], ["p", owner_key]]),
        sign(owner, 0, json.dumps({"name": "rachel", "display_name": "Rachel"}), []),
    ]

    async def relay_request(socket, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        event = json.loads(base64.b64decode(request.headers["Authorization"].split(" ", 1)[1]))
        assert Event.from_json(json.dumps(event)).verify() and auth in event["tags"]
        return socket.respond(200, json.dumps({"self": authority.public_key().to_hex()}))

    async def relay(socket):
        try:
            await socket.send(json.dumps(["AUTH", "challenge"]))
            request = json.loads(await socket.recv())
            assert request[0] == "AUTH" and auth in request[1]["tags"]
            assert Event.from_json(json.dumps(request[1])).verify()
            await socket.send(json.dumps(["OK", request[1]["id"], True, ""]))
            async for frame in socket:
                message = json.loads(frame)
                if message[0] == "REQ":
                    subscription, filters = message[1], message[2:]
                    if filters[0].get("kinds") == [9]:
                        assert filters[0].get("#h") == ["chat"]
                        subscriptions[socket] = subscription
                        subscribed.set()
                    else:
                        for event in metadata:
                            if event["kind"] in filters[0].get("kinds", []) and (
                                "authors" not in filters[0] or event["pubkey"] in filters[0]["authors"]
                            ):
                                await socket.send(json.dumps(["EVENT", subscription, event]))
                    await socket.send(json.dumps(["EOSE", subscription]))
                elif message[0] == "EVENT":
                    event = message[1]
                    assert Event.from_json(json.dumps(event)).verify()
                    assert auth in event["tags"] and event["pubkey"] == pubkey
                    if event["kind"] == 0:
                        await socket.send(json.dumps(["OK", event["id"], False, "Profiles require HTTP"]))
                        continue
                    await socket.send(json.dumps(["OK", event["id"], True, ""]))
                    if event["kind"] == 9:
                        await published.put(event)
        finally:
            subscriptions.pop(socket, None)

    @asynccontextmanager
    async def capture_management(*args, **kwargs):
        nonlocal management_port
        async with serve(*args, **kwargs) as listener:
            management_port = listener.sockets[0].getsockname()[1]
            management_ready.set()
            yield listener

    monkeypatch.setattr(management, "serve", capture_management)
    original_client = httpx.AsyncClient

    class ProfileHTTP(httpx.AsyncBaseTransport):
        def __init__(self):
            self.direct = httpx.AsyncHTTPTransport()

        async def handle_async_request(self, request):
            if request.method != "POST":
                return await self.direct.handle_async_request(request)
            assert request.url.path == "/events"
            event = json.loads(request.content)
            assert event["kind"] == 0 and json.loads(event["content"])["bot"] is True
            assert auth in event["tags"]
            authorization = json.loads(base64.b64decode(request.headers["Authorization"].split(" ", 1)[1]))
            assert Event.from_json(json.dumps(authorization)).verify()
            assert ["payload", hashlib.sha256(request.content).hexdigest()] in authorization["tags"]
            assert json.loads(request.headers["x-auth-tag"]) == auth
            return httpx.Response(200, json={"accepted": True, "event_id": event["id"]})

        async def aclose(self):
            await self.direct.aclose()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=ProfileHTTP(), **kwargs))
    config = make_config(nostr_relays=[])
    history_sizes = []

    def answer(messages, info):
        turns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        history_sizes.append(len(turns))
        assert f'"Rachel" (mention: @{owner.public_key().to_bech32()}; use the name only without @)' in str(turns[-1].content)
        text = (
            "Reply with no desktop client" if len(history_sizes) == 1 else "Recovered the earlier private conversation"
        )
        return ModelResponse(parts=[TextPart(text)])

    monkeypatch.setattr("bot.ai._model_chain", lambda specs: FunctionModel(answer))
    chat = ChatAgent(config)

    async with serve(relay, "127.0.0.1", 0, process_request=relay_request) as listener:
        community = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        path = str(tmp_path / "hosted.db")
        store = BuzzStore(path)
        service = HostedBuzz(config, chat, store, bot, [community])
        control = asyncio.create_task(management.serve_hosted(service, host="127.0.0.1", port=0, token="test-token"))
        try:
            await asyncio.wait_for(management_ready.wait(), 5)
            async with connect(
                f"ws://127.0.0.1:{management_port}/buzz/ws", subprotocols=[management.SUBPROTOCOL]
            ) as socket:
                await socket.send('{"type":"authenticate","token":"test-token"}')
                assert json.loads(await socket.recv()) == {"type": "authenticated"}
                for method, params in [
                    ("provision", {"name": "Grok", "community": community}),
                    ("activate", {"pubkey": pubkey, "name": "Grok", "community": community, "auth": auth}),
                ]:
                    await socket.send(json.dumps({"jsonrpc": "2.0", "id": method, "method": method, "params": params}))
                    response = json.loads(await asyncio.wait_for(socket.recv(), 5))
                    assert "result" in response, response
            # There is now no management connection or desktop/plugin runner.
            with chat._agent.override(model=FunctionModel(answer)):

                async def incoming(text):
                    event = sign(owner, 9, text, [["h", "chat"]])
                    for peer, subscription in list(subscriptions.items()):
                        await peer.send(json.dumps(["EVENT", subscription, event]))
                    return await asyncio.wait_for(published.get(), 5)

                first = await incoming("First private message")
                assert first["content"] == "Reply with no desktop client"
                # Publication ACK is followed by durable completion before process shutdown.
                async with asyncio.timeout(5):
                    while store.next_event(community) is not None:
                        await asyncio.sleep(0)
                await service.close()
                store.close()
                subscribed.clear()
                store = BuzzStore(path)
                service = HostedBuzz(config, chat, store, bot, [community])
                await service.start()
                await service.runners[community].relay.wait_ready()
                await asyncio.wait_for(subscribed.wait(), 5)
                second = await incoming("Second message after service restart")
                assert second["content"] == "Recovered the earlier private conversation"
                assert history_sizes == [1, 2]
                assert first["pubkey"] == second["pubkey"] == pubkey
                assert second["tags"] == [["h", "chat"], auth]
        finally:
            control.cancel()
            with suppress(asyncio.CancelledError):
                await control
            await service.close()
            store.close()
