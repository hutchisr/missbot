"""Authenticated management WebSocket; relay runners outlive every client connection."""

import asyncio
import json
import secrets
from http import HTTPStatus
from typing import Any

import logfire
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

from .service import HostedBuzz

SUBPROTOCOL = Subprotocol("missbot.buzz.v1")


async def serve_hosted(service: HostedBuzz, *, host: str, port: int, token: str) -> None:
    if not token.strip():
        raise ValueError("Hosted management token must be nonempty")
    token = token.strip()
    active = 0

    def process_request(connection: ServerConnection, request: Any):
        if request.path == "/healthz":
            return connection.respond(HTTPStatus.OK, "ok\n")
        if request.path != "/buzz/ws":
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        return None

    async def handler(socket: ServerConnection) -> None:
        nonlocal active
        if active >= 8:
            await socket.close(code=1013, reason="Hosted management connection limit")
            return
        active += 1
        try:
            async with asyncio.timeout(5):
                first = await socket.recv()
            auth = json.loads(first) if isinstance(first, str) and len(first) <= 4096 else None
            if not (
                isinstance(auth, dict)
                and auth.get("type") == "authenticate"
                and isinstance(auth.get("token"), str)
                and secrets.compare_digest(auth["token"].encode(), token.encode())
            ):
                await socket.close(code=1008, reason="Authentication required")
                return
            await socket.send('{"type":"authenticated"}')
            async for frame in socket:
                request_id = None
                try:
                    message = json.loads(frame) if isinstance(frame, str) else None
                    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                        raise ValueError("Expected a JSON-RPC request")
                    request_id = message.get("id")
                    if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
                        raise TypeError("Request id is required")
                    method, params = message.get("method"), message.get("params", {})
                    if not isinstance(method, str) or not isinstance(params, dict):
                        raise TypeError("Invalid management request")
                    async with asyncio.timeout(60):
                        result = await service.call(method, params)
                    response = {"jsonrpc": "2.0", "id": request_id, "result": result}
                except (ValueError, TypeError) as error:
                    response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": str(error)}}
                except RuntimeError as error:
                    # Transport errors are fixed, sanitized strings, never raw relay messages.
                    logfire.error("Hosted Buzz operation refused", reason=str(error))
                    response = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32000, "message": str(error)}}
                except Exception:
                    logfire.exception("Hosted Buzz management operation failed")
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": "Hosted operation failed; check server status and retry"},
                    }
                await socket.send(json.dumps(response))
        except (TimeoutError, ValueError):
            await socket.close(code=1008, reason="Authentication required")
        except ConnectionClosed:
            pass
        finally:
            active -= 1

    async with serve(
        handler,
        host,
        port,
        process_request=process_request,
        subprotocols=[SUBPROTOCOL],
        max_size=65536,
        max_queue=4,
        compression=None,
        open_timeout=10,
        close_timeout=2,
    ):
        logfire.info("Hosted Buzz management listening", host=host, port=port)
        await asyncio.get_running_loop().create_future()
