"""Image generation for autonomous posts.

Speaks an OpenAI-compatible ``POST {base_url}/images/generations`` endpoint (OpenRouter,
OpenAI, or a self-hosted shim) and returns bytes only after they have been checked.

The endpoint is *operator* config, unlike the federated media URLs `bot/net.py` guards, so
there is no SSRF check here — a self-hosted generator on a private address is a legitimate
deployment. What is *not* trusted is the response:

- The body is streamed with a byte cap before it is parsed, so a broken or hostile endpoint
  cannot balloon the pod's memory with one oversized JSON document.
- The media type is sniffed from magic bytes rather than believed. SVG is refused outright:
  it can carry script, and Misskey would serve it from the instance's own origin.
- The decoded image is dropped when it exceeds the configured cap.

A dedicated ``httpx.AsyncClient`` is used per call. The shared ``api_client`` carries the
Misskey token, which must never be sent to a third-party provider. Every failure returns
None so the caller posts text instead of losing the post.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
from dataclasses import dataclass
from typing import Optional

import httpx
import logfire

from .models import Config


# Formats Misskey can serve safely, keyed by their magic bytes. WebP needs a split check
# (``RIFF....WEBP``) and is handled separately in _sniff_media_type.
_MAGIC_MEDIA_TYPES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}

# slack for the JSON envelope around the base64 payload when capping the response body.
_ENVELOPE_ALLOWANCE = 64 * 1024


def _sniff_media_type(data: bytes) -> Optional[str]:
    """Media type from the leading bytes, or None for anything we won't post."""
    for magic, media_type in _MAGIC_MEDIA_TYPES:
        if data.startswith(magic):
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@dataclass(frozen=True)
class GeneratedImage:
    """A generated image and the text that produced and describes it."""

    data: bytes
    media_type: str
    """Sniffed from the bytes, never taken from the provider's response."""
    prompt: str
    """What the model asked for. Kept for logging and provenance."""
    alt_text: str
    """The model's description of the finished image, uploaded as the drive file comment."""

    @property
    def extension(self) -> str:
        """File extension for the upload name. Total because `media_type` is sniffed."""
        return _EXTENSIONS[self.media_type]


class ImageGenerator:
    """Client for one OpenAI-compatible image-generation endpoint."""

    def __init__(self, config: Config, *, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        assert config.image_gen_model, "image_gen_model is required to build an ImageGenerator"
        self._model = config.image_gen_model
        self._url = f"{str(config.image_gen_base_url).rstrip('/')}/images/generations"
        self._size = config.image_gen_size
        self._timeout = config.image_gen_timeout_seconds
        self._max_bytes = config.image_gen_max_bytes
        # Test seam only (mirrors bot/net.py:fetch_image); production passes nothing.
        self._transport = transport
        self._api_key = config.image_gen_api_key or os.environ.get(config.image_gen_api_key_env)
        if not self._api_key:
            logfire.warning(
                "No image generation API key resolved; requests will be unauthenticated",
                env_var=config.image_gen_api_key_env,
            )

    async def generate(self, prompt: str, alt_text: str) -> Optional[GeneratedImage]:
        """Generate one image, or None if anything about it is unusable."""
        payload: dict[str, object] = {"model": self._model, "prompt": prompt, "n": 1}
        if self._size:
            payload["size"] = self._size

        body = await self._post(payload)
        if body is None:
            return None
        encoded = _extract_b64(body)
        if encoded is None:
            return None

        # Check before decoding: base64 carries 3 bytes of image per 4 bytes of text, so an
        # over-cap image is detectable without ever materializing it.
        if len(encoded) > (self._max_bytes * 4) // 3 + 4:
            logfire.warning("Generated image exceeds size cap; dropping", max_bytes=self._max_bytes)
            return None
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            logfire.warning("Image generation returned a non-base64 payload")
            return None
        if len(data) > self._max_bytes:
            logfire.warning("Generated image exceeds size cap; dropping", size=len(data), max_bytes=self._max_bytes)
            return None

        media_type = _sniff_media_type(data)
        if media_type is None:
            logfire.warning("Image generation returned an unsupported format; dropping", head=data[:8].hex())
            return None

        logfire.info("Generated image", model=self._model, media_type=media_type, size=len(data))
        return GeneratedImage(data=data, media_type=media_type, prompt=prompt, alt_text=alt_text)

    async def _post(self, payload: dict[str, object]) -> Optional[dict[str, object]]:
        """POST the request and return the parsed body, capping how much is read."""
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        cap = self._max_bytes * 2 + _ENVELOPE_ALLOWANCE
        chunks: list[bytes] = []
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                transport=self._transport,
            ) as client:
                async with client.stream("POST", self._url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > cap:
                            logfire.warning("Image generation response exceeds byte cap; dropping", cap=cap)
                            return None
                        chunks.append(chunk)
        except httpx.HTTPError:
            # This is the only operator signal for the entire provider leg — every downstream
            # failure is silent by design (fail-soft: the tool reports failure, the post goes
            # out as text). logfire.exception (not .warning) attaches the exception, so a 401
            # bad key, a 404 wrong path, a read timeout, and a DNS failure remain distinguishable.
            logfire.exception("Image generation request failed", model=self._model)
            return None

        try:
            parsed = json.loads(b"".join(chunks))
        except ValueError:
            logfire.warning("Image generation returned invalid JSON")
            return None
        if not isinstance(parsed, dict):
            logfire.warning("Image generation returned a non-object JSON body")
            return None
        return parsed


def _extract_b64(body: dict[str, object]) -> Optional[str]:
    """Pull ``data[0].b64_json`` out of the response, or None if it isn't there."""
    data = body.get("data")
    if not isinstance(data, list):
        logfire.warning("Image generation response 'data' field was not a list")
        return None
    if not data:
        logfire.warning("Image generation response contained no data entries")
        return None
    first = data[0]
    if not isinstance(first, dict):
        logfire.warning("Image generation response data entry was not an object")
        return None
    encoded = first.get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        logfire.warning("Image generation response contained no b64_json payload")
        return None
    return encoded
