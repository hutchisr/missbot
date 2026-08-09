"""Tests for the image-generation provider client."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from bot.imagegen import GeneratedImage, ImageGenerator
from bot.provider import PROJECT_VERSION


_PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64
_JPEG = b"\xff\xd8\xff" + b"y" * 64
_GIF = b"GIF89a" + b"z" * 64
_GIF87 = b"GIF87a" + b"z" * 64
_WEBP = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"w" * 64
_SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'


@pytest.fixture
def image_config(make_config):
    return make_config(
        system_prompt_auto="Post something.",
        image_gen_enabled=True,
        image_gen_model="test/image-model",
        image_gen_api_key="image-key",
    )


def _body(payload: bytes) -> dict:
    return {"data": [{"b64_json": base64.b64encode(payload).decode()}]}


def _responder(body, status: int = 200, captured: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        content = body if isinstance(body, bytes) else json.dumps(body).encode()
        return httpx.Response(status, content=content, headers={"content-type": "application/json"})

    return httpx.MockTransport(handler)


def _generator(image_config, transport) -> ImageGenerator:
    return ImageGenerator(image_config, transport=transport)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("payload", "media_type", "extension"),
    [
        (_PNG, "image/png", "png"),
        (_JPEG, "image/jpeg", "jpg"),
        (_GIF, "image/gif", "gif"),
        (_GIF87, "image/gif", "gif"),
        (_WEBP, "image/webp", "webp"),
    ],
)
async def test_generate_returns_image_with_sniffed_media_type(image_config, payload, media_type, extension):
    gen = _generator(image_config, _responder(_body(payload)))

    image = await gen.generate("a shrimp in a tiny hat", "a shrimp wearing a hat")

    assert image is not None
    assert image == GeneratedImage(
        data=payload,
        media_type=media_type,
        prompt="a shrimp in a tiny hat",
        alt_text="a shrimp wearing a hat",
    )
    assert image.extension == extension


@pytest.mark.anyio
async def test_generate_sends_model_prompt_and_key(image_config):
    captured: list[httpx.Request] = []
    gen = _generator(image_config, _responder(_body(_PNG), captured=captured))

    await gen.generate("draw a shrimp", "a shrimp")

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/v1/images/generations"
    assert request.headers["authorization"] == "Bearer image-key"
    assert request.headers["user-agent"] == f"Missbot/{PROJECT_VERSION}"
    assert request.headers["http-referer"] == "rad://zLseUdKik1qrsiTonrjSoPGYbC6g"
    assert request.headers["x-openrouter-title"] == f"missbot-{PROJECT_VERSION}"
    # The Misskey token must never reach a third-party provider.
    assert "secret-token" not in str(request.headers)
    sent = json.loads(request.content)
    assert sent == {"model": "test/image-model", "prompt": "draw a shrimp", "n": 1}


@pytest.mark.anyio
async def test_generate_sends_size_only_when_configured(make_config):
    captured: list[httpx.Request] = []
    cfg = make_config(
        system_prompt_auto="Post something.",
        image_gen_enabled=True,
        image_gen_model="test/image-model",
        image_gen_api_key="image-key",
        image_gen_size="1024x1024",
    )
    gen = ImageGenerator(cfg, transport=_responder(_body(_PNG), captured=captured))

    await gen.generate("draw a shrimp", "a shrimp")

    assert json.loads(captured[0].content)["size"] == "1024x1024"


@pytest.mark.anyio
async def test_generate_reads_key_from_env(make_config, monkeypatch):
    monkeypatch.setenv("IMAGE_KEY_FROM_ENV", "env-key")
    captured: list[httpx.Request] = []
    cfg = make_config(
        system_prompt_auto="Post something.",
        image_gen_enabled=True,
        image_gen_model="test/image-model",
        image_gen_api_key_env="IMAGE_KEY_FROM_ENV",
    )
    gen = ImageGenerator(cfg, transport=_responder(_body(_PNG), captured=captured))

    await gen.generate("draw a shrimp", "a shrimp")

    assert captured[0].headers["authorization"] == "Bearer env-key"


@pytest.mark.anyio
async def test_generate_omits_auth_header_when_no_key_resolves(make_config, monkeypatch):
    monkeypatch.delenv("MISSING_IMAGE_KEY", raising=False)
    captured: list[httpx.Request] = []
    cfg = make_config(
        system_prompt_auto="Post something.",
        image_gen_enabled=True,
        image_gen_model="test/image-model",
        image_gen_api_key_env="MISSING_IMAGE_KEY",
    )
    gen = ImageGenerator(cfg, transport=_responder(_body(_PNG), captured=captured))

    # A keyless self-hosted endpoint is a supported deployment, so this warns, not raises.
    assert await gen.generate("draw a shrimp", "a shrimp") is not None
    assert "authorization" not in captured[0].headers


@pytest.mark.anyio
async def test_generate_refuses_svg(image_config):
    """SVG can carry script and would be served from the instance's own origin."""
    gen = _generator(image_config, _responder(_body(_SVG)))

    assert await gen.generate("draw a shrimp", "a shrimp") is None


@pytest.mark.anyio
async def test_generate_refuses_unrecognized_format(image_config):
    gen = _generator(image_config, _responder(_body(b"not an image at all")))

    assert await gen.generate("draw a shrimp", "a shrimp") is None


@pytest.mark.anyio
async def test_generate_drops_oversized_image(make_config):
    cfg = make_config(
        system_prompt_auto="Post something.",
        image_gen_enabled=True,
        image_gen_model="test/image-model",
        image_gen_api_key="image-key",
        image_gen_max_bytes=100,
    )
    gen = ImageGenerator(cfg, transport=_responder(_body(_PNG + b"p" * 500)))

    assert await gen.generate("draw a shrimp", "a shrimp") is None


@pytest.mark.anyio
async def test_generate_drops_image_over_post_decode_size_cap(make_config):
    """Distinct from test_generate_drops_oversized_image: that test trips the pre-decode
    base64-length check. This payload is short enough to pass the pre-decode check (136
    base64 chars <= the (max_bytes*4)//3+4 = 137 allowance) but decodes to 102 bytes, over
    the 100-byte cap — so only the post-decode check catches it."""
    cfg = make_config(
        system_prompt_auto="Post something.",
        image_gen_enabled=True,
        image_gen_model="test/image-model",
        image_gen_api_key="image-key",
        image_gen_max_bytes=100,
    )
    payload = b"x" * 102
    encoded = base64.b64encode(payload).decode()
    assert len(encoded) == 136
    gen = ImageGenerator(cfg, transport=_responder({"data": [{"b64_json": encoded}]}))

    assert await gen.generate("draw a shrimp", "a shrimp") is None


@pytest.mark.anyio
async def test_generate_drops_oversized_response_body(make_config):
    """A broken or hostile endpoint must not balloon the pod's memory with one JSON body."""
    cfg = make_config(
        system_prompt_auto="Post something.",
        image_gen_enabled=True,
        image_gen_model="test/image-model",
        image_gen_api_key="image-key",
        image_gen_max_bytes=64,
    )
    huge = b'{"data": [{"b64_json": "' + b"A" * 500_000 + b'"}]}'
    gen = ImageGenerator(cfg, transport=_responder(huge))

    assert await gen.generate("draw a shrimp", "a shrimp") is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": []},
        {"data": [{}]},
        {"data": [{"b64_json": ""}]},
        {"data": "not-a-list"},
        {"data": ["not-an-object"]},
        {"data": [{"b64_json": "!!!not-base64!!!"}]},
    ],
)
async def test_generate_returns_none_on_unusable_payload(image_config, body):
    gen = _generator(image_config, _responder(body))

    assert await gen.generate("draw a shrimp", "a shrimp") is None


@pytest.mark.anyio
async def test_generate_returns_none_on_invalid_json(image_config):
    gen = _generator(image_config, _responder(b"<html>gateway error</html>"))

    assert await gen.generate("draw a shrimp", "a shrimp") is None


@pytest.mark.anyio
async def test_generate_returns_none_on_non_object_json_body(image_config):
    """A 2xx body can be valid JSON that isn't an object (e.g. a bare list or scalar) — the
    top-level `not isinstance(parsed, dict)` guard in `_post`, distinct from the `data`-field
    guards in `_extract_b64` covered above."""
    gen = _generator(image_config, _responder([1, 2, 3]))

    assert await gen.generate("draw a shrimp", "a shrimp") is None


@pytest.mark.anyio
async def test_generate_returns_none_on_http_error(image_config):
    gen = _generator(image_config, _responder({"error": "nope"}, status=500))

    assert await gen.generate("draw a shrimp", "a shrimp") is None


@pytest.mark.anyio
async def test_generate_returns_none_on_transport_error(image_config):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    gen = _generator(image_config, httpx.MockTransport(handler))

    assert await gen.generate("draw a shrimp", "a shrimp") is None
