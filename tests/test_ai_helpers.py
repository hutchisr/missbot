"""Tests for pure helper functions in bot.ai."""

from pydantic_ai import ImageUrl

from bot.ai import _build_user_content, _image_urls_for, _user_handle
from bot.models import MiFile


def test_user_handle_local(make_user):
    assert _user_handle(make_user(username="alice", host=None)) == "alice"


def test_user_handle_remote(make_user):
    assert _user_handle(make_user(username="alice", host="remote.host")) == "alice@remote.host"


def test_image_urls_for_vision_off(make_note):
    note = make_note(files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://x/1.png")])
    assert _image_urls_for(note, vision=False) == []


def test_image_urls_for_filters_non_images(make_note):
    files = [
        MiFile(id="f1", type="image/png", thumbnailUrl="https://x/1.png"),
        MiFile(id="f2", type="video/mp4", thumbnailUrl="https://x/2.mp4"),
        MiFile(id="f3", type="image/jpeg", thumbnailUrl=None),
    ]
    note = make_note(files=files)
    urls = _image_urls_for(note, vision=True)
    assert len(urls) == 1
    assert isinstance(urls[0], ImageUrl)
    assert urls[0].url == "https://x/1.png"


def test_image_urls_for_no_files(make_note):
    note = make_note(files=None)
    assert _image_urls_for(note, vision=True) == []


def test_build_user_content_text_only(make_note, make_user):
    note = make_note(user=make_user(username="alice"), text="hi")
    content = _build_user_content(note, vision=True)
    assert content == "alice: hi"


def test_build_user_content_with_images(make_note, make_user):
    note = make_note(
        user=make_user(username="bob", host="remote.host"),
        text="look",
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://x/1.png")],
    )
    content = _build_user_content(note, vision=True)
    assert isinstance(content, list)
    assert content[0] == "bob@remote.host: look"
    assert isinstance(content[1], ImageUrl)


def test_build_user_content_empty_text(make_note):
    note = make_note(text=None)
    content = _build_user_content(note, vision=True)
    assert content == "alice: "
