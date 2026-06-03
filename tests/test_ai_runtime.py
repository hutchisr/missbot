"""Tests for ChatAgent runtime behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai import ImageUrl

from bot.ai import ChatAgent, build_entity_linker
from bot.extract import ExtractedClaim, Skip
from bot.models import MiFile


@pytest.mark.anyio
async def test_run_accepts_image_only_note(config, make_note):
    agent = ChatAgent(config)
    note = make_note(
        text=None,
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
    )

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="vision reply"))) as run_mock,
    ):
        result = await agent.run(note)

    assert result == "vision reply"
    assert run_mock.await_args is not None
    prompt = run_mock.await_args.args[0]
    assert isinstance(prompt, list)
    assert prompt[0] == "alice: "
    assert isinstance(prompt[1], ImageUrl)


@pytest.mark.anyio
async def test_run_routes_image_prompt_to_vision_model(make_config, make_note):
    cfg = make_config(
        llm_models=[
            {"model": "openrouter:vision/model"},
            {"model": "openrouter:text/model", "vision": False},
        ]
    )
    agent = ChatAgent(cfg)
    assert agent._has_vision_model
    assert agent._vision_model is not None

    note = make_note(
        text=None,
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
    )

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["model"] is agent._vision_model


@pytest.mark.anyio
async def test_run_skips_model_override_when_no_images(make_config, make_note):
    cfg = make_config(
        llm_models=[
            {"model": "openrouter:vision/model"},
            {"model": "openrouter:text/model", "vision": False},
        ]
    )
    agent = ChatAgent(cfg)
    note = make_note(text="hi")

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    assert run_mock.await_args is not None
    assert "model" not in run_mock.await_args.kwargs


@pytest.mark.anyio
async def test_run_drops_images_when_no_vision_model(make_config, make_note):
    cfg = make_config(llm_models=[{"model": "openrouter:text/model", "vision": False}])
    agent = ChatAgent(cfg)
    assert not agent._has_vision_model

    note = make_note(
        text="look",
        files=[MiFile(id="f1", type="image/png", thumbnailUrl="https://media.example/1.png")],
    )

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    assert run_mock.await_args is not None
    prompt = run_mock.await_args.args[0]
    # Image was dropped — prompt collapses to a single string, no ImageUrl.
    assert isinstance(prompt, str)
    assert "model" not in run_mock.await_args.kwargs


@pytest.mark.anyio
async def test_run_flags_unrestricted_for_privileged_author(make_config, make_note, make_user):
    cfg = make_config(social_credit_unrestricted_user_ids=["admin-id"])
    agent = ChatAgent(cfg)
    note = make_note(text="judge bob", user=make_user(id="admin-id", username="boss"))

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["deps"].social_credit_unrestricted is True


@pytest.mark.anyio
async def test_run_restricted_for_non_privileged_author(make_config, make_note, make_user):
    cfg = make_config(social_credit_unrestricted_user_ids=["admin-id"])
    agent = ChatAgent(cfg)
    note = make_note(text="hi", user=make_user(id="rando-id", username="rando"))

    with (
        patch.object(agent, "_get_social_credit_score", AsyncMock(return_value=None)),
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="ok"))) as run_mock,
    ):
        await agent.run(note)

    assert run_mock.await_args is not None
    assert run_mock.await_args.kwargs["deps"].social_credit_unrestricted is False


# ---------------------------------------------------------------------------
# Automatic, injection-resistant message scoring
# ---------------------------------------------------------------------------


def test_score_agent_built_only_with_redis_and_enabled(make_config, fake_redis):
    assert ChatAgent(make_config())._score_agent is None  # no redis
    assert ChatAgent(make_config(), redis_client=fake_redis)._score_agent is not None
    disabled = ChatAgent(make_config(social_credit_auto_score=False), redis_client=fake_redis)
    assert disabled._score_agent is None


def test_score_model_defaults_to_main_model(make_config, fake_redis):
    # With no score_models, the classifier reuses the main reply model chain.
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    assert agent._score_model == "openrouter:test/model"


def test_score_model_uses_separate_chain_when_configured(make_config, fake_redis):
    cfg = make_config(score_models=["openrouter:cheap/classifier"])
    agent = ChatAgent(cfg, redis_client=fake_redis)
    assert agent._score_model == "openrouter:cheap/classifier"
    assert agent._score_agent is not None


@pytest.mark.anyio
async def test_run_auto_scores_non_privileged_message(make_config, make_note, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    note = make_note(text="what a lovely day")  # author = alice

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="good"))) as score_mock,
    ):
        out = await agent.run(note)

    assert out == "reply"
    assert score_mock.await_count == 1
    # "good" -> +5 (see QUALITY_DELTAS); applied to the author and cooldown claimed.
    assert await fake_redis.get("score:alice") == "5"
    assert await fake_redis.exists("score_cooldown:alice")


@pytest.mark.anyio
async def test_run_scoring_respects_cooldown(make_config, make_note, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    await fake_redis.set("score_cooldown:alice", "1")  # already on cooldown
    note = make_note(text="another banger")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="good"))) as score_mock,
    ):
        await agent.run(note)

    # Classifier is not even consulted while on cooldown, and no score is applied.
    assert score_mock.await_count == 0
    assert await fake_redis.get("score:alice") is None


@pytest.mark.anyio
async def test_run_scoring_skips_neutral_without_consuming_cooldown(make_config, make_note, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    note = make_note(text="ok")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="neutral"))),
    ):
        await agent.run(note)

    # Neutral => delta 0 => nothing written and cooldown NOT claimed (free retry).
    assert await fake_redis.get("score:alice") is None
    assert not await fake_redis.exists("score_cooldown:alice")


@pytest.mark.anyio
async def test_run_auto_scores_privileged_author_too(make_config, make_note, make_user, fake_redis):
    """Privileged users are also auto-scored; the flag only gates the manual tool."""
    cfg = make_config(social_credit_unrestricted_user_ids=["user-1"])
    agent = ChatAgent(cfg, redis_client=fake_redis)
    note = make_note(text="great thread", user=make_user(id="user-1", username="operator"))

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(
            agent._score_agent, "run", AsyncMock(return_value=SimpleNamespace(output="exceptional"))
        ) as score_mock,
    ):
        await agent.run(note)

    assert score_mock.await_count == 1
    # "exceptional" -> +10 (see QUALITY_DELTAS).
    assert await fake_redis.get("score:operator") == "10"


@pytest.mark.anyio
async def test_run_scoring_failure_does_not_break_reply(make_config, make_note, fake_redis):
    agent = ChatAgent(make_config(), redis_client=fake_redis)
    note = make_note(text="hi")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._score_agent, "run", AsyncMock(side_effect=RuntimeError("classifier down"))),
    ):
        out = await agent.run(note)

    # Scoring swallows its own errors; the reply still comes back.
    assert out == "reply"
    assert await fake_redis.get("score:alice") is None
    # A failed classification must not burn the cooldown — it should retry next message.
    assert not await fake_redis.exists("score_cooldown:alice")


# --- Note -> world-knowledge ingestion ---


def _memory_cfg(make_config, **extra):
    return make_config(
        memory_enabled=True,
        postgres_url="postgres://u:p@db/x",
        embedding_model="perplexity/pplx-embed-v1-0.6b",
        **extra,
    )


@pytest.mark.anyio
async def test_run_ingests_note_as_claim(make_config, make_note):
    mem = AsyncMock()
    mem.seconds_since_last_write.return_value = None
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    note = make_note(text="Python's latest version is 3.13")  # author = alice
    claim = ExtractedClaim(subject="Python", predicate="latest version", object="3.13")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._extract_agent, "run", AsyncMock(return_value=SimpleNamespace(output=claim))),
    ):
        out = await agent.run(note)

    assert out == "reply"
    mem.add_edge.assert_awaited_once()
    kw = mem.add_edge.await_args.kwargs
    # The note's author is attributed as the claim's author (this drives the agreement count).
    assert kw["author"] == "alice"
    assert kw["subject"] == "Python"
    assert kw["predicate"] == "latest version"
    assert kw["object_text"] == "3.13"


@pytest.mark.anyio
async def test_run_note_ingestion_respects_write_cooldown(make_config, make_note):
    mem = AsyncMock()
    mem.seconds_since_last_write.return_value = 5.0  # within the default 60s cooldown
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    note = make_note(text="some durable world fact")
    claim = ExtractedClaim(subject="X", predicate="y", object="z")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(
            agent._extract_agent, "run", AsyncMock(return_value=SimpleNamespace(output=claim))
        ) as extract_mock,
    ):
        await agent.run(note)

    # Cooldown short-circuits before paying for an extraction call.
    extract_mock.assert_not_awaited()
    mem.add_edge.assert_not_awaited()


@pytest.mark.anyio
async def test_run_note_ingestion_skips_when_extractor_rejects(make_config, make_note):
    mem = AsyncMock()
    mem.seconds_since_last_write.return_value = None
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    note = make_note(text="i really love pizza")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(
            agent._extract_agent, "run", AsyncMock(return_value=SimpleNamespace(output=Skip(reason="personal detail")))
        ),
    ):
        await agent.run(note)

    mem.add_edge.assert_not_awaited()


@pytest.mark.anyio
async def test_run_note_ingestion_disabled_by_flag(make_config, make_note):
    mem = AsyncMock()
    mem.seconds_since_last_write.return_value = None
    agent = ChatAgent(_memory_cfg(make_config, memory_ingest_notes=False), memory=mem)
    note = make_note(text="Python's latest version is 3.13")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._extract_agent, "run", AsyncMock()) as extract_mock,
    ):
        await agent.run(note)

    # Flag off => the ingestion path returns before extracting or writing.
    extract_mock.assert_not_awaited()
    mem.add_edge.assert_not_awaited()


@pytest.mark.anyio
async def test_run_note_ingestion_passes_speaker_to_extractor(make_config, make_note):
    mem = AsyncMock()
    mem.seconds_since_last_write.return_value = None
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    note = make_note(text="I use Arch btw")  # author = alice
    claim = ExtractedClaim(subject="alice", predicate="os", object="Arch Linux")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(
            agent._extract_agent, "run", AsyncMock(return_value=SimpleNamespace(output=claim))
        ) as extract_mock,
    ):
        await agent.run(note)

    # The author's handle is plumbed into the extraction prompt so "I" resolves to them.
    assert extract_mock.await_args is not None
    prompt = extract_mock.await_args.args[0]
    assert "@alice" in prompt
    mem.add_edge.assert_awaited_once()
    assert mem.add_edge.await_args.kwargs["subject"] == "alice"


@pytest.mark.anyio
async def test_run_note_ingestion_drops_sensitive_pii(make_config, make_note):
    mem = AsyncMock()
    mem.seconds_since_last_write.return_value = None
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    note = make_note(text="my email is alice@example.com")
    # Even if the extractor judged it storable, the code backstop must block obvious PII.
    claim = ExtractedClaim(subject="alice", predicate="email", object="alice@example.com")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(agent._extract_agent, "run", AsyncMock(return_value=SimpleNamespace(output=claim))),
    ):
        await agent.run(note)

    mem.add_edge.assert_not_awaited()


@pytest.mark.anyio
async def test_run_note_ingestion_supplies_thread_context(make_config, make_note):
    mem = AsyncMock()
    mem.seconds_since_last_write.return_value = None
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    parent = make_note(id="note-0", text="I have a pet lizard")  # earlier note by alice
    note = make_note(id="note-1", text="her name is Olive")  # follow-up by alice
    claim = ExtractedClaim(subject="alice's lizard", predicate="name", object="Olive")

    with (
        patch.object(agent._agent, "run", AsyncMock(return_value=SimpleNamespace(output="reply"))),
        patch.object(
            agent._extract_agent, "run", AsyncMock(return_value=SimpleNamespace(output=claim))
        ) as extract_mock,
    ):
        await agent.run(note, context=[parent])

    # The prior note rides along as reference context so "her" can resolve to the lizard.
    assert extract_mock.await_args is not None
    prompt = extract_mock.await_args.args[0]
    assert "I have a pet lizard" in prompt
    assert "her name is Olive" in prompt
    assert "@alice" in prompt
    mem.add_edge.assert_awaited_once()


# --- Write-time entity linking ---


def test_entity_linker_wired_when_memory_enabled(make_config):
    mem = AsyncMock()
    agent = ChatAgent(_memory_cfg(make_config), memory=mem)
    assert agent._entity_linker is not None
    assert mem.entity_linker is agent._entity_linker  # store gets the linker callable


def test_entity_linker_absent_without_memory(make_config):
    assert ChatAgent(make_config())._entity_linker is None  # memory disabled


def test_build_entity_linker_only_when_memory_enabled(make_config):
    assert build_entity_linker(make_config()) is None  # disabled => no linker
    assert callable(build_entity_linker(_memory_cfg(make_config)))  # enabled => a callable
