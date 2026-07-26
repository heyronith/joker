"""Non-blocking cognitive runtime: market path continues during slow model calls."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import EventType, make_event
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.runtime.cognitive_agent_runtime import CognitiveAgentRuntime


@pytest.mark.asyncio
async def test_on_event_returns_quickly_while_work_queued() -> None:
    registry = ModelRegistry.with_defaults()
    fake = FakeModelProvider(available=True, latency_seconds=0.5)
    registry.register_provider("fake", fake)
    remapped = {
        n: p.model_copy(update={"provider": "fake"}) for n, p in registry.profiles.items()
    }
    registry.update_config(registry.config.model_copy(update={"profiles": remapped}))
    router = ModelRouter(registry, session_id="nb")
    runtime = CognitiveAgentRuntime(
        session_id="nb",
        run_id="nb",
        router=router,
        config=CognitiveGraphSettings(),
        registry=registry,
    )
    await runtime.start()
    try:
        event = make_event(
            EventType.MARKET_SNAPSHOT_CREATED,
            session_id="nb",
            source="test",
            exchange_timestamp=datetime.now(timezone.utc),
            payload={"snapshot_id": str(uuid4())},
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        await runtime.on_event(event)
        elapsed = loop.time() - started
        assert elapsed < 0.2
        health = await runtime.health()
        assert health.queued_events >= 0
    finally:
        await runtime.shutdown()
