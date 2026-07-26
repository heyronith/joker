"""Tests for CognitiveAgentRuntime nonblocking behaviour."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import pytest

from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import EventType, make_event
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.runtime.cognitive_agent_runtime import CognitiveAgentRuntime


@pytest.mark.asyncio
async def test_on_event_returns_quickly_while_worker_runs() -> None:
    fake = FakeModelProvider(available=True, latency_seconds=0.2)
    registry = ModelRegistry(providers={"fake": fake})
    router = ModelRouter(registry, session_id="sess")
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=5),
        session_id="sess",
        run_id="run",
    )
    runtime = CognitiveAgentRuntime(
        session_id="sess",
        run_id="run",
        router=router,
        config=deps.config,
        graph_deps=deps,
        registry=registry,
    )
    await runtime.start()

    event = make_event(
        EventType.MARKET_SNAPSHOT_CREATED,
        session_id="sess",
        source="test",
        exchange_timestamp=datetime.now(timezone.utc),
        payload={"snapshot_id": "00000000-0000-0000-0000-000000000001"},
    )
    started = time.monotonic()
    await runtime.on_event(event)
    elapsed = time.monotonic() - started
    assert elapsed < 0.1

    await asyncio.sleep(0.05)
    health = await runtime.health()
    assert health.queued_events >= 0

    await runtime.shutdown()
