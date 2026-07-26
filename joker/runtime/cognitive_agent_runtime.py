"""Asynchronous cognitive agent runtime with priority queues."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any
from uuid import uuid4

from joker.cognition.schemas import CognitiveError, CognitiveRuntimeHealth
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import DomainEvent, EventType, make_event
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.position_graph import build_position_graph
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.persistence.cognitive_repositories import (
    DebateRepository,
    DecisionRepository,
    EvidenceRepository,
    HypothesisRepository,
    StrategyRepository,
    WorldModelRepository,
)

logger = logging.getLogger(__name__)


class _Priority(IntEnum):
    CRITICAL = 0
    POSITION_SNAPSHOT = 1
    NEW_ENTRY_SNAPSHOT = 2


@dataclass(order=True)
class _QueuedWork:
    priority: int
    sequence: int
    event: DomainEvent = field(compare=False)
    kind: str = field(compare=False, default="decision")


@dataclass
class _RuntimeCounters:
    active_decision_cycles: int = 0
    active_position_cycles: int = 0
    queued_events: int = 0
    last_success_at: datetime | None = None
    last_error: CognitiveError | None = None


class CognitiveAgentRuntime:
    """Task 2 agent runtime — enqueues work and runs graphs asynchronously."""

    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        router: ModelRouter,
        config: CognitiveGraphSettings,
        graph_deps: CognitiveGraphDeps | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._session_id = session_id
        self._run_id = run_id
        self._router = router
        self._config = config
        self._registry = registry
        self._deps = graph_deps or CognitiveGraphDeps(
            router=router,
            config=config,
            session_id=session_id,
            run_id=run_id,
        )
        self._decision_graph = build_cognitive_graph(self._deps)
        self._position_graph = build_position_graph(self._deps)
        self._queue: asyncio.PriorityQueue[_QueuedWork] = asyncio.PriorityQueue()
        self._sequence = 0
        self._shutdown = False
        self._started = False
        self._worker_task: asyncio.Task[None] | None = None
        self._new_entry_lock = asyncio.Lock()
        self._new_entry_in_flight = False
        self._pending_new_entry_snapshot: str | None = None
        self._counters = _RuntimeCounters()
        self._status: str = "healthy"
        self._received_events: list[DomainEvent] = []

    @property
    def received_events(self) -> list[DomainEvent]:
        return list(self._received_events)

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._shutdown = False
        self._worker_task = asyncio.create_task(self._worker_loop(), name="cognitive-runtime")

    async def shutdown(self) -> None:
        self._status = "shutting_down"
        self._shutdown = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        self._started = False

    async def health(self) -> CognitiveRuntimeHealth:
        local_ok = await self._provider_available(local_only=True)
        remote_ok = await self._provider_available(local_only=False)
        status = self._status
        if not local_ok and not remote_ok:
            status = "unavailable"
        elif self._counters.last_error is not None:
            status = "degraded"
        return CognitiveRuntimeHealth(
            status=status,  # type: ignore[arg-type]
            local_provider_available=local_ok,
            remote_provider_available=remote_ok,
            active_decision_cycles=self._counters.active_decision_cycles,
            active_position_cycles=self._counters.active_position_cycles,
            queued_events=self._queue.qsize(),
            last_success_at=self._counters.last_success_at,
            last_error=self._counters.last_error,
        )

    async def on_event(self, event: DomainEvent) -> None:
        """Enqueue cognitive work and return immediately."""
        self._received_events.append(event)
        if not self._started or self._shutdown:
            return

        priority = self._event_priority(event)
        if (
            priority == _Priority.NEW_ENTRY_SNAPSHOT
            and self._config.market_snapshot_coalescing
        ):
            self._pending_new_entry_snapshot = str(
                event.payload.get("snapshot_id") or event.payload.get("market_snapshot_id") or ""
            )
            if self._new_entry_in_flight:
                return

        self._sequence += 1
        kind = "position" if event.event_type in {
            EventType.POSITION_OPENED,
            EventType.POSITION_CHANGED,
            EventType.POSITION_CLOSED,
            EventType.ORDER_FILLED,
            EventType.ORDER_PARTIALLY_FILLED,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_ACCEPTED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
        } else "decision"
        work = _QueuedWork(priority=priority, sequence=self._sequence, event=event, kind=kind)
        await self._queue.put(work)
        self._counters.queued_events = self._queue.qsize()

    def _event_priority(self, event: DomainEvent) -> int:
        if event.event_type in {
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_ACCEPTED,
            EventType.ORDER_PARTIALLY_FILLED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
            EventType.POSITION_OPENED,
            EventType.POSITION_CHANGED,
            EventType.POSITION_CLOSED,
        }:
            return _Priority.CRITICAL
        if event.event_type == EventType.MARKET_SNAPSHOT_CREATED:
            if event.payload.get("active_position_id"):
                return _Priority.POSITION_SNAPSHOT
            return _Priority.NEW_ENTRY_SNAPSHOT
        return _Priority.NEW_ENTRY_SNAPSHOT

    async def _worker_loop(self) -> None:
        while not self._shutdown:
            try:
                work = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if self._pending_new_entry_snapshot and not self._new_entry_in_flight:
                    await self._run_coalesced_new_entry()
                continue
            try:
                if work.kind == "position" and self._config.position.enabled:
                    await self._run_position_work(work.event)
                elif work.event.event_type == EventType.MARKET_SNAPSHOT_CREATED:
                    await self._run_decision_work(work.event)
                elif work.priority == _Priority.CRITICAL:
                    await self._run_order_work(work.event)
            except Exception as exc:
                logger.exception("cognitive worker error", exc_info=exc)
                self._counters.last_error = CognitiveError(
                    error_code="worker_failure",
                    message=str(exc),
                    recoverable=True,
                )
                self._status = "degraded"
            finally:
                self._queue.task_done()
                self._counters.queued_events = self._queue.qsize()

    async def _run_coalesced_new_entry(self) -> None:
        snapshot_id = self._pending_new_entry_snapshot
        self._pending_new_entry_snapshot = None
        if not snapshot_id:
            return
        fake_event = make_event(
            EventType.MARKET_SNAPSHOT_CREATED,
            session_id=self._session_id,
            source="cognitive_runtime_coalesce",
            exchange_timestamp=datetime.now(timezone.utc),
            payload={"snapshot_id": snapshot_id},
        )
        await self._run_decision_work(fake_event)

    async def _run_decision_work(self, event: DomainEvent) -> None:
        if self._new_entry_in_flight:
            snapshot_id = str(
                event.payload.get("snapshot_id") or event.payload.get("market_snapshot_id") or ""
            )
            self._pending_new_entry_snapshot = snapshot_id or self._pending_new_entry_snapshot
            return
        async with self._new_entry_lock:
            if self._new_entry_in_flight:
                return
            self._new_entry_in_flight = True
            self._counters.active_decision_cycles += 1
            try:
                await self._invoke_decision_graph(event)
            finally:
                self._new_entry_in_flight = False
                self._counters.active_decision_cycles = max(
                    0, self._counters.active_decision_cycles - 1
                )

    async def _invoke_decision_graph(self, event: DomainEvent) -> None:
        snapshot_id = str(
            event.payload.get("snapshot_id")
            or event.payload.get("market_snapshot_id")
            or ""
        )
        if not snapshot_id:
            return
        cycle_id = str(event.payload.get("cycle_id") or uuid4())
        state = initial_cycle_state(
            session_id=self._session_id,
            run_id=self._run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(event.event_id),
            trigger_event_type=event.event_type.value,
            snapshot_id=snapshot_id,
        )
        if self._deps.event_bus is not None:
            await self._deps.event_bus.publish(
                make_event(
                    EventType.COGNITIVE_CYCLE_STARTED,
                    session_id=self._session_id,
                    source="cognitive_runtime",
                    exchange_timestamp=event.exchange_timestamp,
                    correlation_id=event.correlation_id,
                    causation_id=event.event_id,
                    payload={
                        "cycle_id": cycle_id,
                        "snapshot_id": snapshot_id,
                    },
                )
            )
        try:
            await asyncio.wait_for(
                self._decision_graph.ainvoke(state),
                timeout=float(self._config.max_cycle_seconds),
            )
            self._counters.last_success_at = datetime.now(timezone.utc)
            self._status = "healthy"
        except asyncio.TimeoutError:
            self._counters.last_error = CognitiveError(
                error_code="cycle_timeout",
                message=f"cycle exceeded {self._config.max_cycle_seconds}s",
                recoverable=True,
            )
            self._status = "degraded"
        except Exception as exc:
            self._counters.last_error = CognitiveError(
                error_code="cycle_failed",
                message=str(exc),
                recoverable=True,
            )
            self._status = "degraded"

    async def _run_position_work(self, event: DomainEvent) -> None:
        if not self._config.position.enabled:
            return
        position_id = str(event.payload.get("position_id") or "")
        if not position_id:
            return
        snapshot_id = str(event.payload.get("snapshot_id") or event.payload.get("market_snapshot_id") or "")
        if not snapshot_id:
            return
        self._counters.active_position_cycles += 1
        try:
            state: dict[str, Any] = {
                "session_id": self._session_id,
                "run_id": self._run_id,
                "cycle_id": str(uuid4()),
                "snapshot_id": snapshot_id,
                "_position_id": position_id,
                "_contract_id": event.payload.get("contract_id"),
                "_original_strategy_id": event.payload.get("original_strategy_id"),
            }
            await self._position_graph.ainvoke(state)
            self._counters.last_success_at = datetime.now(timezone.utc)
        finally:
            self._counters.active_position_cycles = max(
                0, self._counters.active_position_cycles - 1
            )

    async def _run_order_work(self, event: DomainEvent) -> None:
        client_order_id = str(event.payload.get("client_order_id") or "")
        if not client_order_id:
            return
        # Order-manager cycles are lightweight; full graph wiring uses order_management agent.
        logger.info(
            "cognitive order event received",
            extra={
                "event_type": event.event_type.value,
                "client_order_id": client_order_id,
            },
        )

    async def _provider_available(self, *, local_only: bool) -> bool:
        if self._registry is None:
            return True
        try:
            if local_only:
                for name in ("fake", "ollama"):
                    provider = self._registry._providers.get(name)  # noqa: SLF001
                    if provider is not None:
                        health = await provider.healthcheck()
                        if health.status == "healthy":
                            return True
                return False
            provider = self._registry._providers.get("openai")  # noqa: SLF001
            if provider is None:
                return False
            health = await provider.healthcheck()
            return health.status == "healthy"
        except Exception:
            return False


def build_default_repositories(db_path: str) -> dict[str, Any]:
    """Construct default cognitive repositories for a session database."""
    return {
        "evidence_repo": EvidenceRepository(db_path),
        "world_model_repo": WorldModelRepository(db_path),
        "hypothesis_repo": HypothesisRepository(db_path),
        "strategy_repo": StrategyRepository(db_path),
        "debate_repo": DebateRepository(db_path),
        "decision_repo": DecisionRepository(db_path),
    }
