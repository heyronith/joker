"""Asynchronous cognitive agent runtime with independent decision/position workers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from joker.agents.cognitive.order_management import OrderManagerAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import (
    AgentRole,
    CognitiveError,
    CognitiveRuntimeHealth,
    OrderManagementDecision,
)
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import DomainEvent, EventType, make_event
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.context_hydrate import assemble_role_context, load_snapshot_truth
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import (
    CognitiveCheckpointer,
    ainvoke_config,
)
from joker.graph.position_graph import build_position_graph
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.persistence.cognitive_repositories import (
    DebateRepository,
    DecisionRepository,
    EvidenceRepository,
    HypothesisRepository,
    ModelCallRepository,
    OrderManagementRepository,
    PositionThesisRepository,
    StrategyRepository,
    WorldModelRepository,
)
from joker.cognition.artifacts import CognitiveArtifactStore

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
    """Task 2 agent runtime — independent decision and position workers."""

    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        router: ModelRouter,
        config: CognitiveGraphSettings,
        graph_deps: CognitiveGraphDeps | None = None,
        registry: ModelRegistry | None = None,
        checkpointer_path: Path | str | None = None,
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
        self._checkpointer_helper: CognitiveCheckpointer | None = None
        if checkpointer_path is not None:
            self._checkpointer_helper = CognitiveCheckpointer(Path(checkpointer_path))
        elif self._deps.db_path is not None:
            self._checkpointer_helper = CognitiveCheckpointer(
                Path(self._deps.db_path).with_name(
                    Path(self._deps.db_path).stem + "_cognitive_ckpt.db"
                )
            )
        self._decision_graph = None
        self._position_graph = None
        self._decision_queue: asyncio.PriorityQueue[_QueuedWork] = asyncio.PriorityQueue()
        self._position_queue: asyncio.PriorityQueue[_QueuedWork] = asyncio.PriorityQueue()
        self._sequence = 0
        self._shutdown = False
        self._started = False
        self._decision_worker: asyncio.Task[None] | None = None
        self._position_worker: asyncio.Task[None] | None = None
        self._active_decision_tasks: set[asyncio.Task[None]] = set()
        self._active_position_tasks: set[asyncio.Task[None]] = set()
        self._new_entry_lock = asyncio.Lock()
        self._new_entry_in_flight = False
        self._pending_new_entry_snapshot: str | None = None
        self._counters = _RuntimeCounters()
        self._status: str = "healthy"
        self._received_events: list[DomainEvent] = []
        self._order_decision_ids: set[str] = set()

    @property
    def received_events(self) -> list[DomainEvent]:
        return list(self._received_events)

    @property
    def deps(self) -> CognitiveGraphDeps:
        return self._deps

    async def start(self) -> None:
        if self._started:
            return
        if self._checkpointer_helper is not None:
            saver = await self._checkpointer_helper.open()
            self._deps.checkpointer = saver
        self._decision_graph = build_cognitive_graph(self._deps)
        self._position_graph = build_position_graph(self._deps)
        self._started = True
        self._shutdown = False
        self._decision_worker = asyncio.create_task(
            self._decision_worker_loop(), name="cognitive-decision-worker"
        )
        self._position_worker = asyncio.create_task(
            self._position_worker_loop(), name="cognitive-position-worker"
        )

    async def shutdown(self) -> None:
        """Checkpoint active cycles rather than merely cancelling them."""
        self._status = "shutting_down"
        self._shutdown = True
        # Allow in-flight tasks to finish (or timeout) so LangGraph can persist.
        pending = list(self._active_decision_tasks | self._active_position_tasks)
        if pending:
            done, still = await asyncio.wait(pending, timeout=5.0)
            for task in still:
                task.cancel()
            for task in still:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        for worker in (self._decision_worker, self._position_worker):
            if worker is not None:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
        self._decision_worker = None
        self._position_worker = None
        if self._checkpointer_helper is not None:
            await self._checkpointer_helper.close()
            self._deps.checkpointer = None
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
            queued_events=self._decision_queue.qsize() + self._position_queue.qsize(),
            last_success_at=self._counters.last_success_at,
            last_error=self._counters.last_error,
        )

    async def on_event(self, event: DomainEvent) -> None:
        """Enqueue cognitive work and return immediately — never drops position/order events."""
        self._received_events.append(event)
        if not self._started or self._shutdown:
            return

        priority = self._event_priority(event)
        kind = self._classify(event)

        if (
            kind == "decision"
            and priority == _Priority.NEW_ENTRY_SNAPSHOT
            and self._config.market_snapshot_coalescing
        ):
            self._pending_new_entry_snapshot = str(
                event.payload.get("snapshot_id")
                or event.payload.get("market_snapshot_id")
                or ""
            )
            if self._new_entry_in_flight:
                return

        self._sequence += 1
        work = _QueuedWork(
            priority=priority, sequence=self._sequence, event=event, kind=kind
        )
        if kind in {"position", "order"}:
            await self._position_queue.put(work)
        else:
            await self._decision_queue.put(work)
        self._counters.queued_events = (
            self._decision_queue.qsize() + self._position_queue.qsize()
        )

    def _classify(self, event: DomainEvent) -> str:
        if event.event_type in {
            EventType.POSITION_OPENED,
            EventType.POSITION_CHANGED,
            EventType.POSITION_CLOSED,
        }:
            return "position"
        if event.event_type in {
            EventType.ORDER_FILLED,
            EventType.ORDER_PARTIALLY_FILLED,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_ACCEPTED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
        }:
            return "order"
        if event.event_type == EventType.MARKET_SNAPSHOT_CREATED:
            if event.payload.get("active_position_id"):
                return "position"
            return "decision"
        return "decision"

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

    async def _decision_worker_loop(self) -> None:
        while not self._shutdown:
            try:
                work = await asyncio.wait_for(self._decision_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if self._pending_new_entry_snapshot and not self._new_entry_in_flight:
                    await self._run_coalesced_new_entry()
                continue
            try:
                await self._run_decision_work(work.event)
            except Exception as exc:
                logger.exception("cognitive decision worker error", exc_info=exc)
                self._counters.last_error = CognitiveError(
                    error_code="decision_worker_failure",
                    message=str(exc),
                    recoverable=True,
                )
                self._status = "degraded"
            finally:
                self._decision_queue.task_done()
                self._counters.queued_events = (
                    self._decision_queue.qsize() + self._position_queue.qsize()
                )

    async def _position_worker_loop(self) -> None:
        while not self._shutdown:
            try:
                work = await asyncio.wait_for(self._position_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                if work.kind == "order":
                    await self._run_order_work(work.event)
                else:
                    await self._run_position_work(work.event)
            except Exception as exc:
                logger.exception("cognitive position worker error", exc_info=exc)
                self._counters.last_error = CognitiveError(
                    error_code="position_worker_failure",
                    message=str(exc),
                    recoverable=True,
                )
                self._status = "degraded"
            finally:
                self._position_queue.task_done()
                self._counters.queued_events = (
                    self._decision_queue.qsize() + self._position_queue.qsize()
                )

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
                event.payload.get("snapshot_id")
                or event.payload.get("market_snapshot_id")
                or ""
            )
            self._pending_new_entry_snapshot = snapshot_id or self._pending_new_entry_snapshot
            return
        async with self._new_entry_lock:
            if self._new_entry_in_flight:
                return
            self._new_entry_in_flight = True
            self._counters.active_decision_cycles += 1
            task = asyncio.create_task(self._invoke_decision_graph(event))
            self._active_decision_tasks.add(task)
            try:
                await task
            finally:
                self._active_decision_tasks.discard(task)
                self._new_entry_in_flight = False
                self._counters.active_decision_cycles = max(
                    0, self._counters.active_decision_cycles - 1
                )

    async def _invoke_decision_graph(self, event: DomainEvent) -> None:
        assert self._decision_graph is not None
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
                    payload={"cycle_id": cycle_id, "snapshot_id": snapshot_id},
                )
            )
        config = ainvoke_config(
            session_id=self._session_id,
            graph_kind="decision",
            cycle_id=cycle_id,
        )
        try:
            await asyncio.wait_for(
                self._decision_graph.ainvoke(state, config=config),
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
        assert self._position_graph is not None
        position_id = str(
            event.payload.get("position_id")
            or event.payload.get("active_position_id")
            or ""
        )
        if not position_id:
            return
        snapshot_id = str(
            event.payload.get("snapshot_id")
            or event.payload.get("market_snapshot_id")
            or ""
        )
        if not snapshot_id:
            return
        cycle_id = str(event.payload.get("cycle_id") or uuid4())
        self._counters.active_position_cycles += 1
        state: dict[str, Any] = {
            "session_id": self._session_id,
            "run_id": self._run_id,
            "cycle_id": cycle_id,
            "snapshot_id": snapshot_id,
            "_position_id": position_id,
            "_contract_id": event.payload.get("contract_id"),
            "_original_strategy_id": event.payload.get("original_strategy_id"),
        }
        config = ainvoke_config(
            session_id=self._session_id,
            graph_kind="position",
            cycle_id=cycle_id,
        )
        task = asyncio.create_task(self._position_graph.ainvoke(state, config=config))
        self._active_position_tasks.add(task)
        try:
            await task
            self._counters.last_success_at = datetime.now(timezone.utc)
        finally:
            self._active_position_tasks.discard(task)
            self._counters.active_position_cycles = max(
                0, self._counters.active_position_cycles - 1
            )

    async def _run_order_work(self, event: DomainEvent) -> None:
        client_order_id = str(event.payload.get("client_order_id") or "")
        if not client_order_id:
            return
        snapshot_id = str(
            event.payload.get("snapshot_id")
            or event.payload.get("market_snapshot_id")
            or ""
        )
        # Build order projection from execution runtime when available.
        order_projection: dict[str, Any] = {
            "client_order_id": client_order_id,
            "event_type": event.event_type.value,
            **{k: v for k, v in event.payload.items() if k != "client_order_id"},
        }
        if self._deps.execution_runtime is not None:
            try:
                sync = await self._deps.execution_runtime.poll_order_status(
                    client_order_id
                )
                if sync is not None:
                    order_projection.update(
                        {
                            "status": sync.status,
                            "quantity": sync.quantity,
                            "filled_quantity": getattr(sync, "filled_quantity", None),
                            "limit_price": getattr(sync, "limit_price", None),
                            "contract_id": str(
                                getattr(getattr(sync, "contract", None), "symbol", "")
                            ),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("order_projection_hydrate_failed", extra={"error": str(exc)})

        context: ContextPackage | None = None
        if snapshot_id and self._deps.snapshot_repo is not None:
            try:
                snapshot, data_quality, _surface, surface_slice = await load_snapshot_truth(
                    self._deps, snapshot_id
                )
                context = await assemble_role_context(
                    self._deps,
                    agent_role=AgentRole.ORDER_MANAGER,
                    session_id=self._session_id,
                    cycle_id=str(uuid4()),
                    snapshot=snapshot,
                    data_quality=data_quality,
                    option_surface_slice=surface_slice,
                    order_projection=order_projection,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("order_context_failed", extra={"error": str(exc)})

        if context is None:
            logger.info(
                "cognitive order event received without snapshot context",
                extra={"client_order_id": client_order_id},
            )
            return

        agent = OrderManagerAgent()
        decision = await agent.manage(
            context,
            self._router,
            client_order_id=client_order_id,
            order_projection=order_projection,
        )
        # Duplicate decision idempotency.
        decision_key = f"{client_order_id}:{decision.action}:{decision.rationale_summary}"
        if decision_key in self._order_decision_ids:
            return
        self._order_decision_ids.add(decision_key)
        if self._deps.order_management_repo is not None:
            await self._deps.order_management_repo.append(decision)
        await self._apply_order_decision(decision)

    async def _apply_order_decision(self, decision: OrderManagementDecision) -> None:
        runtime = self._deps.execution_runtime
        if runtime is None:
            return
        action = decision.action
        if action == "continue_waiting":
            return
        if action in {"cancel", "abandon"}:
            await runtime.cancel_order(client_order_id=decision.client_order_id)
            return
        if action in {"replace", "reduce_quantity"}:
            # Cancel then optional replace is operationally supported via cancel today.
            await runtime.cancel_order(client_order_id=decision.client_order_id)
            return

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


def build_default_repositories(db_path: str | Path) -> dict[str, Any]:
    """Construct default cognitive repositories for a session database."""
    store = CognitiveArtifactStore(db_path)
    return {
        "evidence_repo": EvidenceRepository(store),
        "world_model_repo": WorldModelRepository(store),
        "hypothesis_repo": HypothesisRepository(store),
        "strategy_repo": StrategyRepository(store),
        "debate_repo": DebateRepository(store),
        "decision_repo": DecisionRepository(store),
        "position_thesis_repo": PositionThesisRepository(store),
        "order_management_repo": OrderManagementRepository(store),
        "model_call_repo": ModelCallRepository(store),
    }
