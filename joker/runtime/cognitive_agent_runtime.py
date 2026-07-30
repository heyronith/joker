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
        evolution_runtime: Any | None = None,
    ) -> None:
        self._session_id = session_id
        self._run_id = run_id
        self._router = router
        self._config = config
        self._registry = registry
        self._evolution_runtime = evolution_runtime
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
        # When True, market snapshots still drive open-position management but do
        # not enqueue new-entry decision cycles (shadow evidence collection).
        self._suppress_new_entry_snapshots = False
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

    def suppress_new_entry_snapshots(self, suppressed: bool = True) -> None:
        """Block new-entry decision enqueue while still allowing position cycles."""
        self._suppress_new_entry_snapshots = bool(suppressed)
        if suppressed:
            self._pending_new_entry_snapshot = None

    def bind_evolution_runtime(self, evolution_runtime: Any) -> None:
        """Inject Task 3 runtime before workers start (supported public API)."""
        if self._started and self._evolution_runtime is not None:
            raise RuntimeError(
                "cannot rebind evolution runtime after CognitiveAgentRuntime has started"
            )
        self._evolution_runtime = evolution_runtime

    async def start(self) -> None:
        if self._started:
            return
        if self._checkpointer_helper is not None:
            saver = await self._checkpointer_helper.open()
            self._deps.checkpointer = saver
        from joker.runtime.order_action_gateway import ensure_order_action_gateway

        ensure_order_action_gateway(self._deps)
        if self._deps.cycle_registry is None and self._deps.db_path is not None:
            from joker.persistence.cognitive_cycle_registry import CognitiveCycleRegistry

            registry = CognitiveCycleRegistry(
                Path(self._deps.db_path).with_name(
                    Path(self._deps.db_path).stem + "_cognitive_cycles.db"
                )
            )
            await registry.initialize()
            self._deps.cycle_registry = registry
        if (
            self._deps.order_management_action_repo is None
            and self._deps.db_path is not None
        ):
            from joker.persistence.order_management_actions import (
                OrderManagementActionRepository,
            )

            om_repo = OrderManagementActionRepository(
                Path(self._deps.db_path).with_name(
                    Path(self._deps.db_path).stem + "_om_actions.db"
                )
            )
            await om_repo.initialize()
            self._deps.order_management_action_repo = om_repo
        self._decision_graph = build_cognitive_graph(self._deps)
        self._position_graph = build_position_graph(self._deps)
        self._started = True
        self._shutdown = False
        # Resume unfinished cycles before accepting new events.
        try:
            await self._resume_unfinished_cycles()
        except Exception as exc:  # noqa: BLE001
            logger.exception("cognitive_cycle_recovery_failed", exc_info=exc)
            self._status = "degraded"
            self._counters.last_error = CognitiveError(
                error_code="cycle_recovery_failed",
                message=str(exc),
                recoverable=True,
            )
        self._decision_worker = asyncio.create_task(
            self._decision_worker_loop(), name="cognitive-decision-worker"
        )
        self._position_worker = asyncio.create_task(
            self._position_worker_loop(), name="cognitive-position-worker"
        )

    async def _resume_unfinished_cycles(self) -> None:
        registry = self._deps.cycle_registry
        if registry is None or self._deps.checkpointer is None:
            return
        from joker.graph.langgraph_checkpointer import ainvoke_config
        from joker.persistence.cognitive_cycle_registry import CognitiveCycleRecord

        resumable = await registry.list_resumable(self._session_id)
        for record in resumable:
            graph = (
                self._decision_graph
                if record.graph_kind == "decision"
                else self._position_graph
            )
            if graph is None:
                continue
            config = ainvoke_config(
                session_id=record.session_id,
                graph_kind=record.graph_kind,
                cycle_id=record.cycle_id,
            )
            await registry.upsert(
                CognitiveCycleRecord(
                    session_id=record.session_id,
                    graph_kind=record.graph_kind,
                    cycle_id=record.cycle_id,
                    trigger_event_id=record.trigger_event_id,
                    snapshot_id=record.snapshot_id,
                    status="running",
                    checkpoint_thread_id=record.checkpoint_thread_id,
                    last_completed_node=record.last_completed_node,
                    parent_entry_cycle_id=record.parent_entry_cycle_id,
                    original_strategy_id=record.original_strategy_id,
                    original_proposal_id=record.original_proposal_id,
                    payload=record.payload,
                )
            )
            try:
                from joker.cognition.prompt_overrides import pinned_applied_configuration

                applied = None
                payload = record.payload or {}
                cfg_raw = payload.get("configuration_version_id")
                if self._evolution_runtime is not None and cfg_raw:
                    from uuid import UUID as _UUID

                    applied = await self._evolution_runtime.apply_configuration_version(
                        record.cycle_id, _UUID(str(cfg_raw))
                    )
                if applied is not None:
                    with pinned_applied_configuration(applied):
                        resumed = await graph.ainvoke(None, config=config)
                else:
                    resumed = await graph.ainvoke(None, config=config)
                terminal_ok = self._cycle_reached_terminal_outcome(
                    resumed, graph_kind=record.graph_kind
                )
                await registry.upsert(
                    CognitiveCycleRecord(
                        session_id=record.session_id,
                        graph_kind=record.graph_kind,
                        cycle_id=record.cycle_id,
                        trigger_event_id=record.trigger_event_id,
                        snapshot_id=record.snapshot_id,
                        status="completed" if terminal_ok else "running",
                        checkpoint_thread_id=record.checkpoint_thread_id,
                        last_completed_node=record.last_completed_node,
                        parent_entry_cycle_id=record.parent_entry_cycle_id,
                        original_strategy_id=record.original_strategy_id,
                        original_proposal_id=record.original_proposal_id,
                        payload={
                            **(record.payload or {}),
                            "recovery_terminal_ok": terminal_ok,
                        },
                    )
                )
                if not terminal_ok:
                    self._status = "degraded"
                    logger.warning(
                        "cycle_resume_incomplete",
                        extra={
                            "cycle_id": record.cycle_id,
                            "errors": [
                                getattr(e, "error_code", None)
                                for e in (resumed or {}).get("errors") or []
                            ],
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "cycle_resume_failed",
                    extra={"cycle_id": record.cycle_id, "error": str(exc)},
                )
                self._status = "degraded"

    @staticmethod
    def _cycle_reached_terminal_outcome(
        state: dict[str, Any] | None,
        *,
        graph_kind: str,
    ) -> bool:
        """Return True only for a valid terminal graph outcome without blocking errors."""
        if not isinstance(state, dict):
            return False
        errors = state.get("errors") or []
        blocking_codes = {
            "no_submit_callback",
            "no_order_action_gateway",
            "gateway_blocked",
            "submit_validation_failed",
            "validation_failed",
            "cycle_recovery_failed",
        }
        for err in errors:
            code = getattr(err, "error_code", None)
            if code is None and isinstance(err, dict):
                code = err.get("error_code")
            if code in blocking_codes:
                return False
        traces = state.get("node_trace") or []
        terminal_nodes = {
            "persist_cycle",
            "persist_pending_cycle",
            "persist_evidence_request",
            "persist_stale",
            "route_position_action",
        }
        for trace in traces:
            name = getattr(trace, "node_name", None)
            status = getattr(trace, "status", None)
            if name is None and isinstance(trace, dict):
                name = trace.get("node_name")
                status = trace.get("status")
            if name in terminal_nodes and status == "completed":
                return True
        if graph_kind == "decision" and state.get("execution_command_id"):
            return True
        if graph_kind == "position" and state.get("_position_command_id"):
            return True
        # Delayed / evidence / hold without order still count if a persist node ran.
        return False

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
        await self._registry.aclose()
        from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers

        await drain_aiosqlite_workers()
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

        if event.event_type == EventType.MARKET_SNAPSHOT_CREATED:
            # Never await ledger projection on the bus handler path — that can
            # exceed handler_timeout under SQLite contention and cancel routing.
            self._sequence += 1
            await self._decision_queue.put(
                _QueuedWork(
                    priority=_Priority.NEW_ENTRY_SNAPSHOT,
                    sequence=self._sequence,
                    event=event,
                    kind="snapshot_route",
                )
            )
            self._counters.queued_events = (
                self._decision_queue.qsize() + self._position_queue.qsize()
            )
            return

        priority = self._event_priority(event)
        kind = self._classify(event)
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

    async def _enqueue_snapshot_work(self, event: DomainEvent) -> None:
        """Route snapshots using authoritative ledger projection, not event metadata."""
        snapshot_id = str(
            event.payload.get("snapshot_id")
            or event.payload.get("market_snapshot_id")
            or ""
        )
        open_positions = await self._open_position_contract_ids()
        working_entry = await self._has_working_entry_order()
        if open_positions and self._config.position.enabled:
            for contract_id in open_positions:
                enriched = make_event(
                    EventType.MARKET_SNAPSHOT_CREATED,
                    session_id=event.session_id,
                    source="cognitive_runtime_position_route",
                    exchange_timestamp=event.exchange_timestamp,
                    correlation_id=event.correlation_id,
                    causation_id=event.event_id,
                    payload={
                        **dict(event.payload),
                        "snapshot_id": snapshot_id,
                        "position_id": contract_id,
                        "contract_id": contract_id,
                        "active_position_id": contract_id,
                        "trigger_event_id": str(event.event_id),
                    },
                )
                self._sequence += 1
                await self._position_queue.put(
                    _QueuedWork(
                        priority=_Priority.POSITION_SNAPSHOT,
                        sequence=self._sequence,
                        event=enriched,
                        kind="position",
                    )
                )
        # New-entry only when flat, no working entry, and not suppressed (shadow).
        if (
            not open_positions
            and not working_entry
            and not self._suppress_new_entry_snapshots
        ):
            if (
                self._config.market_snapshot_coalescing
                and self._new_entry_in_flight
            ):
                self._pending_new_entry_snapshot = snapshot_id or self._pending_new_entry_snapshot
            else:
                self._sequence += 1
                await self._decision_queue.put(
                    _QueuedWork(
                        priority=_Priority.NEW_ENTRY_SNAPSHOT,
                        sequence=self._sequence,
                        event=event,
                        kind="decision",
                    )
                )
        self._counters.queued_events = (
            self._decision_queue.qsize() + self._position_queue.qsize()
        )

    async def _has_working_entry_order(self) -> bool:
        if self._deps.projection_loader is None:
            return False
        try:
            projection = await self._deps.projection_loader()
        except Exception:
            return False
        from joker.runtime.order_action_gateway import (
            has_working_entry_order,
            working_orders_from_projection,
        )

        return has_working_entry_order(working_orders_from_projection(projection))

    async def _open_position_contract_ids(self) -> list[str]:
        if self._deps.projection_loader is None:
            return []
        try:
            projection = await self._deps.projection_loader()
        except Exception as exc:  # noqa: BLE001
            logger.warning("projection_loader_failed", extra={"error": str(exc)})
            return []
        if projection is None:
            return []
        positions = getattr(projection, "positions", None) or {}
        open_ids: list[str] = []
        items = positions.items() if isinstance(positions, dict) else (
            (getattr(p, "contract_id", None), p) for p in positions
        )
        for key, pos in items:
            qty = getattr(pos, "quantity", None)
            if qty is None and isinstance(pos, dict):
                qty = pos.get("quantity") or pos.get("net_quantity")
            try:
                from decimal import Decimal

                q = Decimal(str(qty)) if qty is not None else Decimal("0")
            except Exception:
                continue
            if q == 0:
                continue
            cid = getattr(pos, "contract_id", None) or key
            if cid is None and isinstance(pos, dict):
                cid = pos.get("contract_id")
            if cid:
                open_ids.append(str(cid))
        return open_ids

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
            if event.payload.get("active_position_id") or event.payload.get("position_id"):
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
            if event.payload.get("active_position_id") or event.payload.get("position_id"):
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
                if work.kind == "snapshot_route":
                    await self._enqueue_snapshot_work(work.event)
                else:
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
        if await self._has_working_entry_order():
            logger.info(
                "decision_cycle_skipped_working_entry",
                extra={"event_id": str(event.event_id)},
            )
            return
        snapshot_id = str(
            event.payload.get("snapshot_id")
            or event.payload.get("market_snapshot_id")
            or ""
        )
        if not snapshot_id:
            return
        cycle_id = str(event.payload.get("cycle_id") or uuid4())
        from joker.graph.langgraph_checkpointer import cognitive_thread_id
        from joker.persistence.cognitive_cycle_registry import CognitiveCycleRecord

        thread_id = cognitive_thread_id(
            session_id=self._session_id, graph_kind="decision", cycle_id=cycle_id
        )
        if self._deps.cycle_registry is not None:
            await self._deps.cycle_registry.upsert(
                CognitiveCycleRecord(
                    session_id=self._session_id,
                    graph_kind="decision",
                    cycle_id=cycle_id,
                    trigger_event_id=str(event.event_id),
                    snapshot_id=snapshot_id,
                    status="running",
                    checkpoint_thread_id=thread_id,
                )
            )
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
            from joker.cognition.prompt_overrides import pinned_applied_configuration

            applied = None
            if self._evolution_runtime is not None:
                applied = await self._evolution_runtime.pin_and_apply_for_cycle(cycle_id)
            if applied is not None:
                with pinned_applied_configuration(applied):
                    result_state = await asyncio.wait_for(
                        self._decision_graph.ainvoke(state, config=config),
                        timeout=float(self._config.max_cycle_seconds),
                    )
            else:
                result_state = await asyncio.wait_for(
                    self._decision_graph.ainvoke(state, config=config),
                    timeout=float(self._config.max_cycle_seconds),
                )
            self._counters.last_success_at = datetime.now(timezone.utc)
            self._status = "healthy"
            if self._deps.cycle_registry is not None:
                terminal_ok = self._cycle_reached_terminal_outcome(
                    result_state, graph_kind="decision"
                )
                await self._deps.cycle_registry.upsert(
                    CognitiveCycleRecord(
                        session_id=self._session_id,
                        graph_kind="decision",
                        cycle_id=cycle_id,
                        trigger_event_id=str(event.event_id),
                        snapshot_id=snapshot_id,
                        status="completed" if terminal_ok else "running",
                        checkpoint_thread_id=thread_id,
                        payload={
                            "configuration_version_id": (
                                str(applied.configuration_version_id)
                                if applied is not None
                                else None
                            )
                        },
                    )
                )
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

    async def _resolve_provenance(
        self, event: DomainEvent
    ) -> dict[str, Any]:
        """Resolve cognitive metadata from registry using Task 1 event fields."""
        payload = dict(event.payload)
        client_order_id = str(payload.get("client_order_id") or "")
        contract_id = str(payload.get("contract_id") or "")
        registry = self._deps.provenance_registry
        record = None
        if registry is not None and client_order_id:
            record = await registry.get_by_client_order_id(client_order_id)
        if record is None and registry is not None and contract_id:
            record = await registry.get_latest_by_contract_id(contract_id)
        if record is not None:
            if not contract_id and record.contract_id:
                contract_id = record.contract_id
            payload.setdefault("snapshot_id", record.snapshot_id)
            payload.setdefault("strategy_id", record.strategy_id)
            payload.setdefault("original_strategy_id", record.strategy_id)
            payload.setdefault("proposal_id", record.proposal_id)
            payload.setdefault("decision_id", record.decision_id)
            payload.setdefault("cycle_id", record.cycle_id)
            if record.contract_id:
                payload.setdefault("contract_id", record.contract_id)
                contract_id = contract_id or record.contract_id
        # Task 1 uses contract_id as the authoritative position identity.
        if contract_id:
            payload.setdefault("position_id", contract_id)
            payload.setdefault("active_position_id", contract_id)
            payload.setdefault("contract_id", contract_id)
        # Fall back to latest market snapshot when provenance lacks snapshot_id.
        if not payload.get("snapshot_id") and self._deps.snapshot_repo is not None:
            try:
                latest = await self._deps.snapshot_repo.get_latest(self._session_id)
                if latest is not None:
                    payload["snapshot_id"] = str(latest.snapshot_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("latest_snapshot_lookup_failed", extra={"error": str(exc)})
        return payload

    async def _sync_objective_reservation(
        self, event: DomainEvent, *, client_order_id: str
    ) -> None:
        """Task 2 observes order events only — financial mutations are Task 1 owned.

        Objective capital projection runs via SessionSupervisor's
        ObjectiveCapitalProjector subscribed to the Task 1 event bus.
        """
        return

    async def _sync_objective_on_position_closed(self, event: DomainEvent) -> None:
        """Task 2 observes closes only — realised PnL / exposure release is Task 1."""
        return

    async def _run_position_work(self, event: DomainEvent) -> None:
        if not self._config.position.enabled:
            return
        assert self._position_graph is not None
        if event.event_type == EventType.POSITION_CLOSED:
            await self._sync_objective_on_position_closed(event)
            return
        resolved = await self._resolve_provenance(event)
        position_id = str(
            resolved.get("position_id")
            or resolved.get("active_position_id")
            or resolved.get("contract_id")
            or ""
        )
        if not position_id:
            logger.info(
                "position_work_skipped_missing_identity",
                extra={"event_type": event.event_type.value},
            )
            return
        snapshot_id = str(
            resolved.get("snapshot_id")
            or resolved.get("market_snapshot_id")
            or ""
        )
        if not snapshot_id:
            logger.info(
                "position_work_skipped_missing_snapshot",
                extra={"position_id": position_id},
            )
            return
        # Position reassessment always gets its own cycle — never reuse entry.
        from joker.persistence.cognitive_cycle_registry import stable_position_cycle_id

        cycle_id = stable_position_cycle_id(
            self._session_id,
            str(event.event_id),
        )
        parent_entry_cycle_id = str(
            resolved.get("parent_entry_cycle_id")
            or resolved.get("cycle_id")
            or ""
        ) or None
        from joker.graph.langgraph_checkpointer import cognitive_thread_id
        from joker.persistence.cognitive_cycle_registry import CognitiveCycleRecord

        thread_id = cognitive_thread_id(
            session_id=self._session_id, graph_kind="position", cycle_id=cycle_id
        )
        if self._deps.cycle_registry is not None:
            await self._deps.cycle_registry.upsert(
                CognitiveCycleRecord(
                    session_id=self._session_id,
                    graph_kind="position",
                    cycle_id=cycle_id,
                    trigger_event_id=str(event.event_id),
                    snapshot_id=snapshot_id,
                    status="running",
                    checkpoint_thread_id=thread_id,
                    parent_entry_cycle_id=parent_entry_cycle_id,
                    original_strategy_id=str(
                        resolved.get("original_strategy_id")
                        or resolved.get("strategy_id")
                        or ""
                    )
                    or None,
                    original_proposal_id=str(resolved.get("proposal_id") or "") or None,
                )
            )
        self._counters.active_position_cycles += 1
        state: dict[str, Any] = {
            "session_id": self._session_id,
            "run_id": self._run_id,
            "cycle_id": cycle_id,
            "snapshot_id": snapshot_id,
            "_position_id": position_id,
            "_contract_id": resolved.get("contract_id") or position_id,
            "_original_strategy_id": resolved.get("original_strategy_id")
            or resolved.get("strategy_id"),
            "_parent_entry_cycle_id": parent_entry_cycle_id,
            "_original_proposal_id": resolved.get("proposal_id"),
            # Clear per-cycle transients explicitly.
            "_position_command_id": None,
            "_position_decision": None,
            "_position_thesis": None,
            "_position_critic_notes": None,
        }
        config = ainvoke_config(
            session_id=self._session_id,
            graph_kind="position",
            cycle_id=cycle_id,
        )
        applied = None
        if self._evolution_runtime is not None:
            contract = str(resolved.get("contract_id") or position_id)
            origin = self._evolution_runtime.originating_configuration_for_contract(
                contract
            )
            if origin is None and parent_entry_cycle_id:
                origin = self._evolution_runtime.get_pinned(parent_entry_cycle_id)
            if origin is not None:
                applied = await self._evolution_runtime.apply_configuration_version(
                    cycle_id, origin
                )
            else:
                applied = await self._evolution_runtime.pin_and_apply_for_cycle(cycle_id)

        from joker.cognition.prompt_overrides import pinned_applied_configuration

        async def _invoke_position():
            if applied is not None:
                with pinned_applied_configuration(applied):
                    return await self._position_graph.ainvoke(state, config=config)
            return await self._position_graph.ainvoke(state, config=config)

        task = asyncio.create_task(_invoke_position())
        self._active_position_tasks.add(task)
        try:
            result_state = await task
            self._counters.last_success_at = datetime.now(timezone.utc)
            if self._deps.cycle_registry is not None:
                terminal_ok = self._cycle_reached_terminal_outcome(
                    result_state, graph_kind="position"
                )
                await self._deps.cycle_registry.upsert(
                    CognitiveCycleRecord(
                        session_id=self._session_id,
                        graph_kind="position",
                        cycle_id=cycle_id,
                        trigger_event_id=str(event.event_id),
                        snapshot_id=snapshot_id,
                        status="completed" if terminal_ok else "running",
                        checkpoint_thread_id=thread_id,
                        parent_entry_cycle_id=parent_entry_cycle_id,
                        original_strategy_id=str(
                            resolved.get("original_strategy_id")
                            or resolved.get("strategy_id")
                            or ""
                        )
                        or None,
                        original_proposal_id=str(resolved.get("proposal_id") or "")
                        or None,
                        payload={
                            "configuration_version_id": (
                                str(applied.configuration_version_id)
                                if applied is not None
                                else None
                            )
                        },
                    )
                )
        finally:
            self._active_position_tasks.discard(task)
            self._counters.active_position_cycles = max(
                0, self._counters.active_position_cycles - 1
            )

    async def _run_order_work(self, event: DomainEvent) -> None:
        resolved = await self._resolve_provenance(event)
        client_order_id = str(resolved.get("client_order_id") or "")
        if not client_order_id:
            return
        await self._sync_objective_reservation(event, client_order_id=client_order_id)
        snapshot_id = str(
            resolved.get("snapshot_id")
            or resolved.get("market_snapshot_id")
            or ""
        )
        order_projection: dict[str, Any] = {
            "client_order_id": client_order_id,
            "event_type": event.event_type.value,
            **{k: v for k, v in resolved.items() if k != "client_order_id"},
        }
        if self._deps.execution_runtime is not None:
            try:
                sync = await self._deps.execution_runtime.poll_order_status(
                    client_order_id
                )
                if sync is not None:
                    contract = getattr(sync, "contract", None)
                    contract_payload = None
                    if contract is not None:
                        dump = getattr(contract, "model_dump", None)
                        contract_payload = (
                            dump(mode="json") if callable(dump) else str(contract)
                        )
                    order_projection.update(
                        {
                            "status": sync.status,
                            "quantity": sync.quantity,
                            "filled_quantity": getattr(sync, "filled_quantity", None),
                            "limit_price": getattr(sync, "limit_price", None),
                            "side": getattr(sync, "side", None),
                            "order_type": getattr(sync, "order_type", None)
                            or getattr(getattr(sync, "intent", None), "order_type", None),
                            "contract": contract_payload,
                            "_contract_obj": contract,
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
                objective_context = None
                if self._deps.objective_state_loader is not None:
                    try:
                        from joker.objectives.schemas import state_to_context

                        obj_state = await self._deps.objective_state_loader()
                        objective_context = state_to_context(obj_state).model_dump_for_hash()
                    except Exception:
                        objective_context = None
                context = await assemble_role_context(
                    self._deps,
                    agent_role=AgentRole.ORDER_MANAGER,
                    session_id=self._session_id,
                    cycle_id=str(uuid4()),
                    snapshot=snapshot,
                    data_quality=data_quality,
                    option_surface_slice=surface_slice,
                    order_projection={
                        k: v
                        for k, v in order_projection.items()
                        if not str(k).startswith("_")
                    },
                    objective_context=objective_context,
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
        applied = None
        if self._evolution_runtime is not None:
            contract = str(
                resolved.get("contract_id")
                or (order_projection or {}).get("contract_id")
                or ""
            )
            origin = None
            if contract:
                origin = self._evolution_runtime.originating_configuration_for_contract(
                    contract
                )
            parent_cycle = str(resolved.get("cycle_id") or "")
            if origin is None and parent_cycle:
                origin = self._evolution_runtime.get_pinned(parent_cycle)
            if origin is not None:
                applied = await self._evolution_runtime.apply_configuration_version(
                    f"om:{client_order_id}", origin
                )
            else:
                applied = await self._evolution_runtime.pin_and_apply_for_cycle(
                    f"om:{client_order_id}"
                )
        from joker.cognition.prompt_overrides import pinned_applied_configuration

        if applied is not None:
            with pinned_applied_configuration(applied):
                decision = await agent.manage(
                    context,
                    self._router,
                    client_order_id=client_order_id,
                    order_projection=order_projection,
                )
        else:
            decision = await agent.manage(
                context,
                self._router,
                client_order_id=client_order_id,
                order_projection=order_projection,
            )
        decision_key = f"{client_order_id}:{decision.action}:{decision.rationale_summary}"
        if decision_key in self._order_decision_ids:
            return
        self._order_decision_ids.add(decision_key)
        if self._deps.order_management_repo is not None:
            await self._deps.order_management_repo.append(decision)
        await self._apply_order_decision(
            decision,
            order_projection=order_projection,
            trigger_event_id=str(event.event_id),
        )

    async def _apply_order_decision(
        self,
        decision: OrderManagementDecision,
        *,
        order_projection: dict[str, Any] | None = None,
        trigger_event_id: str | None = None,
    ) -> None:
        runtime = self._deps.execution_runtime
        if runtime is None:
            return
        action = decision.action
        if action == "continue_waiting":
            return
        source_state = str(
            (order_projection or {}).get("status")
            or (order_projection or {}).get("event_type")
            or "unknown"
        )
        from joker.persistence.order_management_actions import (
            OrderManagementActionRecord,
            make_order_management_action_key,
        )

        action_key = make_order_management_action_key(
            source_order_id=decision.client_order_id,
            source_order_state=source_state,
            trigger_event_id=str(trigger_event_id or ""),
            decision_id=str(decision.decision_id),
            action=str(action),
        )
        om_repo = self._deps.order_management_action_repo
        if om_repo is not None and await om_repo.has_key(action_key):
            logger.info(
                "order_management_action_idempotent_skip",
                extra={"action_key": action_key, "action": action},
            )
            return

        if action in {"cancel", "abandon"}:
            try:
                await runtime.cancel_order(client_order_id=decision.client_order_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "order_cancel_failed",
                    extra={
                        "client_order_id": decision.client_order_id,
                        "error": str(exc),
                    },
                )
                return
            if om_repo is not None:
                await om_repo.record(
                    OrderManagementActionRecord(
                        action_key=action_key,
                        session_id=self._session_id,
                        source_order_id=decision.client_order_id,
                        action=str(action),
                        source_order_state=source_state,
                        trigger_event_id=str(trigger_event_id or ""),
                        decision_id=str(decision.decision_id),
                    )
                )
            return
        if action in {"replace", "reduce_quantity"}:
            if self._deps.order_action_gateway is None:
                logger.warning("order_replace_skipped_no_gateway")
                return
            from joker.runtime.order_action_gateway import (
                OrderActionKind,
                OrderActionRequest,
            )

            projection = order_projection or {}
            contract = projection.get("_contract_obj") or projection.get("contract")
            if isinstance(contract, dict):
                from joker.schemas.domain import OptionContract

                contract = OptionContract.model_validate(contract)
            if contract is None:
                logger.warning(
                    "order_replace_skipped_missing_contract",
                    extra={"client_order_id": decision.client_order_id},
                )
                return
            from joker.runtime.execution_runtime import contract_id_for

            side = str(projection.get("side") or "buy")
            qty = int(
                decision.new_quantity
                if decision.new_quantity is not None
                else projection.get("quantity")
                or 1
            )
            if action == "reduce_quantity":
                open_qty = int(projection.get("quantity") or qty)
                filled = int(projection.get("filled_quantity") or 0)
                remaining = max(0, open_qty - filled)
                if decision.new_quantity is not None:
                    qty = min(int(decision.new_quantity), remaining)
                else:
                    qty = max(1, remaining // 2) if remaining > 1 else remaining
                if qty <= 0:
                    return
            limit = (
                float(decision.new_limit_price)
                if decision.new_limit_price is not None
                else (
                    float(projection["limit_price"])
                    if projection.get("limit_price") is not None
                    else None
                )
            )
            new_client_id = f"{decision.client_order_id}:replace:{decision.decision_id}"
            result = await self._deps.order_action_gateway.submit(
                OrderActionRequest(
                    action=OrderActionKind.REPLACE,
                    snapshot_id=str(decision.snapshot_id),
                    contract_id=contract_id_for(contract),
                    side=side,  # type: ignore[arg-type]
                    quantity=qty,
                    client_order_id=new_client_id,
                    limit_price=limit,
                    order_type="limit" if limit is not None else "market",
                    decision_id=str(decision.decision_id),
                    cycle_id=str(decision.cycle_id),
                    replace_of_client_order_id=decision.client_order_id,
                )
            )
            if result.submitted and om_repo is not None:
                await om_repo.record(
                    OrderManagementActionRecord(
                        action_key=action_key,
                        session_id=self._session_id,
                        source_order_id=decision.client_order_id,
                        action=str(action),
                        source_order_state=source_state,
                        trigger_event_id=str(trigger_event_id or ""),
                        decision_id=str(decision.decision_id),
                        replacement_client_order_id=new_client_id,
                    )
                )
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
