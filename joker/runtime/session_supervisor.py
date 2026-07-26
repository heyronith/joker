"""Session supervisor — wire clock, repos, runtimes; restore and shut down.

Agent boundary is NullAgentRuntime (no-op / event passthrough only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from joker.broker.interface import BrokerClient
from joker.events.bus import InProcessAsyncEventBus
from joker.events.schemas import DomainEvent, EventType, make_event
from joker.graph.checkpoints import CheckpointRecord, SqliteCheckpointStore
from joker.graph.state import JokerGraphState
from joker.ledger.projector import LedgerProjector
from joker.ledger.reconciliation import BrokerReconciler, ReconciliationReport
from joker.ledger.store import SqliteLedgerStore
from joker.market.bars import BarBuilder
from joker.market.option_surface import OptionSurfaceBuilder, OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.persistence.migrations import apply_task1_migrations
from joker.runtime.agent_runtime import AgentRuntime
from joker.runtime.compatibility import NullAgentRuntime
from joker.runtime.execution_runtime import ExecutionRuntime, UnresolvedReconciliation
from joker.runtime.market_runtime import MarketRuntime, MarketRuntimeConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import ExchangeClock, SystemExchangeClock

logger = logging.getLogger(__name__)


@dataclass
class SessionSupervisorConfig:
    """Paths and ids for a Task 1 session (no strategy settings)."""

    db_path: Path
    checkpoint_db_path: Path | None = None
    session_id: str | None = None
    run_id: str | None = None
    broker_account_id: str = "default"
    late_observation_tolerance_seconds: float = 2.0
    event_handler_timeout_seconds: float = 10.0
    auto_apply_reconciliation_corrections: bool = True
    market: MarketRuntimeConfig = field(default_factory=MarketRuntimeConfig)


class SessionSupervisor:
    """Owns session lifecycle: restore → reconcile → run → shutdown."""

    def __init__(
        self,
        *,
        broker: BrokerClient,
        config: SessionSupervisorConfig,
        clock: ExchangeClock | None = None,
        calendar: MarketCalendar | None = None,
        event_bus: InProcessAsyncEventBus | None = None,
        agent_runtime: AgentRuntime | None = None,
    ) -> None:
        self._config = config
        self._calendar = calendar or MarketCalendar()
        self._clock: ExchangeClock = clock or SystemExchangeClock(self._calendar)
        self._broker = broker
        self._session_id = config.session_id or str(uuid4())
        self._run_id = config.run_id or str(uuid4())
        self._bus = event_bus or InProcessAsyncEventBus(
            handler_timeout_seconds=config.event_handler_timeout_seconds
        )
        self._agent = agent_runtime or NullAgentRuntime()
        self._checkpoint_path = config.checkpoint_db_path or (
            Path(config.db_path).with_name(
                Path(config.db_path).stem + "_checkpoints.db"
            )
        )
        self._checkpoints = SqliteCheckpointStore(self._checkpoint_path)
        self._ledger: SqliteLedgerStore | None = None
        self._snapshots: SnapshotRepository | None = None
        self._surfaces: OptionSurfaceRepository | None = None
        self._market: MarketRuntime | None = None
        self._execution: ExecutionRuntime | None = None
        self._started = False
        self._graph_state: JokerGraphState = {
            "run_id": self._run_id,
            "session_id": self._session_id,
            "pending_event_ids": [],
            "errors": [],
        }
        self._last_reconciliation: ReconciliationReport | None = None
        self._unresolved: UnresolvedReconciliation | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def clock(self) -> ExchangeClock:
        return self._clock

    @property
    def event_bus(self) -> InProcessAsyncEventBus:
        return self._bus

    @property
    def market_runtime(self) -> MarketRuntime | None:
        return self._market

    @property
    def execution_runtime(self) -> ExecutionRuntime | None:
        return self._execution

    @property
    def agent_runtime(self) -> AgentRuntime:
        return self._agent

    @property
    def graph_state(self) -> JokerGraphState:
        return self._graph_state

    @property
    def ledger_store(self) -> SqliteLedgerStore | None:
        return self._ledger

    @property
    def snapshot_repository(self) -> SnapshotRepository | None:
        return self._snapshots

    @property
    def option_surface_repository(self) -> OptionSurfaceRepository | None:
        return self._surfaces

    @property
    def unresolved_reconciliation(self) -> UnresolvedReconciliation | None:
        return self._unresolved

    @property
    def claims_recovery(self) -> bool:
        if self._execution is None:
            return False
        return self._execution.claims_recovery() and self._unresolved is None

    async def start(self) -> JokerGraphState:
        """Apply migrations, restore checkpoints, reconcile, start runtimes."""
        apply_task1_migrations(self._config.db_path)
        await self._checkpoints.initialize()

        self._ledger = SqliteLedgerStore(self._config.db_path)
        await self._ledger.initialize()
        self._snapshots = SnapshotRepository(self._config.db_path)
        await self._snapshots.initialize()
        self._surfaces = OptionSurfaceRepository(self._config.db_path)
        await self._surfaces.initialize()

        restored = await self._checkpoints.load_latest(self._session_id)
        if restored is not None:
            self._graph_state = dict(restored.state)  # type: ignore[assignment]
            self._graph_state["session_id"] = self._session_id
            self._graph_state["run_id"] = self._run_id
            logger.info(
                "checkpoint_restored",
                extra={
                    "session_id": self._session_id,
                    "checkpoint_id": restored.checkpoint_id,
                },
            )

        bar_builder = BarBuilder(
            self._clock,
            late_tolerance_seconds=int(self._config.late_observation_tolerance_seconds),
        )
        self._market = MarketRuntime(
            clock=self._clock,
            bar_builder=bar_builder,
            event_bus=self._bus,
            snapshot_repo=self._snapshots,
            surface_builder=OptionSurfaceBuilder(),
            surface_repo=self._surfaces,
            session_id=self._session_id,
            config=self._config.market,
        )
        self._execution = ExecutionRuntime(
            broker=self._broker,
            ledger_store=self._ledger,
            projector=LedgerProjector(),
            reconciler=BrokerReconciler(),
            event_bus=self._bus,
            clock=self._clock,
            session_id=self._session_id,
            broker_account_id=self._config.broker_account_id,
        )
        await self._execution.restore_order_mappings()

        # Agent boundary: forward domain events to injected runtime.
        for event_type in (
            EventType.MARKET_SNAPSHOT_CREATED,
            EventType.ORDER_FILLED,
            EventType.ORDER_PARTIALLY_FILLED,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_ACCEPTED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
            EventType.POSITION_OPENED,
            EventType.POSITION_CHANGED,
            EventType.POSITION_CLOSED,
        ):
            self._bus.subscribe(event_type, self._on_agent_event)

        await self._agent.start()

        now = self._clock.now()
        await self._bus.publish(
            make_event(
                EventType.SESSION_STARTED,
                session_id=self._session_id,
                source="session_supervisor",
                exchange_timestamp=now,
                payload={"run_id": self._run_id},
            )
        )

        report = await self._execution.run_reconciliation()
        if not report.is_consistent:
            if self._config.auto_apply_reconciliation_corrections:
                await self._execution.apply_reconciliation_corrections(report)
                report = await self._execution.run_reconciliation()
            if not report.is_consistent:
                self._unresolved = UnresolvedReconciliation(report=report)
                errors = list(self._graph_state.get("errors") or [])
                errors.append(
                    {
                        "type": "unresolved_reconciliation",
                        "report_id": str(report.report_id),
                        "finding_count": len(report.findings),
                    }
                )
                self._graph_state["errors"] = errors
                logger.error(
                    "startup_reconciliation_unresolved",
                    extra={
                        "session_id": self._session_id,
                        "report_id": str(report.report_id),
                    },
                )

        self._last_reconciliation = report
        self._graph_state["exchange_time"] = now
        await self._checkpoints.save(self._graph_state, self._session_id)
        self._started = True
        return self._graph_state

    async def _on_agent_event(self, event: DomainEvent) -> None:
        """Forward domain events to the injected agent runtime."""
        await self._agent.on_event(event)

    async def checkpoint(self) -> str:
        """Persist current graph state; return checkpoint id."""
        self._graph_state["exchange_time"] = self._clock.now()
        self._graph_state["session_id"] = self._session_id
        self._graph_state["run_id"] = self._run_id
        if self._market and self._market.latest_surface is not None:
            self._graph_state["option_surface_id"] = str(
                self._market.latest_surface.surface_id
            )
        return await self._checkpoints.save(self._graph_state, self._session_id)

    async def restore_latest(self) -> CheckpointRecord | None:
        """Reload latest checkpoint into supervisor graph state."""
        record = await self._checkpoints.load_latest(self._session_id)
        if record is not None:
            self._graph_state = dict(record.state)  # type: ignore[assignment]
        return record

    async def shutdown(self) -> ReconciliationReport | None:
        """Graceful shutdown: session ending → final reconciliation → close bus."""
        now = self._clock.now()
        await self._bus.publish(
            make_event(
                EventType.SESSION_ENDING,
                session_id=self._session_id,
                source="session_supervisor",
                exchange_timestamp=now,
                payload={"run_id": self._run_id},
            )
        )
        report: ReconciliationReport | None = None
        if self._execution is not None:
            report = await self._execution.run_reconciliation()
            self._last_reconciliation = report

        await self.checkpoint()
        await self._agent.shutdown()
        await self._bus.drain()

        await self._bus.publish(
            make_event(
                EventType.SESSION_ENDED,
                session_id=self._session_id,
                source="session_supervisor",
                exchange_timestamp=self._clock.now(),
                payload={
                    "run_id": self._run_id,
                    "reconciliation_consistent": (
                        None if report is None else report.is_consistent
                    ),
                },
            )
        )
        await self._bus.close()

        if self._ledger is not None:
            await self._ledger.close()
        await self._checkpoints.close()
        # Let aiosqlite worker threads observe closed connections before loop teardown.
        import asyncio

        await asyncio.sleep(0)
        self._started = False
        logger.info(
            "session_shutdown_complete",
            extra={
                "session_id": self._session_id,
                "run_id": self._run_id,
                "event_bus_idle": self._bus.is_idle,
                "active_workers": self._bus.active_worker_count,
            },
        )
        return report

    def update_graph_from_snapshot(
        self,
        *,
        market_snapshot_id: str | None = None,
        feature_snapshot_id: str | None = None,
        option_surface_id: str | None = None,
        data_quality_id: str | None = None,
        active_order_id: str | None = None,
        active_position_id: str | None = None,
    ) -> None:
        """Update graph state IDs after market/execution activity (no decisions)."""
        if market_snapshot_id is not None:
            self._graph_state["market_snapshot_id"] = market_snapshot_id
        if feature_snapshot_id is not None:
            self._graph_state["feature_snapshot_id"] = feature_snapshot_id
        if option_surface_id is not None:
            self._graph_state["option_surface_id"] = option_surface_id
        if data_quality_id is not None:
            self._graph_state["data_quality_id"] = data_quality_id
        if active_order_id is not None:
            self._graph_state["active_order_id"] = active_order_id
        if active_position_id is not None:
            self._graph_state["active_position_id"] = active_position_id
        self._graph_state["exchange_time"] = self._clock.now()
