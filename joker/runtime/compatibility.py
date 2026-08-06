"""Compatibility adapters for Task 1 runtime separation.

``live_paper_runner`` remains the CLI façade but must *delegate* market and
execution ownership to ``SessionSupervisor`` / ``MarketRuntime`` /
``ExecutionRuntime`` rather than only emit deprecation comments.
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from joker.broker.interface import BrokerClient
from joker.events.schemas import DomainEvent
from joker.market.observations import UnderlyingObservation
from joker.market.option_surface import compute_mid
from joker.market.snapshots import MarketSnapshot as Task1MarketSnapshot
from joker.market.snapshots import UnderlyingSnapshot
from joker.runtime.execution_runtime import ExecutionCommand, ExecutionRuntime
from joker.runtime.market_runtime import MarketRuntime
from joker.schemas.domain import (
    BrokerOrder,
    MarketSnapshot as LegacyMarketSnapshot,
    OrderIntent,
    Position,
)
from joker.time.clock import ExchangeClock

if TYPE_CHECKING:
    from joker.runtime.agent_runtime import AgentRuntime
    from joker.runtime.session_supervisor import SessionSupervisor


@dataclass
class Task1RuntimeHealth:
    """Tracks whether the Task 1 market truth layer is healthy enough to trade."""

    consecutive_failures: int = 0
    degraded: bool = False
    last_error: str | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    failure_threshold: int = 1
    _history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def allows_execution(self) -> bool:
        return not self.degraded

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.degraded = False
        self.last_error = None
        self.last_success_at = datetime.now(timezone.utc)

    def record_failure(self, reason: str) -> None:
        self.consecutive_failures += 1
        self.last_error = reason
        self.last_failure_at = datetime.now(timezone.utc)
        self._history.append(
            {
                "at": self.last_failure_at.isoformat(),
                "reason": reason,
                "consecutive_failures": self.consecutive_failures,
            }
        )
        if self.consecutive_failures >= self.failure_threshold:
            self.degraded = True


class CompatibilityLivePaperBridge:
    """Active bridge: owns a SessionSupervisor and delegates market/execution.

    The CLI façade (``live_paper_runner``) constructs this bridge and routes
    observations / order requests through it.
    """

    CLI_FACADE = "joker.runtime.live_paper_runner"

    def __init__(
        self,
        *,
        broker: BrokerClient,
        db_path: Path,
        session_id: str | None = None,
        run_id: str | None = None,
        clock: ExchangeClock | None = None,
        broker_account_id: str = "paper",
        broker_account_identity: str | None = None,
        health_failure_threshold: int = 1,
        agent_runtime: AgentRuntime | None = None,
        market_config: Any | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid4())
        self._loop = asyncio.new_event_loop()
        from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig

        supervisor_kwargs: dict[str, Any] = {
            "db_path": db_path,
            "session_id": self.session_id,
            "run_id": run_id,
            "broker_account_id": broker_account_id,
            "broker_account_identity": broker_account_identity,
        }
        if market_config is not None:
            supervisor_kwargs["market"] = market_config
        self._supervisor = SessionSupervisor(
            broker=broker,
            clock=clock,
            config=SessionSupervisorConfig(**supervisor_kwargs),
            agent_runtime=agent_runtime,
        )
        self._started = False
        self.health = Task1RuntimeHealth(failure_threshold=health_failure_threshold)

    @property
    def supervisor(self) -> SessionSupervisor:
        return self._supervisor

    @property
    def market_runtime(self) -> MarketRuntime | None:
        return self._supervisor.market_runtime

    @property
    def execution_runtime(self) -> ExecutionRuntime | None:
        return self._supervisor.execution_runtime

    def run_coro(self, coro: Any) -> Any:
        """Run an async coroutine on the bridge event loop."""
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def start(self, *, start_agent: bool = True) -> None:
        """Start SessionSupervisor (migrations, restore, reconciliation).

        Pass ``start_agent=False`` for two-phase cognitive startup: Task 1
        ExecutionRuntime is created first, cognitive deps/gateway are bound,
        then ``start_agent()`` resumes unfinished cycles.
        """
        self.run_coro(self._supervisor.start(start_agent=start_agent))
        self._started = True

    async def astart(self, *, start_agent: bool = True) -> None:
        """Async start for callers already inside a running event loop."""
        await self._supervisor.start(start_agent=start_agent)
        self._started = True

    def start_agent(self) -> None:
        """Start the injected agent runtime after cognitive binding."""
        if not self._started:
            raise RuntimeError("CompatibilityLivePaperBridge.start() required first")
        self.run_coro(self._supervisor.start_agent_runtime())

    async def astart_agent(self) -> None:
        """Async agent start for callers already inside a running event loop."""
        if not self._started:
            raise RuntimeError("CompatibilityLivePaperBridge.start()/astart() required first")
        await self._supervisor.start_agent_runtime()

    def shutdown(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            if not self._loop.is_closed():
                self.run_coro(self._supervisor.shutdown())
                # Supervisor already drains aiosqlite workers; re-drain in case a
                # cancelled cognitive task opened a short-lived connection.
                from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers

                self.run_coro(drain_aiosqlite_workers())
        finally:
            if not self._loop.is_closed():
                try:
                    pending = asyncio.all_tasks(self._loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        self._loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                from joker.persistence.aiosqlite_lifecycle import join_aiosqlite_workers

                join_aiosqlite_workers()
                self._loop.close()

    def _sync_health_to_graph(self) -> None:
        """Persist truth-layer degradation into session/graph state for audit."""
        errors = list(self._supervisor.graph_state.get("errors") or [])
        errors = [e for e in errors if e.get("code") != "task1_truth_degraded"]
        if self.health.degraded:
            errors.append(
                {
                    "code": "task1_truth_degraded",
                    "message": self.health.last_error or "Task 1 market runtime unhealthy",
                    "consecutive_failures": self.health.consecutive_failures,
                    "at": (
                        self.health.last_failure_at.isoformat()
                        if self.health.last_failure_at
                        else None
                    ),
                }
            )
        self._supervisor.graph_state["errors"] = errors

    def _ingest_ok(self, result: Any) -> Any:
        """Record recovery only after a complete persisted market snapshot."""
        was_degraded = self.health.degraded
        self.health.record_success()
        if was_degraded:
            self._sync_health_to_graph()
        return result

    def _ingest_fail(self, exc: Exception) -> None:
        self.health.record_failure(str(exc))
        self._sync_health_to_graph()

    def ingest_underlying_quote(self, **kwargs: Any) -> Any:
        """Ingest quote observations. Success alone does not restore health."""
        market = self._require_market()
        try:
            return self.run_coro(market.ingest_underlying_quote(**kwargs))
        except Exception as exc:
            self._ingest_fail(exc)
            raise

    def ingest_trade(self, **kwargs: Any) -> Any:
        """Ingest trade prints. Success alone does not restore health."""
        market = self._require_market()
        try:
            return self.run_coro(market.ingest_trade(**kwargs))
        except Exception as exc:
            self._ingest_fail(exc)
            raise

    def ingest_option_quotes(self, quotes: Any) -> Any:
        """Ingest option quotes. Success alone does not restore health."""
        market = self._require_market()
        try:
            return self.run_coro(market.ingest_option_quotes(quotes))
        except Exception as exc:
            self._ingest_fail(exc)
            raise

    def tick(self, now: datetime | None = None) -> Any:
        """Advance MarketRuntime; degrade on failure, recover only on persisted snapshot."""
        market = self._require_market()
        try:
            result = self.run_coro(market.tick(now=now))
        except Exception as exc:
            self._ingest_fail(exc)
            raise
        if getattr(result, "snapshot", None) is not None:
            return self._ingest_ok(result)
        return result

    def submit_execution_command(self, command: ExecutionCommand) -> BrokerOrder:
        if not self.health.allows_execution:
            intent = command.intent
            return BrokerOrder(
                order_id=f"blocked-{command.client_order_id}",
                intent_id=command.client_order_id,
                status="rejected",
                contract=intent.contract,
                side=intent.side,
                quantity=intent.quantity,
                limit_price=intent.limit_price,
            )
        execution = self._require_execution()
        return self.run_coro(execution.submit_execution_command(command))

    def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        execution = self._require_execution()
        return self.run_coro(execution.cancel_order(client_order_id=client_order_id))

    def cancel_order_by_broker_id(self, broker_order_id: str) -> BrokerOrder:
        execution = self._require_execution()
        return self.run_coro(execution.cancel_order_by_broker_id(broker_order_id))

    def record_verified_fill(self, *args: Any, **kwargs: Any) -> Any:
        execution = self._require_execution()
        return self.run_coro(execution.record_verified_fill(*args, **kwargs))

    def poll_order_status(self, client_order_id: str) -> BrokerOrder | None:
        execution = self._require_execution()
        return self.run_coro(execution.poll_order_status(client_order_id))

    def project_session(self) -> Any:
        execution = self._require_execution()
        return self.run_coro(execution.project_session())

    def _require_market(self) -> MarketRuntime:
        if self._supervisor.market_runtime is None:
            raise RuntimeError("MarketRuntime not started; call bridge.start() first")
        return self._supervisor.market_runtime

    def _require_execution(self) -> ExecutionRuntime:
        if self._supervisor.execution_runtime is None:
            raise RuntimeError("ExecutionRuntime not started; call bridge.start() first")
        return self._supervisor.execution_runtime

    def warn_legacy_candle_ownership(self) -> None:
        warnings.warn(
            "Candle/bar construction ownership moved to MarketRuntime / BarBuilder. "
            f"{self.CLI_FACADE} must delegate via CompatibilityLivePaperBridge.",
            DeprecationWarning,
            stacklevel=2,
        )

    def warn_legacy_fill_accounting(self) -> None:
        warnings.warn(
            "Fill accounting ownership moved to ExecutionRuntime / ledger. "
            f"{self.CLI_FACADE} must delegate via CompatibilityLivePaperBridge.",
            DeprecationWarning,
            stacklevel=2,
        )


class ExecutionDelegatingBroker:
    """BrokerClient façade that routes submissions through ExecutionRuntime.

    Used so existing MarketEventHandler / ReactiveEngine call sites continue to
    call ``submit_order`` while ledger truth is owned by ExecutionRuntime.
    """

    def __init__(
        self,
        *,
        inner: BrokerClient,
        bridge: CompatibilityLivePaperBridge,
        broker_account_id: str = "paper",
    ) -> None:
        self._inner = inner
        self._bridge = bridge
        self._broker_account_id = broker_account_id

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        if not self._bridge.health.allows_execution:
            return BrokerOrder(
                order_id=f"blocked-{intent.intent_id}",
                intent_id=intent.intent_id,
                status="rejected",
                contract=intent.contract,
                side=intent.side,
                quantity=intent.quantity,
                limit_price=intent.limit_price,
            )
        command = ExecutionCommand(
            client_order_id=intent.intent_id,
            intent=intent,
            broker_account_id=self._broker_account_id,
        )
        return self._bridge.submit_execution_command(command)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel by broker order id via ExecutionRuntime (ledger + domain events)."""
        return self._bridge.cancel_order_by_broker_id(order_id)

    def get_order(self, order_id: str) -> BrokerOrder | None:
        return self._inner.get_order(order_id)

    def list_open_orders(self) -> list[BrokerOrder]:
        return self._inner.list_open_orders()

    def list_positions(self) -> list[Position]:
        return self._inner.list_positions()

    def get_account_balance(self) -> float:
        return self._inner.get_account_balance()

    def get_daily_pnl(self) -> float:
        return self._inner.get_daily_pnl()

    def get_daily_pnl_available(self) -> tuple[bool, float | None]:
        return self._inner.get_daily_pnl_available()

    def get_fill_price(self, order_id: str) -> float | None:
        getter = getattr(self._inner, "get_fill_price", None)
        if callable(getter):
            return getter(order_id)
        return None

    @property
    def inner(self) -> BrokerClient:
        return self._inner


class NullAgentRuntime:
    """Task 1 agent boundary: no-op / event passthrough only.

    Does not call an LLM, select direction, size risk, or submit orders.
    """

    def __init__(self) -> None:
        self.received_events: list[DomainEvent] = []

    async def start(self) -> None:
        """No-op start for Task 1 null runtime."""

    async def on_event(self, event: DomainEvent) -> None:
        """Accept an event without acting on it (passthrough / audit buffer)."""
        self.received_events.append(event)

    async def on_market_snapshot(self, snapshot_id: str) -> None:
        """No-op hook for snapshot notifications."""
        _ = snapshot_id

    async def tick(self) -> None:
        """No-op periodic agent tick."""
        return None

    async def shutdown(self) -> None:
        """No-op shutdown for Task 1 null runtime."""
        return None


def legacy_market_snapshot_to_underlying_observation(
    legacy: LegacyMarketSnapshot,
    *,
    source: str = "legacy_domain",
    received_timestamp: datetime | None = None,
) -> UnderlyingObservation:
    """Adapt ``joker.schemas.domain.MarketSnapshot`` to an UnderlyingObservation."""
    received = received_timestamp or legacy.timestamp
    return UnderlyingObservation(
        symbol=legacy.symbol,
        source_timestamp=legacy.timestamp,
        received_timestamp=received,
        bid=Decimal(str(legacy.bid)) if legacy.bid is not None else None,
        ask=Decimal(str(legacy.ask)) if legacy.ask is not None else None,
        last=Decimal(str(legacy.price)),
        source=source,
    )


def legacy_market_snapshot_to_underlying_snapshot(
    legacy: LegacyMarketSnapshot,
) -> UnderlyingSnapshot:
    """Adapt legacy domain MarketSnapshot to Task 1 UnderlyingSnapshot."""
    bid = Decimal(str(legacy.bid)) if legacy.bid is not None else None
    ask = Decimal(str(legacy.ask)) if legacy.ask is not None else None
    last = Decimal(str(legacy.price))
    return UnderlyingSnapshot(
        symbol=legacy.symbol,
        exchange_time=legacy.timestamp,
        last=last,
        bid=bid,
        ask=ask,
        mid=compute_mid(bid, ask),
        source="legacy_domain",
    )


def task1_snapshot_summary(snapshot: Task1MarketSnapshot) -> dict[str, Any]:
    """Small JSON-safe summary for compatibility logging (no OPRA dumps)."""
    return {
        "snapshot_id": str(snapshot.snapshot_id),
        "trading_date": snapshot.trading_date.isoformat(),
        "exchange_time": snapshot.exchange_time.isoformat(),
        "symbol": snapshot.underlying.symbol,
        "bars_1m": len(snapshot.bars_1m),
        "bars_5m": len(snapshot.bars_5m),
        "option_surface_id": (
            str(snapshot.option_surface_id) if snapshot.option_surface_id else None
        ),
        "data_quality_id": str(snapshot.data_quality_id),
    }
