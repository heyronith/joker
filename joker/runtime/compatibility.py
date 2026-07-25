"""Compatibility adapters for Task 1 runtime separation.

``live_paper_runner`` remains the CLI façade but must *delegate* market and
execution ownership to ``SessionSupervisor`` / ``MarketRuntime`` /
``ExecutionRuntime`` rather than only emit deprecation comments.
"""

from __future__ import annotations

import asyncio
import warnings
from datetime import datetime
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
    from joker.runtime.session_supervisor import SessionSupervisor


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
    ) -> None:
        self.session_id = session_id or str(uuid4())
        self._loop = asyncio.new_event_loop()
        from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig

        self._supervisor = SessionSupervisor(
            broker=broker,
            clock=clock,
            config=SessionSupervisorConfig(
                db_path=db_path,
                session_id=self.session_id,
                run_id=run_id,
                broker_account_id=broker_account_id,
            ),
        )
        self._started = False

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
        return self._loop.run_until_complete(coro)

    def start(self) -> None:
        """Start SessionSupervisor (migrations, restore, reconciliation)."""
        self.run_coro(self._supervisor.start())
        self._started = True

    def shutdown(self) -> None:
        if not self._started:
            return
        self.run_coro(self._supervisor.shutdown())
        self._started = False
        self._loop.close()

    def ingest_underlying_quote(self, **kwargs: Any) -> Any:
        market = self._require_market()
        return self.run_coro(market.ingest_underlying_quote(**kwargs))

    def ingest_trade(self, **kwargs: Any) -> Any:
        market = self._require_market()
        return self.run_coro(market.ingest_trade(**kwargs))

    def ingest_option_quotes(self, quotes: Any) -> Any:
        market = self._require_market()
        return self.run_coro(market.ingest_option_quotes(quotes))

    def tick(self, now: datetime | None = None) -> Any:
        market = self._require_market()
        return self.run_coro(market.tick(now=now))

    def submit_execution_command(self, command: ExecutionCommand) -> BrokerOrder:
        execution = self._require_execution()
        return self.run_coro(execution.submit_execution_command(command))

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
        command = ExecutionCommand(
            client_order_id=intent.intent_id,
            intent=intent,
            broker_account_id=self._broker_account_id,
        )
        return self._bridge.submit_execution_command(command)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        return self._inner.cancel_order(order_id)

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

    async def on_event(self, event: DomainEvent) -> None:
        """Accept an event without acting on it (passthrough / audit buffer)."""
        self.received_events.append(event)

    async def on_market_snapshot(self, snapshot_id: str) -> None:
        """No-op hook for snapshot notifications."""
        _ = snapshot_id

    async def tick(self) -> None:
        """No-op periodic agent tick."""
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
