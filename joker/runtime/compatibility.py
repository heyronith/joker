"""Compatibility adapters for Task 1 runtime separation.

``live_paper_runner`` remains the CLI façade. New market/execution ownership
lives in MarketRuntime / ExecutionRuntime. Agent cognition stays out of Task 1
via ``NullAgentRuntime``.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from joker.events.schemas import DomainEvent
from joker.market.observations import UnderlyingObservation
from joker.market.snapshots import MarketSnapshot as Task1MarketSnapshot
from joker.market.snapshots import UnderlyingSnapshot
from joker.schemas.domain import MarketSnapshot as LegacyMarketSnapshot


class CompatibilityLivePaperBridge:
    """Documented bridge: live_paper_runner stays the CLI façade.

    Task 1 does not destructively rewrite ``live_paper_runner``. Callers should
    gradually route market ingestion through ``MarketRuntime`` and broker I/O
    through ``ExecutionRuntime``. This class records that boundary and emits
    deprecation warnings when legacy helpers are used for moved responsibilities.
    """

    #: CLI / entrypoint that remains the façade for paper sessions.
    CLI_FACADE = "joker.runtime.live_paper_runner"

    def __init__(self, *, session_id: str | None = None) -> None:
        self.session_id = session_id or str(uuid4())

    def warn_legacy_candle_ownership(self) -> None:
        """Warn that candle construction moved to MarketRuntime / BarBuilder."""
        warnings.warn(
            "Candle/bar construction ownership moved to "
            "joker.runtime.market_runtime.MarketRuntime and "
            "joker.market.bars.BarBuilder. "
            f"{self.CLI_FACADE} remains a CLI façade only.",
            DeprecationWarning,
            stacklevel=2,
        )

    def warn_legacy_fill_accounting(self) -> None:
        """Warn that fill accounting moved to ExecutionRuntime / ledger."""
        warnings.warn(
            "Fill accounting ownership moved to "
            "joker.runtime.execution_runtime.ExecutionRuntime and joker.ledger. "
            f"{self.CLI_FACADE} should not own fill truth going forward.",
            DeprecationWarning,
            stacklevel=2,
        )

    def warn_legacy_broker_direct(self) -> None:
        """Warn that broker calls should go through ExecutionRuntime."""
        warnings.warn(
            "Direct broker interaction from the live paper façade is deprecated "
            "for new code paths; route through ExecutionRuntime.submit_execution_command "
            "and poll_order_status / on_broker_update.",
            DeprecationWarning,
            stacklevel=2,
        )


class NullAgentRuntime:
    """Task 1 agent boundary: no-op / event passthrough only.

    Does not call an LLM, select direction, size risk, or submit orders.
    Downstream Task 2/3 agents replace this boundary.
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
    return UnderlyingSnapshot(
        symbol=legacy.symbol,
        bid=Decimal(str(legacy.bid)) if legacy.bid is not None else None,
        ask=Decimal(str(legacy.ask)) if legacy.ask is not None else None,
        last=Decimal(str(legacy.price)),
        quote_timestamp=legacy.timestamp,
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
