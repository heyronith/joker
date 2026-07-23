"""Deterministic paper exit management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone

from joker.schemas.domain import Position
from joker.schemas.replay import ExitDecision, ExitReason


@dataclass(frozen=True)
class OpenTradeContext:
    position_id: str
    entry_price: float
    stop_price: float
    take_profit_price: float
    entry_time: datetime
    time_stop_minutes: int | None = None
    invalidated: bool = False
    quantity: int = 1
    reserved_notional_usd: float = 0.0


class ExitManager:
    """Evaluate exit conditions for open paper positions."""

    def __init__(
        self,
        *,
        eod_time: time = time(15, 55),
        timezone_info=timezone.utc,
    ) -> None:
        self.eod_time = eod_time
        self.tz = timezone_info

    def check_exit(
        self,
        ctx: OpenTradeContext,
        current_premium: float,
        current_time: datetime,
    ) -> ExitDecision | None:
        ts = current_time if current_time.tzinfo else current_time.replace(tzinfo=self.tz)

        if ctx.invalidated:
            return ExitDecision(
                position_id=ctx.position_id,
                reason=ExitReason.INVALIDATION,
                exit_price=current_premium,
                message="Setup invalidation triggered",
            )

        if current_premium <= ctx.stop_price:
            return ExitDecision(
                position_id=ctx.position_id,
                reason=ExitReason.STOP_LOSS,
                exit_price=current_premium,
                message=f"Premium stop at {ctx.stop_price}",
            )

        if current_premium >= ctx.take_profit_price:
            return ExitDecision(
                position_id=ctx.position_id,
                reason=ExitReason.TAKE_PROFIT,
                exit_price=current_premium,
                message=f"Take profit at {ctx.take_profit_price}",
            )

        if ctx.time_stop_minutes is not None:
            entry = ctx.entry_time if ctx.entry_time.tzinfo else ctx.entry_time.replace(tzinfo=self.tz)
            elapsed_min = (ts - entry).total_seconds() / 60.0
            if elapsed_min >= ctx.time_stop_minutes:
                return ExitDecision(
                    position_id=ctx.position_id,
                    reason=ExitReason.TIME_STOP,
                    exit_price=current_premium,
                    message=f"Time stop after {ctx.time_stop_minutes} minutes",
                )

        if ts.time() >= self.eod_time:
            return ExitDecision(
                position_id=ctx.position_id,
                reason=ExitReason.END_OF_DAY,
                exit_price=current_premium,
                message="End-of-day forced exit",
            )

        return None

    @staticmethod
    def stop_from_entry(entry_price: float, stop_pct: float = 0.5) -> float:
        """Premium stop-loss: e.g. 50% of entry premium."""
        return max(0.01, entry_price * (1.0 - stop_pct))

    @staticmethod
    def target_from_entry(entry_price: float, target_pct: float = 1.0) -> float:
        """Take-profit target: e.g. 100% gain on premium."""
        return entry_price * (1.0 + target_pct)
