"""Deterministic paper exit management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from joker.schemas.replay import ExitDecision, ExitReason

_ET = ZoneInfo("America/New_York")


@dataclass
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
    # Trailing stop state (mutable while position is open)
    peak_premium: float | None = None
    trail_active: bool = False
    trail_stop_price: float | None = None


class ExitManager:
    """Evaluate exit conditions for open paper positions (America/New_York clock)."""

    def __init__(
        self,
        *,
        eod_time: time = time(15, 55),
        timezone_info=None,
        trail_activate_mfe_pct: float = 0.35,
        trail_giveback_pct: float = 0.20,
        late_day_tighten_minutes: float = 45.0,
        late_day_stop_floor_pct: float = 0.35,
    ) -> None:
        self.eod_time = eod_time
        self.tz = timezone_info or _ET
        self.trail_activate_mfe_pct = trail_activate_mfe_pct
        self.trail_giveback_pct = trail_giveback_pct
        self.late_day_tighten_minutes = late_day_tighten_minutes
        self.late_day_stop_floor_pct = late_day_stop_floor_pct

    def _as_tz(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(self.tz)

    def update_trailing(self, ctx: OpenTradeContext, current_premium: float) -> OpenTradeContext:
        """Update peak / trailing stop from latest mid. Returns possibly updated ctx."""
        peak = ctx.peak_premium if ctx.peak_premium is not None else ctx.entry_price
        peak = max(peak, current_premium)
        trail_active = ctx.trail_active
        trail_stop = ctx.trail_stop_price
        if ctx.entry_price > 0:
            mfe_pct = (peak - ctx.entry_price) / ctx.entry_price
            if mfe_pct >= self.trail_activate_mfe_pct:
                trail_active = True
                candidate = peak * (1.0 - self.trail_giveback_pct)
                # Never loosen stop; never trail below initial stop
                floor = max(ctx.stop_price, candidate)
                if trail_stop is None or floor > trail_stop:
                    trail_stop = floor
        return OpenTradeContext(
            position_id=ctx.position_id,
            entry_price=ctx.entry_price,
            stop_price=ctx.stop_price,
            take_profit_price=ctx.take_profit_price,
            entry_time=ctx.entry_time,
            time_stop_minutes=ctx.time_stop_minutes,
            invalidated=ctx.invalidated,
            quantity=ctx.quantity,
            reserved_notional_usd=ctx.reserved_notional_usd,
            peak_premium=peak,
            trail_active=trail_active,
            trail_stop_price=trail_stop,
        )

    def effective_stop(self, ctx: OpenTradeContext, current_time: datetime) -> float:
        """Initial stop, optionally raised by trail and late-day tighten."""
        stop = ctx.stop_price
        if ctx.trail_active and ctx.trail_stop_price is not None:
            stop = max(stop, ctx.trail_stop_price)
        et = self._as_tz(current_time)
        close_m = 16 * 60
        now_m = et.hour * 60 + et.minute
        minutes_to_close = float(close_m - now_m)
        if 0 <= minutes_to_close <= self.late_day_tighten_minutes and ctx.entry_price > 0:
            floor = ctx.entry_price * (1.0 - self.late_day_stop_floor_pct)
            stop = max(stop, floor)
        return stop

    def check_exit(
        self,
        ctx: OpenTradeContext,
        current_premium: float,
        current_time: datetime,
    ) -> ExitDecision | None:
        ts = self._as_tz(current_time)

        if ctx.invalidated:
            return ExitDecision(
                position_id=ctx.position_id,
                reason=ExitReason.INVALIDATION,
                exit_price=current_premium,
                message="Setup invalidation triggered",
            )

        eff_stop = self.effective_stop(ctx, current_time)
        if current_premium <= eff_stop:
            reason = ExitReason.STOP_LOSS
            msg = f"Premium stop at {eff_stop}"
            if ctx.trail_active and ctx.trail_stop_price is not None and eff_stop >= ctx.trail_stop_price - 1e-9:
                msg = f"Trailing stop at {eff_stop}"
            return ExitDecision(
                position_id=ctx.position_id,
                reason=reason,
                exit_price=current_premium,
                message=msg,
            )

        if current_premium >= ctx.take_profit_price:
            return ExitDecision(
                position_id=ctx.position_id,
                reason=ExitReason.TAKE_PROFIT,
                exit_price=current_premium,
                message=f"Take profit at {ctx.take_profit_price}",
            )

        if ctx.time_stop_minutes is not None:
            entry = ctx.entry_time if ctx.entry_time.tzinfo else ctx.entry_time.replace(tzinfo=timezone.utc)
            entry_et = entry.astimezone(self.tz)
            elapsed_min = (ts - entry_et).total_seconds() / 60.0
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
                message="End-of-day forced exit (America/New_York)",
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
