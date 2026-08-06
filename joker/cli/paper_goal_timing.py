"""Resolve paper goal-test objective duration vs absolute deadline.

Pure helpers used by ``joker paper run`` — no broker I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from joker.objectives.deadline import (
    DeadlineParseError,
    deadline_from_duration_minutes,
    resolve_deadline,
    time_remaining_seconds,
)
from joker.time.calendar import MarketCalendar
from joker.time.clock import SessionPhase, SystemExchangeClock

DEFAULT_OBJECTIVE_DURATION_MINUTES = 60.0
DEFAULT_RECONCILIATION_RECOVERY_MINUTES = 30.0
# Allow exits to finish after the objective window without extending the goal.
DEFAULT_SHUTDOWN_GRACE_SECONDS = 120.0


class PaperGoalTimingError(ValueError):
    """Fail-closed objective / runtime duration configuration error."""


@dataclass(frozen=True)
class PaperGoalTiming:
    """Resolved exchange-aware objective and runtime windows."""

    exchange_now: datetime
    objective_deadline: datetime
    objective_duration_minutes: float
    runtime_duration_minutes: float
    session_close: datetime
    remaining_session_seconds: int
    objective_source: str  # default | duration_flag | absolute_deadline
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS

    @property
    def runtime_seconds(self) -> float:
        return float(self.runtime_duration_minutes) * 60.0


def resolve_paper_goal_timing(
    *,
    objective_duration_minutes: float | None,
    target_deadline: str | None,
    duration_minutes: float | None,
    exchange_tz: str = "America/New_York",
    calendar: MarketCalendar | None = None,
    now: datetime | None = None,
    require_regular_session: bool = True,
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
) -> PaperGoalTiming:
    """Resolve objective deadline and aligned runtime duration.

    Rules:
    - ``--objective-duration-minutes`` and ``--target-deadline`` are mutually exclusive.
    - When neither is provided, objective duration defaults to 60 minutes.
    - When ``--duration-minutes`` is omitted, runtime equals objective duration.
    - When both are supplied, require ``duration_minutes >= objective_duration_minutes``.
    - Full objective window must fit before regular-session close (never silently shortened).
    """
    if objective_duration_minutes is not None and target_deadline:
        raise PaperGoalTimingError(
            "--objective-duration-minutes and --target-deadline are mutually exclusive; "
            "provide exactly one"
        )

    cal = calendar or MarketCalendar()
    clock = SystemExchangeClock(calendar=cal)
    exchange_now = now or clock.now()
    if exchange_now.tzinfo is None:
        raise PaperGoalTimingError("exchange now must be timezone-aware")

    phase = clock.session_phase(exchange_now)
    if require_regular_session and phase is not SessionPhase.REGULAR:
        raise PaperGoalTimingError(
            f"paper goal test requires regular market session "
            f"(phase={phase.value}, exchange_now={exchange_now.isoformat()})"
        )

    trading_date = cal.current_or_next_session(exchange_now)
    session_close = cal.session_close(trading_date)
    if session_close.tzinfo is None:
        raise PaperGoalTimingError("session close must be timezone-aware")

    if target_deadline:
        try:
            objective_deadline = resolve_deadline(
                target_deadline,
                exchange_tz=exchange_tz,
                trading_date=trading_date,
                now=exchange_now,
            )
        except DeadlineParseError as exc:
            raise PaperGoalTimingError(str(exc)) from exc
        objective_duration_minutes_resolved = max(
            0.0,
            (objective_deadline - exchange_now).total_seconds() / 60.0,
        )
        objective_source = "absolute_deadline"
    else:
        duration_flag = (
            DEFAULT_OBJECTIVE_DURATION_MINUTES
            if objective_duration_minutes is None
            else float(objective_duration_minutes)
        )
        if duration_flag <= 0:
            raise PaperGoalTimingError(
                f"--objective-duration-minutes must be > 0 (got {duration_flag})"
            )
        try:
            objective_deadline = deadline_from_duration_minutes(
                duration_flag,
                exchange_tz=exchange_tz,
                now=exchange_now,
            )
        except DeadlineParseError as exc:
            raise PaperGoalTimingError(str(exc)) from exc
        objective_duration_minutes_resolved = float(duration_flag)
        objective_source = (
            "default"
            if objective_duration_minutes is None
            else "duration_flag"
        )

    if objective_deadline > session_close:
        max_valid_duration_minutes = max(
            0.0, (session_close - exchange_now).total_seconds() / 60.0
        )
        raise PaperGoalTimingError(
            "objective window does not fit before regular-session close; "
            f"current exchange time={exchange_now.isoformat()} "
            f"requested deadline={objective_deadline.isoformat()} "
            f"regular-session close={session_close.isoformat()} "
            f"maximum valid duration={max_valid_duration_minutes:.2f} minutes. "
            "Do not silently shorten the objective — choose a shorter duration "
            "or an earlier --target-deadline."
        )

    if duration_minutes is None:
        runtime_duration_minutes = objective_duration_minutes_resolved
    else:
        runtime_duration_minutes = float(duration_minutes)
        if runtime_duration_minutes <= 0:
            raise PaperGoalTimingError(
                f"--duration-minutes must be > 0 (got {runtime_duration_minutes})"
            )
        # Allow a tiny float epsilon so 60.0 vs 59.999... does not fail closed.
        if runtime_duration_minutes + 1e-9 < objective_duration_minutes_resolved:
            raise PaperGoalTimingError(
                "runtime --duration-minutes must be >= objective duration so the "
                "process does not end before the objective deadline "
                f"(duration_minutes={runtime_duration_minutes}, "
                f"objective_duration_minutes={objective_duration_minutes_resolved})"
            )

    remaining_session = max(
        0, int((session_close - exchange_now).total_seconds())
    )

    return PaperGoalTiming(
        exchange_now=exchange_now,
        objective_deadline=objective_deadline,
        objective_duration_minutes=objective_duration_minutes_resolved,
        runtime_duration_minutes=runtime_duration_minutes,
        session_close=session_close,
        remaining_session_seconds=remaining_session,
        objective_source=objective_source,
        shutdown_grace_seconds=float(shutdown_grace_seconds),
    )


def format_timing_banner(timing: PaperGoalTiming) -> dict[str, Any]:
    """Redacted operator-facing timing fields for console + evidence."""
    return {
        "exchange_now": timing.exchange_now.isoformat(),
        "objective_deadline": timing.objective_deadline.isoformat(),
        "objective_duration_minutes": timing.objective_duration_minutes,
        "runtime_duration_minutes": timing.runtime_duration_minutes,
        "session_close": timing.session_close.isoformat(),
        "remaining_market_session_seconds": timing.remaining_session_seconds,
        "objective_time_remaining_seconds": time_remaining_seconds(
            timing.objective_deadline, now=timing.exchange_now
        ),
        "objective_source": timing.objective_source,
        "shutdown_grace_seconds": timing.shutdown_grace_seconds,
    }


def resolve_reconciliation_only_timing(
    *,
    original_deadline: datetime,
    duration_minutes: float | None,
    calendar: MarketCalendar | None = None,
    now: datetime | None = None,
    require_regular_session: bool = True,
    default_runtime_minutes: float = DEFAULT_RECONCILIATION_RECOVERY_MINUTES,
    shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
) -> PaperGoalTiming:
    """Build a bounded recovery window without inventing a new entry deadline."""
    cal = calendar or MarketCalendar()
    clock = SystemExchangeClock(calendar=cal)
    exchange_now = now or clock.now()
    if exchange_now.tzinfo is None or original_deadline.tzinfo is None:
        raise PaperGoalTimingError("recovery timing requires timezone-aware datetimes")
    phase = clock.session_phase(exchange_now)
    if require_regular_session and phase is not SessionPhase.REGULAR:
        raise PaperGoalTimingError(
            f"paper goal test requires regular market session "
            f"(phase={phase.value}, exchange_now={exchange_now.isoformat()})"
        )
    trading_date = cal.current_or_next_session(exchange_now)
    session_close = cal.session_close(trading_date)
    remaining_session_minutes = max(
        0.0, (session_close - exchange_now).total_seconds() / 60.0
    )
    runtime_duration_minutes = (
        float(duration_minutes)
        if duration_minutes is not None
        else min(float(default_runtime_minutes), remaining_session_minutes)
    )
    if runtime_duration_minutes <= 0:
        raise PaperGoalTimingError("reconciliation-only recovery runtime must be > 0")
    if runtime_duration_minutes - remaining_session_minutes > 1e-9:
        raise PaperGoalTimingError(
            "reconciliation-only recovery runtime must fit before regular-session close "
            f"(runtime_minutes={runtime_duration_minutes}, "
            f"remaining_session_minutes={remaining_session_minutes:.2f})"
        )
    return PaperGoalTiming(
        exchange_now=exchange_now,
        objective_deadline=original_deadline,
        objective_duration_minutes=0.0,
        runtime_duration_minutes=runtime_duration_minutes,
        session_close=session_close,
        remaining_session_seconds=max(0, int((session_close - exchange_now).total_seconds())),
        objective_source="reconciliation_only",
        shutdown_grace_seconds=float(shutdown_grace_seconds),
    )
