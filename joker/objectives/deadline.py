"""Parse and validate session objective deadlines in exchange time."""

from __future__ import annotations

import re
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

class DeadlineParseError(ValueError):
    """Ambiguous, past, or unparseable deadline."""


_SAME_DAY = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s*(ET|EDT|EST|America/New_York)?\s*$",
    re.IGNORECASE,
)


def resolve_deadline(
    raw: str | datetime,
    *,
    exchange_tz: str = "America/New_York",
    trading_date: date | None = None,
    now: datetime | None = None,
) -> datetime:
    """
    Resolve an absolute timezone-aware deadline.

    Accepts:
    - timezone-aware datetime
    - ISO-8601 string with offset/Z
    - same-day exchange clock like ``15:30 ET``
    """
    tz = ZoneInfo(exchange_tz)
    clock = now or datetime.now(tz=tz)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=tz)
    else:
        clock = clock.astimezone(tz)

    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            raise DeadlineParseError("deadline datetime must be timezone-aware")
        deadline = raw.astimezone(tz)
    else:
        text = str(raw).strip()
        if not text:
            raise DeadlineParseError("deadline is empty")
        m = _SAME_DAY.match(text)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2))
            if hour > 23 or minute > 59:
                raise DeadlineParseError(f"invalid clock time: {text!r}")
            day = trading_date or clock.date()
            deadline = datetime.combine(day, time(hour=hour, minute=minute), tzinfo=tz)
        else:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise DeadlineParseError(f"unparseable deadline: {text!r}") from exc
            if parsed.tzinfo is None:
                raise DeadlineParseError(
                    "ISO deadline must include a timezone offset (ambiguous otherwise)"
                )
            deadline = parsed.astimezone(tz)

    if deadline <= clock:
        raise DeadlineParseError(
            f"deadline already passed: {deadline.isoformat()} (now={clock.isoformat()})"
        )
    return deadline


def time_remaining_seconds(
    deadline: datetime,
    *,
    now: datetime | None = None,
    exchange_tz: str = "America/New_York",
) -> int:
    tz = ZoneInfo(exchange_tz)
    clock = now or datetime.now(tz=tz)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=tz)
    else:
        clock = clock.astimezone(tz)
    if deadline.tzinfo is None:
        raise DeadlineParseError("deadline must be timezone-aware")
    remaining = int((deadline.astimezone(tz) - clock).total_seconds())
    return max(0, remaining)
