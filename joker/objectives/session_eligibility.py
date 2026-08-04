"""Authoritative exchange-session eligibility for objective entry gates.

Similarity buckets (open/midday/close) are for historical comparison only.
Physical trading permission comes exclusively from ExchangeClock + MarketCalendar.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from joker.time.clock import ExchangeClock, SessionPhase

# Agent / historical similarity labels that may appear during regular RTH.
REGULAR_SIMILARITY_BUCKETS: frozenset[str] = frozenset(
    {
        "open",
        "midday",
        "close",
        "regular",
        "REGULAR",
    }
)

NON_REGULAR_SIMILARITY_BUCKETS: frozenset[str] = frozenset(
    {
        "premarket",
        "post_market",
        "postmarket",
        "closed",
        "holiday",
    }
)


class ObjectiveSessionEligibility(StrEnum):
    REGULAR = "regular"
    NOT_REGULAR = "not_regular"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ObjectiveSessionState:
    """Typed session truth for objective / target-attainment decisions."""

    exchange_phase: SessionPhase | None
    similarity_bucket: str | None
    eligibility: ObjectiveSessionEligibility
    entries_permitted: bool
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "exchange_phase": (
                self.exchange_phase.value if self.exchange_phase is not None else None
            ),
            "similarity_bucket": self.similarity_bucket,
            "eligibility": self.eligibility.value,
            "entries_permitted": self.entries_permitted,
            "reason_codes": list(self.reason_codes),
        }


def resolve_objective_session_state(
    *,
    clock: ExchangeClock | None,
    similarity_bucket: str | None = None,
) -> ObjectiveSessionState:
    """Resolve eligibility from the exchange clock; never from agent labels alone."""
    bucket = (similarity_bucket or "").strip() or None
    if clock is None:
        return ObjectiveSessionState(
            exchange_phase=None,
            similarity_bucket=bucket,
            eligibility=ObjectiveSessionEligibility.UNKNOWN,
            entries_permitted=False,
            reason_codes=("exchange_session_truth_unavailable",),
        )
    try:
        phase = clock.session_phase()
    except Exception:
        return ObjectiveSessionState(
            exchange_phase=None,
            similarity_bucket=bucket,
            eligibility=ObjectiveSessionEligibility.UNKNOWN,
            entries_permitted=False,
            reason_codes=("exchange_session_truth_unavailable",),
        )

    if phase is SessionPhase.REGULAR:
        return ObjectiveSessionState(
            exchange_phase=phase,
            similarity_bucket=bucket,
            eligibility=ObjectiveSessionEligibility.REGULAR,
            entries_permitted=True,
            reason_codes=(),
        )
    return ObjectiveSessionState(
        exchange_phase=phase,
        similarity_bucket=bucket,
        eligibility=ObjectiveSessionEligibility.NOT_REGULAR,
        entries_permitted=False,
        reason_codes=("market_not_regular", f"exchange_phase={phase.value}"),
    )


def similarity_bucket_for_history(exchange_phase: SessionPhase | None, bucket: str | None) -> str:
    """Historical comparison label — never used for physical entry permission."""
    if bucket and bucket.strip():
        return bucket.strip()
    if exchange_phase is SessionPhase.REGULAR:
        return "regular"
    if exchange_phase is None:
        return "unknown"
    return exchange_phase.value
