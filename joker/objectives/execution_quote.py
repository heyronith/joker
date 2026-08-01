"""Authoritative Task-1 quote loading for execution-time EV repricing."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from joker.market.option_surface import compute_relative_spread


class CurrentExecutionQuote(BaseModel):
    """Latest Task-1 option quote used for gateway EV revalidation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: UUID
    option_surface_id: UUID
    contract_id: str

    bid: Decimal
    ask: Decimal
    mid: Decimal
    relative_spread: Decimal
    quote_timestamp: datetime
    quote_age_seconds: int

    data_quality_id: UUID
    usable_for_execution: bool
    invalidation_reasons: tuple[str, ...] = ()


CurrentOptionQuoteLoader = Callable[
    [str],
    Awaitable[CurrentExecutionQuote | None],
]
CurrentDataQualityLoader = Callable[
    [UUID],
    Awaitable[Any | None],
]


def execution_premium_from_quote(quote: CurrentExecutionQuote) -> Decimal:
    """Documented long-option entry premium: ask (pay the offer)."""
    return quote.ask.quantize(Decimal("0.01"))


async def load_current_execution_quote(
    *,
    session_id: str,
    contract_id: str,
    snapshot_repo: Any,
    option_surface_repo: Any,
    data_quality_repo: Any | None,
    now: datetime | None = None,
    max_quote_age_seconds: int = 30,
    max_relative_spread: float = 0.25,
) -> CurrentExecutionQuote | None:
    """Reload the latest Task-1 surface quote for ``contract_id``."""
    if snapshot_repo is None or option_surface_repo is None:
        return None
    clock = now or datetime.now(timezone.utc)
    snapshot = await snapshot_repo.get_latest(session_id)
    if snapshot is None:
        return None
    surface_id = getattr(snapshot, "option_surface_id", None)
    if surface_id is None:
        return None
    surface = await option_surface_repo.get_by_id(surface_id)
    if surface is None:
        return None

    contract = None
    for row in getattr(surface, "contracts", ()) or ():
        if str(getattr(row, "contract_id", "")) == str(contract_id):
            contract = row
            break
    reasons: list[str] = []
    if contract is None:
        reasons.append("contract_absent_from_latest_surface")
        return CurrentExecutionQuote(
            snapshot_id=snapshot.snapshot_id,
            option_surface_id=UUID(str(surface_id)),
            contract_id=str(contract_id),
            bid=Decimal("0"),
            ask=Decimal("0"),
            mid=Decimal("0"),
            relative_spread=Decimal("1"),
            quote_timestamp=getattr(surface, "exchange_time", clock),
            quote_age_seconds=10**9,
            data_quality_id=UUID(str(snapshot.data_quality_id)),
            usable_for_execution=False,
            invalidation_reasons=tuple(reasons),
        )

    bid = Decimal(str(contract.bid)) if getattr(contract, "bid", None) is not None else None
    ask = Decimal(str(contract.ask)) if getattr(contract, "ask", None) is not None else None
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        reasons.append("invalid_or_crossed_market")
    mid = (
        ((bid + ask) / Decimal("2")).quantize(Decimal("0.01"))
        if bid is not None and ask is not None
        else Decimal("0")
    )
    rel = compute_relative_spread(bid, ask)
    if rel is None:
        reasons.append("relative_spread_unavailable")
        rel = Decimal("1")
    elif float(rel) > float(max_relative_spread):
        reasons.append("spread_unacceptable")

    q_ts = getattr(contract, "quote_timestamp", None) or getattr(
        surface, "exchange_time", clock
    )
    if q_ts.tzinfo is None:
        reasons.append("quote_timestamp_naive")
        age = 10**9
    else:
        age = max(0, int((clock - q_ts.astimezone(timezone.utc)).total_seconds()))
    if age > int(max_quote_age_seconds):
        reasons.append("quote_stale")

    usable = True
    dq_id = UUID(str(snapshot.data_quality_id))
    if data_quality_repo is not None:
        report = await data_quality_repo.get_by_id(dq_id)
        if report is None:
            reasons.append("data_quality_missing")
            usable = False
        else:
            usable = bool(getattr(report, "usable_for_execution", False))
            if not usable:
                reasons.append("data_unusable_for_execution")
    if reasons:
        usable = False

    return CurrentExecutionQuote(
        snapshot_id=snapshot.snapshot_id,
        option_surface_id=UUID(str(surface_id)),
        contract_id=str(contract_id),
        bid=bid or Decimal("0"),
        ask=ask or Decimal("0"),
        mid=mid,
        relative_spread=Decimal(str(rel)),
        quote_timestamp=q_ts if getattr(q_ts, "tzinfo", None) else clock,
        quote_age_seconds=age,
        data_quality_id=dq_id,
        usable_for_execution=usable and not reasons,
        invalidation_reasons=tuple(reasons),
    )


def build_current_option_quote_loader(
    deps: Any,
    *,
    max_quote_age_seconds: int = 30,
    max_relative_spread: float = 0.25,
) -> CurrentOptionQuoteLoader:
    """Bind deps into a public CurrentOptionQuoteLoader callback."""

    async def _load(contract_id: str) -> CurrentExecutionQuote | None:
        now = None
        clock = getattr(deps, "clock", None)
        if clock is not None and hasattr(clock, "now"):
            now = clock.now()
        return await load_current_execution_quote(
            session_id=str(deps.session_id),
            contract_id=contract_id,
            snapshot_repo=getattr(deps, "snapshot_repo", None),
            option_surface_repo=getattr(deps, "option_surface_repo", None),
            data_quality_repo=getattr(deps, "data_quality_repo", None),
            now=now,
            max_quote_age_seconds=max_quote_age_seconds,
            max_relative_spread=max_relative_spread,
        )

    return _load
