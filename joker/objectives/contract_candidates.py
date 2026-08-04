"""Build bounded strategy × contract candidates from the linked option surface."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID

from joker.objectives.target_attainment import TargetAttainmentContractCandidate


def _d(value: Any, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    return Decimal(str(value))


def _option_type_for_direction(direction: str | None) -> str | None:
    d = (direction or "").lower()
    if d in {"bullish", "long", "call", "up"}:
        return "call"
    if d in {"bearish", "short", "put", "down"}:
        return "put"
    return None


def build_contract_candidates_for_strategies(
    *,
    strategies: list[Any],
    surface_slice: list[Any],
    trading_date: date | None,
    estimates_by_strategy: dict[str, dict[str, Any]] | None = None,
    max_contracts_per_strategy: int = 5,
    max_relative_spread: float = 0.35,
    max_quote_age_seconds: float | None = 30.0,
    now: Any | None = None,
) -> list[TargetAttainmentContractCandidate]:
    """Deterministic contract candidates from strategy legs ∩ linked surface."""
    by_id: dict[str, Any] = {}
    for contract in surface_slice or []:
        cid = str(getattr(contract, "contract_id", "") or "")
        if cid:
            by_id[cid] = contract

    out: list[TargetAttainmentContractCandidate] = []
    estimates_by_strategy = estimates_by_strategy or {}

    for strategy in strategies:
        sid = getattr(strategy, "strategy_id", None)
        if sid is None:
            continue
        sid_u = UUID(str(sid))
        direction = str(getattr(getattr(strategy, "direction", None), "value", None)
                        or getattr(strategy, "direction", "")
                        or "")
        wanted_type = _option_type_for_direction(direction)
        est = estimates_by_strategy.get(str(sid_u)) or {}
        horizon = int(
            est.get("estimated_resolution_seconds")
            or getattr(strategy, "expected_horizon_seconds", 600)
            or 600
        )
        legs = tuple(getattr(strategy, "candidate_legs", ()) or ())
        seen: set[str] = set()
        for leg in legs:
            if len(seen) >= max_contracts_per_strategy:
                break
            cid = str(getattr(leg, "contract_id", "") or "")
            if not cid or cid in seen:
                continue
            contract = by_id.get(cid)
            if contract is None:
                continue
            reason = _validate_contract(
                contract,
                trading_date=trading_date,
                wanted_option_type=wanted_type
                or str(getattr(leg, "option_type", "") or "").lower() or None,
                max_relative_spread=max_relative_spread,
                max_quote_age_seconds=max_quote_age_seconds,
                now=now,
            )
            if reason is not None:
                continue
            bid = _d(getattr(contract, "bid", None))
            ask = _d(getattr(contract, "ask", None))
            mid = ((bid + ask) / Decimal("2")).quantize(Decimal("0.0001"))
            spread = (
                ((ask - bid) / ask).quantize(Decimal("0.0001"))
                if ask > 0
                else Decimal("1")
            )
            # Affordability / worst-case premium uses current ask for long buys.
            premium = ask
            max_loss = (premium * Decimal("100")).quantize(Decimal("0.01"))
            out.append(
                TargetAttainmentContractCandidate(
                    strategy_id=sid_u,
                    contract_id=cid,
                    option_type=str(
                        getattr(contract, "option_type", None)
                        or getattr(leg, "option_type", "")
                        or ""
                    ).lower(),
                    strike=_d(getattr(contract, "strike", None) or getattr(leg, "strike", 0)),
                    premium_per_contract_usd=premium,
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    relative_spread=spread,
                    estimated_win_probability=(
                        _d(est["estimated_win_probability"])
                        if est.get("estimated_win_probability") is not None
                        else None
                    ),
                    expected_value_usd=(
                        _d(est["expected_value_usd"])
                        if est.get("expected_value_usd") is not None
                        else None
                    ),
                    estimated_payoff_ratio=(
                        _d(est["estimated_payoff_ratio"])
                        if est.get("estimated_payoff_ratio") is not None
                        else None
                    ),
                    estimated_useful_upside_usd=(
                        _d(est["useful_upside_usd"])
                        if est.get("useful_upside_usd") is not None
                        else None
                    ),
                    estimated_resolution_seconds=horizon,
                    maximum_loss_usd_per_contract=max_loss,
                    historical_sample_count=int(est.get("sample_count") or 0),
                    historical_hit_rate=(
                        _d(est["historical_hit_rate"])
                        if est.get("historical_hit_rate") is not None
                        else None
                    ),
                    direction=direction or None,
                    quote_timestamp=str(
                        getattr(contract, "quote_timestamp", None)
                        or getattr(contract, "source_timestamp", None)
                        or ""
                    )
                    or None,
                    calculation_method=str(est.get("calculation_method") or "surface_ask"),
                    assumptions=("execution_reference_premium=ask",),
                )
            )
            seen.add(cid)
    return out


def _validate_contract(
    contract: Any,
    *,
    trading_date: date | None,
    wanted_option_type: str | None,
    max_relative_spread: float,
    max_quote_age_seconds: float | None = 30.0,
    now: Any | None = None,
) -> str | None:
    symbol = str(getattr(contract, "symbol", "") or "").upper()
    if symbol and symbol != "SPY":
        return "wrong_underlying"
    if getattr(contract, "usable_for_execution", True) is False:
        return "contract_unusable_for_execution"
    expiry = getattr(contract, "expiry", None) or getattr(contract, "expiration", None)
    if trading_date is not None and expiry is not None:
        exp_date = expiry if isinstance(expiry, date) else None
        if exp_date is None:
            try:
                exp_date = date.fromisoformat(str(expiry)[:10])
            except Exception:
                return "invalid_expiration"
        if exp_date != trading_date:
            return "non_0dte_contract"
    bid = getattr(contract, "bid", None)
    ask = getattr(contract, "ask", None)
    if bid is None or ask is None:
        return "missing_bid_ask"
    try:
        bid_d = Decimal(str(bid))
        ask_d = Decimal(str(ask))
    except Exception:
        return "invalid_bid_ask"
    if bid_d <= 0 or ask_d <= 0 or ask_d < bid_d:
        return "invalid_bid_ask"
    if ask_d > 0 and float((ask_d - bid_d) / ask_d) > max_relative_spread:
        return "unusable_spread"
    opt = str(getattr(contract, "option_type", "") or "").lower()
    if wanted_option_type and opt and opt != wanted_option_type.lower():
        return "wrong_option_type_for_strategy"
    if max_quote_age_seconds is not None:
        age = getattr(contract, "quote_age_seconds", None)
        if age is None and now is not None:
            ts = getattr(contract, "quote_timestamp", None) or getattr(
                contract, "source_timestamp", None
            )
            if ts is not None:
                try:
                    age = (now - ts).total_seconds()
                except Exception:
                    age = None
        if age is not None and float(age) > float(max_quote_age_seconds):
            return "stale_contract"
        stale_flag = getattr(contract, "stale", None)
        if stale_flag is True:
            return "stale_contract"
        dq = str(getattr(contract, "data_quality_code", "") or "").lower()
        if dq in {"stale", "stale_quote", "quote_stale"}:
            return "stale_contract"
    return None
