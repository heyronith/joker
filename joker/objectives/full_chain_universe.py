"""Deterministic SPY 0DTE full-chain eligibility and stratification.

This module is a market-truth boundary. Contract identifiers are accepted only
from the linked option surface; strategy-agent ``candidate_legs`` are hints and
are never used to discover executable contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Sequence
from uuid import UUID

from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot


class MoneynessBucket(StrEnum):
    ITM = "itm"
    ATM = "atm"
    OTM = "otm"


@dataclass(frozen=True)
class ContractSelectionSpec:
    """Non-authoritative contract preferences derived from a strategy thesis."""

    option_types: tuple[Literal["call", "put"], ...]
    direction: str
    strategy_family: str
    resolution_horizon_seconds: int
    minimum_moneyness_pct: Decimal = Decimal("-100")
    maximum_moneyness_pct: Decimal = Decimal("100")
    preferred_delta_min: Decimal | None = None
    preferred_delta_max: Decimal | None = None
    maximum_relative_spread: Decimal = Decimal("0.25")
    maximum_quote_age_seconds: int = 30
    maximum_premium_usd: Decimal | None = None
    minimum_liquidity_score: float = 0.0

    @classmethod
    def from_strategy(
        cls,
        strategy: Any,
        *,
        maximum_relative_spread: float,
        maximum_quote_age_seconds: int,
        maximum_premium_usd: Decimal | None = None,
    ) -> ContractSelectionSpec:
        direction = str(
            getattr(getattr(strategy, "direction", None), "value", None)
            or getattr(strategy, "direction", "")
        ).lower()
        option_types: tuple[Literal["call", "put"], ...]
        if direction in {"bullish", "long", "up", "call"}:
            option_types = ("call",)
        elif direction in {"bearish", "short", "down", "put"}:
            option_types = ("put",)
        else:
            option_types = ("call", "put")
        preferences = tuple(
            getattr(strategy, "contract_selection_preferences", ()) or ()
        )
        preference = preferences[0] if preferences else None
        preferred_types = tuple(getattr(preference, "option_types", ()) or ())
        if preferred_types:
            compatible = tuple(
                item for item in option_types if item in preferred_types
            )
            if compatible:
                option_types = compatible
        return cls(
            option_types=option_types,
            direction=direction or "neutral",
            strategy_family=str(getattr(strategy, "strategy_family", None) or "unknown"),
            resolution_horizon_seconds=max(
                1, int(getattr(strategy, "expected_horizon_seconds", 600) or 600)
            ),
            minimum_moneyness_pct=Decimal(
                str(
                    getattr(preference, "minimum_moneyness_pct", None)
                    if getattr(preference, "minimum_moneyness_pct", None)
                    is not None
                    else "-100"
                )
            ),
            maximum_moneyness_pct=Decimal(
                str(
                    getattr(preference, "maximum_moneyness_pct", None)
                    if getattr(preference, "maximum_moneyness_pct", None)
                    is not None
                    else "100"
                )
            ),
            preferred_delta_min=getattr(
                preference, "preferred_delta_min", None
            ),
            preferred_delta_max=getattr(
                preference, "preferred_delta_max", None
            ),
            maximum_relative_spread=Decimal(str(maximum_relative_spread)),
            maximum_quote_age_seconds=max(1, int(maximum_quote_age_seconds)),
            maximum_premium_usd=maximum_premium_usd,
        )


@dataclass(frozen=True)
class FullChainContract:
    """Validated contract from the exact linked surface."""

    surface_id: UUID
    contract_id: str
    symbol: str
    expiration: date
    option_type: Literal["call", "put"]
    strike: Decimal
    underlying_price: Decimal
    distance_from_spot: Decimal
    distance_from_spot_pct: Decimal
    moneyness_bucket: MoneynessBucket
    premium_bucket: str
    delta_bucket: str
    bid: Decimal
    ask: Decimal
    mid: Decimal
    relative_spread: Decimal
    quote_timestamp: datetime
    quote_age_seconds: Decimal
    evaluated_at_exchange_time: datetime
    liquidity_score: float
    delta: Decimal | None = None
    gamma: Decimal | None = None
    theta: Decimal | None = None
    implied_volatility: Decimal | None = None
    quality_flags: tuple[str, ...] = ()
    stratum: tuple[str, ...] = ()

    @property
    def maximum_loss_usd_per_contract(self) -> Decimal:
        return (self.ask * Decimal("100")).quantize(Decimal("0.01"))

    def as_summary(self) -> dict[str, Any]:
        """Sanitized optimizer/agent summary; excludes raw provider payloads."""
        return {
            "contract_id": self.contract_id,
            "option_type": self.option_type,
            "strike": str(self.strike),
            "distance_from_spot_pct": str(self.distance_from_spot_pct),
            "moneyness_bucket": self.moneyness_bucket.value,
            "premium_bucket": self.premium_bucket,
            "delta_bucket": self.delta_bucket,
            "bid": str(self.bid),
            "ask": str(self.ask),
            "relative_spread": str(self.relative_spread),
            "quote_age_seconds": str(self.quote_age_seconds),
            "evaluated_at_exchange_time": self.evaluated_at_exchange_time.isoformat(),
            "liquidity_score": self.liquidity_score,
        }


@dataclass(frozen=True)
class FullChainUniverse:
    snapshot_id: UUID
    surface_id: UUID
    trading_date: date
    underlying_price: Decimal
    evaluated_at_exchange_time: datetime
    source_contract_count: int
    eligible_contract_count: int
    contracts: tuple[FullChainContract, ...]
    exclusion_counts: dict[str, int] = field(default_factory=dict)
    stratified: bool = False
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": str(self.snapshot_id),
            "surface_id": str(self.surface_id),
            "trading_date": self.trading_date.isoformat(),
            "underlying_price": str(self.underlying_price),
            "evaluated_at_exchange_time": self.evaluated_at_exchange_time.isoformat(),
            "source_contract_count": self.source_contract_count,
            "eligible_contract_count": self.eligible_contract_count,
            "evaluated_contract_count": len(self.contracts),
            "exclusion_counts": dict(sorted(self.exclusion_counts.items())),
            "stratified": self.stratified,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class FullChainUniverseSettings:
    maximum_quote_age_seconds: int = 30
    maximum_surface_age_seconds: int = 30
    maximum_future_timestamp_seconds: int = 1
    maximum_relative_spread: Decimal = Decimal("0.25")
    maximum_contracts_evaluated: int = 200
    moneyness_buckets: tuple[Decimal, ...] = (
        Decimal("-2"),
        Decimal("-0.5"),
        Decimal("0.5"),
        Decimal("2"),
    )
    premium_buckets: tuple[Decimal, ...] = (
        Decimal("0.10"),
        Decimal("0.20"),
        Decimal("0.50"),
        Decimal("1.00"),
        Decimal("2.00"),
        Decimal("5.00"),
    )
    delta_buckets: tuple[Decimal, ...] = (
        Decimal("0.10"),
        Decimal("0.25"),
        Decimal("0.50"),
        Decimal("0.75"),
    )


def build_full_chain_universe(
    *,
    snapshot_id: UUID,
    surface: OptionSurfaceSnapshot,
    current_exchange_time: datetime,
    current_trading_date: date,
    available_capital_usd: Decimal,
    settings: FullChainUniverseSettings | None = None,
) -> FullChainUniverse:
    """Validate every row, then deterministically stratify if a bound is needed."""
    cfg = settings or FullChainUniverseSettings()
    spot = surface.underlying_price
    reasons: list[str] = []
    if current_exchange_time.tzinfo is None:
        reasons.append("current_exchange_time_naive")
    if surface.underlying_symbol.upper() != "SPY":
        reasons.append("wrong_underlying")
    if surface.trading_date != current_trading_date:
        reasons.append("surface_trading_date_mismatch")
    if current_exchange_time.tzinfo is not None:
        surface_age = Decimal(
            str((current_exchange_time - surface.exchange_time).total_seconds())
        )
        if surface_age < -Decimal(cfg.maximum_future_timestamp_seconds):
            reasons.append("surface_timestamp_in_future")
        elif surface_age > Decimal(cfg.maximum_surface_age_seconds):
            reasons.append("stale_option_surface")
    if spot is None or spot <= 0:
        reasons.append("underlying_price_unavailable")
    if reasons:
        return FullChainUniverse(
            snapshot_id=snapshot_id,
            surface_id=surface.surface_id,
            trading_date=current_trading_date,
            underlying_price=spot or Decimal("0"),
            evaluated_at_exchange_time=current_exchange_time,
            source_contract_count=len(surface.contracts),
            eligible_contract_count=0,
            contracts=(),
            reason_codes=tuple(reasons),
        )

    exclusions: dict[str, int] = {}
    eligible: list[FullChainContract] = []
    for row in surface.contracts:
        reason = _validate_row(
            row,
            trading_date=current_trading_date,
            current_exchange_time=current_exchange_time,
            available_capital_usd=available_capital_usd,
            settings=cfg,
        )
        if reason is not None:
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        assert row.bid is not None and row.ask is not None
        mid = row.mid if row.mid is not None else (row.bid + row.ask) / Decimal("2")
        spread = (
            row.relative_spread
            if row.relative_spread is not None
            else (row.ask - row.bid) / mid
        )
        distance = row.strike - spot
        distance_pct = (distance / spot * Decimal("100")).quantize(Decimal("0.0001"))
        moneyness = _moneyness(row.option_type, distance_pct)
        premium_bucket = _bucket(row.ask, cfg.premium_buckets, "premium")
        delta_abs = abs(row.delta) if row.delta is not None else None
        delta_bucket = (
            _bucket(delta_abs, cfg.delta_buckets, "delta")
            if delta_abs is not None
            else "delta:unknown"
        )
        stratum = (
            row.option_type,
            moneyness.value,
            premium_bucket,
            delta_bucket,
            _distance_bucket(distance_pct, cfg.moneyness_buckets),
            _liquidity_bucket(row.liquidity_score),
        )
        quote_age = max(
            Decimal("0"),
            Decimal(str((current_exchange_time - row.quote_timestamp).total_seconds())),
        )
        eligible.append(
            FullChainContract(
                surface_id=surface.surface_id,
                contract_id=row.contract_id,
                symbol="SPY",
                expiration=row.expiry,
                option_type=row.option_type,
                strike=row.strike,
                underlying_price=spot,
                distance_from_spot=distance,
                distance_from_spot_pct=distance_pct,
                moneyness_bucket=moneyness,
                premium_bucket=premium_bucket,
                delta_bucket=delta_bucket,
                bid=row.bid,
                ask=row.ask,
                mid=mid,
                relative_spread=spread,
                quote_timestamp=row.quote_timestamp,
                quote_age_seconds=quote_age,
                evaluated_at_exchange_time=current_exchange_time,
                liquidity_score=float(row.liquidity_score),
                delta=row.delta,
                gamma=row.gamma,
                theta=row.theta,
                implied_volatility=row.implied_volatility,
                quality_flags=tuple(sorted(row.quality_flags)),
                stratum=stratum,
            )
        )

    eligible.sort(key=_canonical_contract_key)
    selected = _deterministic_stratified_sample(
        eligible, maximum=max(1, int(cfg.maximum_contracts_evaluated))
    )
    return FullChainUniverse(
        snapshot_id=snapshot_id,
        surface_id=surface.surface_id,
        trading_date=current_trading_date,
        underlying_price=spot,
        evaluated_at_exchange_time=current_exchange_time,
        source_contract_count=len(surface.contracts),
        eligible_contract_count=len(eligible),
        contracts=tuple(selected),
        exclusion_counts=exclusions,
        stratified=len(selected) < len(eligible),
        reason_codes=() if selected else ("no_valid_contract_candidates",),
    )


def contracts_for_spec(
    universe: FullChainUniverse,
    spec: ContractSelectionSpec,
) -> tuple[FullChainContract, ...]:
    """Return deterministic thesis-compatible contracts without using agent legs."""
    selected: list[FullChainContract] = []
    for contract in universe.contracts:
        if contract.option_type not in spec.option_types:
            continue
        if not (
            spec.minimum_moneyness_pct
            <= contract.distance_from_spot_pct
            <= spec.maximum_moneyness_pct
        ):
            continue
        if contract.relative_spread > spec.maximum_relative_spread:
            continue
        if contract.quote_age_seconds > spec.maximum_quote_age_seconds:
            continue
        if contract.liquidity_score < spec.minimum_liquidity_score:
            continue
        if (
            spec.maximum_premium_usd is not None
            and contract.maximum_loss_usd_per_contract > spec.maximum_premium_usd
        ):
            continue
        if contract.delta is not None:
            abs_delta = abs(contract.delta)
            if (
                spec.preferred_delta_min is not None
                and abs_delta < spec.preferred_delta_min
            ):
                continue
            if (
                spec.preferred_delta_max is not None
                and abs_delta > spec.preferred_delta_max
            ):
                continue
        selected.append(contract)
    return tuple(sorted(selected, key=_canonical_contract_key))


def stratify_contracts(
    contracts: Sequence[FullChainContract],
    *,
    maximum: int,
) -> tuple[FullChainContract, ...]:
    """Public deterministic bound preserving the universe's contract strata."""
    return tuple(
        _deterministic_stratified_sample(
            sorted(contracts, key=_canonical_contract_key),
            maximum=max(1, int(maximum)),
        )
    )


def _validate_row(
    row: OptionContractSnapshot,
    *,
    trading_date: date,
    current_exchange_time: datetime,
    available_capital_usd: Decimal,
    settings: FullChainUniverseSettings,
) -> str | None:
    if not row.contract_id.strip():
        return "missing_contract_id"
    if not _valid_executable_contract_id(row):
        return "invalid_contract_id"
    if not row.contract_id.startswith("SPY:"):
        return "wrong_underlying"
    if row.expiry != trading_date:
        return "non_0dte_contract"
    if row.bid is None or row.ask is None:
        return "missing_bid_ask"
    if row.bid <= 0 or row.ask <= 0 or row.ask < row.bid:
        return "invalid_bid_ask"
    if row.quote_timestamp.tzinfo is None:
        return "quote_timestamp_naive"
    quote_age = Decimal(
        str((current_exchange_time - row.quote_timestamp).total_seconds())
    )
    if quote_age < -Decimal(settings.maximum_future_timestamp_seconds):
        return "quote_timestamp_in_future"
    if quote_age > Decimal(settings.maximum_quote_age_seconds):
        return "stale_contract"
    mid = row.mid if row.mid is not None else (row.bid + row.ask) / Decimal("2")
    if mid <= 0:
        return "mid_unavailable"
    spread = (
        row.relative_spread
        if row.relative_spread is not None
        else (row.ask - row.bid) / mid
    )
    if spread > settings.maximum_relative_spread:
        return "excessive_spread"
    unusable = {
        "crossed_market",
        "execution_unusable",
        "stale_quote",
        "quote_stale",
    }
    if unusable.intersection({f.lower() for f in row.quality_flags}):
        return "execution_unusable"
    if row.ask * Decimal("100") > available_capital_usd:
        return "unaffordable_contract"
    return None


def _valid_executable_contract_id(row: OptionContractSnapshot) -> bool:
    parts = row.contract_id.split(":")
    if len(parts) != 4:
        return False
    symbol, expiry_raw, strike_raw, option_type = parts
    try:
        expiry = date.fromisoformat(expiry_raw)
        strike = Decimal(strike_raw)
    except Exception:
        return False
    return (
        symbol.upper() == "SPY"
        and expiry == row.expiry
        and strike == row.strike
        and option_type == row.option_type
    )


def _moneyness(option_type: str, distance_pct: Decimal) -> MoneynessBucket:
    if abs(distance_pct) <= Decimal("0.50"):
        return MoneynessBucket.ATM
    if option_type == "call":
        return MoneynessBucket.ITM if distance_pct < 0 else MoneynessBucket.OTM
    return MoneynessBucket.ITM if distance_pct > 0 else MoneynessBucket.OTM


def _bucket(value: Decimal, thresholds: Sequence[Decimal], prefix: str) -> str:
    for threshold in sorted(thresholds):
        if value <= threshold:
            return f"{prefix}:<={threshold}"
    return f"{prefix}:>{max(thresholds)}" if thresholds else f"{prefix}:all"


def _distance_bucket(value: Decimal, thresholds: Sequence[Decimal]) -> str:
    return _bucket(value, thresholds, "distance_pct")


def _liquidity_bucket(score: float) -> str:
    if score >= 0.67:
        return "liquidity:high"
    if score >= 0.34:
        return "liquidity:medium"
    return "liquidity:low"


def _canonical_contract_key(c: FullChainContract) -> tuple[Any, ...]:
    return (
        c.option_type,
        c.moneyness_bucket.value,
        c.strike,
        c.ask,
        c.contract_id,
    )


def _deterministic_stratified_sample(
    contracts: Sequence[FullChainContract],
    *,
    maximum: int,
) -> list[FullChainContract]:
    if len(contracts) <= maximum:
        return list(contracts)
    strata: dict[tuple[str, ...], list[FullChainContract]] = {}
    for contract in contracts:
        strata.setdefault(contract.stratum, []).append(contract)
    for rows in strata.values():
        rows.sort(
            key=lambda c: (
                -c.liquidity_score,
                c.relative_spread,
                abs(c.distance_from_spot_pct),
                c.ask,
                c.contract_id,
            )
        )
    selected: list[FullChainContract] = []
    ordered_strata = sorted(strata)
    index = 0
    while len(selected) < maximum:
        progressed = False
        for key in ordered_strata:
            rows = strata[key]
            if index < len(rows):
                selected.append(rows[index])
                progressed = True
                if len(selected) == maximum:
                    break
        if not progressed:
            break
        index += 1
    return sorted(selected, key=_canonical_contract_key)
