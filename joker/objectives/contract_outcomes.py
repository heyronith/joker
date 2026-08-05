"""Contract-specific, scenario-based SPY 0DTE outcome estimates.

Estimates are explicitly heuristic unless backed by contract-level empirical
calibration. Missing Greeks are never fabricated; conservative quote/intrinsic
fallbacks are labeled in the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Sequence
from uuid import UUID, uuid4

from joker.objectives.full_chain_universe import FullChainContract
from joker.objectives.shared_scenarios import (
    SharedUnderlyingScenarioGrid,
    build_shared_underlying_scenario_grid,
)

ContractEstimateType = Literal[
    "market_greeks_scenario",
    "quote_intrinsic_fallback",
    "empirical_contract",
    "unknown",
]


@dataclass(frozen=True)
class ContractScenarioOutcome:
    scenario_id: str
    probability: Decimal
    underlying_price: Decimal
    estimated_option_price: Decimal
    pnl_per_contract_usd: Decimal
    horizon_seconds: int = 0
    shared_scenario_grid_hash: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "scenario_id": self.scenario_id,
            "probability": str(self.probability),
            "underlying_price": str(self.underlying_price),
            "estimated_option_price": str(self.estimated_option_price),
            "pnl_per_contract_usd": str(self.pnl_per_contract_usd),
            "horizon_seconds": str(self.horizon_seconds),
            "shared_scenario_grid_hash": self.shared_scenario_grid_hash,
        }


@dataclass(frozen=True)
class ContractOutcomeEstimate:
    estimate_id: UUID
    strategy_id: UUID
    strategy_family: str
    contract_id: str
    strike: Decimal
    option_type: str
    distance_from_spot: Decimal
    bid: Decimal
    ask: Decimal
    midpoint: Decimal
    relative_spread: Decimal
    liquidity_score: float
    delta: Decimal | None
    gamma: Decimal | None
    theta: Decimal | None
    implied_volatility: Decimal | None
    evaluation_premium: Decimal
    maximum_loss_usd_per_contract: Decimal
    estimated_useful_upside_usd: Decimal
    expected_pnl_usd: Decimal
    estimated_resolution_seconds: int
    required_contract_price: Decimal
    probability_reaches_required_price: Decimal | None
    probability_closes_goal_gap: Decimal | None
    lower_probability_bound: Decimal | None
    upper_probability_bound: Decimal | None
    estimate_type: ContractEstimateType
    assumptions: tuple[str, ...]
    uncertainty_reasons: tuple[str, ...]
    evidence_ids: tuple[UUID, ...]
    historical_sample_count: int
    scenarios: tuple[ContractScenarioOutcome, ...]
    usable_for_ranking: bool
    shared_scenario_grid_hash: str | None = None

    def as_dict(self, *, include_scenarios: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "estimate_id": str(self.estimate_id),
            "strategy_id": str(self.strategy_id),
            "strategy_family": self.strategy_family,
            "contract_id": self.contract_id,
            "strike": str(self.strike),
            "option_type": self.option_type,
            "distance_from_spot": str(self.distance_from_spot),
            "bid": str(self.bid),
            "ask": str(self.ask),
            "midpoint": str(self.midpoint),
            "relative_spread": str(self.relative_spread),
            "liquidity_score": self.liquidity_score,
            "delta": str(self.delta) if self.delta is not None else None,
            "gamma": str(self.gamma) if self.gamma is not None else None,
            "theta": str(self.theta) if self.theta is not None else None,
            "implied_volatility": (
                str(self.implied_volatility)
                if self.implied_volatility is not None
                else None
            ),
            "evaluation_premium": str(self.evaluation_premium),
            "maximum_loss_usd_per_contract": str(
                self.maximum_loss_usd_per_contract
            ),
            "estimated_useful_upside_usd": str(self.estimated_useful_upside_usd),
            "expected_pnl_usd": str(self.expected_pnl_usd),
            "estimated_resolution_seconds": self.estimated_resolution_seconds,
            "required_contract_price": str(self.required_contract_price),
            "probability_reaches_required_price": (
                str(self.probability_reaches_required_price)
                if self.probability_reaches_required_price is not None
                else None
            ),
            "probability_closes_goal_gap": (
                str(self.probability_closes_goal_gap)
                if self.probability_closes_goal_gap is not None
                else None
            ),
            "lower_probability_bound": (
                str(self.lower_probability_bound)
                if self.lower_probability_bound is not None
                else None
            ),
            "upper_probability_bound": (
                str(self.upper_probability_bound)
                if self.upper_probability_bound is not None
                else None
            ),
            "estimate_type": self.estimate_type,
            "assumptions": list(self.assumptions),
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "evidence_ids": [str(e) for e in self.evidence_ids],
            "historical_sample_count": self.historical_sample_count,
            "shared_scenario_grid_hash": self.shared_scenario_grid_hash,
            "usable_for_ranking": self.usable_for_ranking,
        }
        if include_scenarios:
            payload["scenarios"] = [s.as_dict() for s in self.scenarios]
        return payload


@dataclass(frozen=True)
class ContractOutcomeSettings:
    fallback_underlying_move_pct: Decimal = Decimal("1.00")
    minimum_probability_bound_width: Decimal = Decimal("0.10")
    maximum_probability_bound_width: Decimal = Decimal("0.30")
    minimum_scenario_probability: Decimal = Decimal("0.001")


def estimate_contract_outcome(
    *,
    strategy: Any,
    contract: FullChainContract,
    remaining_goal_gap_usd: Decimal,
    time_remaining_seconds: int,
    settings: ContractOutcomeSettings | None = None,
    historical_contract_hit_rate: Decimal | None = None,
    historical_sample_count: int = 0,
    evidence_ids: Sequence[UUID] = (),
    shared_scenario_grid: SharedUnderlyingScenarioGrid | None = None,
) -> ContractOutcomeEstimate:
    """Estimate one strategy × contract using shared-underlying scenarios."""
    cfg = settings or ContractOutcomeSettings()
    strategy_id = UUID(str(getattr(strategy, "strategy_id")))
    family = str(getattr(strategy, "strategy_family", None) or "unknown")
    requested_horizon = min(
        max(1, int(getattr(strategy, "expected_horizon_seconds", 600) or 600)),
        max(0, int(time_remaining_seconds)),
    )
    scenario_grid = shared_scenario_grid
    if scenario_grid is None and requested_horizon > 0:
        scenario_grid = build_shared_underlying_scenario_grid(
            strategies=(strategy,),
            reference_underlying_price=contract.underlying_price,
            evaluation_time=contract.evaluated_at_exchange_time,
            horizon_seconds=requested_horizon,
            base_move_pct=cfg.fallback_underlying_move_pct,
        )
    horizon = scenario_grid.horizon_seconds if scenario_grid is not None else 0
    assumptions: list[str] = [
        "long_option_max_loss_equals_ask_premium",
        "scenario_probabilities_are_heuristic_not_calibrated",
        "shared_underlying_scenarios_preserve_contract_correlation",
        "bounded_linear_interpolation_between_shared_scenario_points",
    ]
    uncertainty: list[str] = []
    if horizon <= 0 or contract.underlying_price <= 0 or contract.ask <= 0:
        return _unknown_estimate(
            strategy_id=strategy_id,
            family=family,
            contract=contract,
            horizon=max(0, horizon),
            remaining_goal_gap_usd=remaining_goal_gap_usd,
            evidence_ids=tuple(evidence_ids),
            reasons=("insufficient_market_truth_for_contract_estimate",),
        )

    has_greeks = contract.delta is not None and contract.implied_volatility is not None
    if has_greeks:
        estimate_type: ContractEstimateType = "market_greeks_scenario"
        assumptions.append("provider_delta_and_iv_used_when_present")
    else:
        estimate_type = "quote_intrinsic_fallback"
        uncertainty.append("missing_greeks_or_implied_volatility")
        assumptions.append("conservative_moneyness_response_fallback")

    scenarios = _price_shared_scenarios(
        contract=contract,
        shared_scenario_grid=scenario_grid,
        has_greeks=has_greeks,
    )
    if not scenarios:
        return _unknown_estimate(
            strategy_id=strategy_id,
            family=family,
            contract=contract,
            horizon=horizon,
            remaining_goal_gap_usd=remaining_goal_gap_usd,
            evidence_ids=tuple(evidence_ids),
            reasons=("scenario_construction_failed",),
        )

    required_price = (
        contract.ask + remaining_goal_gap_usd / Decimal("100")
    ).quantize(Decimal("0.0001"))
    p_required = _interpolated_hit_probability(
        scenarios,
        threshold=required_price,
        value=lambda scenario: scenario.estimated_option_price,
    )
    p_close = _interpolated_hit_probability(
        scenarios,
        threshold=remaining_goal_gap_usd,
        value=lambda scenario: scenario.pnl_per_contract_usd,
    )
    expected = sum(
        (s.probability * s.pnl_per_contract_usd for s in scenarios), Decimal("0")
    ).quantize(Decimal("0.01"))
    upside = max(
        Decimal("0"),
        max((s.pnl_per_contract_usd for s in scenarios), default=Decimal("0")),
    ).quantize(Decimal("0.01"))

    # Contract-specific empirical evidence may tighten, but never replaces, the
    # contract response model. Strategy-level rates are intentionally not used.
    if historical_contract_hit_rate is not None and historical_sample_count > 0:
        weight = min(Decimal("0.50"), Decimal(historical_sample_count) / Decimal("200"))
        p_close = (
            p_close * (Decimal("1") - weight)
            + historical_contract_hit_rate * weight
        ).quantize(Decimal("0.0001"))
        assumptions.append("contract_specific_empirical_rate_blended")
        if historical_sample_count >= 20:
            estimate_type = "empirical_contract"
        else:
            uncertainty.append("contract_empirical_sample_below_calibration_threshold")

    width = _uncertainty_width(contract=contract, estimate_type=estimate_type, cfg=cfg)
    lower = max(Decimal("0"), p_close - width).quantize(Decimal("0.0001"))
    upper = min(Decimal("1"), p_close + width).quantize(Decimal("0.0001"))
    return ContractOutcomeEstimate(
        estimate_id=uuid4(),
        strategy_id=strategy_id,
        strategy_family=family,
        contract_id=contract.contract_id,
        strike=contract.strike,
        option_type=contract.option_type,
        distance_from_spot=contract.distance_from_spot,
        bid=contract.bid,
        ask=contract.ask,
        midpoint=contract.mid,
        relative_spread=contract.relative_spread,
        liquidity_score=contract.liquidity_score,
        delta=contract.delta,
        gamma=contract.gamma,
        theta=contract.theta,
        implied_volatility=contract.implied_volatility,
        evaluation_premium=contract.ask,
        maximum_loss_usd_per_contract=contract.maximum_loss_usd_per_contract,
        estimated_useful_upside_usd=upside,
        expected_pnl_usd=expected,
        estimated_resolution_seconds=horizon,
        required_contract_price=required_price,
        probability_reaches_required_price=p_required,
        probability_closes_goal_gap=p_close,
        lower_probability_bound=lower,
        upper_probability_bound=upper,
        estimate_type=estimate_type,
        assumptions=tuple(assumptions),
        uncertainty_reasons=tuple(uncertainty),
        evidence_ids=tuple(evidence_ids),
        historical_sample_count=max(0, int(historical_sample_count)),
        shared_scenario_grid_hash=scenario_grid.grid_hash,
        scenarios=scenarios,
        usable_for_ranking=True,
    )


def _price_shared_scenarios(
    *,
    contract: FullChainContract,
    shared_scenario_grid: SharedUnderlyingScenarioGrid | None,
    has_greeks: bool,
) -> tuple[ContractScenarioOutcome, ...]:
    if shared_scenario_grid is None:
        return ()
    outcomes: list[ContractScenarioOutcome] = []
    for scenario in shared_scenario_grid.scenarios:
        scenario_spot = scenario.underlying_price
        underlying_move = scenario_spot - contract.underlying_price
        option_price = _scenario_option_price(
            contract=contract,
            scenario_spot=scenario_spot,
            underlying_move=underlying_move,
            horizon_seconds=scenario.horizon_seconds,
            has_greeks=has_greeks,
        )
        pnl = ((option_price - contract.ask) * Decimal("100")).quantize(
            Decimal("0.01")
        )
        outcomes.append(
            ContractScenarioOutcome(
                scenario_id=scenario.scenario_id,
                probability=scenario.probability,
                underlying_price=scenario_spot.quantize(Decimal("0.0001")),
                estimated_option_price=option_price,
                pnl_per_contract_usd=max(
                    -contract.maximum_loss_usd_per_contract, pnl
                ),
                horizon_seconds=scenario.horizon_seconds,
                shared_scenario_grid_hash=shared_scenario_grid.grid_hash,
            )
        )
    return tuple(outcomes)


def _scenario_option_price(
    *,
    contract: FullChainContract,
    scenario_spot: Decimal,
    underlying_move: Decimal,
    horizon_seconds: int,
    has_greeks: bool,
) -> Decimal:
    intrinsic = (
        max(Decimal("0"), scenario_spot - contract.strike)
        if contract.option_type == "call"
        else max(Decimal("0"), contract.strike - scenario_spot)
    )
    if has_greeks and contract.delta is not None:
        gamma = contract.gamma or Decimal("0")
        theta = contract.theta or Decimal("0")
        # Provider theta is generally per day. Clamp residual time value at zero.
        projected = (
            contract.mid
            + contract.delta * underlying_move
            + Decimal("0.5") * gamma * underlying_move * underlying_move
            + theta * Decimal(horizon_seconds) / Decimal(86400)
        )
        return max(intrinsic, projected, Decimal("0")).quantize(Decimal("0.0001"))

    current_intrinsic = (
        max(Decimal("0"), contract.underlying_price - contract.strike)
        if contract.option_type == "call"
        else max(Decimal("0"), contract.strike - contract.underlying_price)
    )
    time_value = max(Decimal("0"), contract.mid - current_intrinsic)
    # Conservative fallback: intrinsic plus no more than half current time value.
    residual_time_value = time_value * Decimal("0.50")
    return max(Decimal("0"), intrinsic + residual_time_value).quantize(
        Decimal("0.0001")
    )


def _uncertainty_width(
    *,
    contract: FullChainContract,
    estimate_type: ContractEstimateType,
    cfg: ContractOutcomeSettings,
) -> Decimal:
    spread_component = min(Decimal("0.20"), contract.relative_spread / Decimal("2"))
    missing_component = (
        Decimal("0.15") if estimate_type == "quote_intrinsic_fallback" else Decimal("0.05")
    )
    return max(
        cfg.minimum_probability_bound_width,
        min(cfg.maximum_probability_bound_width, spread_component + missing_component),
    )


def _interpolated_hit_probability(
    scenarios: Sequence[ContractScenarioOutcome],
    *,
    threshold: Decimal,
    value: Any,
) -> Decimal:
    """P(value >= threshold) via CDF interpolation on the shared scenario grid.

    Softens only the single adjacent pair that straddles the threshold so a
    coarse common underlying grid does not create false exact ties across
    contracts with materially different priced outcomes.
    """
    if not scenarios:
        return Decimal("0.0000")
    buckets: dict[Decimal, Decimal] = {}
    for scenario in scenarios:
        level = value(scenario)
        buckets[level] = buckets.get(level, Decimal("0")) + scenario.probability
    ordered_levels = sorted(buckets)
    if threshold <= ordered_levels[0]:
        return Decimal("1.0000")
    if threshold > ordered_levels[-1]:
        return Decimal("0.0000")

    cumulative = Decimal("0")
    previous_level = ordered_levels[0]
    previous_cdf = Decimal("0")
    for level in ordered_levels:
        cumulative += buckets[level]
        if level < threshold:
            previous_level = level
            previous_cdf = cumulative
            continue
        if level == threshold:
            return (Decimal("1") - cumulative + buckets[level]).quantize(
                Decimal("0.0001")
            )
        span = level - previous_level
        if span <= 0:
            return (Decimal("1") - previous_cdf).quantize(Decimal("0.0001"))
        # Linear CDF between the last point below threshold and the first above.
        frac = (threshold - previous_level) / span
        cdf_at_threshold = previous_cdf + frac * (cumulative - previous_cdf)
        return max(
            Decimal("0"),
            min(Decimal("1"), Decimal("1") - cdf_at_threshold),
        ).quantize(Decimal("0.0001"))
    return Decimal("0.0000")


def _unknown_estimate(
    *,
    strategy_id: UUID,
    family: str,
    contract: FullChainContract,
    horizon: int,
    remaining_goal_gap_usd: Decimal,
    evidence_ids: tuple[UUID, ...],
    reasons: tuple[str, ...],
) -> ContractOutcomeEstimate:
    return ContractOutcomeEstimate(
        estimate_id=uuid4(),
        strategy_id=strategy_id,
        strategy_family=family,
        contract_id=contract.contract_id,
        strike=contract.strike,
        option_type=contract.option_type,
        distance_from_spot=contract.distance_from_spot,
        bid=contract.bid,
        ask=contract.ask,
        midpoint=contract.mid,
        relative_spread=contract.relative_spread,
        liquidity_score=contract.liquidity_score,
        delta=contract.delta,
        gamma=contract.gamma,
        theta=contract.theta,
        implied_volatility=contract.implied_volatility,
        evaluation_premium=contract.ask,
        maximum_loss_usd_per_contract=contract.maximum_loss_usd_per_contract,
        estimated_useful_upside_usd=Decimal("0"),
        expected_pnl_usd=Decimal("0"),
        estimated_resolution_seconds=horizon,
        required_contract_price=(
            contract.ask + remaining_goal_gap_usd / Decimal("100")
        ),
        probability_reaches_required_price=None,
        probability_closes_goal_gap=None,
        lower_probability_bound=None,
        upper_probability_bound=None,
        estimate_type="unknown",
        assumptions=(),
        uncertainty_reasons=reasons,
        evidence_ids=evidence_ids,
        historical_sample_count=0,
        shared_scenario_grid_hash=None,
        scenarios=(),
        usable_for_ranking=False,
    )
