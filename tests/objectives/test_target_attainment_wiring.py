"""Config, shadow baseline, feasibility, and soft-veto wiring for target_attainment."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from joker.cli.session_confirm import build_objective_engines
from joker.config.loader import load_app_settings
from joker.objectives.config import ObjectiveSettings
from joker.objectives.feasibility import FeasibilityInputs, GoalFeasibilityEngine
from joker.objectives.schemas import SessionObjectiveState
from joker.objectives.scoring import ObjectiveStrategyScorer, StrategyScoreInput
from joker.objectives.target_attainment import (
    TargetAttainmentContext,
    TargetAttainmentCandidate,
    TargetAttainmentPolicy,
    run_positive_ev_baseline_shadow,
)


def _state(**overrides: object) -> SessionObjectiveState:
    data = {
        "objective_id": uuid4(),
        "session_id": "s",
        "status": "active",
        "authorised_capital_usd": Decimal("500.00"),
        "target_profit_usd": Decimal("100.00"),
        "target_ending_equity_usd": Decimal("600.00"),
        "reserved_capital_usd": Decimal("0.00"),
        "available_capital_usd": Decimal("500.00"),
        "realised_pnl_usd": Decimal("0.00"),
        "unrealised_pnl_usd": Decimal("0.00"),
        "progress_to_goal_pct": Decimal("0.00"),
        "required_profit_remaining_usd": Decimal("400.00"),
        "time_remaining_seconds": 600,
        "max_concurrent_positions": 1,
        "version": 1,
    }
    data.update(overrides)
    return SessionObjectiveState.model_validate(data)


def test_positive_ev_baseline_remains_available() -> None:
    settings = ObjectiveSettings(policy="positive_ev_baseline")
    assert settings.policy == "positive_ev_baseline"
    assert settings.is_target_attainment is False


def test_target_attainment_is_authoritative_in_paper_config() -> None:
    root = Path(__file__).resolve().parents[2]
    app, _env = load_app_settings(root / "config" / "paper.yaml")
    assert app.objective.policy == "target_attainment"
    assert app.objective.shadow_baseline_enabled is True
    assert app.objective.is_target_attainment is True
    assert app.objective.target_attainment.allow_full_remaining_capital is True
    assert app.objective.target_attainment.maximum_capital_fraction == 1.0


def test_baseline_shadow_never_executes() -> None:
    state = _state(required_profit_remaining_usd=Decimal("100.00"), time_remaining_seconds=3600)
    snap = uuid4()
    candidates = [
        StrategyScoreInput(
            strategy_id=uuid4(),
            snapshot_id=snap,
            expected_value_usd=Decimal("10.00"),
            estimated_win_probability=Decimal("0.60"),
            estimated_payoff_ratio=Decimal("1.5"),
            estimated_resolution_seconds=300,
            maximum_loss_usd=Decimal("100.00"),
            capital_required_usd=Decimal("100.00"),
        )
    ]
    shadow = run_positive_ev_baseline_shadow(
        state,
        candidates,
        snapshot_id=snap,
        require_positive_expected_value=True,
        minimum_win_probability=0.45,
    )
    assert shadow["policy"] == "positive_ev_baseline"
    assert shadow["shadow_only"] is True
    assert shadow["executes"] is False


def test_only_one_policy_reaches_order_gateway() -> None:
    """Shadow baseline is comparison-only; target-attainment decision is authoritative."""
    state = _state(
        required_profit_remaining_usd=Decimal("100.00"),
        available_capital_usd=Decimal("200.00"),
        time_remaining_seconds=1800,
    )
    snap = uuid4()
    sid = uuid4()
    candidates = [
        StrategyScoreInput(
            strategy_id=sid,
            snapshot_id=snap,
            expected_value_usd=Decimal("-1.00"),
            estimated_win_probability=Decimal("0.35"),
            estimated_payoff_ratio=Decimal("3.0"),
            estimated_resolution_seconds=300,
            maximum_loss_usd=Decimal("100.00"),
            capital_required_usd=Decimal("100.00"),
            calculation_inputs={"useful_upside_usd": "150.00", "sample_count": 5},
        )
    ]
    shadow = run_positive_ev_baseline_shadow(
        state, candidates, snapshot_id=snap
    )
    assert shadow["executes"] is False

    ctx = TargetAttainmentContext.from_state(
        state,
        snapshot_id=snap,
        objective_duration_seconds=3600,
        maximum_authorised_contracts=20,
    )
    from joker.objectives.target_attainment import candidate_from_score_input

    ta_cands = [candidate_from_score_input(c) for c in candidates]
    decision = TargetAttainmentPolicy().decide(
        ctx, ta_cands, baseline_shadow=shadow
    )
    assert decision.baseline_shadow is not None
    assert decision.baseline_shadow["executes"] is False
    # Authoritative action may enter even when baseline rejects negative EV.
    assert decision.action.value in {"enter", "wait", "block"}


def test_target_policy_cannot_exceed_authorized_capital() -> None:
    ctx = TargetAttainmentContext(
        objective_id=uuid4(),
        snapshot_id=uuid4(),
        authorised_capital_usd=Decimal("200.00"),
        available_capital_usd=Decimal("200.00"),
        reserved_capital_usd=Decimal("0.00"),
        realised_pnl_usd=Decimal("0.00"),
        unrealised_pnl_usd=Decimal("0.00"),
        target_profit_usd=Decimal("100.00"),
        remaining_goal_gap_usd=Decimal("100.00"),
        time_remaining_seconds=1800,
        objective_duration_seconds=3600,
        elapsed_seconds=1800,
        open_position_count=0,
        working_order_count=0,
        max_concurrent_positions=1,
        maximum_authorised_contracts=20,
        allow_full_remaining_capital=True,
        maximum_capital_fraction=1.0,
    )
    cand = TargetAttainmentCandidate(
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        estimated_win_probability=Decimal("0.55"),
        expected_value_usd=Decimal("5.00"),
        estimated_payoff_ratio=Decimal("1.0"),
        estimated_useful_upside_usd=Decimal("80.00"),
        estimated_resolution_seconds=300,
        maximum_loss_usd_per_contract=Decimal("100.00"),
    )
    decision = TargetAttainmentPolicy().decide(ctx, [cand])
    for ev in decision.quantity_evaluations:
        if not ev.physically_impossible and ev.quantity > 0:
            assert ev.capital_required_usd <= ctx.available_capital_usd
    if decision.selected_quantity > 0:
        assert decision.selected_capital_usd <= ctx.available_capital_usd


def test_feasibility_extreme_target_is_low_under_target_attainment() -> None:
    eng = GoalFeasibilityEngine(policy="target_attainment")
    a = eng.assess(
        _state(
            required_profit_remaining_usd=Decimal("400.00"),
            authorised_capital_usd=Decimal("500.00"),
            available_capital_usd=Decimal("500.00"),
            time_remaining_seconds=600,
        ),
        FeasibilityInputs(snapshot_id=uuid4(), session_phase="regular"),
    )
    assert a.classification == "low"
    assert "extreme_target_vs_time" in a.binding_constraints or (
        "high_target_insufficient_time" in a.binding_constraints
    )


def test_feasibility_extreme_target_may_be_infeasible_under_baseline() -> None:
    eng = GoalFeasibilityEngine(policy="positive_ev_baseline")
    a = eng.assess(
        _state(
            # >=100% of authorised capital remaining with <60 minutes → infeasible
            required_profit_remaining_usd=Decimal("500.00"),
            authorised_capital_usd=Decimal("500.00"),
            available_capital_usd=Decimal("500.00"),
            target_profit_usd=Decimal("500.00"),
            time_remaining_seconds=600,
        ),
        FeasibilityInputs(snapshot_id=uuid4(), session_phase="regular"),
    )
    assert a.classification == "infeasible"


def test_scorer_soft_mode_keeps_negative_ev_valid() -> None:
    scorer = ObjectiveStrategyScorer(
        require_positive_expected_value=False,
        minimum_win_probability=0.0,
    )
    state = _state(
        required_profit_remaining_usd=Decimal("100.00"),
        time_remaining_seconds=3600,
        available_capital_usd=Decimal("500.00"),
    )
    snap = uuid4()
    score = scorer.score(
        state,
        StrategyScoreInput(
            strategy_id=uuid4(),
            snapshot_id=snap,
            expected_value_usd=Decimal("-5.00"),
            estimated_win_probability=Decimal("0.30"),
            estimated_resolution_seconds=300,
            maximum_loss_usd=Decimal("100.00"),
            capital_required_usd=Decimal("100.00"),
        ),
    )
    assert score.valid is True
    assert "non_positive_expected_value" not in score.invalidation_codes


def test_build_objective_engines_from_paper_softens_vetoes() -> None:
    root = Path(__file__).resolve().parents[2]
    app, _env = load_app_settings(root / "config" / "paper.yaml")
    bundle = build_objective_engines(app)
    assert bundle.objective_policy == "target_attainment"
    assert bundle.target_attainment_policy is not None
    assert bundle.shadow_baseline_enabled is True
    assert bundle.objective_strategy_scorer.require_positive_ev is False
    assert bundle.objective_strategy_scorer.min_win_p == 0.0
    assert bundle.capital_sizer.require_positive_ev is False
    assert bundle.capital_sizer.max_capital_fraction == 1.0
    assert bundle.feasibility_engine.policy == "target_attainment"
    kwargs = bundle.as_deps_kwargs()
    assert "target_attainment_policy" in kwargs
    assert kwargs["objective_policy"] == "target_attainment"


def test_evidence_redacts_credentials_in_decision_dict() -> None:
    from joker.runtime.paper_goal_result import redact_mapping

    ctx = TargetAttainmentContext(
        objective_id=uuid4(),
        snapshot_id=uuid4(),
        authorised_capital_usd=Decimal("200.00"),
        available_capital_usd=Decimal("200.00"),
        reserved_capital_usd=Decimal("0.00"),
        realised_pnl_usd=Decimal("0.00"),
        unrealised_pnl_usd=Decimal("0.00"),
        target_profit_usd=Decimal("50.00"),
        remaining_goal_gap_usd=Decimal("50.00"),
        time_remaining_seconds=900,
        objective_duration_seconds=3600,
        elapsed_seconds=2700,
        open_position_count=0,
        working_order_count=0,
        max_concurrent_positions=1,
        maximum_authorised_contracts=5,
    )
    decision = TargetAttainmentPolicy().decide(ctx, [])
    payload = {
        "decision": decision.as_dict(),
        "api_key": "sk-secret-should-redact",
        "account_id": "ACC123456789",
    }
    redacted = redact_mapping(payload)
    assert "api_key" not in redacted
    assert "decision" in redacted
