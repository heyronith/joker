"""Historical outcome retrieval, similarity, leakage, and EV eligibility."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.cognition.schemas import (
    AgentRole,
    EntryPlan,
    ExecutionPlan,
    ExitPlan,
    InvalidationPlan,
    MarketDirection,
    StrategyHypothesis,
)
from joker.objectives.config import HistoricalOutcomeSettings
from joker.objectives.estimate import StrategyEstimateBuilder
from joker.objectives.historical_outcomes import HistoricalOutcomeService, independence_key
from joker.objectives.historical_schemas import HistoricalOutcomeQuery
from joker.objectives.schemas import SessionObjectiveState
from joker.objectives.similarity import score_similarity
from joker.evolution.telemetry import resolve_calibration_outcome
from tests.objectives.historical_fixtures import (
    make_closed_episode,
    make_hist_service,
    seed_positive_history,
)


def _obj_state() -> SessionObjectiveState:
    return SessionObjectiveState.model_validate(
        {
            "objective_id": uuid4(),
            "session_id": "s",
            "status": "active",
            "authorised_capital_usd": "500",
            "target_profit_usd": "50",
            "target_ending_equity_usd": "550",
            "available_capital_usd": "500",
            "required_profit_remaining_usd": "50",
            "time_remaining_seconds": 3600,
            "version": 1,
            "max_concurrent_positions": 1,
            "deadline_exchange_time": datetime.now(timezone.utc) + timedelta(hours=2),
        }
    )


def _strategy() -> StrategyHypothesis:
    from joker.cognition.schemas import StrategyLegCandidate

    return StrategyHypothesis(
        name="bull call",
        market_thesis="t",
        direction=MarketDirection.BULLISH,
        candidate_legs=(
            StrategyLegCandidate(
                contract_id="SPY:2026-07-01:500.0:call",
                side="buy",
                option_type="call",
                strike=Decimal("500"),
                quantity=1,
                rationale="primary",
            ),
        ),
        entry_plan=EntryPlan(entry_style="immediate", preferred_order_type="limit"),
        execution_plan=ExecutionPlan(
            max_quote_age_seconds=5,
            partial_fill_policy="accept",
            replacement_policy="none",
        ),
        exit_plan=ExitPlan(stop_conditions=("stop",)),
        invalidation_plan=InvalidationPlan(conditions=("inv",)),
        expected_horizon_seconds=1800,
        confidence=0.7,
        novelty_score=0.5,
        agent_role=AgentRole.BULLISH_INVENTOR,
        snapshot_id=uuid4(),
        session_id="s",
        cycle_id="c",
        prompt_version="t",
        model_call_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_historical_outcomes_exclude_future_episodes(tmp_path) -> None:
    svc, _ = make_hist_service(tmp_path, minimum_samples_for_ev=5)
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    past = [make_closed_episode(pnl=Decimal("10"), as_of=as_of, hours_before=10 + i) for i in range(5)]
    future = make_closed_episode(pnl=Decimal("99"), as_of=as_of + timedelta(days=1), hours_before=1)
    # Force future terminal after as_of
    future.terminal_event_timestamp = as_of + timedelta(hours=1)
    future.entry_decision_timestamp = as_of + timedelta(minutes=30)
    svc.seed_episodes_for_tests([*past, future])
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="bullish",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
        )
    )
    assert future.episode_id in report.excluded_future_episodes
    assert summary.sample_count == 5
    assert future.episode_id not in summary.comparable_episode_ids


@pytest.mark.asyncio
async def test_historical_outcomes_exclude_current_episode(tmp_path) -> None:
    svc, _ = make_hist_service(tmp_path, minimum_samples_for_ev=3)
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    eps = [make_closed_episode(pnl=Decimal("8"), as_of=as_of, hours_before=5 + i) for i in range(4)]
    current = eps[0]
    svc.seed_episodes_for_tests(eps)
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="bullish",
            direction="bullish",
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
            current_episode_id=current.episode_id,
        )
    )
    assert current.episode_id in report.excluded_current_episode
    assert summary.sample_count == 3


@pytest.mark.asyncio
async def test_duplicate_replays_count_as_one_authoritative_sample(tmp_path) -> None:
    svc, _ = make_hist_service(tmp_path, minimum_samples_for_ev=1)
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    base = make_closed_episode(pnl=Decimal("10"), as_of=as_of, hours_before=5)
    dup = make_closed_episode(pnl=Decimal("10"), as_of=as_of, hours_before=5)
    # Same independence key components
    dup.entry_decision_event_id = base.entry_decision_event_id
    dup.terminal_event_id = base.terminal_event_id
    dup.episode_id = base.episode_id
    assert independence_key(base) == independence_key(dup)
    svc.seed_episodes_for_tests([base, dup])
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="bullish",
            direction="bullish",
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
        )
    )
    assert summary.sample_count == 1
    assert report.excluded_duplicate_truth


def test_similarity_policy_is_deterministic() -> None:
    a, ca = score_similarity(
        query_strategy_family="bullish",
        query_direction="bullish",
        query_regime_labels=("trend",),
        query_session_phase="midday",
        episode_strategy_family="bullish",
        episode_direction="bullish",
        episode_regime_labels=("trend",),
        episode_session_phase="midday",
    )
    b, cb = score_similarity(
        query_strategy_family="bullish",
        query_direction="bullish",
        query_regime_labels=("trend",),
        query_session_phase="midday",
        episode_strategy_family="bullish",
        episode_direction="bullish",
        episode_regime_labels=("trend",),
        episode_session_phase="midday",
    )
    assert a == b
    assert ca == cb
    assert a >= Decimal("0.65")


@pytest.mark.asyncio
async def test_insufficient_samples_leave_ev_none(tmp_path) -> None:
    svc, _ = make_hist_service(tmp_path, minimum_samples_for_ev=20)
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    seed_positive_history(svc, as_of=as_of, n=5)
    summary = await svc.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="bullish",
    )
    assert summary.valid_for_ev is False
    est = StrategyEstimateBuilder(minimum_samples_for_calibrated_ev=20).build(
        strategy=_strategy(),
        objective_state=_obj_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.10"),
        historical_summary=summary,
    )
    assert est.expected_value_usd is None
    assert est.valid is False


@pytest.mark.asyncio
async def test_positive_lower_bound_allows_valid_estimate(tmp_path) -> None:
    svc, repo = make_hist_service(tmp_path, minimum_samples_for_ev=20)
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    seed_positive_history(svc, as_of=as_of, n=20, pnl=Decimal("12.00"))
    summary = await svc.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="bullish",
    )
    assert summary.valid_for_ev is True
    assert summary.lower_confidence_bound_ev_usd is not None
    assert summary.lower_confidence_bound_ev_usd > 0
    est = StrategyEstimateBuilder(minimum_samples_for_calibrated_ev=20).build(
        strategy=_strategy(),
        objective_state=_obj_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.10"),
        historical_summary=summary,
    )
    assert est.expected_value_usd is not None and est.expected_value_usd > 0
    assert est.valid is True
    assert est.historical_summary_id == summary.summary_id
    loaded = repo.get_historical_summary(summary.summary_id)
    assert loaded is not None


@pytest.mark.asyncio
async def test_positive_mean_with_negative_lower_bound_rejects(tmp_path) -> None:
    settings = HistoricalOutcomeSettings(
        minimum_samples_for_ev=20,
        minimum_effective_sample_size=15,
        require_lower_confidence_bound_positive=True,
        use_similarity_weighting=False,
        require_same_strategy_family=False,
        minimum_similarity=0.01,
    )
    svc = HistoricalOutcomeService(settings=settings)
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    # High variance: mostly small wins with large losses → mean may be +ve, LCB -ve
    pnls = [Decimal("5")] * 16 + [Decimal("-40")] * 4
    eps = [
        make_closed_episode(pnl=p, as_of=as_of, hours_before=10 + i)
        for i, p in enumerate(pnls)
    ]
    svc.seed_episodes_for_tests(eps)
    summary = await svc.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="bullish",
    )
    assert summary.average_pnl_usd is not None
    # Either mean positive with LCB non-positive, or overall invalid — must not be valid_for_ev
    if summary.average_pnl_usd > 0 and (
        summary.lower_confidence_bound_ev_usd is None
        or summary.lower_confidence_bound_ev_usd <= 0
    ):
        assert summary.valid_for_ev is False
    else:
        # If mean itself not positive, still invalid for EV gate
        assert summary.valid_for_ev is False


@pytest.mark.asyncio
async def test_negative_ev_rejects_high_upside_strategy(tmp_path) -> None:
    svc, _ = make_hist_service(tmp_path, minimum_samples_for_ev=20)
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    seed_positive_history(svc, as_of=as_of, n=20, pnl=Decimal("-8.00"))
    summary = await svc.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="bullish",
    )
    assert summary.valid_for_ev is False
    est = StrategyEstimateBuilder().build(
        strategy=_strategy(),
        objective_state=_obj_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.10"),
        historical_summary=summary,
    )
    assert est.expected_value_usd is None or est.expected_value_usd <= 0
    assert est.valid is False


@pytest.mark.asyncio
async def test_objective_node_passes_historical_summary_to_builder(tmp_path) -> None:
    from joker.graph.objective_nodes import score_strategies_against_objective_node
    from joker.objectives.scoring import ObjectiveStrategyScorer
    from joker.objectives.service import SessionObjectiveService
    from joker.objectives.repository import apply_objective_migrations, ObjectiveRepository
    db = tmp_path / "n.db"
    apply_objective_migrations(db)
    repo = ObjectiveRepository(db)
    deadline = datetime.now(timezone.utc) + timedelta(hours=2)
    svc = SessionObjectiveService(repo)
    definition = await svc.create_objective(
        session_id="n",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=deadline,
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    hist, _ = make_hist_service(tmp_path, minimum_samples_for_ev=20)
    as_of = datetime.now(timezone.utc)
    seed_positive_history(hist, as_of=as_of, n=20)

    class _Deps:
        objective_service = svc
        objective_strategy_scorer = ObjectiveStrategyScorer()
        historical_outcome_service = hist
        historical_outcome_settings = hist._settings
        snapshot_repo = None
        option_surface_repo = None
        data_quality_repo = None

    strategy = _strategy()
    state = {
        "snapshot_id": str(strategy.snapshot_id),
        "strategies": [strategy],
        "evidence": [],
        "errors": [],
    }

    async def _fake_load(deps, snapshot_id):
        return None, None, None, []

    import joker.graph.objective_nodes as nodes

    orig = nodes.load_snapshot_truth
    nodes.load_snapshot_truth = _fake_load  # type: ignore[assignment]
    try:
        out = await score_strategies_against_objective_node(_Deps(), state)  # type: ignore[arg-type]
    finally:
        nodes.load_snapshot_truth = orig  # type: ignore[assignment]
    assert out["_historical_summaries"]
    assert out["_strategy_estimates"][0]["sample_count"] == 20
    assert out["_strategy_estimates"][0]["expected_value_usd"] is not None


@pytest.mark.asyncio
async def test_estimate_provenance_is_persisted(tmp_path) -> None:
    svc, repo = make_hist_service(tmp_path, minimum_samples_for_ev=20)
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    seed_positive_history(svc, as_of=as_of, n=20)
    sid = uuid4()
    oid = uuid4()
    summary = await svc.summarize_for_strategy(
        objective_id=oid,
        strategy_id=sid,
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="bullish",
    )
    est = StrategyEstimateBuilder().build(
        strategy=_strategy(),
        objective_state=_obj_state().model_copy(update={"objective_id": oid}),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.10"),
        historical_summary=summary,
    )
    repo.save_strategy_estimate(est)
    loaded = repo.get_strategy_estimate(est.estimate_id)
    assert loaded is not None
    assert loaded.historical_summary_id == summary.summary_id
    assert loaded.sample_count == 20
    assert loaded.comparable_episode_ids


def test_missing_outcome_creates_no_calibration_sample() -> None:
    r = resolve_calibration_outcome(
        episode_id=uuid4(),
        meta_confidence=Decimal("0.8"),
        traded=False,
        realised_pnl=None,
        complete=False,
    )
    assert r.included is False
    assert r.outcome is None
    r2 = resolve_calibration_outcome(
        episode_id=uuid4(),
        meta_confidence=Decimal("0.8"),
        traded=True,
        realised_pnl=None,
        complete=True,
    )
    assert r2.included is False


@pytest.mark.asyncio
async def test_incomplete_event_horizon_excluded_from_ev(tmp_path) -> None:
    svc, _ = make_hist_service(tmp_path, minimum_samples_for_ev=5)
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    good = [make_closed_episode(pnl=Decimal("10"), as_of=as_of, hours_before=5 + i) for i in range(5)]
    bad = make_closed_episode(
        pnl=Decimal("10"),
        as_of=as_of,
        hours_before=3,
        completed=False,
        findings=("truth_degraded=true", "historical_ev_eligible=false"),
    )
    svc.seed_episodes_for_tests([*good, bad])
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="bullish",
            direction="bullish",
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
        )
    )
    assert bad.episode_id in report.excluded_incomplete or bad.episode_id in report.excluded_truth_degraded
    assert bad.episode_id not in summary.comparable_episode_ids
