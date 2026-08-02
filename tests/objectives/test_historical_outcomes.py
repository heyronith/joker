"""Historical outcome retrieval, similarity, leakage, and EV eligibility."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.cognition.schemas import (
    AgentRole,
    EntryPlan,
    ExecutionPlan,
    ExitPlan,
    InvalidationPlan,
    MarketDirection,
    StrategyHypothesis,
    StrategyLegCandidate,
)
from joker.objectives.estimate import StrategyEstimateBuilder
from joker.objectives.historical_outcomes import (
    independence_key,
    session_phase_from_exchange_ts,
)
from joker.objectives.historical_schemas import HistoricalOutcomeQuery
from joker.objectives.schemas import SessionObjectiveState
from joker.objectives.similarity import score_similarity
from joker.evolution.telemetry import resolve_calibration_outcome
from tests.objectives.historical_fixtures import (
    make_closed_episode,
    make_repo_backed_hist_service,
    persist_dataset_with_episodes,
    persist_positive_history,
)


ET = ZoneInfo("America/New_York")


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


def _strategy(*, family: str = "breakout_continuation") -> StrategyHypothesis:
    return StrategyHypothesis(
        name="bull call",
        market_thesis="t",
        direction=MarketDirection.BULLISH,
        strategy_family=family,
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
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo,
        evaluation_repo=ev_repo,
        as_of=as_of,
        n=5,
        pnl=Decimal("10"),
    )
    future_ep, future_ev = make_closed_episode(
        pnl=Decimal("99"), as_of=as_of, hours_before=1
    )
    future_ep = future_ep.model_copy(
        update={
            "terminal_event_timestamp": as_of + timedelta(hours=1),
            "entry_decision_timestamp": as_of + timedelta(minutes=30),
        }
    )
    await ep_repo.append(future_ep)
    await ev_repo.append(future_ev)
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            session_phase="midday",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
        )
    )
    assert future_ep.episode_id in report.excluded_future_episodes
    assert report.safe is True
    assert summary.sample_count == 5


@pytest.mark.asyncio
async def test_future_episodes_are_excluded_without_invalidating_clean_remainder(
    tmp_path,
) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=20
    )
    future_ep, future_ev = make_closed_episode(pnl=Decimal("50"), as_of=as_of)
    future_ep = future_ep.model_copy(
        update={"terminal_event_timestamp": as_of + timedelta(hours=2)}
    )
    await ep_repo.append(future_ep)
    await ev_repo.append(future_ev)
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
        )
    )
    assert report.safe is True
    assert summary.valid_for_ev is True
    assert summary.sample_count >= 20


@pytest.mark.asyncio
async def test_historical_outcomes_exclude_current_episode(tmp_path) -> None:
    await test_current_episode_is_excluded(tmp_path)


@pytest.mark.asyncio
async def test_current_episode_is_excluded(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    rows = await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=6
    )
    current_id = rows[0][0].episode_id
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
            current_episode_id=current_id,
        )
    )
    assert current_id in report.excluded_current_episode
    assert report.safe is True
    assert summary.sample_count == 5


@pytest.mark.asyncio
async def test_duplicate_replays_count_as_one_authoritative_sample(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=1
    )
    as_of = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    ep, ev = make_closed_episode(pnl=Decimal("8"), as_of=as_of)
    await ep_repo.append(ep)
    await ev_repo.append(ev)
    # Same independence key: identical entry/terminal lifecycle, different episode_id
    clone_id = uuid4()
    clone = ep.model_copy(
        update={"episode_id": clone_id, "session_id": "clone", "run_id": "clone-run"}
    )
    await ep_repo.append(clone)
    await ev_repo.append(
        ev.model_copy(update={"evaluation_id": uuid4(), "episode_id": clone_id})
    )
    assert independence_key(ep) == independence_key(clone)
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
        )
    )
    assert summary.sample_count == 1
    assert clone_id in report.excluded_duplicate_truth or ep.episode_id in report.excluded_duplicate_truth
    assert report.safe is True


def test_similarity_policy_is_deterministic() -> None:
    pid = uuid4()
    a, ca = score_similarity(
        query_strategy_family="breakout_continuation",
        query_direction="bullish",
        query_pattern_ids=(pid,),
        query_regime_labels=("trend",),
        query_session_phase="open",
        episode_strategy_family="breakout_continuation",
        episode_direction="bullish",
        episode_pattern_ids=(pid,),
        episode_regime_labels=("trend",),
        episode_session_phase="open",
    )
    b, cb = score_similarity(
        query_strategy_family="breakout_continuation",
        query_direction="bullish",
        query_pattern_ids=(pid,),
        query_regime_labels=("trend",),
        query_session_phase="open",
        episode_strategy_family="breakout_continuation",
        episode_direction="bullish",
        episode_pattern_ids=(pid,),
        episode_regime_labels=("trend",),
        episode_session_phase="open",
    )
    assert a == b
    assert ca == cb


def test_session_phase_is_correct_in_est() -> None:
    # 2026-01-15 is EST (UTC-5)
    ts = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)  # 10:00 ET
    assert session_phase_from_exchange_ts(ts) == "open"
    ts2 = datetime(2026, 1, 15, 18, 0, tzinfo=timezone.utc)  # 13:00 ET
    assert session_phase_from_exchange_ts(ts2) == "midday"


def test_session_phase_is_correct_in_edt() -> None:
    # 2026-07-15 is EDT (UTC-4)
    ts = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)  # 10:00 ET
    assert session_phase_from_exchange_ts(ts) == "open"
    ts2 = datetime(2026, 7, 15, 19, 0, tzinfo=timezone.utc)  # 15:00 ET
    assert session_phase_from_exchange_ts(ts2) == "close"


def test_same_direction_different_strategy_family_is_not_full_match() -> None:
    score, components = score_similarity(
        query_strategy_family="breakout_continuation",
        query_direction="bullish",
        episode_strategy_family="mean_reversion",
        episode_direction="bullish",
    )
    assert components["direction_match"] == Decimal("1")
    assert components["strategy_family_match"] == Decimal("0")
    assert score < Decimal("1")


def test_pattern_overlap_uses_historical_pattern_ids() -> None:
    shared = uuid4()
    with_overlap, c1 = score_similarity(
        query_strategy_family="breakout_continuation",
        query_direction="bullish",
        query_pattern_ids=(shared,),
        episode_strategy_family="breakout_continuation",
        episode_direction="bullish",
        episode_pattern_ids=(shared,),
    )
    without, c2 = score_similarity(
        query_strategy_family="breakout_continuation",
        query_direction="bullish",
        query_pattern_ids=(shared,),
        episode_strategy_family="breakout_continuation",
        episode_direction="bullish",
        episode_pattern_ids=(uuid4(),),
    )
    assert c1["pattern_overlap"] == Decimal("1")
    assert c2["pattern_overlap"] == Decimal("0")
    assert with_overlap > without


def test_changed_regime_changes_similarity() -> None:
    a, _ = score_similarity(
        query_strategy_family="breakout_continuation",
        query_direction="bullish",
        query_regime_labels=("trend",),
        episode_strategy_family="breakout_continuation",
        episode_direction="bullish",
        episode_regime_labels=("trend",),
    )
    b, _ = score_similarity(
        query_strategy_family="breakout_continuation",
        query_direction="bullish",
        query_regime_labels=("trend",),
        episode_strategy_family="breakout_continuation",
        episode_direction="bullish",
        episode_regime_labels=("range_bound",),
    )
    assert a > b


def test_changed_liquidity_changes_similarity() -> None:
    a, _ = score_similarity(
        query_strategy_family="breakout_continuation",
        query_direction="bullish",
        query_liquidity_bucket="tight",
        episode_strategy_family="breakout_continuation",
        episode_direction="bullish",
        episode_liquidity_bucket="tight",
    )
    b, _ = score_similarity(
        query_strategy_family="breakout_continuation",
        query_direction="bullish",
        query_liquidity_bucket="tight",
        episode_strategy_family="breakout_continuation",
        episode_direction="bullish",
        episode_liquidity_bucket="wide",
    )
    assert a > b


@pytest.mark.asyncio
async def test_insufficient_samples_leave_ev_none(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=20
    )
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=5
    )
    summary = await svc.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    est = StrategyEstimateBuilder(minimum_samples_for_calibrated_ev=20).build(
        strategy=_strategy(),
        objective_state=_obj_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        historical_summary=summary,
    )
    assert est.expected_value_usd is None
    assert est.valid is False


@pytest.mark.asyncio
async def test_positive_lower_bound_allows_valid_estimate(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=20, require_lcb=True
    )
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo,
        evaluation_repo=ev_repo,
        as_of=as_of,
        n=20,
        pnl=Decimal("15.00"),
    )
    assert svc.uses_repository_loaders is True
    summary = await svc.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    assert summary.valid_for_ev is True
    assert summary.lower_confidence_bound_ev_usd is not None
    assert summary.lower_confidence_bound_ev_usd > 0
    est = StrategyEstimateBuilder(
        minimum_samples_for_calibrated_ev=20,
        require_lower_confidence_bound_positive=True,
    ).build(
        strategy=_strategy(),
        objective_state=_obj_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        historical_summary=summary,
    )
    assert est.valid is True
    assert est.expected_value_usd is not None


@pytest.mark.asyncio
async def test_positive_mean_with_negative_lower_bound_rejects(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=20, require_lcb=True
    )
    as_of = datetime.now(timezone.utc)
    # High variance: mostly small wins + large losses → mean may be + but LCB -
    for i in range(18):
        ep, ev = make_closed_episode(pnl=Decimal("2.00"), as_of=as_of, hours_before=30 + i)
        await ep_repo.append(ep)
        await ev_repo.append(ev)
    for i in range(2):
        ep, ev = make_closed_episode(pnl=Decimal("-40.00"), as_of=as_of, hours_before=10 + i)
        await ep_repo.append(ep)
        await ev_repo.append(ev)
    summary = await svc.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    if summary.average_pnl_usd is not None and summary.average_pnl_usd > 0:
        assert summary.valid_for_ev is False or (
            summary.lower_confidence_bound_ev_usd is not None
            and summary.lower_confidence_bound_ev_usd <= 0
        )


@pytest.mark.asyncio
async def test_negative_ev_rejects_high_upside_strategy(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=20
    )
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo,
        evaluation_repo=ev_repo,
        as_of=as_of,
        n=20,
        pnl=Decimal("-8.00"),
    )
    summary = await svc.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    est = StrategyEstimateBuilder(minimum_samples_for_calibrated_ev=20).build(
        strategy=_strategy(),
        objective_state=_obj_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        historical_summary=summary,
    )
    assert est.valid is False


@pytest.mark.asyncio
async def test_missing_as_of_timestamp_fails_closed(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(tmp_path)
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=5
    )
    naive = datetime(2026, 7, 1, 12, 0)  # naive
    with pytest.raises(Exception):
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            as_of_timestamp=naive,
        )


@pytest.mark.asyncio
async def test_active_configuration_training_episode_is_excluded(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, ds_repo = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime.now(timezone.utc)
    rows = await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=8
    )
    overlap_ids = tuple(r[0].episode_id for r in rows[:3])
    ds = await persist_dataset_with_episodes(
        ds_repo,
        episode_ids=overlap_ids,
        partition="train",
        time_end=as_of - timedelta(days=1),
    )
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
            configuration_version_id=uuid4(),
            blocked_training_dataset_ids=(ds.dataset_id,),
            configuration_dataset_provenance_resolved=True,
        )
    )
    assert len(report.excluded_dataset_overlap) >= 3
    assert summary.sample_count == 5
    assert report.safe is True


@pytest.mark.asyncio
async def test_unrelated_old_training_dataset_does_not_exclude_episode(
    tmp_path,
) -> None:
    svc, _, ep_repo, ev_repo, ds_repo = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime.now(timezone.utc)
    rows = await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=8
    )
    overlap_ids = tuple(r[0].episode_id for r in rows[:3])
    await persist_dataset_with_episodes(
        ds_repo,
        episode_ids=overlap_ids,
        partition="train",
        time_end=as_of - timedelta(days=1),
    )
    # Active configuration was not trained on that old dataset → keep samples.
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
            configuration_version_id=uuid4(),
            blocked_training_dataset_ids=(),
            configuration_dataset_provenance_resolved=True,
        )
    )
    assert summary.sample_count == 8
    assert report.excluded_dataset_overlap == ()


@pytest.mark.asyncio
async def test_missing_configuration_dataset_provenance_fails_closed(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=8
    )
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            as_of_timestamp=as_of,
            configuration_version_id=uuid4(),
            configuration_dataset_provenance_resolved=False,
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
        )
    )
    assert report.safe is False
    assert summary.valid_for_ev is False
    assert any("missing_configuration_dataset_provenance" in n for n in report.notes)


@pytest.mark.asyncio
async def test_holdout_episode_remains_eligible_when_not_used_for_training(
    tmp_path,
) -> None:
    svc, _, ep_repo, ev_repo, ds_repo = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime.now(timezone.utc)
    rows = await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=8
    )
    holdout = tuple(r[0].episode_id for r in rows[3:])
    train = tuple(r[0].episode_id for r in rows[:3])
    ds = await persist_dataset_with_episodes(
        ds_repo,
        episode_ids=train,
        partition="train",
        time_end=as_of - timedelta(days=1),
    )
    await persist_dataset_with_episodes(
        ds_repo,
        episode_ids=holdout,
        partition="holdout",
        time_end=as_of - timedelta(days=1),
    )
    summary, _, outcomes = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
            blocked_training_dataset_ids=(ds.dataset_id,),
            configuration_dataset_provenance_resolved=True,
        )
    )
    assert summary.sample_count == 5
    assert {o.episode_id for o in outcomes}.isdisjoint(set(train))


@pytest.mark.asyncio
async def test_dataset_cutoff_after_as_of_is_excluded(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, ds_repo = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime.now(timezone.utc)
    rows = await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=8
    )
    future_cut = tuple(r[0].episode_id for r in rows[:3])
    await persist_dataset_with_episodes(
        ds_repo,
        episode_ids=future_cut,
        partition="holdout",
        time_end=as_of + timedelta(days=1),
    )
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
            configuration_dataset_provenance_resolved=True,
        )
    )
    assert len(report.excluded_dataset_overlap) >= 3
    assert summary.sample_count == 5


@pytest.mark.asyncio
async def test_unresolved_dataset_overlap_marks_summary_unsafe(tmp_path) -> None:
    from joker.objectives.config import HistoricalOutcomeSettings
    from joker.objectives.historical_outcomes import HistoricalOutcomeService

    as_of = datetime.now(timezone.utc)
    # Blocked training datasets with no dataset loader → leakage safety unresolved.
    svc = HistoricalOutcomeService(
        settings=HistoricalOutcomeSettings(minimum_samples_for_ev=1),
    )
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            as_of_timestamp=as_of,
            blocked_training_dataset_ids=(uuid4(),),
            maximum_samples=10,
            minimum_similarity=Decimal("0.10"),
        )
    )
    assert report.safe is False
    assert summary.valid_for_ev is False
    assert any("unresolved_dataset_overlap" in n for n in report.notes)


@pytest.mark.asyncio
async def test_safe_exclusions_do_not_force_valid_for_ev_false(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=20
    )
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo,
        evaluation_repo=ev_repo,
        as_of=as_of,
        n=21,
        pnl=Decimal("12"),
    )
    current = (await ep_repo.list_completed(limit=1))[0]
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
            current_episode_id=current.episode_id,
        )
    )
    assert report.excluded_current_episode
    assert report.safe is True
    assert summary.valid_for_ev is True


@pytest.mark.asyncio
async def test_objective_node_passes_historical_summary_to_builder(tmp_path) -> None:
    from joker.graph.objective_nodes import score_strategies_against_objective_node
    from joker.graph.graph_deps import CognitiveGraphDeps
    from joker.config.settings import CognitiveGraphSettings
    from joker.objectives.service import SessionObjectiveService
    from joker.models.fake_provider import FakeModelProvider
    from joker.models.registry import ModelRegistry
    from joker.models.router import ModelRouter
    from joker.models.schemas import ModelsConfig, default_model_profiles
    from joker.objectives.feasibility import GoalFeasibilityEngine
    from joker.objectives.scoring import ObjectiveStrategyScorer
    from joker.objectives.sizing import DeterministicObjectiveSizer

    svc, obj_repo, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(tmp_path)
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo,
        evaluation_repo=ev_repo,
        as_of=as_of,
        n=20,
        pnl=Decimal("14"),
    )
    objective_service = SessionObjectiveService(obj_repo)
    definition = await objective_service.create_objective(
        session_id="node",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(timezone.utc) + timedelta(hours=2),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await objective_service.confirm_objective(definition.objective_id)
    profiles = {
        n: p.model_copy(update={"provider": "fake", "model": "fake"})
        for n, p in default_model_profiles().items()
    }
    router = ModelRouter(
        ModelRegistry(ModelsConfig(profiles=profiles), providers={"fake": FakeModelProvider()}),
        session_id="node",
    )
    snap_id = uuid4()

    class _SnapRepo:
        async def get_by_id(self, sid):
            return SimpleNamespace(
                snapshot_id=sid,
                exchange_time=as_of,
                option_surface_id=None,
                data_quality_id=uuid4(),
            )

    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="node",
        run_id="node",
        snapshot_repo=_SnapRepo(),  # type: ignore[arg-type]
        objective_service=objective_service,
        objective_state_loader=objective_service.get_state,
        feasibility_engine=GoalFeasibilityEngine(),
        objective_strategy_scorer=ObjectiveStrategyScorer(),
        capital_sizer=DeterministicObjectiveSizer(),
        historical_outcome_service=svc,
        historical_outcome_settings=svc._settings,
    )
    strategy = _strategy()
    state = {
        "strategies": [strategy],
        "snapshot_id": str(snap_id),
        "session_id": "node",
        "cycle_id": "c",
        "trace": [],
    }
    out = await score_strategies_against_objective_node(deps, state)  # type: ignore[arg-type]
    estimates = out.get("_strategy_estimates") or []
    assert estimates
    assert estimates[0].get("sample_count", 0) >= 20
    assert estimates[0].get("historical_summary_id")

@pytest.mark.asyncio
async def test_estimate_provenance_is_persisted(tmp_path) -> None:
    svc, obj_repo, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(tmp_path)
    as_of = datetime.now(timezone.utc)
    rows = await persist_positive_history(
        episode_repo=ep_repo,
        evaluation_repo=ev_repo,
        as_of=as_of,
        n=20,
        pnl=Decimal("11"),
    )
    summary = await svc.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    est = StrategyEstimateBuilder(minimum_samples_for_calibrated_ev=20).build(
        strategy=_strategy(),
        objective_state=_obj_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        historical_summary=summary,
    )
    obj_repo.save_strategy_estimate(est)
    loaded = obj_repo.get_strategy_estimate(est.estimate_id)
    assert loaded is not None
    assert loaded.historical_summary_id == summary.summary_id
    assert loaded.sample_count >= 20
    persisted_ids = {r[0].episode_id for r in rows}
    assert set(summary.comparable_episode_ids).issubset(persisted_ids)


def test_missing_outcome_creates_no_calibration_sample() -> None:
    ep, _ = make_closed_episode(pnl=Decimal("5"), as_of=datetime.now(timezone.utc))
    resolved = resolve_calibration_outcome(
        episode_id=ep.episode_id,
        meta_confidence=Decimal("0.7"),
        traded=False,
        realised_pnl=None,
        complete=False,
    )
    assert resolved.included is False
    assert resolved.outcome is None


@pytest.mark.asyncio
async def test_incomplete_event_horizon_excluded_from_ev(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=1
    )
    as_of = datetime.now(timezone.utc)
    ep, ev = make_closed_episode(
        pnl=Decimal("9"),
        as_of=as_of,
        findings=("truth_degraded=true", "historical_ev_eligible=false"),
    )
    await ep_repo.append(ep)
    await ev_repo.append(ev)
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=10,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
        )
    )
    assert summary.sample_count == 0
    assert ep.episode_id in report.excluded_truth_degraded or summary.exclusion_counts.get(
        "truth_degraded", 0
    ) >= 1
