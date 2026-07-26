"""Episode compiler and evaluation tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evaluation.dataset_builder import DatasetBuilder, DatasetBuilderError
from joker.evaluation.graph import EvaluationGraphRunner
from joker.evaluation.metrics import compute_deterministic_metrics
from joker.evolution.episode_compiler import EpisodeCompiler
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import TradingEpisode


@pytest.mark.asyncio
async def test_closed_trade_and_no_trade_episodes(tmp_path) -> None:
    db = tmp_path / "ep.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    await repos["episodes"].initialize()
    await repos["traces"].initialize()
    compiler = EpisodeCompiler(repos["episodes"], repos["traces"])
    cfg = uuid4()
    snap = uuid4()
    closed = await compiler.compile_closed_trade(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=cfg,
        initial_snapshot_id=snap,
        terminal_snapshot_id=snap,
        contract_id="SPY:2026-07-01:500:call",
        direction="bullish",
        entry_order_ids=("e1",),
        exit_order_ids=("x1",),
        entry_price=Decimal("1.00"),
        exit_price=Decimal("1.50"),
        entry_quantity=Decimal("1"),
        exit_quantity=Decimal("1"),
        remaining_quantity=Decimal("0"),
        realised_pnl=Decimal("50"),
        max_favourable_excursion=Decimal("60"),
        max_adverse_excursion=Decimal("-10"),
        holding_seconds=600,
        terminal_event_id="term-1",
    )
    assert closed.completed is True
    assert closed.action_class == "closed_trade"
    metrics = compute_deterministic_metrics(closed)
    assert metrics.realised_pnl == Decimal("50")
    assert metrics.profit_capture_ratio is not None

    incomplete = await compiler.compile_closed_trade(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=cfg,
        initial_snapshot_id=snap,
        terminal_snapshot_id=None,
        contract_id="SPY:2026-07-01:501:call",
        direction="bullish",
        entry_order_ids=("e2",),
        exit_order_ids=("x2",),
        entry_price=Decimal("1.00"),
        exit_price=Decimal("0.50"),
        entry_quantity=Decimal("2"),
        exit_quantity=Decimal("1"),
        remaining_quantity=Decimal("0"),  # mismatched vs entry-exit
        realised_pnl=Decimal("-50"),
        terminal_event_id="term-2",
    )
    assert incomplete.completed is False
    assert "quantity_identity_mismatch" in incomplete.completeness_findings

    no_trade = await compiler.compile_no_trade(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=cfg,
        initial_snapshot_id=snap,
        terminal_event_id="term-3",
        rejection_codes=("low_confidence",),
        decision_rationale="insufficient evidence",
        confidence_values={"meta": Decimal("0.4")},
    )
    assert no_trade.action_class == "no_trade"
    trace = await repos["traces"].get_by_episode(no_trade.episode_id)
    assert trace is not None
    assert "chain_of_thought" not in trace.model_dump()


@pytest.mark.asyncio
async def test_profitable_unsupported_vs_losing_calibrated(tmp_path) -> None:
    db = tmp_path / "eval.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    await repos["evaluations"].initialize()
    runner = EvaluationGraphRunner(repos["evaluations"])
    cfg = uuid4()
    good_loss = TradingEpisode(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        initial_snapshot_id=uuid4(),
        action_class="closed_trade",
        configuration_version_id=cfg,
        quantity=Decimal("1"),
        realised_pnl=Decimal("-20"),
        entry_price=Decimal("1.0"),
        exit_price=Decimal("0.8"),
        completed=True,
        idempotency_key="loss",
    )
    eval_loss = await runner.evaluate(
        good_loss,
        agent_scores={
            "thesis_quality": Decimal("0.8"),
            "evidence_grounding_score": Decimal("0.9"),
            "calibration_score": Decimal("0.85"),
        },
    )
    assert eval_loss.valid is True
    assert eval_loss.thesis_quality == Decimal("0.8")

    bad_profit = TradingEpisode(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        initial_snapshot_id=uuid4(),
        action_class="closed_trade",
        configuration_version_id=cfg,
        quantity=Decimal("1"),
        realised_pnl=Decimal("40"),
        entry_price=Decimal("1.0"),
        exit_price=Decimal("1.4"),
        completed=True,
        idempotency_key="profit",
    )
    eval_profit = await runner.evaluate(
        bad_profit,
        agent_scores={
            "thesis_quality": Decimal("0.2"),
            "evidence_grounding_score": Decimal("0.1"),
        },
    )
    assert eval_profit.outcome_quality is not None
    assert eval_profit.evidence_grounding_score == Decimal("0.1")
    assert "chain_of_thought" not in eval_profit.model_dump()


@pytest.mark.asyncio
async def test_dataset_no_overlap_and_leakage_rejection(tmp_path) -> None:
    db = tmp_path / "ds.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    builder = DatasetBuilder(repos["datasets"])
    cfg = uuid4()
    episodes = [
        TradingEpisode(
            session_id="s",
            run_id="r",
            trading_date=date(2026, 7, 1),
            initial_snapshot_id=uuid4(),
            action_class="no_trade",
            configuration_version_id=cfg,
            completed=True,
            idempotency_key=f"k{i}",
            created_at=datetime_for(i),
            market_regime_tags=("trending_up",),
        )
        for i in range(12)
    ]
    dataset = await builder.build_and_persist(episodes, random_seed=7, minimum_holdout=2)
    seen = set()
    for ids in dataset.partition_map.values():
        for eid in ids:
            assert eid not in seen
            seen.add(eid)
    future = TradingEpisode(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 2),
        initial_snapshot_id=uuid4(),
        action_class="no_trade",
        configuration_version_id=cfg,
        completed=True,
        idempotency_key="future",
        created_at=datetime_for(100),
    )
    with pytest.raises(DatasetBuilderError, match="temporal leakage"):
        builder.build(
            [*episodes, future],
            proposal_time=datetime_for(50),
            minimum_holdout=1,
        )


def datetime_for(i: int):
    from datetime import datetime, timedelta, timezone

    return datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc) + timedelta(minutes=i)
