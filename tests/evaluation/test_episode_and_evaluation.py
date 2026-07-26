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
from tests.evolution.projection_helpers import (
    FakeExecutionProjection,
    closed_trade_projection,
)


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
    contract = "SPY:2026-07-01:500:call"
    event_id = str(uuid4())
    closed = await compiler.compile_from_position_closed(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=cfg,
        event_payload={
            "contract_id": contract,
            "client_order_id": "exit-1",
            "realized_pnl": "50",
        },
        event_id=event_id,
        execution=FakeExecutionProjection(
            closed_trade_projection(
                contract_id=contract,
                realised_pnl=Decimal("50"),
                entry_price=Decimal("1.00"),
                exit_price=Decimal("1.50"),
            )
        ),
        initial_snapshot_id=snap,
    )
    assert closed.completed is True
    assert closed.action_class == "closed_trade"
    assert closed.realised_pnl == Decimal("50")
    metrics = compute_deterministic_metrics(closed)
    assert metrics.realised_pnl == Decimal("50")

    incomplete = await compiler.compile_from_position_closed(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=cfg,
        event_payload={
            "contract_id": "SPY:2026-07-01:501:call",
            "client_order_id": "exit-2",
            "realized_pnl": "-50",
        },
        event_id=str(uuid4()),
        execution=FakeExecutionProjection(
            closed_trade_projection(
                contract_id="SPY:2026-07-01:501:call",
                entry_id="e2",
                exit_id="x2",
                qty=Decimal("2"),
                remaining_mismatch=True,
                realised_pnl=Decimal("-50"),
                exit_price=Decimal("0.50"),
            )
        ),
        initial_snapshot_id=snap,
    )
    assert incomplete.completed is False
    assert "quantity_identity_mismatch" in incomplete.completeness_findings

    no_trade = await compiler.compile_from_no_trade_cycle(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=cfg,
        cycle_id="cycle-3",
        snapshot_id=snap,
        event_id=str(uuid4()),
        outcome="no_trade",
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
