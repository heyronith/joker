"""Promotion gate and durable experiment resume tests."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.experiment_results_store import ExperimentEpisodeResultStore
from joker.evolution.experiment_runner import ExperimentRunner
from joker.evolution.improvement import ImprovementProposalService
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.promotion_gate import PromotionEligibilityGate
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import ExperimentDefinition, ExperimentResult, PromptPatch, TradingEpisode


def test_tail_loss_more_negative_is_regression() -> None:
    gate = PromotionEligibilityGate()
    result = ExperimentResult(
        experiment_id=uuid4(),
        champion_metrics={
            "tail_loss": Decimal("-10"),
            "calibration_error": Decimal("0.1"),
            "latency_ms": 100,
            "cost_gbp": Decimal("1"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-40"),
            "calibration_error": Decimal("0.1"),
            "latency_ms": 100,
            "cost_gbp": Decimal("1"),
        },
    )
    eligibility = gate.evaluate(
        result=result,
        completed_episode_count=100,
        holdout_episode_count=50,
        adversarial_passed=True,
    )
    assert eligibility.eligible is False
    assert "tail_loss_regression" in eligibility.gate_codes
    assert "catastrophic_tail_loss_regression" in eligibility.gate_codes


def test_missing_critical_metric_fails_closed() -> None:
    gate = PromotionEligibilityGate()
    result = ExperimentResult(
        experiment_id=uuid4(),
        champion_metrics={"calibration_error": Decimal("0.1")},
        challenger_metrics={"calibration_error": Decimal("0.1")},
    )
    eligibility = gate.evaluate(
        result=result,
        completed_episode_count=100,
        holdout_episode_count=50,
        adversarial_passed=True,
    )
    assert eligibility.eligible is False
    assert any(c.startswith("missing_critical_metric") for c in eligibility.gate_codes)


@pytest.mark.asyncio
async def test_experiment_keys_survive_process_restart(tmp_path) -> None:
    db = tmp_path / "resume.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    svc = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, challenger = await svc.propose(
        parent_champion=champ,
        weakness="w",
        hypothesis="h",
        patch=PromptPatch(
            role="meta_decision",
            parent_prompt_version_id=uuid4(),
            replacement_template="Prefer calibrated no-trade.",
            change_rationale="calibration",
        ),
    )
    episodes = [
        TradingEpisode(
            session_id="s",
            run_id="r",
            trading_date=date(2026, 7, 1),
            initial_snapshot_id=uuid4(),
            action_class="closed_trade",
            configuration_version_id=champ.configuration_version_id,
            quantity=Decimal("1"),
            realised_pnl=Decimal("1"),
            completed=True,
            idempotency_key=f"e{i}",
            created_at=datetime(2026, 7, 1, 12, 0, i % 60, tzinfo=timezone.utc),
        )
        for i in range(4)
    ]
    from joker.evaluation.dataset_builder import DatasetBuilder

    dataset = await DatasetBuilder(repos["datasets"]).build_and_persist(
        episodes, random_seed=1, minimum_holdout=1, allow_incomplete=False
    )
    definition = ExperimentDefinition(
        experiment_id=uuid4(),
        champion_version_id=champ.configuration_version_id,
        challenger_version_id=challenger.configuration_version_id,
        dataset_id=dataset.dataset_id,
        maximum_cost_gbp=Decimal("25"),
    )
    calls = {"n": 0}

    async def replay(ep, cfg_id, sample):
        calls["n"] += 1
        return {
            "realised_pnl": ep.realised_pnl or 0,
            "model_calls": 1,
            "cost_gbp": "0.01",
            "broker_submit": False,
        }

    runner1 = ExperimentRunner(repos["experiments"], repeated_samples=1, db_path=db)
    await runner1.create(definition)
    await runner1.run(
        definition.experiment_id,
        episodes=episodes,
        partition_map=dataset.partition_map,
        replay_fn=replay,
    )
    first_calls = calls["n"]
    assert first_calls > 0

    # Fresh process: new runner + store reconstruct completed keys from SQLite.
    store = ExperimentEpisodeResultStore(db)
    await store.initialize()
    keys = await store.list_keys(definition.experiment_id)
    assert len(keys) == first_calls

    runner2 = ExperimentRunner(repos["experiments"], repeated_samples=1, db_path=db)
    await runner2.run(
        definition.experiment_id,
        episodes=episodes,
        partition_map=dataset.partition_map,
        replay_fn=replay,
    )
    assert calls["n"] == first_calls
    await registry.close()
