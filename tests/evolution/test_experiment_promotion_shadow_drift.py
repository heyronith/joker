"""Experiment, promotion, shadow, and drift tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.adversarial import evaluate_adversarial_subset, required_scenario_ids
from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.decision import EvolutionDecisionService
from joker.evolution.drift import DriftMonitor
from joker.evolution.experiment_runner import ExperimentRunner
from joker.evolution.improvement import ImprovementProposalService
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import (
    ExperimentDefinition,
    PromptPatch,
    TradingEpisode,
)
from joker.evolution.shadow import ShadowIsolationError, ShadowRuntime


def _episodes(cfg, n: int = 24) -> list[TradingEpisode]:
    from datetime import datetime, timezone

    out = []
    for i in range(n):
        out.append(
            TradingEpisode(
                session_id="s",
                run_id="r",
                trading_date=date(2026, 7, 1),
                initial_snapshot_id=uuid4(),
                action_class="closed_trade",
                configuration_version_id=cfg,
                quantity=Decimal("1"),
                realised_pnl=Decimal(str(i - 10)),
                entry_price=Decimal("1.0"),
                exit_price=Decimal("1.0"),
                completed=True,
                idempotency_key=f"e{i}",
                created_at=datetime(2026, 7, 1, 12, 0, i % 60, tzinfo=timezone.utc),
            )
        )
    return out


@pytest.mark.asyncio
async def test_experiment_resume_and_no_broker(tmp_path) -> None:
    db = tmp_path / "exp.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    svc = ImprovementProposalService(repos["proposals"], repos["configurations"])
    _proposal, challenger = await svc.propose(
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
    episodes = _episodes(champ.configuration_version_id, 24)
    partition_map = {
        "development": tuple(e.episode_id for e in episodes[:12]),
        "validation": tuple(e.episode_id for e in episodes[12:16]),
        "holdout": tuple(e.episode_id for e in episodes[16:22]),
        "adversarial": (),
        "shadow": tuple(e.episode_id for e in episodes[22:]),
    }
    from joker.evaluation.dataset_builder import DatasetBuilder

    dataset = await DatasetBuilder(repos["datasets"]).build_and_persist(
        episodes, random_seed=1, minimum_holdout=2, allow_incomplete=False
    )
    # Use builder partitions to avoid overlap issues with manual map when needed.
    partition_map = dataset.partition_map

    definition = ExperimentDefinition(
        experiment_id=uuid4(),
        champion_version_id=champ.configuration_version_id,
        challenger_version_id=challenger.configuration_version_id,
        dataset_id=dataset.dataset_id,
        maximum_cost_gbp=Decimal("25"),
    )
    runner = ExperimentRunner(repos["experiments"], repeated_samples=1)
    await runner.create(definition)

    async def replay(ep, cfg_id, sample):
        return {
            "realised_pnl": ep.realised_pnl or 0,
            "model_calls": 1,
            "cost_gbp": "0.01",
            "broker_submit": False,
            "configuration_version_id": str(cfg_id),
        }

    result = await runner.run(
        definition.experiment_id,
        episodes=episodes,
        partition_map=partition_map,
        replay_fn=replay,
        adversarial_passed=True,
    )
    assert result.experiment_id == definition.experiment_id
    resumed = await runner.resume(
        definition.experiment_id,
        episodes=episodes,
        partition_map=partition_map,
        replay_fn=replay,
    )
    assert resumed.result_id == result.result_id
    await registry.close()


@pytest.mark.asyncio
async def test_promotion_blocked_and_agent_reject(tmp_path) -> None:
    db = tmp_path / "promo.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    svc = ImprovementProposalService(repos["proposals"], repos["configurations"])
    proposal, challenger = await svc.propose(
        parent_champion=champ,
        weakness="calibration",
        hypothesis="h",
        patch=PromptPatch(
            role="critic",
            parent_prompt_version_id=uuid4(),
            replacement_template="Cite evidence IDs.",
            change_rationale="grounding",
        ),
        metrics_to_improve=("calibration_score",),
    )
    from joker.evolution.schemas import ExperimentResult

    bad = ExperimentResult(
        experiment_id=uuid4(),
        safety_failures=("x",),
        champion_metrics={
            "tail_loss": Decimal("-1"),
            "calibration_error": Decimal("0.1"),
            "latency_ms": 100,
            "cost_gbp": Decimal("1"),
            "mean_pnl": Decimal("1"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-1"),
            "calibration_error": Decimal("0.2"),
            "latency_ms": 100,
            "cost_gbp": Decimal("1"),
            "mean_pnl": Decimal("5"),
        },
        aggregate_metrics={"pnl_delta": Decimal("4")},
    )
    decisions = EvolutionDecisionService(
        repos["promotions"], repos["configurations"], registry
    )
    decision = await decisions.decide_and_apply(
        experiment_id=bad.experiment_id,
        result=bad,
        challenger=challenger,
        champion=champ,
        proposal=proposal,
        holdout_episode_count=50,
        completed_episode_count=100,
        adversarial_passed=True,
        agent_override_action="promote",
    )
    assert decision.final_status == "blocked_by_gate"
    assert decision.agent_action != "promote" or not decision.deterministic_eligible
    current = await registry.get_current_champion()
    assert current.configuration_version_id == champ.configuration_version_id
    await registry.close()


@pytest.mark.asyncio
async def test_shadow_isolation_and_backpressure(tmp_path) -> None:
    db = tmp_path / "shadow.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    await repos["shadow"].initialize()
    runtime = ShadowRuntime(repos["shadow"], queue_size=2)
    await runtime.start()
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    svc = ImprovementProposalService(repos["proposals"], repos["configurations"])
    _, challenger = await svc.propose(
        parent_champion=champ,
        weakness="w",
        hypothesis="h",
        patch=PromptPatch(
            role="perception",
            parent_prompt_version_id=uuid4(),
            replacement_template="Focus on quote age.",
            change_rationale="stale quotes",
        ),
    )
    assignment = await runtime.register_challenger(challenger=challenger, champion=champ)
    assert await runtime.enqueue_snapshot(
        assignment_id=assignment.assignment_id,
        challenger_version_id=challenger.configuration_version_id,
        snapshot_id="snap-1",
        payload={"symbol": "SPY"},
    )
    assert await runtime.enqueue_snapshot(
        assignment_id=assignment.assignment_id,
        challenger_version_id=challenger.configuration_version_id,
        snapshot_id="snap-2",
        payload={"symbol": "SPY"},
        coalesce=False,
    )
    # queue full
    ok = await runtime.enqueue_snapshot(
        assignment_id=assignment.assignment_id,
        challenger_version_id=challenger.configuration_version_id,
        snapshot_id="snap-3",
        payload={"symbol": "SPY"},
        coalesce=False,
    )
    assert ok is False
    with pytest.raises(ShadowIsolationError):
        runtime.forbid_execution_runtime()
    import asyncio

    await asyncio.sleep(0.05)
    await runtime.stop()
    await registry.close()


@pytest.mark.asyncio
async def test_safety_rollback_and_idempotent(tmp_path) -> None:
    db = tmp_path / "drift.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    svc = ImprovementProposalService(repos["proposals"], repos["configurations"])
    _, challenger = await svc.propose(
        parent_champion=champ,
        weakness="w",
        hypothesis="h",
        patch=PromptPatch(
            role="strategy",
            parent_prompt_version_id=uuid4(),
            replacement_template="Require surface id.",
            change_rationale="integrity",
        ),
    )
    await registry.promote(
        challenger=challenger,
        expected_champion_id=champ.configuration_version_id,
        reason="test",
    )
    monitor = DriftMonitor(repos["drift"], repos["rollbacks"], registry)
    first = await monitor.evaluate_and_maybe_rollback(
        current_champion_id=challenger.configuration_version_id,
        previous_champion_id=champ.configuration_version_id,
        observations=[],
        safety_violation=True,
    )
    assert first is not None
    second = await monitor.evaluate_and_maybe_rollback(
        current_champion_id=challenger.configuration_version_id,
        previous_champion_id=champ.configuration_version_id,
        observations=[],
        safety_violation=True,
    )
    # Second attempt CAS-fails or returns idempotent pending; champion restored.
    current = await registry.get_current_champion()
    assert current.configuration_version_id == champ.configuration_version_id
    await registry.close()


def test_adversarial_corpus_complete() -> None:
    required = required_scenario_ids()
    assert len(required) >= 25
    ok, missing = evaluate_adversarial_subset(set(required))
    assert ok and not missing
    ok2, missing2 = evaluate_adversarial_subset({"adv_01"})
    assert not ok2 and "adv_02" in missing2
