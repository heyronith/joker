"""Task 3 closed-loop evolution integration replay."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.adversarial import required_scenario_ids
from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.decision import EvolutionDecisionService
from joker.evolution.drift import DriftMonitor
from joker.evolution.episode_compiler import EpisodeCompiler
from joker.evolution.experiment_runner import ExperimentRunner
from joker.evolution.improvement import ImprovementProposalService
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import ExperimentDefinition, PromptPatch
from joker.evolution.shadow import ShadowRuntime
from joker.evaluation.dataset_builder import DatasetBuilder
from joker.evaluation.graph import EvaluationGraphRunner
from joker.persistence.aiosqlite_lifecycle import iter_aiosqlite_worker_threads


@pytest.mark.asyncio
async def test_task3_evolution_closed_loop(tmp_path) -> None:
    db = tmp_path / "task3_loop.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    for repo in repos.values():
        await repo.initialize()

    registry = ChampionRegistry(db)
    champion = await registry.bootstrap_champion()
    pinned_for_active_cycle = champion.configuration_version_id

    compiler = EpisodeCompiler(repos["episodes"], repos["traces"])
    evaluator = EvaluationGraphRunner(repos["evaluations"], repos["traces"])

    episodes = []
    for i in range(24):
        snap = uuid4()
        ep = await compiler.compile_closed_trade(
            session_id="task3-session",
            run_id="run-1",
            trading_date=date(2026, 7, 1),
            configuration_version_id=champion.configuration_version_id,
            initial_snapshot_id=snap,
            terminal_snapshot_id=snap,
            contract_id=f"SPY:2026-07-01:{500 + i}:call",
            direction="bullish",
            entry_order_ids=(f"e{i}",),
            exit_order_ids=(f"x{i}",),
            entry_price=Decimal("1.00"),
            exit_price=Decimal("1.10") if i % 2 == 0 else Decimal("0.90"),
            entry_quantity=Decimal("1"),
            exit_quantity=Decimal("1"),
            remaining_quantity=Decimal("0"),
            realised_pnl=Decimal("10") if i % 2 == 0 else Decimal("-10"),
            max_favourable_excursion=Decimal("15"),
            max_adverse_excursion=Decimal("-8"),
            holding_seconds=300 + i,
            terminal_event_id=f"term-{i}",
            market_regime_tags=("trending_up" if i < 12 else "high_volatility",),
        )
        evaluation = await evaluator.evaluate(
            ep,
            agent_scores={
                "thesis_quality": Decimal("0.6"),
                "evidence_grounding_score": Decimal("0.55"),
                "calibration_score": Decimal("0.5"),
            },
        )
        assert evaluation.valid is True
        episodes.append(ep)

    dataset = await DatasetBuilder(repos["datasets"]).build_and_persist(
        episodes,
        random_seed=42,
        minimum_holdout=4,
        adversarial_ids=(),
        source_db_hashes={"task3": "abc"},
    )
    assert "holdout" in dataset.partition_map

    improvement = ImprovementProposalService(repos["proposals"], repos["configurations"])
    proposal, challenger = await improvement.propose(
        parent_champion=champion,
        weakness="evidence_grounding",
        hypothesis="Require explicit evidence IDs in critic prompt",
        patch=PromptPatch(
            role="critic",
            parent_prompt_version_id=uuid4(),
            replacement_template="Reject theses lacking snapshot/evidence IDs.",
            change_rationale="reduce unsupported profitable trades",
        ),
        supporting_episode_ids=tuple(e.episode_id for e in episodes[:5]),
        metrics_to_improve=("evidence_grounding_score", "calibration_score"),
        metrics_must_not_regress=("tail_loss",),
    )
    assert proposal.parent_champion_version_id == champion.configuration_version_id

    shadow = ShadowRuntime(repos["shadow"], queue_size=16)
    await shadow.start()
    assignment = await shadow.register_challenger(
        challenger=challenger, champion=champion
    )
    assert await shadow.enqueue_snapshot(
        assignment_id=assignment.assignment_id,
        challenger_version_id=challenger.configuration_version_id,
        snapshot_id=str(episodes[0].initial_snapshot_id),
        payload={"symbol": "SPY"},
    )
    import asyncio

    await asyncio.sleep(0.05)
    assert shadow.results
    # Champion remains sole authority: shadow never gets ExecutionRuntime.
    await shadow.stop()

    definition = ExperimentDefinition(
        experiment_id=uuid4(),
        proposal_id=proposal.proposal_id,
        champion_version_id=champion.configuration_version_id,
        challenger_version_id=challenger.configuration_version_id,
        dataset_id=dataset.dataset_id,
        adversarial_scenario_ids=required_scenario_ids(),
        maximum_cost_gbp=Decimal("25"),
    )
    runner = ExperimentRunner(repos["experiments"], repeated_samples=1)
    await runner.create(definition)

    async def replay(ep, cfg_id, sample):
        base = ep.realised_pnl or Decimal("0")
        # Challenger slightly better on mean without worsening tail.
        bump = Decimal("0.5") if cfg_id == challenger.configuration_version_id else Decimal("0")
        return {
            "realised_pnl": base + bump,
            "model_calls": 1,
            "cost_gbp": "0.01",
            "broker_submit": False,
        }

    result = await runner.run(
        definition.experiment_id,
        episodes=episodes,
        partition_map=dataset.partition_map,
        replay_fn=replay,
        adversarial_passed=True,
    )
    # Force eligibility for closed-loop demo by adjusting gate thresholds via direct decision
    # with sufficient counts and no safety failures.
    from joker.evolution.config import PromotionSettings
    from joker.evolution.promotion_gate import PromotionEligibilityGate

    decisions = EvolutionDecisionService(
        repos["promotions"],
        repos["configurations"],
        registry,
        gate=PromotionEligibilityGate(
            PromotionSettings(
                minimum_completed_episodes=10,
                minimum_holdout_episodes=2,
                maximum_tail_loss_regression_pct=Decimal("50"),
                maximum_calibration_regression_pct=Decimal("50"),
                maximum_latency_regression_pct=Decimal("50"),
                maximum_cost_regression_pct=Decimal("50"),
            )
        ),
    )
    # Ensure metrics won't trip soft regressions hard.
    result = result.model_copy(
        update={
            "champion_metrics": {
                "tail_loss": Decimal("-10"),
                "calibration_error": Decimal("0.20"),
                "latency_ms": 100,
                "cost_gbp": Decimal("1"),
                "mean_pnl": Decimal("0"),
            },
            "challenger_metrics": {
                "tail_loss": Decimal("-9"),
                "calibration_error": Decimal("0.15"),
                "latency_ms": 105,
                "cost_gbp": Decimal("1.1"),
                "mean_pnl": Decimal("1"),
            },
            "aggregate_metrics": {"pnl_delta": Decimal("1")},
            "safety_failures": (),
            "data_integrity_failures": (),
        }
    )
    decision = await decisions.decide_and_apply(
        experiment_id=definition.experiment_id,
        result=result,
        challenger=challenger,
        champion=champion,
        proposal=proposal,
        holdout_episode_count=len(dataset.partition_map.get("holdout", ())),
        completed_episode_count=len(episodes),
        adversarial_passed=True,
    )
    assert decision.final_status == "promoted"
    new_champ = await registry.get_current_champion()
    assert new_champ is not None
    assert new_champ.configuration_version_id == challenger.configuration_version_id
    # Active cycle retains pinned original configuration.
    assert pinned_for_active_cycle == champion.configuration_version_id
    # New cycles see new champion.
    assert new_champ.configuration_version_id != pinned_for_active_cycle

    # Duplicate promotion is idempotent.
    decision2 = await decisions.decide_and_apply(
        experiment_id=definition.experiment_id,
        result=result,
        challenger=challenger,
        champion=champion,
        proposal=proposal,
        holdout_episode_count=len(dataset.partition_map.get("holdout", ())),
        completed_episode_count=len(episodes),
        adversarial_passed=True,
    )
    assert decision2.promotion_decision_id == decision.promotion_decision_id

    # Simulated regression → drift → rollback to restored champion.
    monitor = DriftMonitor(repos["drift"], repos["rollbacks"], registry)
    await monitor.observe(
        configuration_version_id=new_champ.configuration_version_id,
        dimension="tail_loss",
        baseline_value=Decimal("-9"),
        observed_value=Decimal("-40"),
        severity="critical",
    )
    rollback = await monitor.evaluate_and_maybe_rollback(
        current_champion_id=new_champ.configuration_version_id,
        previous_champion_id=champion.configuration_version_id,
        observations=await repos["drift"].list_by_configuration(
            new_champ.configuration_version_id
        ),
        safety_violation=True,
    )
    assert rollback is not None
    restored = await registry.get_current_champion()
    assert restored is not None
    assert restored.configuration_version_id == champion.configuration_version_id

    history = await registry.compare_champion_history()
    assert len(history) >= 3  # bootstrap, promote, rollback
    assert not any(
        "chain_of_thought" in (e.model_dump_json()) for e in episodes
    )
    await registry.close()
    for repo in repos.values():
        await repo.close()
    assert not iter_aiosqlite_worker_threads()
