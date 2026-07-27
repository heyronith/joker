"""Unit tests for separated promotion decision vs champion activation."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.config import PromotionSettings
from joker.evolution.decision import EvolutionDecisionService
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.promotion_gate import PromotionEligibilityGate
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import ExperimentResult, PromptPatch
from joker.evolution.improvement import ImprovementProposalService


def _eligible_result(experiment_id):
    return ExperimentResult(
        experiment_id=experiment_id,
        champion_metrics={
            "tail_loss": Decimal("-10"),
            "latency_ms": 100,
            "cost_gbp": Decimal("1"),
            "cost_known": True,
            "mean_pnl": Decimal("0"),
            "brier_score": Decimal("0.2"),
            "expected_calibration_error": Decimal("0.1"),
            "calibration_sample_count": Decimal("20"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-9"),
            "latency_ms": 100,
            "cost_gbp": Decimal("1"),
            "cost_known": True,
            "mean_pnl": Decimal("1"),
            "brier_score": Decimal("0.2"),
            "expected_calibration_error": Decimal("0.1"),
            "calibration_sample_count": Decimal("20"),
        },
        aggregate_metrics={"pnl_delta": Decimal("1")},
        gate_rejection_codes=(),
    )


@pytest.mark.asyncio
async def test_decision_persistence_does_not_activate_champion(tmp_path) -> None:
    db = tmp_path / "dec.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    for r in repos.values():
        await r.initialize()
    registry = ChampionRegistry(db)
    champion = await registry.bootstrap_champion()
    improvement = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, challenger = await improvement.propose(
        parent_champion=champion,
        weakness="x",
        hypothesis="y",
        patch=PromptPatch(
            role="falsifier",
            parent_prompt_version_id=uuid4(),
            replacement_template="t",
            change_rationale="r",
        ),
    )
    experiment_id = uuid4()
    service = EvolutionDecisionService(
        repos["promotions"],
        repos["configurations"],
        registry,
        gate=PromotionEligibilityGate(
            PromotionSettings(
                minimum_completed_episodes=1,
                minimum_holdout_episodes=0,
                require_known_cost=False,
                minimum_calibration_samples=0,
                require_brier_score=False,
                require_expected_calibration_error=False,
            )
        ),
    )
    decision = await service.decide(
        experiment_id=experiment_id,
        result=_eligible_result(experiment_id),
        challenger=challenger,
        champion=champion,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
        agent_override_action="promote",
    )
    assert decision.final_status == "promoted"
    current = await registry.get_current_champion()
    assert current is not None
    assert current.configuration_version_id == champion.configuration_version_id


@pytest.mark.asyncio
async def test_apply_persisted_decision_activates_once(tmp_path) -> None:
    db = tmp_path / "act.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    for r in repos.values():
        await r.initialize()
    registry = ChampionRegistry(db)
    champion = await registry.bootstrap_champion()
    improvement = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, challenger = await improvement.propose(
        parent_champion=champion,
        weakness="x",
        hypothesis="y",
        patch=PromptPatch(
            role="falsifier",
            parent_prompt_version_id=uuid4(),
            replacement_template="t",
            change_rationale="r",
        ),
    )
    experiment_id = uuid4()
    service = EvolutionDecisionService(
        repos["promotions"],
        repos["configurations"],
        registry,
        gate=PromotionEligibilityGate(
            PromotionSettings(
                minimum_completed_episodes=1,
                minimum_holdout_episodes=0,
                require_known_cost=False,
                minimum_calibration_samples=0,
                require_brier_score=False,
                require_expected_calibration_error=False,
            )
        ),
    )
    decision = await service.decide(
        experiment_id=experiment_id,
        result=_eligible_result(experiment_id),
        challenger=challenger,
        champion=champion,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
        agent_override_action="promote",
    )
    await service.apply_persisted_decision(promotion_decision_id=decision.promotion_decision_id)
    again = await service.apply_persisted_decision(
        promotion_decision_id=decision.promotion_decision_id
    )
    assert again.promotion_decision_id == decision.promotion_decision_id
    current = await registry.get_current_champion()
    assert current.configuration_version_id == challenger.configuration_version_id
    history = await registry.compare_champion_history(limit=10)
    assert sum(1 for h in history if h.reason == "agent_promote") == 1


@pytest.mark.asyncio
async def test_activation_restart_reuses_existing_decision(tmp_path) -> None:
    db = tmp_path / "reuse.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    for r in repos.values():
        await r.initialize()
    registry = ChampionRegistry(db)
    champion = await registry.bootstrap_champion()
    improvement = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, challenger = await improvement.propose(
        parent_champion=champion,
        weakness="x",
        hypothesis="y",
        patch=PromptPatch(
            role="falsifier",
            parent_prompt_version_id=uuid4(),
            replacement_template="t",
            change_rationale="r",
        ),
    )
    experiment_id = uuid4()
    service = EvolutionDecisionService(
        repos["promotions"],
        repos["configurations"],
        registry,
        gate=PromotionEligibilityGate(
            PromotionSettings(
                minimum_completed_episodes=1,
                minimum_holdout_episodes=0,
                require_known_cost=False,
                minimum_calibration_samples=0,
                require_brier_score=False,
                require_expected_calibration_error=False,
            )
        ),
    )
    first = await service.decide(
        experiment_id=experiment_id,
        result=_eligible_result(experiment_id),
        challenger=challenger,
        champion=champion,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
        agent_override_action="promote",
    )
    second = await service.decide(
        experiment_id=experiment_id,
        result=_eligible_result(experiment_id),
        challenger=challenger,
        champion=champion,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
        agent_override_action="promote",
    )
    assert first.promotion_decision_id == second.promotion_decision_id


@pytest.mark.asyncio
async def test_rejected_decision_never_activates(tmp_path) -> None:
    db = tmp_path / "rej.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    for r in repos.values():
        await r.initialize()
    registry = ChampionRegistry(db)
    champion = await registry.bootstrap_champion()
    improvement = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, challenger = await improvement.propose(
        parent_champion=champion,
        weakness="x",
        hypothesis="y",
        patch=PromptPatch(
            role="falsifier",
            parent_prompt_version_id=uuid4(),
            replacement_template="t",
            change_rationale="r",
        ),
    )
    experiment_id = uuid4()
    service = EvolutionDecisionService(
        repos["promotions"],
        repos["configurations"],
        registry,
        gate=PromotionEligibilityGate(
            PromotionSettings(
                minimum_completed_episodes=1,
                minimum_holdout_episodes=0,
                require_known_cost=False,
                minimum_calibration_samples=0,
                require_brier_score=False,
                require_expected_calibration_error=False,
            )
        ),
    )
    decision = await service.decide(
        experiment_id=experiment_id,
        result=_eligible_result(experiment_id),
        challenger=challenger,
        champion=champion,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
        agent_override_action="reject",
    )
    await service.apply_persisted_decision(promotion_decision_id=decision.promotion_decision_id)
    current = await registry.get_current_champion()
    assert current.configuration_version_id == champion.configuration_version_id
