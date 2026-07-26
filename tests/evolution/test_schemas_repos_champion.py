"""Task 3 schema, repository, champion, and gate unit tests."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.champion_registry import ChampionRegistry, ChampionRegistryError
from joker.evolution.hashing import hash_model
from joker.evolution.improvement import ImprovementError, ImprovementProposalService
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.promotion_gate import PromotionEligibilityGate
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import (
    PROHIBITED_MUTATION_TARGETS,
    ExperimentResult,
    PromptPatch,
    TradingEpisode,
    assert_no_chain_of_thought,
)
from joker.persistence.aiosqlite_lifecycle import iter_aiosqlite_worker_threads


def test_assert_no_chain_of_thought_rejects_hidden_keys() -> None:
    with pytest.raises(ValueError, match="chain-of-thought"):
        assert_no_chain_of_thought({"chain_of_thought": "secret"})


def test_prohibited_mutation_targets_cover_safety_surfaces() -> None:
    assert "order_action_gateway" in PROHIBITED_MUTATION_TARGETS
    assert "live_money_flags" in PROHIBITED_MUTATION_TARGETS


@pytest.mark.asyncio
async def test_episode_append_idempotent_and_hash_stable(tmp_path) -> None:
    db = tmp_path / "t3.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    await repos["episodes"].initialize()
    cfg = uuid4()
    ep = TradingEpisode(
        session_id="s1",
        run_id="r1",
        trading_date=date(2026, 7, 1),
        initial_snapshot_id=uuid4(),
        action_class="no_trade",
        configuration_version_id=cfg,
        completed=True,
        idempotency_key="ep-key-1",
    )
    assert await repos["episodes"].append(ep) is True
    assert await repos["episodes"].append(ep) is False
    loaded = await repos["episodes"].get_by_id(ep.episode_id)
    assert loaded is not None
    assert loaded.idempotency_key == "ep-key-1"
    await repos["episodes"].close()
    assert not iter_aiosqlite_worker_threads()


@pytest.mark.asyncio
async def test_champion_cas_and_uniqueness(tmp_path) -> None:
    db = tmp_path / "champ.db"
    registry = ChampionRegistry(db)
    first = await registry.bootstrap_champion()
    second = await registry.bootstrap_champion()
    assert first.configuration_version_id == second.configuration_version_id
    history = await registry.compare_champion_history()
    assert len(history) == 1

    repos = build_evolution_repositories(db)
    await repos["configurations"].initialize()
    svc = ImprovementProposalService(repos["proposals"], repos["configurations"], registry.policy_store)
    proposal, challenger = await svc.propose(
        parent_champion=first,
        weakness="calibration",
        hypothesis="tighten critic",
        patch=PromptPatch(
            role="critic",
            parent_prompt_version_id=uuid4(),
            replacement_template="Be stricter about evidence IDs.",
            change_rationale="reduce unsupported theses",
        ),
    )
    assert proposal.status == "registered"
    transition = await registry.promote(
        challenger=challenger,
        expected_champion_id=first.configuration_version_id,
        reason="test",
    )
    assert transition.new_version_id == challenger.configuration_version_id
    with pytest.raises(ChampionRegistryError, match="CAS"):
        await registry.promote(
            challenger=challenger,
            expected_champion_id=first.configuration_version_id,
            reason="stale",
        )
    current = await registry.get_current_champion()
    assert current is not None
    assert current.configuration_version_id == challenger.configuration_version_id
    await registry.close()


@pytest.mark.asyncio
async def test_improvement_rejects_prohibited_mutation(tmp_path) -> None:
    db = tmp_path / "mut.db"
    repos = build_evolution_repositories(db)
    await repos["configurations"].initialize()
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    svc = ImprovementProposalService(repos["proposals"], repos["configurations"], registry.policy_store)
    with pytest.raises(ImprovementError):
        await svc.propose(
            parent_champion=champ,
            weakness="x",
            hypothesis="y",
            patch={"patch_type": "prompt", "mutation_target": "order_action_gateway"},
        )
    await registry.close()


def test_promotion_gate_blocks_safety_and_insufficient_samples() -> None:
    gate = PromotionEligibilityGate()
    result = ExperimentResult(
        experiment_id=uuid4(),
        safety_failures=("duplicate_broker_action",),
        champion_metrics={"tail_loss": Decimal("-1"), "calibration_error": Decimal("0.1"), "latency_ms": 100, "cost_gbp": Decimal("1")},
        challenger_metrics={"tail_loss": Decimal("-1"), "calibration_error": Decimal("0.1"), "latency_ms": 100, "cost_gbp": Decimal("1")},
    )
    eligibility = gate.evaluate(
        result=result,
        completed_episode_count=100,
        holdout_episode_count=50,
        adversarial_passed=True,
    )
    assert eligibility.eligible is False
    assert "safety_violation" in eligibility.gate_codes


@pytest.mark.asyncio
async def test_migration_idempotent(tmp_path) -> None:
    db = tmp_path / "mig.db"
    apply_task3_migrations(db)
    apply_task3_migrations(db)
    assert db.exists()
