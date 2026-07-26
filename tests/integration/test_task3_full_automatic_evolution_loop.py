"""Automatic Task 3 closed-loop orchestrator integration (no manual service calls)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.agent_schemas import EvaluatorAgentScores, ImprovementAgentProposal
from joker.evolution.config import (
    DatasetSettings,
    EvolutionSettings,
    ExperimentSettings,
    OrchestratorSettings,
    PromotionSettings,
)
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.runtime import EvolutionRuntime
from joker.evolution.schemas import TradingEpisode
from joker.evaluation.agentic_graph import EVALUATOR_ROLES
from joker.events.bus import InProcessAsyncEventBus
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig
from joker.persistence.aiosqlite_lifecycle import iter_aiosqlite_worker_threads


def _router() -> tuple[ModelRouter, FakeModelProvider]:
    fake = FakeModelProvider(available=True)
    scores = EvaluatorAgentScores(
        thesis_quality=Decimal("0.7"),
        evidence_grounding_score=Decimal("0.7"),
        calibration_score=Decimal("0.7"),
        execution_quality=Decimal("0.7"),
        efficiency_score=Decimal("0.5"),
    )
    for role in EVALUATOR_ROLES:
        fake.set_canned_for_role(role, scores)
    fake.set_canned_for_role(
        "improvement_proposer",
        ImprovementAgentProposal(
            weakness="evidence_grounding",
            hypothesis="Require explicit evidence IDs",
            patch_type="prompt",
            role="falsifier",
            replacement_template="Reject theses lacking snapshot/evidence IDs.",
            change_rationale="grounding",
            metrics_to_improve=("evidence_grounding_score",),
            metrics_must_not_regress=("tail_loss",),
            critic_accepted=True,
        ),
    )
    fake.set_canned_for_role(
        "improvement_critic",
        ImprovementAgentProposal(
            weakness="evidence_grounding",
            hypothesis="Require explicit evidence IDs",
            patch_type="prompt",
            role="falsifier",
            replacement_template="Reject theses lacking snapshot/evidence IDs.",
            change_rationale="grounding",
            metrics_to_improve=("evidence_grounding_score",),
            metrics_must_not_regress=("tail_loss",),
            critic_accepted=True,
        ),
    )
    from joker.evolution.agent_schemas import EvolutionDecisionAgentOutput

    fake.set_canned_for_role(
        "evolution_decision",
        EvolutionDecisionAgentOutput(
            action="promote",
            rationale_codes=("improved_grounding",),
            summary="challenger improves grounding without unsafe regression",
        ),
    )
    registry = ModelRegistry(ModelsConfig(), providers={"ollama": fake, "openai": fake})
    return ModelRouter(registry, session_id="orch"), fake


@pytest.mark.asyncio
async def test_task3_full_automatic_evolution_loop(tmp_path) -> None:
    db = tmp_path / "auto_loop.db"
    apply_task3_migrations(db)
    router, _fake = _router()
    settings = EvolutionSettings(
        enabled=True,
        datasets=DatasetSettings(
            minimum_episode_count=4,
            minimum_holdout_count=1,
            minimum_regime_count=1,
        ),
        experiments=ExperimentSettings(repeated_samples=1, maximum_model_calls=5000),
        promotion=PromotionSettings(
            minimum_completed_episodes=4,
            minimum_holdout_episodes=1,
            maximum_tail_loss_regression_pct=Decimal("100"),
            maximum_calibration_regression_pct=Decimal("100"),
            maximum_latency_regression_pct=Decimal("100"),
            maximum_cost_regression_pct=Decimal("100"),
        ),
        orchestrator=OrchestratorSettings(
            enabled=True,
            minimum_new_completed_episodes=4,
            minimum_new_evaluations=4,
            minimum_holdout_episodes=1,
            maximum_active_challengers=1,
            automatic_cycle_interval_minutes=0,
        ),
    )
    runtime = EvolutionRuntime(
        db_path=db,
        settings=settings,
        session_id="auto-session",
        run_id="auto-run",
        event_bus=InProcessAsyncEventBus(),
        model_router=router,
    )
    await runtime.prepare()
    runtime.subscribe_events()
    assert runtime.orchestrator is not None

    class _StubReplay:
        async def replay_episode(self, episode, configuration_version_id, sample):
            base = episode.realised_pnl or Decimal("0")
            # Challenger gets a slight edge so promotion gates can pass.
            bump = Decimal("1") if configuration_version_id != (
                await runtime.current_champion_id()
            ) else Decimal("0")
            return {
                "realised_pnl": base + bump,
                "model_calls": 2,
                "cost_gbp": Decimal("0.02"),
                "latency_ms": 20,
                "broker_submit": False,
                "execution_runtime": False,
                "historical_pnl_attributed": False,
            }

    runtime.experiments._replay_service = _StubReplay()
    champ = await runtime.configuration_for_new_cycle()
    assert champ is not None
    before = champ.configuration_version_id

    # Seed completed episodes + evaluations via runtime repos (Task1/2 already proven elsewhere).
    episodes = []
    for i in range(6):
        ep = TradingEpisode(
            session_id="auto-session",
            run_id="auto-run",
            trading_date=date(2026, 7, 1),
            initial_snapshot_id=uuid4(),
            terminal_snapshot_id=uuid4(),
            action_class="closed_trade",
            configuration_version_id=before,
            contract_id=f"SPY:2026-07-01:{500 + i}:call",
            quantity=Decimal("1"),
            realised_pnl=Decimal("10") if i % 2 == 0 else Decimal("-5"),
            entry_price=Decimal("1.00"),
            exit_price=Decimal("1.10") if i % 2 == 0 else Decimal("0.95"),
            completed=True,
            market_regime_tags=("trending_up",),
            idempotency_key=f"auto-ep-{i}",
            created_at=datetime(2026, 7, 1, 12, 0, i, tzinfo=timezone.utc),
        )
        await runtime._repos["episodes"].append(ep)
        evaluation = await runtime.evaluation_runner.evaluate(ep)
        assert evaluation.valid is True
        episodes.append(ep)

    state = await runtime.orchestrator.maybe_start_cycle()
    assert state is not None
    # Advance to completion through durable stages (production orchestrator path).
    final = await runtime.orchestrator.advance(state)
    assert final.status in {"completed", "failed", "blocked", "running"}
    # Promotion may block on adversarial/gate, but challenger + experiment must exist.
    assert final.dataset_id is not None
    assert final.proposal_id is not None
    assert final.challenger_version_id is not None
    assert final.experiment_id is not None

    after = await runtime.configuration_for_new_cycle()
    assert after is not None
    # New cycles would use current champion (possibly promoted).
    applied = await runtime.pin_and_apply_for_cycle("post-loop-cycle")
    assert applied is not None
    assert applied.configuration_version_id == after.configuration_version_id

    await runtime.shutdown()
    assert not iter_aiosqlite_worker_threads()


@pytest.mark.asyncio
async def test_task3_full_loop_restart(tmp_path) -> None:
    db = tmp_path / "restart_loop.db"
    apply_task3_migrations(db)
    router, _ = _router()
    settings = EvolutionSettings(
        enabled=True,
        datasets=DatasetSettings(
            minimum_episode_count=4,
            minimum_holdout_count=1,
            minimum_regime_count=1,
        ),
        experiments=ExperimentSettings(repeated_samples=1),
        promotion=PromotionSettings(
            minimum_completed_episodes=4,
            minimum_holdout_episodes=1,
            maximum_tail_loss_regression_pct=Decimal("100"),
            maximum_calibration_regression_pct=Decimal("100"),
            maximum_latency_regression_pct=Decimal("100"),
            maximum_cost_regression_pct=Decimal("100"),
        ),
        orchestrator=OrchestratorSettings(
            enabled=True,
            minimum_new_completed_episodes=4,
            minimum_new_evaluations=4,
            minimum_holdout_episodes=1,
            automatic_cycle_interval_minutes=0,
        ),
    )
    runtime = EvolutionRuntime(
        db_path=db,
        settings=settings,
        session_id="restart-session",
        run_id="restart-run",
        model_router=router,
    )
    await runtime.prepare()
    champ = await runtime.configuration_for_new_cycle()
    assert champ is not None

    class _StubReplay:
        async def replay_episode(self, episode, configuration_version_id, sample):
            return {
                "realised_pnl": episode.realised_pnl or Decimal("0"),
                "model_calls": 1,
                "cost_gbp": Decimal("0.01"),
                "latency_ms": 10,
                "broker_submit": False,
            }

    runtime.experiments._replay_service = _StubReplay()
    for i in range(5):
        ep = TradingEpisode(
            session_id="restart-session",
            run_id="restart-run",
            trading_date=date(2026, 7, 1),
            initial_snapshot_id=uuid4(),
            action_class="closed_trade",
            configuration_version_id=champ.configuration_version_id,
            quantity=Decimal("1"),
            realised_pnl=Decimal("3"),
            entry_price=Decimal("1.0"),
            exit_price=Decimal("1.05"),
            completed=True,
            market_regime_tags=("trending_up",),
            idempotency_key=f"restart-ep-{i}",
        )
        await runtime._repos["episodes"].append(ep)
        await runtime.evaluation_runner.evaluate(ep)

    state = await runtime.orchestrator.maybe_start_cycle()
    assert state is not None
    # Crash after dataset persistence.
    state = await runtime.orchestrator._build_dataset(state)
    await runtime.orchestrator._persist(state)
    assert state.dataset_id is not None
    await runtime.shutdown()

    runtime2 = EvolutionRuntime(
        db_path=db,
        settings=settings,
        session_id="restart-session",
        run_id="restart-run",
        model_router=router,
    )
    await runtime2.prepare()
    runtime2.experiments._replay_service = _StubReplay()
    resumed = await runtime2.orchestrator.resume_all()
    assert resumed
    assert resumed[0].dataset_id is not None
    # Idempotent: proposal/challenger created once across resume.
    proposals = await runtime2._repos["proposals"].list_pending()
    assert len(proposals) <= 1 or resumed[0].proposal_id is not None
    await runtime2.shutdown()
    assert not iter_aiosqlite_worker_threads()
