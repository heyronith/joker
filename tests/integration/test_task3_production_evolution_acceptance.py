"""Production Task 3 evolution acceptance — public runtime path, no replay stub."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.config.settings import CognitiveGraphSettings
from joker.evolution.agent_schemas import (
    EvolutionDecisionAgentOutput,
    EvaluatorAgentScores,
    ImprovementAgentProposal,
)
from joker.evolution.config import (
    DatasetSettings,
    EvolutionSettings,
    ExperimentSettings,
    OrchestratorSettings,
    PromotionSettings,
    ShadowSettings,
)
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.runtime import EvolutionRuntime
from joker.evaluation.agentic_graph import EVALUATOR_ROLES
from joker.events.bus import InProcessAsyncEventBus
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.data_quality_store import DataQualityRepository
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig
from joker.persistence.aiosqlite_lifecycle import iter_aiosqlite_worker_threads
from tests.evolution.projection_helpers import (
    FakeExecutionProjection,
    closed_trade_projection,
)


def _router() -> ModelRouter:
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
    proposal = ImprovementAgentProposal(
        weakness="evidence_grounding",
        hypothesis="Require explicit evidence IDs",
        patch_type="prompt",
        role="falsifier",
        replacement_template="Reject theses lacking snapshot/evidence IDs.",
        change_rationale="grounding",
        metrics_to_improve=("evidence_grounding_score",),
        metrics_must_not_regress=("tail_loss",),
        critic_accepted=True,
    )
    fake.set_canned_for_role("improvement_proposer", proposal)
    fake.set_canned_for_role("improvement_critic", proposal)
    fake.set_canned_for_role(
        "evolution_decision",
        EvolutionDecisionAgentOutput(
            action="promote",
            rationale_codes=("improved_grounding",),
            summary="promote challenger",
        ),
    )
    return ModelRouter(
        ModelRegistry(ModelsConfig(), providers={"ollama": fake, "openai": fake}),
        session_id="accept",
    )


def _settings() -> EvolutionSettings:
    return EvolutionSettings(
        enabled=True,
        datasets=DatasetSettings(
            minimum_episode_count=2,
            minimum_holdout_count=1,
            minimum_regime_count=1,
        ),
        experiments=ExperimentSettings(repeated_samples=1),
        promotion=PromotionSettings(
            minimum_completed_episodes=2,
            minimum_holdout_episodes=1,
            maximum_tail_loss_regression_pct=Decimal("100"),
            maximum_calibration_regression_pct=Decimal("100"),
            maximum_latency_regression_pct=Decimal("100"),
            maximum_cost_regression_pct=Decimal("100"),
        ),
        shadow=ShadowSettings(
            minimum_completed_cycles=0,
            minimum_traded_cycles=0,
            minimum_regime_coverage=0,
            minimum_observation_minutes=0,
        ),
        orchestrator=OrchestratorSettings(
            enabled=True,
            minimum_new_completed_episodes=2,
            minimum_new_evaluations=2,
            minimum_holdout_episodes=1,
        ),
    )


@pytest.mark.asyncio
async def test_task3_production_evolution_acceptance(tmp_path) -> None:
    """Public orchestrator path with durable saver, claims, adversarial, shadow gate.

    Episodes are compiled via EpisodeCompiler + lifecycle-stamped projections
    (same identity path as POSITION_CLOSED), then evaluated and advanced only
    through EvolutionRuntime public APIs. No replay_fn stub is injected.
    """
    db = tmp_path / "accept.db"
    apply_task3_migrations(db)
    router = _router()
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=30),
        session_id="accept",
        run_id="accept",
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=DataQualityRepository(db),
    )
    runtime = EvolutionRuntime(
        db_path=db,
        settings=_settings(),
        session_id="accept",
        run_id="accept",
        event_bus=InProcessAsyncEventBus(),
        model_router=router,
        cognitive_graph_deps=deps,
    )
    await runtime.prepare()
    assert runtime.orchestrator is not None
    assert runtime.orchestrator._checkpointer is not None
    assert runtime.evidence_claims is not None
    assert runtime.adversarial_suite is not None
    assert runtime.shadow_ledger is not None

    champ = await runtime.configuration_for_new_cycle()
    assert champ is not None
    before = champ.configuration_version_id

    # Compile lifecycle-faithful episodes (production compiler path).
    for i in range(3):
        contract = f"SPY:2026-07-01:{500 + i}:call"
        ep = await runtime.episode_compiler.compile_from_position_closed(
            session_id="accept",
            run_id="accept",
            trading_date=date(2026, 7, 1),
            configuration_version_id=before,
            event_payload={
                "contract_id": contract,
                "client_order_id": f"exit-{i}",
                "realized_pnl": "50",
            },
            event_id=str(uuid4()),
            execution=FakeExecutionProjection(
                closed_trade_projection(
                    contract_id=contract,
                    entry_id=f"entry-{i}",
                    exit_id=f"exit-{i}",
                    realised_pnl=Decimal("50"),
                )
            ),
            initial_snapshot_id=uuid4(),
        )
        assert ep.completed is True
        assert ep.position_lifecycle_id is not None
        evaluation = await runtime.evaluation_runner.evaluate(ep)
        assert evaluation.valid is True

    state = await runtime.orchestrator.maybe_start_cycle()
    assert state is not None
    # Evidence claims must exist after claim node runs via advance.
    final = await runtime.orchestrator.advance(state)
    assert final.dataset_id is not None
    assert final.proposal_id is not None
    assert final.challenger_version_id is not None
    assert final.experiment_id is not None

    definition = await runtime._repos["experiments"].get_definition(final.experiment_id)
    assert definition is not None
    assert len(definition.adversarial_scenario_ids) == 25

    claims = await runtime.evidence_claims.list_by_cycle(final.cycle_id)
    assert claims
    assert final.adversarial_passed is not None

    # Production replay service is wired (no stub).
    assert runtime.experiments._replay_service is runtime.replay

    after = await runtime.configuration_for_new_cycle()
    assert after is not None
    applied = await runtime.pin_and_apply_for_cycle("post-accept")
    assert applied is not None

    # Completed evaluations cannot be auto-reclaimed.
    owned = await runtime.evidence_claims.list_unclaimed_evaluation_ids()
    assert owned

    await runtime.shutdown()
    assert not iter_aiosqlite_worker_threads()
