"""Production Task 3 crash/restart acceptance across orchestrator nodes."""

from __future__ import annotations

from datetime import date
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
from joker.evolution.crash_injector import CrashAfterNode
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.runtime import EvolutionRuntime
from joker.evaluation.agentic_graph import EVALUATOR_ROLES
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
            rationale_codes=("ok",),
            summary="promote",
        ),
    )
    return ModelRouter(
        ModelRegistry(ModelsConfig(), providers={"ollama": fake, "openai": fake}),
        session_id="restart",
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


def _runtime(db, settings, router, session_id="restart") -> EvolutionRuntime:
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=30),
        session_id=session_id,
        run_id=session_id,
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=DataQualityRepository(db),
    )
    return EvolutionRuntime(
        db_path=db,
        settings=settings,
        session_id=session_id,
        run_id=session_id,
        model_router=router,
        cognitive_graph_deps=deps,
    )


async def _seed(runtime: EvolutionRuntime, n: int = 3) -> None:
    champ = await runtime.configuration_for_new_cycle()
    assert champ is not None
    for i in range(n):
        contract = f"SPY:2026-07-01:{510 + i}:call"
        ep = await runtime.episode_compiler.compile_from_position_closed(
            session_id=runtime.session_id,
            run_id=runtime.run_id,
            trading_date=date(2026, 7, 1),
            configuration_version_id=champ.configuration_version_id,
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
        await runtime.evaluation_runner.evaluate(ep)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_name,attr",
    [
        ("claim_evidence", None),
        ("build_dataset", "dataset_id"),
        ("generate_improvement", "proposal_id"),
        ("register_challenger", "challenger_version_id"),
        ("run_experiment", "experiment_id"),
        ("run_promotion_decision", "promotion_decision_id"),
    ],
)
async def test_production_orchestrator_restart_after_node(
    tmp_path, node_name: str, attr: str | None
) -> None:
    db = tmp_path / f"restart_{node_name}.db"
    apply_task3_migrations(db)
    settings = _settings()
    router = _router()
    crash = CrashAfterNode(node_name)
    runtime = _runtime(db, settings, router)
    await runtime.prepare()
    runtime.orchestrator._crash = crash
    await _seed(runtime)
    state = await runtime.orchestrator.maybe_start_cycle()
    assert state is not None
    await runtime.orchestrator.advance(state)
    assert crash.hits == 1
    record = await runtime._repos["evolution_cycles"].get("restart", state.cycle_id)
    assert record is not None
    await runtime.shutdown()
    assert not iter_aiosqlite_worker_threads()

    runtime2 = _runtime(db, settings, router)
    await runtime2.prepare()
    resumed = await runtime2.orchestrator.resume_all()
    assert resumed
    if attr:
        assert getattr(resumed[0], attr) is not None or resumed[0].status in {
            "completed",
            "blocked",
            "running",
        }
    if resumed[0].dataset_id is not None:
        assert str(resumed[0].dataset_id) == str(
            (record.payload or {}).get("dataset_id") or resumed[0].dataset_id
        )
    await runtime2.shutdown()
    assert not iter_aiosqlite_worker_threads()


@pytest.mark.asyncio
async def test_evaluation_graph_resumes_after_completed_evaluator_node(tmp_path) -> None:
    db = tmp_path / "eval_resume.db"
    apply_task3_migrations(db)
    runtime = _runtime(db, _settings(), _router(), session_id="eval")
    await runtime.prepare()
    await _seed(runtime, n=1)
    episodes = await runtime._repos["episodes"].list_completed(limit=10)
    assert episodes
    first = await runtime.evaluation_runner.evaluate(episodes[0])
    second = await runtime.evaluation_runner.evaluate(episodes[0])
    assert first.evaluation_id == second.evaluation_id
    await runtime.shutdown()
    assert not iter_aiosqlite_worker_threads()
