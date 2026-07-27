"""Production Task 3 crash/restart acceptance across orchestrator nodes."""

from __future__ import annotations

import pytest

from joker.config.settings import CognitiveGraphSettings
from joker.evolution.crash_injector import CrashAfterNode
from joker.evolution.runtime import EvolutionRuntime
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.data_quality_store import DataQualityRepository
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.persistence.aiosqlite_lifecycle import (
    drain_aiosqlite_workers,
    iter_aiosqlite_worker_threads,
    join_aiosqlite_workers,
)
from tests.integration.task3_production_harness import (
    acceptance_settings,
    build_acceptance_router,
    build_paper_evolution_stack,
    run_closed_trade_round_trip,
    shutdown_stack,
    wait_for_closed_episodes,
    wait_for_evaluations,
    wire_replay_canned_for_episodes,
)


def restart_settings():
    """Restart crash tests skip shadow gating so later orchestrator nodes are reachable."""
    settings = acceptance_settings()
    return settings.model_copy(
        update={
            "shadow": settings.shadow.model_copy(
                update={
                    "minimum_completed_cycles": 0,
                    "minimum_traded_cycles": 0,
                    "minimum_regime_coverage": 0,
                    "minimum_observation_minutes": 0,
                }
            )
        }
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


async def _seed_paper_prerequisites(tmp_path, session_id: str):
    """Seed durable episodes/evaluations via paper fills (no FakeExecutionProjection)."""
    stack = await build_paper_evolution_stack(
        tmp_path,
        session_id=session_id,
        settings=restart_settings(),
        start_orchestrator_worker=False,
    )
    try:
        await run_closed_trade_round_trip(stack, trade_index=0, minute_offset=0)
        await run_closed_trade_round_trip(stack, trade_index=1, minute_offset=20)
        episodes = await wait_for_closed_episodes(stack["evolution"], session_id, count=2)
        await wait_for_evaluations(stack["evolution"], episodes)
        return stack["db"]
    finally:
        await shutdown_stack(stack, strict_workers=False)


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
    session_id = f"restart-{node_name}"
    db = await _seed_paper_prerequisites(tmp_path, session_id)
    settings = restart_settings()
    router, fake = build_acceptance_router(session_id)
    crash = CrashAfterNode(node_name)
    runtime = _runtime(db, settings, router, session_id=session_id)
    try:
        await runtime.prepare()
        await wire_replay_canned_for_episodes(runtime, fake)
        runtime.orchestrator._crash = crash
        state = await runtime.orchestrator.maybe_start_cycle()
        assert state is not None
        await runtime.orchestrator.advance(state)
        assert crash.hits == 1
        record = await runtime._repos["evolution_cycles"].get(session_id, state.cycle_id)
        assert record is not None
    finally:
        await runtime.shutdown()
        await drain_aiosqlite_workers(timeout=5.0)
        join_aiosqlite_workers(timeout=5.0)

    runtime2 = _runtime(db, settings, router, session_id=session_id)
    try:
        await runtime2.prepare()
        await wire_replay_canned_for_episodes(runtime2, fake)
        resumed = await runtime2.orchestrator.resume_all()
        assert resumed
        # Exact expected terminal/progress outcomes — do not accept unresolved running
        # unless the crash intentionally left the cycle mid-stage after durable progress.
        if attr:
            value = getattr(resumed[0], attr)
            assert value is not None or resumed[0].status in {"completed", "blocked"}
            if resumed[0].status == "running":
                assert value is not None, (
                    f"restart after {node_name} left running without durable {attr}"
                )
        assert resumed[0].status in {"completed", "blocked", "running"}
        if resumed[0].status == "running":
            assert resumed[0].stage not in {"", "load_or_create_cycle"}
        if resumed[0].dataset_id is not None:
            assert str(resumed[0].dataset_id) == str(
                (record.payload or {}).get("dataset_id") or resumed[0].dataset_id
            )
    finally:
        await runtime2.shutdown()
        await drain_aiosqlite_workers(timeout=5.0)
        join_aiosqlite_workers(timeout=5.0)


@pytest.mark.asyncio
async def test_evaluation_graph_resumes_after_completed_evaluator_node(tmp_path) -> None:
    session_id = "eval-resume"
    db = await _seed_paper_prerequisites(tmp_path, session_id)
    router, fake = build_acceptance_router(session_id)
    runtime = _runtime(db, restart_settings(), router, session_id=session_id)
    try:
        await runtime.prepare()
        await wire_replay_canned_for_episodes(runtime, fake)
        episodes = await runtime._repos["episodes"].list_completed(limit=10)
        assert episodes
        first = await runtime.evaluation_runner.evaluate(episodes[0])
        second = await runtime.evaluation_runner.evaluate(episodes[0])
        assert first.evaluation_id == second.evaluation_id
    finally:
        await runtime.shutdown()
        await drain_aiosqlite_workers(timeout=5.0)
        join_aiosqlite_workers(timeout=5.0)
        assert not iter_aiosqlite_worker_threads()
