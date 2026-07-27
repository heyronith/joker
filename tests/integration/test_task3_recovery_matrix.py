"""Table-driven Task 3 recovery proofs across orchestrator, evaluation, and rollback."""

from __future__ import annotations

from decimal import Decimal

import pytest

from joker.config.settings import CognitiveGraphSettings
from joker.evolution.crash_injector import CrashAfterNode
from joker.evolution.orchestrator import EvolutionCycleState
from joker.evolution.runtime import EvolutionRuntime
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.data_quality_store import DataQualityRepository
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.persistence.aiosqlite_lifecycle import (
    drain_aiosqlite_workers,
    join_aiosqlite_workers,
)
from tests.integration.task3_production_harness import (
    acceptance_settings,
    build_paper_evolution_stack,
    build_restart_evolution_runtime,
    build_router_from_fake,
    rebuild_paper_evolution_stack,
    restart_settings,
    run_closed_trade_round_trip,
    shutdown_stack,
    wait_for_automatic_evolution,
    wait_for_closed_episodes,
    wait_for_evaluations,
    wire_replay_canned_for_episodes,
)


async def _runtime(db, settings, router, session_id: str, *, fake=None):
    runtime, _ = await build_restart_evolution_runtime(
        db,
        session_id=session_id,
        settings=settings,
        router=router,
        fake=fake,
    )
    return runtime


async def _seed_paper_db(tmp_path, session_id: str):
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
        return stack["db"], stack["fake"]
    finally:
        await shutdown_stack(stack, strict_workers=False)


async def _crash_and_resume_orchestrator(
    tmp_path,
    *,
    session_id: str,
    node_name: str,
    attr: str | None,
) -> EvolutionCycleState:
    db, fake = await _seed_paper_db(tmp_path, session_id)
    settings = restart_settings()
    router = build_router_from_fake(fake, session_id)
    crash = CrashAfterNode(node_name)

    runtime = await _runtime(db, settings, router, session_id, fake=fake)
    try:
        await runtime.prepare()
        await wire_replay_canned_for_episodes(runtime, fake)
        runtime.orchestrator._crash = crash
        state = await runtime.orchestrator.maybe_start_cycle()
        assert state is not None
        await runtime.orchestrator.advance(state)
        assert crash.hits == 1
        if attr:
            record = await runtime._repos["evolution_cycles"].get(
                session_id, state.cycle_id
            )
            assert record is not None
            crashed = EvolutionCycleState.from_record(record)
            assert getattr(crashed, attr) is not None
    finally:
        await runtime.shutdown()
        await drain_aiosqlite_workers(timeout=5.0)
        join_aiosqlite_workers(timeout=5.0)

    runtime2 = await _runtime(db, settings, router, session_id, fake=fake)
    try:
        await runtime2.prepare()
        await wire_replay_canned_for_episodes(runtime2, fake)
        resumed_list = await runtime2.orchestrator.resume_all()
        assert resumed_list
        terminal = resumed_list[0]
        assert terminal.status == "completed", (
            f"{node_name} recovery expected completed, got {terminal.status}"
        )
        if attr:
            assert getattr(terminal, attr) is not None
        return terminal
    finally:
        await runtime2.shutdown()
        await drain_aiosqlite_workers(timeout=5.0)
        join_aiosqlite_workers(timeout=5.0)


ORCHESTRATOR_RECOVERY_CASES = [
    pytest.param("build_dataset", "dataset_id", id="dataset"),
    pytest.param("run_experiment", "experiment_id", id="experiment"),
    pytest.param("run_promotion_decision", "promotion_decision_id", id="promotion"),
    pytest.param("apply_promotion_decision", "promotion_decision_id", id="activation"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("node_name,attr", ORCHESTRATOR_RECOVERY_CASES)
async def test_recovery_matrix_orchestrator(
    tmp_path, node_name: str, attr: str
) -> None:
    await _crash_and_resume_orchestrator(
        tmp_path,
        session_id=f"matrix-orch-{node_name}",
        node_name=node_name,
        attr=attr,
    )


@pytest.mark.asyncio
async def test_recovery_matrix_evaluation_model_call_dedupe(tmp_path) -> None:
    from uuid import uuid4

    from joker.evaluation import agentic_graph

    session_id = "matrix-eval"
    db, fake = await _seed_paper_db(tmp_path, session_id)
    router = build_router_from_fake(fake, session_id)
    settings = restart_settings()
    runtime = await _runtime(db, settings, router, session_id, fake=fake)
    try:
        await runtime.prepare()
        template = (await runtime._repos["episodes"].list_completed(limit=1))[0]
        episode = template.model_copy(
            update={
                "episode_id": uuid4(),
                "idempotency_key": f"matrix-eval-crash-{uuid4().hex}",
            }
        )
        await runtime._repos["episodes"].append(episode)

        original_invoke = agentic_graph.invoke_evolution_agent
        hits = {"count": 0}

        async def crashing_invoke(*args, **kwargs):
            role = kwargs.get("role")
            result = await original_invoke(*args, **kwargs)
            if role == "evaluator_calibration":
                hits["count"] += 1
                if hits["count"] == 1:
                    raise RuntimeError("injected_crash:evaluator_calibration")
            return result

        agentic_graph.invoke_evolution_agent = crashing_invoke
        try:
            with pytest.raises(RuntimeError, match="injected_crash"):
                await runtime.evaluation_runner.evaluate(episode)
        finally:
            agentic_graph.invoke_evolution_agent = original_invoke
    finally:
        await runtime.shutdown()
        await drain_aiosqlite_workers(timeout=5.0)
        join_aiosqlite_workers(timeout=5.0)

    runtime2 = await _runtime(db, settings, router, session_id, fake=fake)
    try:
        await runtime2.prepare()
        episode = next(
            e
            for e in await runtime2._repos["episodes"].list_completed(limit=10)
            if e.idempotency_key.startswith("matrix-eval-crash-")
        )
        first = await runtime2.evaluation_runner.evaluate(episode)
        second = await runtime2.evaluation_runner.evaluate(episode)
        assert first.evaluation_id == second.evaluation_id
        repo = runtime2.cognitive_graph_deps.model_call_repo
        assert repo is not None
        calls = await repo.list_by_cycle(str(episode.episode_id))
        assert len(calls) >= 4
    finally:
        await runtime2.shutdown()
        await drain_aiosqlite_workers(timeout=5.0)
        join_aiosqlite_workers(timeout=5.0)


@pytest.mark.asyncio
async def test_recovery_matrix_rollback_idempotent_after_promotion(tmp_path) -> None:
    """Promotion + rollback persisted; fresh runtime must not duplicate rollback history."""
    session_id = "matrix-rollback"
    stack = await build_paper_evolution_stack(
        tmp_path,
        session_id=session_id,
        settings=acceptance_settings(),
        start_orchestrator_worker=True,
    )
    db = stack["db"]
    try:
        evolution = stack["evolution"]
        evolution.orchestrator.pause()
        before = (await evolution.configuration_for_new_cycle()).configuration_version_id
        await run_closed_trade_round_trip(stack, trade_index=0, minute_offset=0)
        await run_closed_trade_round_trip(stack, trade_index=1, minute_offset=20)
        episodes = await wait_for_closed_episodes(evolution, session_id, count=2)
        await wait_for_evaluations(evolution, episodes)
        await wire_replay_canned_for_episodes(evolution, stack["fake"])
        evolution.orchestrator.resume_scheduling()
        evolution.wake_orchestrator(reason="matrix_rollback")
        final = await wait_for_automatic_evolution(stack)
        assert final.status == "completed"
        champ_b = (await evolution.configuration_for_new_cycle()).configuration_version_id
        assert champ_b != before

        assert evolution.drift is not None
        await evolution.drift.observe(
            configuration_version_id=champ_b,
            dimension="safety",
            baseline_value=Decimal("0"),
            observed_value=Decimal("1"),
            severity="critical",
            evidence={"reason": "matrix_rollback"},
        )
        await evolution.drift.evaluate_and_maybe_rollback(
            current_champion_id=champ_b,
            previous_champion_id=before,
            observations=[],
            safety_violation=True,
        )
        history = await evolution.champion_registry.compare_champion_history(limit=20)
        rollbacks = [
            t
            for t in history
            if t.previous_version_id == champ_b
            and t.new_version_id == before
            and str(t.reason).startswith("rollback")
        ]
        assert len(rollbacks) == 1
    finally:
        await shutdown_stack(stack, strict_workers=False)

    stack2 = await rebuild_paper_evolution_stack(
        tmp_path,
        db=db,
        session_id=session_id,
        settings=acceptance_settings(),
        start_orchestrator_worker=False,
    )
    try:
        evolution2 = stack2["evolution"]
        assert evolution2.drift is not None
        await evolution2.drift.evaluate_and_maybe_rollback(
            current_champion_id=champ_b,
            previous_champion_id=before,
            observations=[],
            safety_violation=True,
        )
        history2 = await evolution2.champion_registry.compare_champion_history(limit=20)
        rollbacks2 = [
            t
            for t in history2
            if t.previous_version_id == champ_b
            and t.new_version_id == before
            and str(t.reason).startswith("rollback")
        ]
        assert len(rollbacks2) == 1
        restored = await evolution2.configuration_for_new_cycle()
        assert restored.configuration_version_id == before
    finally:
        await shutdown_stack(stack2, strict_workers=False)
