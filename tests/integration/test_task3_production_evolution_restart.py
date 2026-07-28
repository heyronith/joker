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
from joker.evolution.orchestrator import EvolutionCycleState
from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers
from tests.integration.task3_production_harness import (
    build_paper_evolution_stack,
    build_restart_evolution_runtime,
    build_router_from_fake,
    restart_settings,
    run_closed_trade_round_trip,
    shutdown_stack,
    wait_for_closed_episodes,
    wait_for_evaluations,
    wire_replay_canned_for_episodes,
)


async def _runtime(db, settings, router, session_id="restart", *, fake=None):
    runtime, _ = await build_restart_evolution_runtime(
        db,
        session_id=session_id,
        settings=settings,
        router=router,
        fake=fake,
    )
    return runtime


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
        return stack["db"], stack["fake"]
    finally:
        await shutdown_stack(stack, strict_workers=False)


def _assert_terminal_cycle(state, *, node_name: str, attr: str | None) -> None:
    assert state.status == "completed", (
        f"restart after {node_name} must reach completed, got {state.status} "
        f"stage={state.stage} failures={state.failure_codes}"
    )
    if attr is not None:
        assert getattr(state, attr) is not None, (
            f"completed cycle after {node_name} must retain durable {attr}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "node_name,attr",
    [
        ("claim_evidence", None),
        ("build_dataset", "dataset_id"),
        ("generate_improvement", "proposal_id"),
        ("register_challenger", "challenger_version_id"),
        ("run_experiment", "experiment_id"),
        ("run_adversarial_suite", "experiment_id"),
        ("run_promotion_decision", "promotion_decision_id"),
        ("apply_promotion_decision", "promotion_decision_id"),
    ],
)
async def test_production_orchestrator_restart_after_node(
    tmp_path, node_name: str, attr: str | None
) -> None:
    session_id = f"restart-{node_name}"
    db, fake = await _seed_paper_prerequisites(tmp_path, session_id)
    settings = restart_settings()
    router = build_router_from_fake(fake, session_id)
    crash = CrashAfterNode(node_name)
    runtime = await _runtime(db, settings, router, session_id=session_id, fake=fake)
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
        if attr:
            crashed_state = _cycle_from_record(record)
            assert getattr(crashed_state, attr) is not None, (
                f"crash after {node_name} must leave durable {attr}"
            )
    finally:
        await runtime.shutdown()
        await drain_aiosqlite_workers(timeout=0.5)

    runtime2 = await _runtime(db, settings, router, session_id=session_id, fake=fake)
    try:
        await runtime2.prepare()
        await wire_replay_canned_for_episodes(runtime2, fake)
        resumed_list = await runtime2.orchestrator.resume_all()
        assert resumed_list
        resumed = resumed_list[0]
        _assert_terminal_cycle(resumed, node_name=node_name, attr=attr)
        if attr and record is not None:
            persisted = (record.payload or {}).get(attr)
            resumed_val = getattr(resumed, attr)
            if persisted is not None and resumed_val is not None:
                assert str(resumed_val) == str(persisted)
        if node_name == "apply_promotion_decision":
            assert resumed.promotion_decision_id is not None
            activation = await runtime2._repos["activations"].get_by_decision_id(
                resumed.promotion_decision_id
            )
            assert activation is not None
            assert activation.completed is True
    finally:
        await runtime2.shutdown()
        await drain_aiosqlite_workers(timeout=0.5)


def _cycle_from_record(record):
    return EvolutionCycleState.from_record(record)


@pytest.mark.asyncio
async def test_evaluation_graph_resumes_after_evaluator_node_crash(tmp_path) -> None:
    """Crash after first evaluator node; fresh runtime resumes without duplicate model calls."""
    from uuid import uuid4

    from joker.evaluation import agentic_graph

    session_id = "eval-resume"
    db, fake = await _seed_paper_prerequisites(tmp_path, session_id)
    router = build_router_from_fake(fake, session_id)
    settings = restart_settings()
    runtime = await _runtime(db, settings, router, session_id=session_id, fake=fake)
    try:
        await runtime.prepare()
        await wire_replay_canned_for_episodes(runtime, fake)
        template = (await runtime._repos["episodes"].list_completed(limit=1))[0]
        episode = template.model_copy(
            update={
                "episode_id": uuid4(),
                "idempotency_key": f"eval-crash-{uuid4().hex}",
            }
        )
        await runtime._repos["episodes"].append(episode)
        existing = await runtime._repos["evaluations"].list_by_episode(episode.episode_id)
        assert not existing

        original_invoke = agentic_graph.invoke_evolution_agent
        crash_role = "evaluator_thesis"
        hits = {"count": 0}

        async def crashing_invoke(*args, **kwargs):
            role = kwargs.get("role")
            result = await original_invoke(*args, **kwargs)
            if role == crash_role:
                hits["count"] += 1
                if hits["count"] == 1:
                    raise RuntimeError("injected_crash:evaluator_thesis")
            return result

        agentic_graph.invoke_evolution_agent = crashing_invoke
        try:
            with pytest.raises(RuntimeError, match="injected_crash"):
                await runtime.evaluation_runner.evaluate(episode)
        finally:
            agentic_graph.invoke_evolution_agent = original_invoke

        persisted = await runtime._repos["evaluations"].list_by_episode(episode.episode_id)
        assert not persisted, "evaluation must not persist before graph completes"
    finally:
        await runtime.shutdown()
        await drain_aiosqlite_workers(timeout=0.5)

    runtime2 = await _runtime(db, settings, router, session_id=session_id, fake=fake)
    try:
        await runtime2.prepare()
        await wire_replay_canned_for_episodes(runtime2, fake)
        episodes = await runtime2._repos["episodes"].list_completed(limit=10)
        crash_ep = next(
            e for e in episodes if e.idempotency_key.startswith("eval-crash-")
        )
        evaluation = await runtime2.evaluation_runner.evaluate(crash_ep)
        assert evaluation.valid
        assert evaluation.evaluation_id is not None
        repo = runtime2.cognitive_graph_deps.model_call_repo
        assert repo is not None
        calls = await repo.list_by_cycle(str(crash_ep.episode_id))
        assert len(calls) >= 4

        second = await runtime2.evaluation_runner.evaluate(crash_ep)
        assert second.evaluation_id == evaluation.evaluation_id
        calls_after = await repo.list_by_cycle(str(crash_ep.episode_id))
        assert len(calls_after) == len(calls)
    finally:
        await runtime2.shutdown()
        await drain_aiosqlite_workers(timeout=0.5)
