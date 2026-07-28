"""Regression: shadow soft-pause must not replay experiment or adversarial suite."""

from __future__ import annotations

import pytest

from joker.evolution.runtime import EvolutionRuntime
from tests.integration.task3_production_harness import (
    _wait_shadow_threshold as wait_shadow_threshold,
    acceptance_settings,
    build_paper_evolution_stack,
    build_restart_evolution_runtime,
    feed_shadow_snapshots_via_market,
    run_closed_trade_round_trip,
    shutdown_stack,
    wait_for_closed_episodes,
    wait_for_evaluations,
    wire_replay_canned_for_episodes,
)


def _wrap_execution_counters(runtime: EvolutionRuntime, counter: dict[str, int]) -> None:
    if runtime.experiments is not None:
        original_resume = runtime.experiments.resume

        async def counted_resume(*args, **kwargs):
            counter["experiment"] += 1
            return await original_resume(*args, **kwargs)

        runtime.experiments.resume = counted_resume  # type: ignore[method-assign]

    if runtime.adversarial_suite is not None:
        original_adv = runtime.adversarial_suite.run_for_experiment

        async def counted_adv(*args, **kwargs):
            counter["adversarial"] += 1
            return await original_adv(*args, **kwargs)

        runtime.adversarial_suite.run_for_experiment = counted_adv  # type: ignore[method-assign]


async def _advance_until_shadow_pause(orch, state, *, max_rounds: int = 50):
    for _ in range(max_rounds):
        if state.stage == "collect_shadow_evidence" and state.status == "running":
            return state
        if state.status in {"completed", "failed", "blocked"}:
            pytest.fail(
                f"cycle reached terminal state before shadow pause: "
                f"stage={state.stage} status={state.status}"
            )
        state = await orch.advance(state)
    pytest.fail("orchestrator never reached collect_shadow_evidence")


@pytest.mark.asyncio
async def test_shadow_soft_pause_does_not_replay_experiment_or_adversarial(
    tmp_path,
) -> None:
    """Repeated ticks and a runtime restart during shadow wait run heavy nodes once."""
    session_id = "shadow-resume"
    settings = acceptance_settings()
    stack = await build_paper_evolution_stack(
        tmp_path,
        session_id=session_id,
        settings=settings,
        start_orchestrator_worker=False,
    )
    counter = {"experiment": 0, "adversarial": 0}
    evolution = stack["evolution"]
    orch = evolution.orchestrator
    assert orch is not None

    try:
        await run_closed_trade_round_trip(stack, trade_index=0, minute_offset=0)
        await run_closed_trade_round_trip(stack, trade_index=1, minute_offset=20)
        await stack["supervisor"].event_bus.drain(timeout=10.0)
        episodes = await wait_for_closed_episodes(evolution, session_id, count=2)
        await wait_for_evaluations(evolution, episodes)
        await wire_replay_canned_for_episodes(evolution, stack["fake"])

        _wrap_execution_counters(evolution, counter)
        state = await orch.maybe_start_cycle()
        assert state is not None
        state = await _advance_until_shadow_pause(orch, state)
        assert counter["experiment"] == 1
        assert counter["adversarial"] == 1

        for _ in range(6):
            state = await orch.advance(state)
            assert counter["experiment"] == 1
            assert counter["adversarial"] == 1
            assert state.stage == "collect_shadow_evidence"
            assert state.status == "running"

        cycle_id = state.cycle_id
        experiment_id = state.experiment_id
        assert experiment_id is not None

        await evolution.shutdown()

        runtime2, fake = await build_restart_evolution_runtime(
            stack["db"],
            session_id=session_id,
            settings=settings,
            router=stack["router"],
            fake=stack["fake"],
        )
        await runtime2.prepare()
        if runtime2.shadow is not None:
            await runtime2.shadow.restore_from_ledger()
            await runtime2.shadow.start()
        await wire_replay_canned_for_episodes(runtime2, fake)
        _wrap_execution_counters(runtime2, counter)
        orch2 = runtime2.orchestrator
        assert orch2 is not None

        resumed = await orch2.resume_all()
        assert resumed
        state = resumed[0]
        assert state.cycle_id == cycle_id
        assert state.stage == "collect_shadow_evidence"

        for _ in range(6):
            state = await orch2.advance(state)
            assert counter["experiment"] == 1
            assert counter["adversarial"] == 1
            assert state.stage == "collect_shadow_evidence"
            assert state.status == "running"

        stack["evolution"] = runtime2
        await feed_shadow_snapshots_via_market(stack, cycles=4)
        await wait_shadow_threshold(runtime2, timeout=30.0)

        for _ in range(20):
            state = await orch2.advance(state)
            if state.status in {"completed", "failed", "blocked"}:
                break
        else:
            pytest.fail(
                f"cycle did not terminalise after shadow feed: "
                f"stage={state.stage} status={state.status}"
            )

        assert counter["experiment"] == 1
        assert counter["adversarial"] == 1
        assert state.status == "completed"
        assert state.experiment_id == experiment_id

        adv_results = await runtime2.adversarial_suite._store.list_for_experiment(
            experiment_id
        )
        assert adv_results
    finally:
        await shutdown_stack(stack)
