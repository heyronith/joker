"""Production promotion → drift → rollback → restored champion use."""

from __future__ import annotations

from decimal import Decimal

import pytest

from joker.persistence.aiosqlite_lifecycle import iter_aiosqlite_worker_threads
from tests.integration.task3_production_harness import (
    acceptance_settings,
    build_paper_evolution_stack,
    run_closed_trade_round_trip,
    shutdown_stack,
    wait_for_automatic_evolution,
    wait_for_closed_episodes,
    wait_for_evaluations,
    wire_replay_canned_for_episodes,
)


@pytest.mark.asyncio
async def test_full_loop_drift_and_rollback(tmp_path) -> None:
    session_id = "rollback"
    stack = await build_paper_evolution_stack(
        tmp_path,
        session_id=session_id,
        settings=acceptance_settings(),
        start_orchestrator_worker=True,
    )
    try:
        evolution = stack["evolution"]
        evolution.orchestrator.pause()

        champ_a = await evolution.configuration_for_new_cycle()
        assert champ_a is not None
        before = champ_a.configuration_version_id

        await run_closed_trade_round_trip(stack, trade_index=0, minute_offset=0)
        await run_closed_trade_round_trip(stack, trade_index=1, minute_offset=20)
        await stack["supervisor"].event_bus.drain(timeout=10.0)
        episodes = await wait_for_closed_episodes(evolution, session_id, count=2)
        await wait_for_evaluations(evolution, episodes)
        await wire_replay_canned_for_episodes(evolution, stack["fake"])
        evolution.orchestrator.resume_scheduling()
        evolution.wake_orchestrator(reason="rollback_test")
        final = await wait_for_automatic_evolution(stack)
        assert final.status == "completed"
        assert final.challenger_version_id is not None

        champ_b = await evolution.configuration_for_new_cycle()
        assert champ_b is not None
        assert champ_b.configuration_version_id == final.challenger_version_id
        assert champ_b.configuration_version_id != before

        # Open-position pin under B.
        evolution.remember_position_configuration(
            "SPY:rollback:1", champ_b.configuration_version_id
        )
        evolution._pinned_cycle_configs["open-under-b"] = champ_b.configuration_version_id

        assert evolution.drift is not None
        await evolution.drift.observe(
            configuration_version_id=champ_b.configuration_version_id,
            dimension="safety",
            baseline_value=Decimal("0"),
            observed_value=Decimal("1"),
            severity="critical",
            evidence={"reason": "acceptance_safety_trigger"},
        )
        record = await evolution.drift.evaluate_and_maybe_rollback(
            current_champion_id=champ_b.configuration_version_id,
            previous_champion_id=before,
            observations=[],
            safety_violation=True,
        )
        assert record is not None
        assert record.rolled_back_version_id == champ_b.configuration_version_id
        assert record.restored_version_id == before

        restored = await evolution.configuration_for_new_cycle()
        assert restored is not None
        assert restored.configuration_version_id == before

        # Existing open position retains B; new cycle uses A.
        assert (
            evolution.originating_configuration_for_contract("SPY:rollback:1")
            == champ_b.configuration_version_id
        )
        pinned_new = await evolution.pin_and_apply_for_cycle("post-rollback-entry")
        assert pinned_new is not None
        assert evolution.get_pinned("post-rollback-entry") == before

        # Idempotent restart of rollback.
        await evolution.drift.evaluate_and_maybe_rollback(
            current_champion_id=before,
            previous_champion_id=before,
            observations=[],
            safety_violation=True,
        )
        # Current champion is already A — either no-op or fail closed without mutating open pin.
        assert (
            evolution.originating_configuration_for_contract("SPY:rollback:1")
            == champ_b.configuration_version_id
        )

        history = await evolution.champion_registry.compare_champion_history(limit=10)
        promotions = [
            t
            for t in history
            if t.previous_version_id == before
            and t.new_version_id == champ_b.configuration_version_id
        ]
        rollbacks = [
            t
            for t in history
            if t.previous_version_id == champ_b.configuration_version_id
            and t.new_version_id == before
            and str(t.reason).startswith("rollback")
        ]
        assert len(promotions) >= 1
        assert len(rollbacks) >= 1
    finally:
        await shutdown_stack(stack)
        assert not iter_aiosqlite_worker_threads()
