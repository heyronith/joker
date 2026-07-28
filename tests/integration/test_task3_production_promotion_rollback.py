"""Production promotion → drift → rollback → restored champion use."""

from __future__ import annotations

from decimal import Decimal

import pytest

from tests.cognitive.task2_canned import CONTRACT_ID
from tests.integration.task3_production_harness import (
    acceptance_settings,
    build_paper_evolution_stack,
    ensure_flat_position,
    install_paper_path_factories,
    rebuild_paper_evolution_stack,
    register_evolution_router_canned,
    run_closed_trade_round_trip,
    run_open_trade_entry_only,
    shutdown_stack,
    wait_for_automatic_evolution,
    wait_for_closed_episodes,
    wait_for_evaluations,
    wire_replay_canned_for_episodes,
)


def _rollback_transitions(history, *, rolled_back, restored):
    return [
        t
        for t in history
        if t.previous_version_id == rolled_back
        and t.new_version_id == restored
        and str(t.reason).startswith("rollback")
    ]


@pytest.mark.asyncio
async def test_full_loop_drift_and_rollback(tmp_path) -> None:
    session_id = "rollback"
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

        if final.promotion_decision_id is not None:
            await evolution.decisions.apply_persisted_decision(
                promotion_decision_id=final.promotion_decision_id
            )

        champ_b = await evolution.configuration_for_new_cycle()
        assert champ_b is not None
        assert champ_b.configuration_version_id == final.challenger_version_id
        assert champ_b.configuration_version_id != before

        install_paper_path_factories(stack["fake"], session_id=session_id)
        register_evolution_router_canned(stack["fake"])

        evolution.orchestrator.pause()
        await stack["supervisor"].event_bus.drain(timeout=10.0)
        await ensure_flat_position(stack, trade_index=99)

        # Real paper entry under promoted champion B (via _on_position_opened pin path).
        await run_open_trade_entry_only(stack, trade_index=2, minute_offset=40)
        await stack["supervisor"].event_bus.drain(timeout=10.0)
        projection = await stack["supervisor"].execution_runtime.project_session()
        pos = projection.positions.get(CONTRACT_ID)
        assert pos is not None and pos.quantity != 0
        assert (
            evolution.originating_configuration_for_contract(CONTRACT_ID)
            == champ_b.configuration_version_id
        )

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

        # Existing open position retains B; new cycle uses restored champion A.
        assert (
            evolution.originating_configuration_for_contract(CONTRACT_ID)
            == champ_b.configuration_version_id
        )
        pinned_new = await evolution.pin_and_apply_for_cycle("post-rollback-entry")
        assert pinned_new is not None
        assert evolution.get_pinned("post-rollback-entry") == before

        history_before_restart = await evolution.champion_registry.compare_champion_history(
            limit=20
        )
        rollbacks_before = _rollback_transitions(
            history_before_restart,
            rolled_back=champ_b.configuration_version_id,
            restored=before,
        )
        assert len(rollbacks_before) == 1

        # Fresh-process idempotency: shutdown, rebuild, repeat rollback evaluation.
        await shutdown_stack(stack, strict_workers=False)
        stack = await rebuild_paper_evolution_stack(
            tmp_path,
            db=db,
            session_id=session_id,
            settings=acceptance_settings(),
            start_orchestrator_worker=False,
        )
        evolution = stack["evolution"]
        assert evolution.drift is not None

        second = await evolution.drift.evaluate_and_maybe_rollback(
            current_champion_id=champ_b.configuration_version_id,
            previous_champion_id=before,
            observations=[],
            safety_violation=True,
        )
        assert second is not None
        assert second.rolled_back_version_id == champ_b.configuration_version_id
        assert second.restored_version_id == before

        restored_after = await evolution.configuration_for_new_cycle()
        assert restored_after is not None
        assert restored_after.configuration_version_id == before

        history_after = await evolution.champion_registry.compare_champion_history(limit=20)
        rollbacks_after = _rollback_transitions(
            history_after,
            rolled_back=champ_b.configuration_version_id,
            restored=before,
        )
        assert len(rollbacks_after) == 1

        promotions = [
            t
            for t in history_after
            if t.previous_version_id == before
            and t.new_version_id == champ_b.configuration_version_id
        ]
        assert len(promotions) >= 1
    finally:
        await shutdown_stack(stack)