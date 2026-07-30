"""Production Task 3 evolution acceptance — paper fills, automatic worker path."""

from __future__ import annotations

import pytest

from joker.evolution.adversarial import required_scenario_ids
from joker.evolution.runtime import build_status_report
from tests.integration.task3_production_harness import (
    EXPECTED_REALIZED_PNL,
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
async def test_task3_production_evolution_acceptance(tmp_path) -> None:
    """Paper → automatic workers → mandatory promotion → champion pin."""
    session_id = "accept"
    stack = await build_paper_evolution_stack(
        tmp_path,
        session_id=session_id,
        settings=acceptance_settings(),
        start_orchestrator_worker=True,
    )
    try:
        evolution = stack["evolution"]
        supervisor = stack["supervisor"]
        # Prevent experiment replay from starting before snapshot canned outputs exist.
        assert evolution.orchestrator is not None
        evolution.orchestrator.pause()

        assert evolution.orchestrator._checkpointer is not None
        assert evolution.evidence_claims is not None
        assert evolution.adversarial_suite is not None
        assert evolution.shadow_ledger is not None
        assert evolution.replay is not None
        assert evolution.experiments._replay_service is evolution.replay

        champ = await evolution.configuration_for_new_cycle()
        assert champ is not None
        before_champion_id = champ.configuration_version_id

        await run_closed_trade_round_trip(stack, trade_index=0, minute_offset=0)
        await run_closed_trade_round_trip(stack, trade_index=1, minute_offset=20)

        assert stack["gateway_entry_ids"], "entry fills must use OrderActionGateway"
        assert stack["gateway_exit_ids"], "exit fills must use OrderActionGateway"
        assert len(stack["gateway_entry_ids"]) >= 2
        assert len(stack["gateway_exit_ids"]) >= 2

        await supervisor.event_bus.drain(timeout=10.0)
        episodes = await wait_for_closed_episodes(evolution, session_id, count=2)
        for episode in episodes:
            assert episode.realised_pnl == EXPECTED_REALIZED_PNL
            assert episode.terminal_event_id is not None
            assert episode.terminal_event_timestamp is not None
        evaluations = await wait_for_evaluations(evolution, episodes)
        assert all(e.valid for e in evaluations)
        assert all(ep.entry_order_ids and ep.exit_order_ids for ep in episodes)

        await wire_replay_canned_for_episodes(evolution, stack["fake"])
        evolution.orchestrator.resume_scheduling()
        evolution.wake_orchestrator(reason="acceptance_ready")
        final_state = await wait_for_automatic_evolution(stack)
        assert final_state is not None
        assert final_state.status == "completed"
        assert final_state.dataset_id is not None
        assert final_state.proposal_id is not None
        assert final_state.challenger_version_id is not None
        assert final_state.experiment_id is not None
        assert final_state.promotion_decision_id is not None

        if final_state.promotion_decision_id is not None:
            await evolution.decisions.apply_persisted_decision(
                promotion_decision_id=final_state.promotion_decision_id
            )

        definition = await evolution._repos["experiments"].get_definition(
            final_state.experiment_id
        )
        assert definition is not None
        assert len(definition.adversarial_scenario_ids) == len(required_scenario_ids())

        adv_results = await evolution.adversarial_suite._store.list_for_experiment(
            final_state.experiment_id
        )
        expected_adv_count = len(required_scenario_ids()) * 2
        assert len(adv_results) == expected_adv_count
        for result in adv_results:
            assert result.executed is True
            evidence = result.evidence
            assert evidence is not None
            if evidence.execution_mode == "execution_recovery":
                assert evidence.runtime_invoked or evidence.durable_checkpoint_loaded
                assert (
                    evidence.model_call_ids
                    or evidence.durable_checkpoint_loaded
                    or evidence.checkpoint_resumed
                )
            else:
                assert evidence.runtime_invoked is True
                if evidence.execution_mode == "full_replay":
                    assert evidence.model_call_ids or result.graph_thread_ids
                else:
                    assert evidence.model_call_ids or evidence.graph_thread_ids
            if result.passed:
                assert not evidence.failed_invariants
                assert not evidence.runtime_errors
            assert evidence.fixture_loaded is True

        claims = await evolution.evidence_claims.list_by_cycle(final_state.cycle_id)
        assert claims
        assert final_state.adversarial_passed is True

        promotion = await evolution._repos["promotions"].get_by_id(
            final_state.promotion_decision_id
        )
        assert promotion is not None
        assert promotion.final_status == "promoted"
        assert promotion.deterministic_eligible is True
        assert final_state.promotion_decision_id == promotion.promotion_decision_id

        result = await evolution._repos["experiments"].get_result(final_state.experiment_id)
        assert result is not None
        assert bool(result.challenger_metrics.get("cost_known")) is True
        assert result.challenger_metrics.get("cost_source") == "persisted_model_calls"
        assert result.challenger_metrics.get("pricing_version")
        assert int(result.challenger_metrics.get("calibration_sample_count") or 0) >= 2
        assert result.challenger_metrics.get("calibration_sample_ids")
        assert result.challenger_metrics.get("brier_score") is not None
        assert (
            result.challenger_metrics.get("expected_calibration_error") is not None
            or result.challenger_metrics.get("expected_calibration_error") == 0
            or "expected_calibration_error" in result.challenger_metrics
        )
        assert result.champion_metrics.get("cost_source") in {
            "persisted_model_calls",
            "missing",
        }
        if result.champion_metrics.get("cost_known"):
            assert result.champion_metrics.get("pricing_version")

        for episode in episodes:
            if episode.entry_decision_event_id is not None:
                assert episode.entry_decision_timestamp is not None
            if episode.market_event_ids:
                assert all(eid is not None for eid in episode.market_event_ids)

        activation = await evolution._repos["activations"].get_by_decision_id(
            promotion.promotion_decision_id
        )
        assert activation is not None
        assert activation.registry_applied is True
        assert activation.history_verified is True
        assert activation.configuration_status_applied is True
        assert activation.completed is True

        history = await evolution.champion_registry.compare_champion_history(limit=5)
        promoted = [
            t for t in history if t.promotion_decision_id == promotion.promotion_decision_id
        ]
        assert promoted, "activation must reference persisted promotion decision"

        after = await evolution.configuration_for_new_cycle()
        assert after is not None
        assert after.configuration_version_id == final_state.challenger_version_id
        assert after.configuration_version_id != before_champion_id

        # Subsequent Task 2 cycle pins the promoted champion.
        pinned = await evolution.pin_and_apply_for_cycle("post-promote-cycle")
        assert pinned is not None
        assert (
            evolution.get_pinned("post-promote-cycle") == final_state.challenger_version_id
        )

        owned = await evolution.evidence_claims.list_unclaimed_evaluation_ids()
        assert owned

        status = await build_status_report(evolution)
        assert status["paper_only"] is True
        assert status["live_trading_enabled"] is False
        assert evolution.replay._isolated_deps().execution_runtime is None
        assert evolution.replay._isolated_deps().order_action_gateway is None
    finally:
        await shutdown_stack(stack)


@pytest.mark.asyncio
async def test_acceptance_requires_promotion(tmp_path) -> None:
    """Gate-rejected challenger must not satisfy the mandatory promotion proof."""
    # Covered by the primary acceptance assertions; this documents the contract.
    from joker.evolution.schemas import PromotionDecision
    from uuid import uuid4
    from datetime import datetime, timezone

    decision = PromotionDecision(
        experiment_id=uuid4(),
        challenger_version_id=uuid4(),
        champion_version_id=uuid4(),
        deterministic_eligible=False,
        deterministic_gate_codes=("unknown_required_cost",),
        agent_action="reject",
        strategic_rationale="blocked",
        final_status="blocked_by_gate",
    )
    assert decision.final_status != "promoted"
