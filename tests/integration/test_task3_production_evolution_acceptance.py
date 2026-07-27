"""Production Task 3 evolution acceptance — paper fills, public runtime path."""

from __future__ import annotations

import pytest

from joker.evolution.adversarial import required_scenario_ids
from joker.evolution.runtime import build_status_report
from joker.persistence.aiosqlite_lifecycle import iter_aiosqlite_worker_threads
from tests.cognitive.task2_canned import CONTRACT_ID
from tests.integration.task3_production_harness import (
    EXPECTED_REALIZED_PNL,
    acceptance_settings,
    build_paper_evolution_stack,
    drain_evolution_orchestrator,
    run_closed_trade_round_trip,
    shutdown_stack,
    wait_for_closed_episodes,
    wait_for_evaluations,
    wire_replay_canned_for_episodes,
)


@pytest.mark.asyncio
async def test_task3_production_evolution_acceptance(tmp_path) -> None:
    """Paper session → 2 closed trades → automatic workers → evolution cycle."""
    session_id = "accept"
    stack = await build_paper_evolution_stack(
        tmp_path,
        session_id=session_id,
        settings=acceptance_settings(),
        start_orchestrator_worker=False,
    )
    evolution = stack["evolution"]
    supervisor = stack["supervisor"]

    assert evolution.orchestrator is not None
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
    evaluations = await wait_for_evaluations(evolution, episodes)
    assert all(e.valid for e in evaluations)
    assert all(ep.entry_order_ids and ep.exit_order_ids for ep in episodes)

    await wire_replay_canned_for_episodes(evolution, stack["fake"])
    evolution.orchestrator.resume_scheduling()
    final_state = await drain_evolution_orchestrator(stack)
    assert final_state is not None
    assert final_state.dataset_id is not None
    assert final_state.proposal_id is not None
    assert final_state.challenger_version_id is not None
    assert final_state.experiment_id is not None

    definition = await evolution._repos["experiments"].get_definition(
        final_state.experiment_id
    )
    assert definition is not None
    assert len(definition.adversarial_scenario_ids) == 25

    adv_results = await evolution.adversarial_suite._store.list_for_experiment(
        final_state.experiment_id
    )
    assert len(adv_results) == len(required_scenario_ids()) * 2
    assert all(r.executed for r in adv_results)

    claims = await evolution.evidence_claims.list_by_cycle(final_state.cycle_id)
    assert claims
    assert final_state.adversarial_passed is True

    promotion = None
    if final_state.promotion_decision_id is not None:
        promotion = await evolution._repos["promotions"].get_by_id(
            final_state.promotion_decision_id
        )
    if promotion is None and final_state.experiment_id is not None:
        promotion = await evolution._repos["promotions"].get_by_experiment(
            final_state.experiment_id
        )
    assert promotion is not None
    if promotion.final_status == "promoted":
        history = await evolution.champion_registry.compare_champion_history(limit=5)
        promoted = [t for t in history if t.promotion_decision_id == promotion.promotion_decision_id]
        assert promoted, "activation must reference persisted promotion decision"
        assert promotion.created_at <= promoted[0].activated_at

    after = await evolution.configuration_for_new_cycle()
    assert after is not None

    owned = await evolution.evidence_claims.list_unclaimed_evaluation_ids()
    assert owned

    status = await build_status_report(evolution)
    assert status["paper_only"] is True
    assert status["live_trading_enabled"] is False
    assert evolution.replay._isolated_deps().execution_runtime is None
    assert evolution.replay._isolated_deps().order_action_gateway is None

    await shutdown_stack(stack)
    assert not iter_aiosqlite_worker_threads()
