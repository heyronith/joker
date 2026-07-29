"""Dispatcher-level adversarial semantics: no label-only full-replay passes."""

from __future__ import annotations

from uuid import uuid4

import pytest

from joker.config.settings import CognitiveGraphSettings
from joker.evolution.adversarial import required_scenario_ids
from joker.evolution.adversarial_fixtures import (
    ADVERSARIAL_DEFINITIONS,
    AdversarialFixtureRepository,
)
from joker.evolution.adversarial_runners import (
    AdversarialRunnerDispatcher,
    FullReplayAdversarialRunner,
    _evaluate_replay_invariants,
)
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.policy_store import PolicyVersionStore
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig


def _router():
    fake = FakeModelProvider(available=True)
    registry = ModelRegistry(
        ModelsConfig(), providers={"ollama": fake, "openai": fake, "fake": fake}
    )
    return ModelRouter(registry, session_id="adv-dispatch"), fake


def _deps(router, db_path=None) -> CognitiveGraphDeps:
    return CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="adv",
        run_id="adv",
        context_assembler=None,
        snapshot_repo=None,
        option_surface_repo=None,
        data_quality_repo=None,
        evidence_repo=None,
        world_model_repo=None,
        hypothesis_repo=None,
        strategy_repo=None,
        debate_repo=None,
        decision_repo=None,
        position_thesis_repo=None,
        order_management_repo=None,
        model_call_repo=None,
        execution_runtime=None,
        submit_callback=None,
        event_bus=None,
        clock=None,
        db_path=db_path,
        checkpointer=None,
        data_quality_loader=None,
        projection_loader=None,
        provenance_registry=None,
        order_action_gateway=None,
        cycle_registry=None,
        order_management_action_repo=None,
    )


async def _champion(tmp_path):
    apply_task3_migrations(tmp_path / "c.db")
    from joker.evolution.champion_registry import ChampionRegistry

    reg = ChampionRegistry(tmp_path / "c.db")
    await reg.bootstrap_champion()
    champ = await reg.get_current_champion()
    assert champ is not None
    return champ


def test_every_required_scenario_has_explicit_execution_mode():
    required = set(required_scenario_ids())
    by_id = {d.scenario_id: d for d in ADVERSARIAL_DEFINITIONS}
    assert required <= set(by_id)
    for sid in required:
        assert by_id[sid].execution_mode in {
            "entry_graph",
            "position_graph",
            "order_management",
            "execution_recovery",
            "full_replay",
        }
    # Previously unmapped scenarios must not silently default to full_replay.
    for sid in (
        "adv_10",
        "adv_11",
        "adv_12",
        "adv_13",
        "adv_14",
        "adv_02",
        "adv_04",
        "adv_05",
        "adv_06",
        "adv_20",
        "adv_21",
        "adv_24",
        "adv_25",
    ):
        assert by_id[sid].execution_mode != "full_replay" or sid in {"adv_22", "adv_23"}


def test_generic_clean_replay_payload_does_not_satisfy_arbitrary_labels():
    expected = (
        "false_consensus_resisted",
        "provider_timeout_recovered",
        "duplicate_order_prevented",
        "urgent_exit_priority",
        "calibrated_loss_accepted",
    )
    payload = {
        "ran_task2_graph": True,
        "integrity_findings": (),
        "broker_submit": False,
        "traded": False,
        "realised_pnl": "0",
        "calibration_sample_count": 0,
        "calibration_pairs": (),
        "ran_position_graph": False,
        "meta_decision_action": "abandon",
    }
    satisfied, failed = _evaluate_replay_invariants(expected, payload=payload)
    assert "false_consensus_resisted" not in satisfied
    assert "provider_timeout_recovered" not in satisfied
    assert "duplicate_order_prevented" not in satisfied
    assert "urgent_exit_priority" not in satisfied
    assert "calibrated_loss_accepted" not in satisfied
    assert satisfied == () or set(satisfied).isdisjoint(set(expected) - {"justified_no_trade"})


@pytest.mark.asyncio
async def test_full_replay_does_not_copy_expected_invariants_into_findings(tmp_path):
    router, _ = _router()
    cfg = await _champion(tmp_path)

    class CleanReplay:
        async def replay_episode(self, **kwargs):
            return {
                "ran_task2_graph": True,
                "entry_graph_thread_id": "replay:clean",
                "fill_ids": (),
                "integrity_findings": (),
                "broker_submit": False,
                "traded": False,
                "realised_pnl": "0",
                "calibration_sample_count": 0,
                "calibration_pairs": (),
                "ran_position_graph": False,
                "meta_decision_action": "abandon",
            }

    # Pick a definition that expects a non-generic label if any full_replay remains;
    # otherwise use a synthetic definition-like object via an existing full_replay scenario.
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == "adv_22")
    # Temporarily evaluate with a fixture that does NOT request concrete augmentation.
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    # Clear calibration stimulus so CleanReplay path is used without augmentation.
    fixture = fixture.model_copy(
        update={"stimulus": {"scenario_id": fixture.scenario_id, "baseline_safe": True}}
    )
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = FullReplayAdversarialRunner(
        template_deps=_deps(router, tmp_path / "replay.db"),
        policy_store=store,
        replay_service=CleanReplay(),
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.runtime_invoked
    for inv in definition.expected_invariants:
        assert inv not in evidence.findings
    assert evidence.passed is False
    assert "calibrated_loss_accepted" not in evidence.satisfied_invariants


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_id", sorted(required_scenario_ids()))
async def test_dispatcher_routes_every_required_scenario(tmp_path, scenario_id: str):
    router, _ = _router()
    cfg = await _champion(tmp_path)
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == scenario_id)
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    assert fixture.execution_mode == definition.execution_mode

    class SoftReplay:
        async def replay_episode(self, **kwargs):
            return {
                "ran_task2_graph": True,
                "entry_graph_thread_id": f"replay:{scenario_id}",
                "fill_ids": (),
                "integrity_findings": (),
                "broker_submit": False,
                "traded": False,
                "realised_pnl": "0",
                "calibration_sample_count": 0,
                "calibration_pairs": (),
                "ran_position_graph": False,
                "open_at_end": False,
                "meta_decision_action": "abandon",
            }

    store = PolicyVersionStore(tmp_path / "c.db")
    dispatcher = AdversarialRunnerDispatcher(
        template_deps=_deps(router, tmp_path / f"d-{scenario_id}.db"),
        policy_store=store,
        replay_service=SoftReplay(),
    )
    runner = dispatcher.for_mode(definition.execution_mode)
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.execution_mode == definition.execution_mode
    assert evidence.fixture_loaded
    # Must not satisfy expected labels merely by echoing them into findings.
    for inv in definition.expected_invariants:
        assert inv not in evidence.findings or inv in evidence.satisfied_invariants
    if evidence.passed:
        assert set(definition.expected_invariants) <= set(evidence.satisfied_invariants)
