"""Regressions for adversarial semantic honesty (Task 3 final blockers)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from joker.config.settings import CognitiveGraphSettings
from joker.evolution.adversarial_fixtures import (
    ADVERSARIAL_DEFINITIONS,
    AdversarialFixtureRepository,
)
from joker.evolution.adversarial_model_path import (
    install_adversarial_model_path,
    install_scenario_specific_observations,
)
from joker.evolution.adversarial_runners import (
    EntryGraphAdversarialRunner,
    FullReplayAdversarialRunner,
    _evaluate_replay_invariants,
    _observe_entry_findings_from_graph,
)
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.policy_store import PolicyVersionStore
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig
from joker.cognition.schemas import MetaDecisionAction


def _router():
    fake = FakeModelProvider(available=True)
    registry = ModelRegistry(
        ModelsConfig(), providers={"ollama": fake, "openai": fake, "fake": fake}
    )
    return ModelRouter(registry, session_id="adv-sem"), fake


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


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_id", ["adv_22", "adv_23"])
async def test_full_replay_exception_always_fails(tmp_path, scenario_id: str):
    router, _ = _router()
    cfg = await _champion(tmp_path)
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == scenario_id)
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )

    class BoomReplay:
        async def replay_episode(self, **kwargs):
            raise RuntimeError("forced_replay_failure")

    store = PolicyVersionStore(tmp_path / "c.db")
    runner = FullReplayAdversarialRunner(
        template_deps=_deps(router, tmp_path / f"boom-{scenario_id}.db"),
        policy_store=store,
        replay_service=BoomReplay(),
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.passed is False
    assert evidence.completed is False
    assert evidence.runtime_errors
    assert any("full_replay:" in e for e in evidence.runtime_errors)
    assert "calibrated_loss_accepted" not in evidence.satisfied_invariants
    assert "regime_shift_handled" not in evidence.satisfied_invariants


def test_replay_invariants_never_treat_fabricated_graph_flags_as_success():
    """ran_* flags alone without real calibration confidence must not pass adv_22."""
    payload = {
        "ran_task2_graph": True,
        "ran_position_graph": True,
        "broker_submit": False,
        "traded": True,
        "realised_pnl": "-1.5",
        "calibration_sample_count": 1,
        "calibration_pairs": [("0.65", True)],  # fabricated confidence
        "entry_confidence": None,  # not from a persisted model decision
        "integrity_findings": (),
        "open_at_end": False,
        "meta_decision_action": "execute",
    }
    satisfied, _ = _evaluate_replay_invariants(
        ("calibrated_loss_accepted",),
        payload=payload,
    )
    assert "calibrated_loss_accepted" not in satisfied


def test_calibration_requires_persisted_entry_confidence():
    payload = {
        "ran_task2_graph": True,
        "ran_position_graph": True,
        "broker_submit": False,
        "traded": True,
        "realised_pnl": "-1.5",
        "calibration_sample_count": 1,
        "calibration_pairs": [("0.42", 0)],
        "entry_confidence": "0.42",
        "integrity_findings": (),
        "open_at_end": False,
        "meta_decision_action": "execute",
    }
    satisfied, _ = _evaluate_replay_invariants(
        ("calibrated_loss_accepted",),
        payload=payload,
    )
    assert "calibrated_loss_accepted" in satisfied


def test_generic_abandon_result_does_not_prove_scenario_labels():
    """A bare ABANDON meta decision must not satisfy conflict/consensus/etc."""
    from joker.cognition.schemas import MetaDecision

    bare = {
        "meta_decision": MetaDecision(
            session_id="s",
            snapshot_id=uuid4(),
            decision_id=uuid4(),
            prompt_version="2.0.0",
            model_call_id=uuid4(),
            cycle_id="c",
            action=MetaDecisionAction.ABANDON,
            selected_strategy_id=None,
            confidence=0.2,
            rationale_summary="generic abandon",
        ),
        "world_model": None,
        "reviews": [],
        "strategies": [],
    }
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == "adv_02")
    # Sync load via corpus — repository is async; use definition fixture id lookup.
    from joker.evolution.adversarial_fixtures import FIXTURE_CORPUS

    fixture = FIXTURE_CORPUS[definition.frozen_truth_fixture_id]
    findings = _observe_entry_findings_from_graph(
        bare, fixture=fixture, entry_submitted=False
    )
    for label in (
        "conflicting_evidence_handled",
        "false_consensus_resisted",
        "thin_liquidity_rejected",
        "unsupported_reasoning_rejected",
        "narrow_overfit_rejected",
    ):
        assert label not in findings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_id,label",
    [
        ("adv_02", "conflicting_evidence_handled"),
        ("adv_04", "false_consensus_resisted"),
        ("adv_05", "thin_liquidity_rejected"),
        ("adv_21", "unsupported_reasoning_rejected"),
        ("adv_25", "narrow_overfit_rejected"),
    ],
)
async def test_generic_abandon_path_cannot_pass_scenario(
    tmp_path, scenario_id: str, label: str, monkeypatch
):
    """Install only the generic ABANDON path — no scenario overlays — must not pass."""
    router, fake = _router()
    cfg = await _champion(tmp_path)
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == scenario_id)
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )

    # Force the runner to skip scenario-specific observation factories.
    monkeypatch.setattr(
        "joker.evolution.adversarial_runners.install_scenario_specific_observations",
        lambda *a, **k: None,
    )
    install_adversarial_model_path(
        fake, session_id="adv", meta_action=MetaDecisionAction.ABANDON
    )

    store = PolicyVersionStore(tmp_path / "c.db")
    runner = EntryGraphAdversarialRunner(
        template_deps=_deps(router, tmp_path / f"gen-{scenario_id}.db"),
        policy_store=store,
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.passed is False
    assert label not in evidence.satisfied_invariants


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_id,label",
    [
        ("adv_02", "conflicting_evidence_handled"),
        ("adv_04", "false_consensus_resisted"),
        ("adv_05", "thin_liquidity_rejected"),
        ("adv_21", "unsupported_reasoning_rejected"),
        ("adv_25", "narrow_overfit_rejected"),
    ],
)
async def test_removing_scenario_evidence_causes_failure(
    tmp_path, scenario_id: str, label: str, monkeypatch
):
    """Strip observed scenario findings after the graph runs → must fail."""
    router, _ = _router()
    cfg = await _champion(tmp_path)
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == scenario_id)
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )

    real_observe = _observe_entry_findings_from_graph

    def _strip_label(result, *, fixture, entry_submitted):
        return [
            f
            for f in real_observe(
                result, fixture=fixture, entry_submitted=entry_submitted
            )
            if f != label
        ]

    monkeypatch.setattr(
        "joker.evolution.adversarial_runners._observe_entry_findings_from_graph",
        _strip_label,
    )
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = EntryGraphAdversarialRunner(
        template_deps=_deps(router, tmp_path / f"strip-{scenario_id}.db"),
        policy_store=store,
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.passed is False
    assert label not in evidence.satisfied_invariants


def test_ran_graph_flags_require_actual_execution_semantics():
    """Evaluator must not satisfy regime_shift without ran_position_graph from payload."""
    payload = {
        "ran_task2_graph": True,
        "ran_position_graph": False,  # not actually run
        "broker_submit": False,
        "traded": True,
        "realised_pnl": "-1.0",
        "open_at_end": False,
        "integrity_findings": (),
        "calibration_pairs": (),
        "calibration_sample_count": 0,
        "entry_confidence": None,
        "meta_decision_action": "execute",
    }
    satisfied, _ = _evaluate_replay_invariants(
        ("regime_shift_handled",),
        payload=payload,
    )
    assert "regime_shift_handled" not in satisfied

    payload["ran_position_graph"] = True
    satisfied2, _ = _evaluate_replay_invariants(
        ("regime_shift_handled",),
        payload=payload,
    )
    assert "regime_shift_handled" in satisfied2
