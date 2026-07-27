"""Adversarial mode runners must invoke real Task 2 cognition."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.adversarial_fixtures import (
    ADVERSARIAL_DEFINITIONS,
    AdversarialFixtureRepository,
)
from joker.evolution.adversarial_runners import (
    AdversarialExecutionEvidence,
    EntryGraphAdversarialRunner,
    ExecutionRecoveryAdversarialRunner,
    FullReplayAdversarialRunner,
    OrderManagementAdversarialRunner,
    PositionGraphAdversarialRunner,
)
from joker.evolution.adversarial_suite import _executed_from_evidence as suite_executed
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import CognitiveConfigurationVersion
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.config.settings import CognitiveGraphSettings
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig


def _router():
    fake = FakeModelProvider(available=True)
    registry = ModelRegistry(
        ModelsConfig(), providers={"ollama": fake, "openai": fake, "fake": fake}
    )
    return ModelRouter(registry, session_id="adv-test"), fake


async def _config(tmp_path) -> CognitiveConfigurationVersion:
    apply_task3_migrations(tmp_path / "c.db")
    repos = build_evolution_repositories(tmp_path / "c.db")
    await repos["configurations"].initialize()
    store = PolicyVersionStore(tmp_path / "c.db")
    await store.initialize()
    # Bootstrap via champion registry path used elsewhere.
    from joker.evolution.champion_registry import ChampionRegistry

    reg = ChampionRegistry(tmp_path / "c.db")
    await reg.bootstrap_champion()
    champ = await reg.get_current_champion()
    assert champ is not None
    return champ


def _deps(router) -> CognitiveGraphDeps:
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
        db_path=None,
        checkpointer=None,
        data_quality_loader=None,
        projection_loader=None,
        provenance_registry=None,
        order_action_gateway=None,
        cycle_registry=None,
        order_management_action_repo=None,
    )


@pytest.mark.asyncio
async def test_entry_adversarial_fixture_invokes_cognitive_graph(tmp_path, monkeypatch):
    router, _ = _router()
    cfg = await _config(tmp_path)
    calls = {"n": 0}
    real_build = __import__(
        "joker.graph.cognitive_graph", fromlist=["build_cognitive_graph"]
    ).build_cognitive_graph

    def spy_build(deps):
        calls["n"] += 1
        return real_build(deps)

    monkeypatch.setattr(
        "joker.evolution.adversarial_runners.build_cognitive_graph", spy_build
    )
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.execution_mode == "entry_graph")
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = EntryGraphAdversarialRunner(
        template_deps=_deps(router), policy_store=store
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert calls["n"] >= 1
    assert evidence.completed
    assert evidence.runtime_invoked
    assert evidence.graph_kind == "entry"
    assert evidence.graph_thread_ids
    assert suite_executed(evidence)


@pytest.mark.asyncio
async def test_position_adversarial_fixture_invokes_position_graph(tmp_path, monkeypatch):
    router, _ = _router()
    cfg = await _config(tmp_path)
    calls = {"n": 0}

    class FakeGraph:
        async def ainvoke(self, state, config=None):
            calls["n"] += 1
            return {"_position_decision": type("D", (), {"recommended_action": "REDUCE", "recommended_quantity": 1})()}

    monkeypatch.setattr(
        "joker.evolution.adversarial_runners.build_position_graph",
        lambda deps: FakeGraph(),
    )
    definition = next(
        d for d in ADVERSARIAL_DEFINITIONS if d.execution_mode == "position_graph"
    )
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = PositionGraphAdversarialRunner(
        template_deps=_deps(router), policy_store=store
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert calls["n"] >= 1
    assert evidence.graph_kind == "position"
    assert suite_executed(evidence)


@pytest.mark.asyncio
async def test_order_management_fixture_invokes_order_manager(tmp_path, monkeypatch):
    router, _ = _router()
    cfg = await _config(tmp_path)
    calls = {"n": 0}

    class SpyOM:
        def __init__(self, *a, **k):
            pass

        async def manage(self, **kwargs):
            calls["n"] += 1
            raise RuntimeError("replay_om_missing_snapshot_context")

    monkeypatch.setattr(
        "joker.evolution.adversarial_runners.ReplayOrderManagementRunner", SpyOM
    )
    definition = next(
        d for d in ADVERSARIAL_DEFINITIONS if d.execution_mode == "order_management"
    )
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = OrderManagementAdversarialRunner(
        template_deps=_deps(router), policy_store=store
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert calls["n"] >= 1
    assert evidence.graph_kind == "order_management"
    assert evidence.runtime_invoked
    assert evidence.passed is False
    assert evidence.runtime_errors
    assert suite_executed(evidence)


@pytest.mark.asyncio
async def test_entry_graph_exception_does_not_auto_pass(tmp_path, monkeypatch):
    router, _ = _router()
    cfg = await _config(tmp_path)

    class BoomGraph:
        async def ainvoke(self, state, config=None):
            raise RuntimeError("forced_graph_failure")

    monkeypatch.setattr(
        "joker.evolution.adversarial_runners.build_cognitive_graph",
        lambda deps: BoomGraph(),
    )
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.execution_mode == "entry_graph")
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = EntryGraphAdversarialRunner(
        template_deps=_deps(router), policy_store=store
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.runtime_invoked
    assert evidence.passed is False
    assert any(f.startswith("graph_fail_closed:") for f in evidence.findings)
    assert evidence.runtime_errors
    assert definition.expected_invariants
    assert set(definition.expected_invariants) != set(evidence.findings)


@pytest.mark.asyncio
async def test_full_replay_missing_service_fails_closed(tmp_path):
    router, _ = _router()
    cfg = await _config(tmp_path)
    definition = next(
        d for d in ADVERSARIAL_DEFINITIONS if d.execution_mode == "full_replay"
    )
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = FullReplayAdversarialRunner(
        template_deps=_deps(router), policy_store=store, replay_service=None
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.passed is False
    assert evidence.runtime_invoked is False
    assert "replay_service_missing" in evidence.runtime_errors
    assert suite_executed(evidence) is False


@pytest.mark.asyncio
async def test_recovery_fixture_creates_fresh_runtime(tmp_path, monkeypatch):
    router, _ = _router()
    cfg = await _config(tmp_path)
    definition = next(
        d for d in ADVERSARIAL_DEFINITIONS if d.execution_mode == "execution_recovery"
    )
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    deps = replace(_deps(router), db_path=tmp_path / "recovery_runner.db")
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = ExecutionRecoveryAdversarialRunner(
        template_deps=deps, policy_store=store
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.crash_injected
    assert evidence.fresh_runtime_created
    assert evidence.durable_checkpoint_loaded
    assert evidence.checkpoint_resumed
    assert evidence.runtime_invoked
    assert suite_executed(evidence)


@pytest.mark.asyncio
async def test_full_replay_fixture_invokes_cognitive_replay_service(tmp_path):
    router, _ = _router()
    cfg = await _config(tmp_path)
    called = {"n": 0}

    class FakeReplay:
        async def replay_episode(self, **kwargs):
            called["n"] += 1
            assert "experiment_id" in kwargs
            return {
                "ran_task2_graph": True,
                "entry_graph_thread_id": "replay:x:entry",
                "fill_ids": (),
                "integrity_findings": ("snapshot_not_found",),
            }

    definition = next(
        d for d in ADVERSARIAL_DEFINITIONS if d.execution_mode == "full_replay"
    )
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = FullReplayAdversarialRunner(
        template_deps=_deps(router), policy_store=store, replay_service=FakeReplay()
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert called["n"] == 1
    assert evidence.graph_kind == "full_replay"
    assert evidence.runtime_invoked
    assert suite_executed(evidence)


@pytest.mark.asyncio
async def test_provider_timeout_is_observed_by_real_graph_path(tmp_path, monkeypatch):
    router, fake = _router()
    cfg = await _config(tmp_path)
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == "adv_10")
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    assert fixture.provider_behaviour == "timeout"
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = EntryGraphAdversarialRunner(
        template_deps=_deps(router), policy_store=store
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.completed
    assert evidence.graph_thread_ids
    assert suite_executed(evidence)


@pytest.mark.asyncio
async def test_missing_data_quality_is_observed_by_real_graph_path(tmp_path):
    router, _ = _router()
    cfg = await _config(tmp_path)
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == "adv_17")
    fixture = await AdversarialFixtureRepository().load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    store = PolicyVersionStore(tmp_path / "c.db")
    runner = EntryGraphAdversarialRunner(
        template_deps=_deps(router), policy_store=store
    )
    evidence = await runner.execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert "missing_data_quality_fail_closed" in evidence.findings
    assert suite_executed(evidence)


def test_execution_mode_label_without_trace_is_not_eligible():
    evidence = AdversarialExecutionEvidence(
        experiment_id=uuid4(),
        scenario_id="adv_x",
        scenario_version="3.1.0",
        configuration_version_id=uuid4(),
        sample_number=1,
        execution_mode="entry_graph",
        fixture_loaded=True,
        completed=True,
        passed=True,
        runtime_invoked=False,
        graph_thread_ids=(),
    )
    assert suite_executed(evidence) is False
