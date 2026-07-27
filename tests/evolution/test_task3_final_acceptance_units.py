"""Task 3 final-acceptance remediation unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.adversarial_fixtures import (
    ADVERSARIAL_DEFINITIONS,
    AdversarialFixtureRepository,
)
from joker.evolution.adversarial_suite import AdversarialResultStore, AdversarialSuiteRunner
from joker.evolution.evidence_claims import EvidenceClaim, EvidenceClaimStore
from joker.evolution.config import PromotionSettings
from joker.evolution.lifecycle_dag import build_lifecycle_order_graph, unique_fill_accounting
from joker.evolution.promotion_gate import PromotionEligibilityGate
from joker.evolution.schemas import ExperimentResult
from joker.ledger.projector import OrderLifecycle, OrderStatus


def _order(
    oid: str,
    *,
    side: str = "buy",
    status: OrderStatus = OrderStatus.FILLED,
    filled: str = "1",
    parent: str | None = None,
    lifecycle: str | None = "life-1",
    price: str = "1.00",
    fees: str = "0.65",
) -> OrderLifecycle:
    return OrderLifecycle(
        client_order_id=oid,
        contract_id="C",
        side=side,  # type: ignore[arg-type]
        status=status,
        submitted_qty=Decimal(filled),
        filled_qty=Decimal(filled),
        avg_fill_price=Decimal(price),
        fees=Decimal(fees),
        parent_client_order_id=parent,
        position_lifecycle_id=lifecycle,
        originating_entry_client_order_id="e1" if oid != "e1" else "e1",
    )


@pytest.mark.asyncio
async def test_each_required_adversarial_fixture_is_loadable() -> None:
    repo = AdversarialFixtureRepository()
    for definition in ADVERSARIAL_DEFINITIONS:
        if not definition.required:
            continue
        fixture = await repo.load(
            definition.frozen_truth_fixture_id,
            expected_version=definition.version,
        )
        assert fixture.scenario_id == definition.scenario_id
        assert fixture.frames


@pytest.mark.asyncio
async def test_adversarial_suite_runs_entry_graph_fixture(tmp_path) -> None:
    """Suite requires configuration + template deps; without them scenarios fail closed."""
    runner = AdversarialSuiteRunner(AdversarialResultStore(str(tmp_path / "e.db")))
    passed, results = await runner.run_for_experiment(
        experiment_id=uuid4(),
        champion_version_id=uuid4(),
        challenger_version_id=uuid4(),
    )
    assert passed is False
    assert results
    assert all(not r.executed for r in results)


@pytest.mark.asyncio
async def test_adversarial_suite_runs_position_graph_fixture(tmp_path) -> None:
    runner = AdversarialSuiteRunner(AdversarialResultStore(str(tmp_path / "p.db")))
    passed, results = await runner.run_for_experiment(
        experiment_id=uuid4(),
        champion_version_id=uuid4(),
        challenger_version_id=uuid4(),
    )
    assert passed is False
    pos = [r for r in results if r.execution_mode == "position_graph"]
    assert pos and all(not r.executed for r in pos)


@pytest.mark.asyncio
async def test_adversarial_suite_runs_order_management_fixture(tmp_path) -> None:
    runner = AdversarialSuiteRunner(AdversarialResultStore(str(tmp_path / "o.db")))
    passed, _ = await runner.run_for_experiment(
        experiment_id=uuid4(),
        champion_version_id=uuid4(),
        challenger_version_id=uuid4(),
    )
    assert passed is False


@pytest.mark.asyncio
async def test_adversarial_suite_runs_crash_recovery_fixture(tmp_path) -> None:
    runner = AdversarialSuiteRunner(AdversarialResultStore(str(tmp_path / "c.db")))
    passed, results = await runner.run_for_experiment(
        experiment_id=uuid4(),
        champion_version_id=uuid4(),
        challenger_version_id=uuid4(),
    )
    assert passed is False
    rec = [r for r in results if r.execution_mode == "execution_recovery"]
    assert rec and all(not r.executed for r in rec)


@pytest.mark.asyncio
async def test_missing_fixture_blocks_promotion(tmp_path) -> None:
    class BrokenRepo(AdversarialFixtureRepository):
        async def load(self, fixture_id, *, expected_version: str):  # type: ignore[override]
            raise LookupError("missing_fixture")

    runner = AdversarialSuiteRunner(
        AdversarialResultStore(str(tmp_path / "m.db")),
        fixtures=BrokenRepo(),
    )
    passed, _ = await runner.run_for_experiment(
        experiment_id=uuid4(),
        champion_version_id=uuid4(),
        challenger_version_id=uuid4(),
    )
    assert passed is False


@pytest.mark.asyncio
async def test_fixture_version_mismatch_blocks_promotion(tmp_path) -> None:
    class BadVersionRepo(AdversarialFixtureRepository):
        async def load(self, fixture_id, *, expected_version: str):  # type: ignore[override]
            raise ValueError(f"fixture_version_mismatch:0.0.1!={expected_version}")

    runner = AdversarialSuiteRunner(
        AdversarialResultStore(str(tmp_path / "v.db")),
        fixtures=BadVersionRepo(),
    )
    passed, _ = await runner.run_for_experiment(
        experiment_id=uuid4(),
        champion_version_id=uuid4(),
        challenger_version_id=uuid4(),
    )
    assert passed is False


@pytest.mark.asyncio
async def test_invented_contract_scenario_fails_unsafe_configuration(tmp_path) -> None:
    from joker.evolution.adversarial_runners import EntryGraphAdversarialRunner
    from joker.evolution.champion_registry import ChampionRegistry
    from joker.evolution.policy_store import PolicyVersionStore
    from joker.config.settings import CognitiveGraphSettings
    from joker.graph.graph_deps import CognitiveGraphDeps
    from joker.models.fake_provider import FakeModelProvider
    from joker.models.registry import ModelRegistry
    from joker.models.router import ModelRouter
    from joker.models.schemas import ModelsConfig

    apply_task3_migrations = __import__(
        "joker.evolution.migrations", fromlist=["apply_task3_migrations"]
    ).apply_task3_migrations
    apply_task3_migrations(tmp_path / "inv.db")
    reg = ChampionRegistry(tmp_path / "inv.db")
    await reg.bootstrap_champion()
    cfg = await reg.get_current_champion()
    assert cfg is not None
    fake = FakeModelProvider(available=True)
    router = ModelRouter(
        ModelRegistry(
            ModelsConfig(), providers={"ollama": fake, "openai": fake, "fake": fake}
        ),
        session_id="inv",
    )
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="inv",
        run_id="inv",
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
    repo = AdversarialFixtureRepository()
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == "adv_03")
    fixture = await repo.load(
        definition.frozen_truth_fixture_id, expected_version=definition.version
    )
    evidence = await EntryGraphAdversarialRunner(
        template_deps=deps, policy_store=PolicyVersionStore(tmp_path / "inv.db")
    ).execute(
        experiment_id=uuid4(),
        definition=definition,
        fixture=fixture,
        configuration=cfg,
        sample_number=1,
    )
    assert evidence.completed
    assert evidence.graph_thread_ids
    assert "invented_contract_rejected" in evidence.findings or evidence.passed


@pytest.mark.asyncio
async def test_adversarial_results_resume_without_duplicate_execution(tmp_path) -> None:
    store = AdversarialResultStore(str(tmp_path / "r.db"))
    # Pre-seed executed results so resume is a no-op without template deps.
    from joker.evolution.adversarial_suite import AdversarialScenarioResult

    experiment_id = uuid4()
    champ = uuid4()
    chall = uuid4()
    for cfg in (champ, chall):
        for definition in ADVERSARIAL_DEFINITIONS:
            if not definition.required:
                continue
            await store.upsert(
                AdversarialScenarioResult(
                    result_id=uuid4(),
                    experiment_id=experiment_id,
                    scenario_id=definition.scenario_id,
                    scenario_version=definition.version,
                    configuration_version_id=cfg,
                    passed=True,
                    executed=True,
                    frozen_truth_loaded=True,
                    replay_finished=True,
                    findings=definition.expected_invariants,
                    execution_mode=definition.execution_mode,
                    graph_thread_ids=("thread",),
                )
            )
    runner = AdversarialSuiteRunner(store)
    _, first = await runner.run_for_experiment(
        experiment_id=experiment_id,
        champion_version_id=champ,
        challenger_version_id=chall,
    )
    _, second = await runner.run_for_experiment(
        experiment_id=experiment_id,
        champion_version_id=champ,
        challenger_version_id=chall,
    )
    assert {r.result_id for r in first} == {r.result_id for r in second}


def test_entry_replacement_is_not_double_counted() -> None:
    orders = [
        _order("e1"),
        _order("e1r", parent="e1", filled="1", price="1.05"),
        _order("x1", side="sell", price="1.20"),
    ]
    graph = build_lifecycle_order_graph(
        orders, originating_entry_id="e1", terminal_exit_id="x1", lifecycle_id="life-1"
    )
    assert not graph.categories_overlap()
    assert "e1" in graph.original_entry_ids
    assert "e1r" in graph.entry_replacement_ids
    entry_qty, _exit_qty, _buy, _sell, fees = unique_fill_accounting(graph)
    assert entry_qty == Decimal("2")
    assert fees == Decimal("1.95")


def test_multilevel_entry_replacement_chain() -> None:
    orders = [
        _order("e1", filled="0", status=OrderStatus.CANCELLED),
        _order("e2", parent="e1", filled="0", status=OrderStatus.CANCELLED),
        _order("e3", parent="e2", filled="1"),
    ]
    graph = build_lifecycle_order_graph(
        orders, originating_entry_id="e1", terminal_exit_id=None, lifecycle_id="life-1"
    )
    assert "e3" in graph.entry_replacement_ids
    assert graph.categories_overlap() is False


def test_multilevel_exit_replacement_chain() -> None:
    orders = [
        _order("e1"),
        _order("x1", side="sell", filled="0", status=OrderStatus.CANCELLED),
        _order("x2", side="sell", parent="x1", filled="1", price="1.2"),
    ]
    graph = build_lifecycle_order_graph(
        orders, originating_entry_id="e1", terminal_exit_id="x1", lifecycle_id="life-1"
    )
    assert "x2" in graph.exit_replacement_ids or "x2" in graph.terminal_exit_ids


def test_scale_in_is_separate_from_replacement() -> None:
    orders = [
        _order("e1"),
        _order("add1", filled="1"),
        _order("e1r", parent="e1"),
    ]
    graph = build_lifecycle_order_graph(
        orders, originating_entry_id="e1", terminal_exit_id=None, lifecycle_id="life-1"
    )
    assert "add1" in graph.scale_in_ids
    assert "e1r" in graph.entry_replacement_ids


def test_unique_fill_accounting_prevents_duplicate_pnl() -> None:
    orders = [
        _order("e1", fees="0.65"),
        _order("e1r", parent="e1", fees="0.65"),
        _order("x1", side="sell", fees="0.65", price="1.5"),
    ]
    graph = build_lifecycle_order_graph(
        orders, originating_entry_id="e1", terminal_exit_id="x1", lifecycle_id="life-1"
    )
    _eq, _xq, _bc, _sp, fees = unique_fill_accounting(graph)
    assert fees == Decimal("1.95")


def test_cross_lifecycle_parent_is_rejected() -> None:
    orders = [
        _order("e1", lifecycle="life-a"),
        _order("e2", parent="e1", lifecycle="life-b"),
    ]
    graph = build_lifecycle_order_graph(
        orders, originating_entry_id="e1", terminal_exit_id=None, lifecycle_id="life-b"
    )
    assert any(f.startswith("cross_lifecycle_parent") for f in graph.findings)


def test_order_parent_cycle_marks_episode_incomplete() -> None:
    a = _order("a", parent="b")
    b = _order("b", parent="a")
    graph = build_lifecycle_order_graph(
        [a, b], originating_entry_id="a", terminal_exit_id=None, lifecycle_id="life-1"
    )
    assert any("order_parent_cycle" in f for f in graph.findings)


def test_nonsequential_order_ids_have_no_effect() -> None:
    orders = [_order("z9"), _order("a1", parent="z9")]
    graph = build_lifecycle_order_graph(
        orders, originating_entry_id="z9", terminal_exit_id=None, lifecycle_id="life-1"
    )
    assert "z9" in graph.original_entry_ids
    assert "a1" in graph.entry_replacement_ids


def test_unknown_required_cost_blocks_promotion() -> None:
    gate = PromotionEligibilityGate(
        PromotionSettings(
            minimum_completed_episodes=1,
            minimum_holdout_episodes=0,
            require_known_cost=True,
            minimum_calibration_samples=0,
            require_brier_score=False,
            require_expected_calibration_error=False,
        )
    )
    result = ExperimentResult(
        experiment_id=uuid4(),
        champion_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "cost_known": False,
            "mean_pnl": Decimal("1"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "cost_known": False,
            "mean_pnl": Decimal("2"),
        },
    )
    out = gate.evaluate(
        result=result,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
    )
    assert out.eligible is False
    assert "unknown_required_cost" in out.gate_codes


def test_known_zero_cost_is_not_unknown() -> None:
    gate = PromotionEligibilityGate(
        PromotionSettings(
            minimum_completed_episodes=1,
            minimum_holdout_episodes=0,
            require_known_cost=True,
            minimum_calibration_samples=0,
            require_brier_score=False,
            require_expected_calibration_error=False,
            maximum_cost_regression_pct=Decimal("100"),
        )
    )
    result = ExperimentResult(
        experiment_id=uuid4(),
        champion_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "cost_gbp": Decimal("0"),
            "cost_known": True,
            "mean_pnl": Decimal("1"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "cost_gbp": Decimal("0"),
            "cost_known": True,
            "mean_pnl": Decimal("2"),
        },
    )
    out = gate.evaluate(
        result=result,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
    )
    assert "unknown_required_cost" not in out.gate_codes


def test_missing_calibration_pairs_blocks_promotion() -> None:
    gate = PromotionEligibilityGate(
        PromotionSettings(
            minimum_completed_episodes=1,
            minimum_holdout_episodes=0,
            require_known_cost=False,
            minimum_calibration_samples=1,
            require_brier_score=True,
            require_expected_calibration_error=True,
        )
    )
    result = ExperimentResult(
        experiment_id=uuid4(),
        champion_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "calibration_sample_count": Decimal("0"),
            "mean_pnl": Decimal("1"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "calibration_sample_count": Decimal("0"),
            "mean_pnl": Decimal("2"),
        },
    )
    out = gate.evaluate(
        result=result,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
    )
    assert "insufficient_calibration_samples" in out.gate_codes
    assert "missing_brier_score" in out.gate_codes
    assert "missing_expected_calibration_error" in out.gate_codes


def test_insufficient_calibration_samples_blocks_promotion() -> None:
    gate = PromotionEligibilityGate(
        PromotionSettings(
            minimum_completed_episodes=1,
            minimum_holdout_episodes=0,
            require_known_cost=False,
            minimum_calibration_samples=20,
            require_brier_score=True,
            require_expected_calibration_error=True,
        )
    )
    result = ExperimentResult(
        experiment_id=uuid4(),
        champion_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "brier_score": Decimal("0.1"),
            "expected_calibration_error": Decimal("0.1"),
            "calibration_sample_count": Decimal("5"),
            "mean_pnl": Decimal("1"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "brier_score": Decimal("0.1"),
            "expected_calibration_error": Decimal("0.1"),
            "calibration_sample_count": Decimal("5"),
            "mean_pnl": Decimal("2"),
        },
    )
    out = gate.evaluate(
        result=result,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
    )
    assert "insufficient_calibration_samples" in out.gate_codes


def test_pnl_dispersion_does_not_satisfy_calibration_gate() -> None:
    gate = PromotionEligibilityGate(
        PromotionSettings(
            minimum_completed_episodes=1,
            minimum_holdout_episodes=0,
            require_known_cost=False,
            minimum_calibration_samples=1,
            require_brier_score=True,
            require_expected_calibration_error=True,
        )
    )
    result = ExperimentResult(
        experiment_id=uuid4(),
        champion_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "pnl_mean_absolute_deviation": Decimal("0.01"),
            "calibration_error": Decimal("0.01"),
            "calibration_sample_count": Decimal("5"),
            "mean_pnl": Decimal("1"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "pnl_mean_absolute_deviation": Decimal("0.01"),
            "calibration_error": Decimal("0.01"),
            "calibration_sample_count": Decimal("5"),
            "mean_pnl": Decimal("2"),
        },
    )
    out = gate.evaluate(
        result=result,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
    )
    assert "missing_brier_score" in out.gate_codes
    assert "missing_expected_calibration_error" in out.gate_codes


def test_brier_regression_blocks_promotion() -> None:
    gate = PromotionEligibilityGate(
        PromotionSettings(
            minimum_completed_episodes=1,
            minimum_holdout_episodes=0,
            require_known_cost=False,
            minimum_calibration_samples=1,
            require_brier_score=True,
            require_expected_calibration_error=False,
            maximum_calibration_regression_pct=Decimal("1"),
        )
    )
    result = ExperimentResult(
        experiment_id=uuid4(),
        champion_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "brier_score": Decimal("0.10"),
            "calibration_sample_count": Decimal("20"),
            "mean_pnl": Decimal("1"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "brier_score": Decimal("0.50"),
            "calibration_sample_count": Decimal("20"),
            "mean_pnl": Decimal("2"),
        },
    )
    out = gate.evaluate(
        result=result,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
    )
    assert any("brier_score" in c for c in out.gate_codes)


def test_ece_regression_blocks_promotion() -> None:
    gate = PromotionEligibilityGate(
        PromotionSettings(
            minimum_completed_episodes=1,
            minimum_holdout_episodes=0,
            require_known_cost=False,
            minimum_calibration_samples=1,
            require_brier_score=False,
            require_expected_calibration_error=True,
            maximum_calibration_regression_pct=Decimal("1"),
        )
    )
    result = ExperimentResult(
        experiment_id=uuid4(),
        champion_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "expected_calibration_error": Decimal("0.10"),
            "calibration_sample_count": Decimal("20"),
            "mean_pnl": Decimal("1"),
        },
        challenger_metrics={
            "tail_loss": Decimal("-1"),
            "latency_ms": 10,
            "expected_calibration_error": Decimal("0.50"),
            "calibration_sample_count": Decimal("20"),
            "mean_pnl": Decimal("2"),
        },
    )
    out = gate.evaluate(
        result=result,
        holdout_episode_count=0,
        completed_episode_count=1,
        adversarial_passed=True,
    )
    assert any("expected_calibration_error" in c for c in out.gate_codes)


@pytest.mark.asyncio
async def test_explicit_reuse_preserves_original_claim(tmp_path) -> None:
    store = EvidenceClaimStore(tmp_path / "e.db")
    eid = uuid4()
    ep = uuid4()
    await store.claim_batch(
        evolution_cycle_id="c1",
        claims=[EvidenceClaim(evaluation_id=eid, episode_id=ep, evolution_cycle_id="c1")],
        minimum_count=1,
    )
    await store.mark_consumed("c1")
    original = (await store.list_history(eid))[0]
    reused = await store.explicit_reuse(
        evaluation_id=eid,
        evolution_cycle_id="c2",
        episode_id=ep,
        reuse_reason="audit-reuse",
    )
    history = await store.list_history(eid)
    assert len(history) == 2
    assert history[0].claim_id == original.claim_id
    assert reused.prior_claim_id == original.claim_id


@pytest.mark.asyncio
async def test_explicit_reuse_links_prior_claim(tmp_path) -> None:
    store = EvidenceClaimStore(tmp_path / "e2.db")
    eid = uuid4()
    ep = uuid4()
    ok, claims = await store.claim_batch(
        evolution_cycle_id="c1",
        claims=[EvidenceClaim(evaluation_id=eid, episode_id=ep, evolution_cycle_id="c1")],
        minimum_count=1,
    )
    assert ok
    reused = await store.explicit_reuse(
        evaluation_id=eid,
        evolution_cycle_id="c2",
        episode_id=ep,
        reuse_reason="linked",
    )
    assert reused.prior_claim_id == claims[0].claim_id


@pytest.mark.asyncio
async def test_automatic_cycle_cannot_reuse_consumed_evaluation(tmp_path) -> None:
    store = EvidenceClaimStore(tmp_path / "e3.db")
    eid = uuid4()
    ep = uuid4()
    await store.claim_batch(
        evolution_cycle_id="c1",
        claims=[EvidenceClaim(evaluation_id=eid, episode_id=ep, evolution_cycle_id="c1")],
        minimum_count=1,
    )
    await store.mark_consumed("c1")
    ok, _ = await store.claim_batch(
        evolution_cycle_id="c2",
        claims=[EvidenceClaim(evaluation_id=eid, episode_id=ep, evolution_cycle_id="c2")],
        minimum_count=1,
    )
    assert ok is False


@pytest.mark.asyncio
async def test_concurrent_automatic_claims_remain_exclusive(tmp_path) -> None:
    store = EvidenceClaimStore(tmp_path / "e4.db")
    eid = uuid4()
    ep = uuid4()
    ok1, _ = await store.claim_batch(
        evolution_cycle_id="a",
        claims=[EvidenceClaim(evaluation_id=eid, episode_id=ep, evolution_cycle_id="a")],
        minimum_count=1,
    )
    ok2, _ = await store.claim_batch(
        evolution_cycle_id="b",
        claims=[EvidenceClaim(evaluation_id=eid, episode_id=ep, evolution_cycle_id="b")],
        minimum_count=1,
    )
    assert ok1 is True
    assert ok2 is False


@pytest.mark.asyncio
async def test_reuse_history_is_append_only(tmp_path) -> None:
    store = EvidenceClaimStore(tmp_path / "e5.db")
    eid = uuid4()
    ep = uuid4()
    await store.claim_batch(
        evolution_cycle_id="c1",
        claims=[EvidenceClaim(evaluation_id=eid, episode_id=ep, evolution_cycle_id="c1")],
        minimum_count=1,
    )
    await store.explicit_reuse(
        evaluation_id=eid, evolution_cycle_id="c2", episode_id=ep, reuse_reason="r1"
    )
    await store.explicit_reuse(
        evaluation_id=eid, evolution_cycle_id="c3", episode_id=ep, reuse_reason="r2"
    )
    history = await store.list_history(eid)
    assert len(history) == 3
    assert history[0].claim_reason == "automatic_cycle"
    assert all(h.claim_reason != "" for h in history)
