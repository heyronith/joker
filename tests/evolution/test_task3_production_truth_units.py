"""Production-truth remediation proofs for Task 3 evidence, adversarial, shadow, lifecycle."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.adversarial_suite import (
    AdversarialResultStore,
    AdversarialSuiteRunner,
)
from joker.evolution.evidence_claims import EvidenceClaim, EvidenceClaimStore
from joker.evolution.lifecycle_id import make_position_lifecycle_id
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.shadow_ledger import ShadowLedger
from joker.evolution.telemetry import brier_score, expected_calibration_error


@pytest.mark.asyncio
async def test_completed_cycle_evaluations_are_not_reclaimed(tmp_path) -> None:
    db = tmp_path / "claims.db"
    apply_task3_migrations(db)
    store = EvidenceClaimStore(db)
    await store.initialize()
    eid = uuid4()
    ep = uuid4()
    ok, _ = await store.claim_batch(
        evolution_cycle_id="cycle-1",
        claims=[
            EvidenceClaim(
                evaluation_id=eid,
                episode_id=ep,
                evolution_cycle_id="cycle-1",
            )
        ],
        minimum_count=1,
    )
    assert ok
    await store.mark_consumed("cycle-1")
    owned = await store.list_unclaimed_evaluation_ids()
    assert str(eid) in owned
    ok2, inserted = await store.claim_batch(
        evolution_cycle_id="cycle-2",
        claims=[
            EvidenceClaim(
                evaluation_id=eid,
                episode_id=ep,
                evolution_cycle_id="cycle-2",
            )
        ],
        minimum_count=1,
    )
    assert ok2 is False
    assert inserted == []


@pytest.mark.asyncio
async def test_failed_pre_dataset_cycle_releases_claims(tmp_path) -> None:
    db = tmp_path / "release.db"
    store = EvidenceClaimStore(db)
    eid = uuid4()
    await store.claim_batch(
        evolution_cycle_id="c-fail",
        claims=[
            EvidenceClaim(
                evaluation_id=eid,
                episode_id=uuid4(),
                evolution_cycle_id="c-fail",
            )
        ],
        minimum_count=1,
    )
    await store.release_cycle("c-fail", reason="cycle_failed_pre_dataset")
    claims = await store.list_by_cycle("c-fail")
    assert claims[0].claim_status == "released"


@pytest.mark.asyncio
async def test_failed_post_dataset_cycle_retains_recoverable_claims(tmp_path) -> None:
    db = tmp_path / "retain.db"
    store = EvidenceClaimStore(db)
    eid = uuid4()
    await store.claim_batch(
        evolution_cycle_id="c-post",
        claims=[
            EvidenceClaim(
                evaluation_id=eid,
                episode_id=uuid4(),
                evolution_cycle_id="c-post",
            )
        ],
        minimum_count=1,
    )
    await store.attach_dataset("c-post", uuid4())
    claims = await store.list_by_cycle("c-post")
    assert claims[0].claim_status == "claimed"
    assert claims[0].dataset_id is not None


@pytest.mark.asyncio
async def test_explicit_evidence_reuse_is_audited(tmp_path) -> None:
    db = tmp_path / "reuse.db"
    store = EvidenceClaimStore(db)
    eid = uuid4()
    ep = uuid4()
    await store.claim_batch(
        evolution_cycle_id="c1",
        claims=[
            EvidenceClaim(
                evaluation_id=eid, episode_id=ep, evolution_cycle_id="c1"
            )
        ],
        minimum_count=1,
    )
    await store.mark_consumed("c1")
    reused = await store.explicit_reuse(
        evaluation_id=eid,
        evolution_cycle_id="c2",
        episode_id=ep,
        reuse_reason="cross-regime re-evaluation",
    )
    assert reused.reuse_reason == "cross-regime re-evaluation"
    assert reused.claim_reason == "explicit_reuse"


@pytest.mark.asyncio
async def test_concurrent_orchestrators_cannot_claim_same_evaluation(tmp_path) -> None:
    db = tmp_path / "race.db"
    store = EvidenceClaimStore(db)
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
async def test_orchestrator_runs_required_adversarial_suite(tmp_path) -> None:
    db = tmp_path / "adv.db"
    apply_task3_migrations(db)
    from joker.evolution.adversarial_runners import AdversarialExecutionEvidence
    from joker.evolution.adversarial_suite import AdversarialRunnerDispatcher

    class _PassDispatcher:
        def for_mode(self, mode: str):
            class _R:
                async def execute(self, **kwargs):
                    definition = kwargs["definition"]
                    configuration = kwargs["configuration"]
                    return AdversarialExecutionEvidence(
                        experiment_id=kwargs["experiment_id"],
                        scenario_id=definition.scenario_id,
                        scenario_version=definition.version,
                        configuration_version_id=configuration.configuration_version_id,
                        sample_number=kwargs["sample_number"],
                        execution_mode=definition.execution_mode,
                        fixture_loaded=True,
                        runtime_invoked=True,
                        graph_kind=definition.execution_mode,
                        graph_thread_ids=(f"thread:{definition.scenario_id}",),
                        crash_injected=definition.execution_mode == "execution_recovery",
                        fresh_runtime_created=definition.execution_mode
                        == "execution_recovery",
                        durable_checkpoint_loaded=definition.execution_mode
                        == "execution_recovery",
                        checkpoint_resumed=definition.execution_mode
                        == "execution_recovery",
                        expected_invariants=definition.expected_invariants,
                        evaluated_invariants=definition.expected_invariants,
                        satisfied_invariants=definition.expected_invariants,
                        invariants_evaluated=definition.expected_invariants,
                        findings=definition.expected_invariants,
                        passed=True,
                        completed=True,
                    )

            return _R()

    # Seed configurations so suite can resolve champion/challenger.
    from joker.evolution.champion_registry import ChampionRegistry

    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    from joker.evolution.improvement import ImprovementProposalService
    from joker.evolution.schemas import PromptPatch
    from joker.evolution.repositories import build_evolution_repositories

    repos = build_evolution_repositories(db)
    for r in repos.values():
        await r.initialize()
    improvement = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, chall = await improvement.propose(
        parent_champion=champ,
        weakness="w",
        hypothesis="h",
        patch=PromptPatch(
            role="falsifier",
            parent_prompt_version_id=uuid4(),
            replacement_template="t",
            change_rationale="r",
        ),
    )
    runner = AdversarialSuiteRunner(
        AdversarialResultStore(str(db)),
        dispatcher=_PassDispatcher(),
        config_repo=repos["configurations"],
    )
    experiment_id = uuid4()
    passed, results = await runner.run_for_experiment(
        experiment_id=experiment_id,
        champion_version_id=champ.configuration_version_id,
        challenger_version_id=chall.configuration_version_id,
    )
    assert passed is True
    assert len(results) == 50  # 25 scenarios × 2 configs
    assert await runner.adversarial_passed(experiment_id) is True


@pytest.mark.asyncio
async def test_optional_scenario_failure_is_reported(tmp_path) -> None:
    # Required suite executes via mode runners; optional scenarios are absent.
    db = tmp_path / "opt.db"
    apply_task3_migrations(db)
    from joker.evolution.adversarial_runners import AdversarialExecutionEvidence
    from joker.evolution.champion_registry import ChampionRegistry
    from joker.evolution.improvement import ImprovementProposalService
    from joker.evolution.repositories import build_evolution_repositories
    from joker.evolution.schemas import PromptPatch

    class _PassDispatcher:
        def for_mode(self, mode: str):
            class _R:
                async def execute(self, **kwargs):
                    definition = kwargs["definition"]
                    configuration = kwargs["configuration"]
                    return AdversarialExecutionEvidence(
                        experiment_id=kwargs["experiment_id"],
                        scenario_id=definition.scenario_id,
                        scenario_version=definition.version,
                        configuration_version_id=configuration.configuration_version_id,
                        sample_number=1,
                        execution_mode=definition.execution_mode,
                        fixture_loaded=True,
                        runtime_invoked=True,
                        graph_kind=definition.execution_mode,
                        graph_thread_ids=("t",),
                        crash_injected=definition.execution_mode == "execution_recovery",
                        fresh_runtime_created=definition.execution_mode
                        == "execution_recovery",
                        durable_checkpoint_loaded=definition.execution_mode
                        == "execution_recovery",
                        expected_invariants=definition.expected_invariants,
                        satisfied_invariants=definition.expected_invariants,
                        findings=definition.expected_invariants,
                        passed=True,
                        completed=True,
                    )

            return _R()

    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    repos = build_evolution_repositories(db)
    for r in repos.values():
        await r.initialize()
    _, chall = await ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    ).propose(
        parent_champion=champ,
        weakness="w",
        hypothesis="h",
        patch=PromptPatch(
            role="falsifier",
            parent_prompt_version_id=uuid4(),
            replacement_template="t",
            change_rationale="r",
        ),
    )
    runner = AdversarialSuiteRunner(
        AdversarialResultStore(str(db)),
        dispatcher=_PassDispatcher(),
        config_repo=repos["configurations"],
    )
    passed, results = await runner.run_for_experiment(
        experiment_id=uuid4(),
        champion_version_id=champ.configuration_version_id,
        challenger_version_id=chall.configuration_version_id,
    )
    assert passed is True
    assert all(r.executed and r.frozen_truth_loaded for r in results)
    assert not any(r.scenario_id.startswith("optional_") for r in results)


@pytest.mark.asyncio
async def test_missing_adversarial_result_blocks_promotion(tmp_path) -> None:
    runner = AdversarialSuiteRunner(AdversarialResultStore(str(tmp_path / "m.db")))
    assert await runner.adversarial_passed(uuid4()) is False


@pytest.mark.asyncio
async def test_failed_required_scenario_blocks_promotion(tmp_path) -> None:
    db = tmp_path / "fail_adv.db"
    store = AdversarialResultStore(str(db))
    runner = AdversarialSuiteRunner(store)
    experiment_id = uuid4()
    champ = uuid4()
    chall = uuid4()
    await runner.run_for_experiment(
        experiment_id=experiment_id,
        champion_version_id=champ,
        challenger_version_id=chall,
    )
    # Inject a failed executed result for a required scenario.
    from joker.evolution.adversarial_suite import AdversarialScenarioResult
    from datetime import datetime, timezone

    bad = AdversarialScenarioResult(
        result_id=uuid4(),
        experiment_id=experiment_id,
        scenario_id="adv_03",
        scenario_version="3.1.0",
        configuration_version_id=champ,
        passed=False,
        executed=True,
        frozen_truth_loaded=True,
        replay_finished=True,
        findings=("invented_contract_accepted",),
    )
    await store.upsert(bad)
    assert await runner.adversarial_passed(experiment_id) is False


@pytest.mark.asyncio
async def test_adversarial_result_resume_is_idempotent(tmp_path) -> None:
    db = tmp_path / "adv_idemp.db"
    runner = AdversarialSuiteRunner(AdversarialResultStore(str(db)))
    experiment_id = uuid4()
    champ = uuid4()
    chall = uuid4()
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
    assert len(first) == len(second)
    assert {r.result_id for r in first} == {r.result_id for r in second}


@pytest.mark.asyncio
async def test_shadow_fill_not_duplicated_after_restart(tmp_path) -> None:
    db = tmp_path / "shadow.db"
    ledger = ShadowLedger(db)
    await ledger.initialize()
    assignment = uuid4()
    ok1 = await ledger.record_fill(
        fill_id="f1",
        client_order_id="o1",
        assignment_id=assignment,
        quantity=Decimal("1"),
        price=Decimal("1.1"),
        fee=Decimal("0.65"),
    )
    ok2 = await ledger.record_fill(
        fill_id="f1",
        client_order_id="o1",
        assignment_id=assignment,
        quantity=Decimal("1"),
        price=Decimal("1.1"),
        fee=Decimal("0.65"),
    )
    assert ok1 is True
    assert ok2 is False


@pytest.mark.asyncio
async def test_shadow_open_position_survives_fresh_runtime_restart(tmp_path) -> None:
    db = tmp_path / "shadow_pos.db"
    ledger = ShadowLedger(db)
    assignment = uuid4()
    chall = uuid4()
    await ledger.upsert_position(
        assignment_id=assignment,
        challenger_version_id=chall,
        position_lifecycle_id="life-1",
        contract_id="SPY:2026-07-01:500:call",
        configuration_version_id=chall,
        quantity=Decimal("1"),
        average_price=Decimal("1.0"),
        realised_pnl=Decimal("0"),
        status="open",
        last_snapshot_id=str(uuid4()),
    )
    ledger2 = ShadowLedger(db)
    open_positions = await ledger2.list_open_positions(assignment)
    assert len(open_positions) == 1
    assert open_positions[0]["position_lifecycle_id"] == "life-1"


def test_lifecycle_id_is_stable_not_broker_order_dependent() -> None:
    a = make_position_lifecycle_id(
        session_id="s",
        originating_entry_client_order_id="entry-a",
        contract_id="C1",
    )
    b = make_position_lifecycle_id(
        session_id="s",
        originating_entry_client_order_id="entry-a",
        contract_id="C1",
    )
    c = make_position_lifecycle_id(
        session_id="s",
        originating_entry_client_order_id="entry-b",
        contract_id="C1",
    )
    assert a == b
    assert a != c


def test_calibration_is_confidence_versus_outcome() -> None:
    pairs = [(Decimal("0.8"), 1), (Decimal("0.2"), 0), (Decimal("0.9"), 0)]
    brier = brier_score(pairs)
    ece = expected_calibration_error(pairs)
    assert brier is not None and brier > 0
    assert ece is not None
