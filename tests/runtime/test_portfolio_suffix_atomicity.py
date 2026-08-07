"""Atomic suffix/request recovery and fresh fill provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
    PortfolioComponentResolutionStatus,
    PortfolioComponentStatus,
    PortfolioExecutionComponentRecord,
    PortfolioExecutionOwner,
    PortfolioReoptimizationStatus,
)
from joker.runtime.portfolio_recovery import PortfolioRecoveryCoordinator
from joker.runtime.recovery_mode import RecoveryMode
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock

NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)
OWNER = PortfolioExecutionOwner(
    session_id="session-a",
    broker_account_identity="paper-a",
    trading_date="2026-08-05",
)


def _component(
    tuple_id: str,
    *,
    index: int,
    count: int = 2,
    status: PortfolioComponentStatus = PortfolioComponentStatus.AUTHORIZED,
    filled_quantity: int = 0,
) -> PortfolioExecutionComponentRecord:
    submitted = filled_quantity
    return PortfolioExecutionComponentRecord(
        session_id=OWNER.session_id,
        origin_run_id="run-a",
        broker_account_identity=OWNER.broker_account_identity,
        trading_date=OWNER.trading_date,
        target_portfolio_decision_id="decision-a",
        selected_portfolio_id="portfolio-a",
        authorized_position_tuple_id=tuple_id,
        component_index=index,
        component_count=count,
        strategy_id=f"strategy-{tuple_id}",
        contract_id=f"contract-{tuple_id}",
        authorized_quantity=1,
        capital_allocation=Decimal("100"),
        client_order_id=f"client-{tuple_id}",
        status=status,
        remaining_quantity=1 - filled_quantity,
        submitted_quantity=submitted,
        filled_quantity=filled_quantity,
        original_decision_snapshot_id="snapshot-a",
        evaluated_objective_version=1,
        evaluated_timestamp=NOW.isoformat(),
    )


def _clock() -> FrozenExchangeClock:
    return FrozenExchangeClock(NOW, calendar=MarketCalendar())


@pytest.mark.asyncio
async def test_nonterminal_suffix_request_is_atomic_with_component_transition(
    tmp_path,
) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1))
    runtime = SimpleNamespace(
        session_id="session-a",
        project_session=AsyncMock(return_value=SimpleNamespace(orders={}, positions={})),
        poll_order_status=AsyncMock(),
        unresolved_reconciliation=None,
        kill_switch=False,
        entry_permission=None,
        _broker=None,
        _bridge_poll_order_status=None,
    )
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=runtime,  # type: ignore[arg-type]
        provenance_registry=registry,
        stable_owner=OWNER,
        clock=_clock(),
        objective_service=None,
        recovery_mode=RecoveryMode.NORMAL,
    )

    resolution = await coordinator.request_suffix_reoptimization(
        decision_id="decision-a",
        reason="material_truth_changed",
        origin_run_id="run-a",
        latest_snapshot_id="snapshot-b",
    )

    assert resolution.request is not None
    assert resolution.request.status == PortfolioReoptimizationStatus.PENDING
    component = await registry.portfolio_executions.get("tuple-b")
    assert component is not None
    assert component.status == PortfolioComponentStatus.REOPTIMIZATION_REQUIRED
    assert component.resolution_status == PortfolioComponentResolutionStatus.UNRESOLVED


@pytest.mark.asyncio
async def test_interrupted_reoptimization_required_without_request_is_repaired(
    tmp_path,
) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1))
    await registry.portfolio_executions.transition(
        "tuple-b",
        owner=OWNER,
        status=PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
        failure_reoptimization_reason="crash_before_request",
    )
    runtime = SimpleNamespace(
        session_id="session-a",
        project_session=AsyncMock(return_value=SimpleNamespace(orders={}, positions={})),
        poll_order_status=AsyncMock(),
        unresolved_reconciliation=None,
        kill_switch=False,
        entry_permission=None,
        _broker=None,
        _bridge_poll_order_status=None,
    )
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=runtime,  # type: ignore[arg-type]
        provenance_registry=registry,
        stable_owner=OWNER,
        clock=_clock(),
        objective_service=None,
        recovery_mode=RecoveryMode.NORMAL,
    )

    repaired = await coordinator.repair_interrupted_suffix_states(
        decision_id="decision-a",
        origin_run_id="run-b",
        latest_snapshot_id="snapshot-b",
    )

    assert repaired.request is not None
    assert repaired.request.status == PortfolioReoptimizationStatus.PENDING
    assert repaired.request.remaining_authorized_tuple_ids == ("tuple-b",)
    component = await registry.portfolio_executions.get("tuple-b")
    assert component is not None
    assert component.status == PortfolioComponentStatus.REOPTIMIZATION_REQUIRED
    assert component.resolution_status == PortfolioComponentResolutionStatus.UNRESOLVED


@pytest.mark.asyncio
async def test_fresh_fill_appears_in_already_filled_tuple_ids(tmp_path) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-a", index=0))
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1))
    submitted = await registry.portfolio_executions.transition(
        "tuple-a",
        owner=OWNER,
        status=PortfolioComponentStatus.SUBMITTED,
        submitted_quantity=1,
    )
    await registry.portfolio_executions.transition(
        "tuple-a",
        owner=OWNER,
        status=PortfolioComponentStatus.FILLED,
        submitted_quantity=1,
        filled_quantity=1,
        post_fill_objective_version=3,
        post_fill_objective_fingerprint='{"objective_id":"obj"}',
        post_fill_snapshot_id="snapshot-post-fill",
        post_fill_exchange_time=NOW.isoformat(),
        reconciled_filled_quantity=1,
        continuation_ready=True,
        expected_state_version=submitted.state_version,
    )

    class Obj:
        version = 4
        status = "active"

        def as_dict(self):
            return {"version": 4, "status": "active"}

    runtime = SimpleNamespace(
        session_id="session-a",
        project_session=AsyncMock(return_value=SimpleNamespace(orders={}, positions={})),
        poll_order_status=AsyncMock(),
        unresolved_reconciliation=None,
        kill_switch=False,
        entry_permission=None,
        _broker=None,
        _bridge_poll_order_status=None,
    )
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=runtime,  # type: ignore[arg-type]
        provenance_registry=registry,
        stable_owner=OWNER,
        clock=_clock(),
        objective_service=SimpleNamespace(get_state=AsyncMock(return_value=Obj())),
        recovery_mode=RecoveryMode.NORMAL,
    )

    # Stale pre-fill component list must not be trusted; reload from durable store.
    stale_components = [
        _component("tuple-a", index=0, status=PortfolioComponentStatus.WORKING),
        _component("tuple-b", index=1),
    ]
    resolution = await coordinator.request_suffix_reoptimization(
        decision_id="decision-a",
        reason="continue_after_fill",
        origin_run_id="run-a",
        latest_snapshot_id="snapshot-post-fill",
        source_components=None,
    )
    assert resolution.request is not None
    assert "tuple-a" in resolution.request.already_filled_tuple_ids
    assert resolution.request.remaining_authorized_tuple_ids == ("tuple-b",)
    # Prove stale WORKING view would have missed the fill.
    assert not any(
        component.status == PortfolioComponentStatus.FILLED
        for component in stale_components
    )


@pytest.mark.asyncio
async def test_terminal_suffix_does_not_pre_transition_outside_transaction(
    tmp_path,
) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1))

    class Obj:
        version = 2
        status = "deadline_reached"

        def as_dict(self):
            return {"version": 2, "status": "deadline_reached"}

    runtime = SimpleNamespace(
        session_id="session-a",
        project_session=AsyncMock(return_value=SimpleNamespace(orders={}, positions={})),
        poll_order_status=AsyncMock(),
        unresolved_reconciliation=None,
        kill_switch=False,
        entry_permission=None,
        _broker=None,
        _bridge_poll_order_status=None,
    )
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=runtime,  # type: ignore[arg-type]
        provenance_registry=registry,
        stable_owner=OWNER,
        clock=_clock(),
        objective_service=SimpleNamespace(get_state=AsyncMock(return_value=Obj())),
        recovery_mode=RecoveryMode.RECONCILIATION_ONLY,
    )
    before = await registry.portfolio_executions.get("tuple-b")
    assert before is not None
    assert before.status == PortfolioComponentStatus.AUTHORIZED

    resolution = await coordinator.request_suffix_reoptimization(
        decision_id="decision-a",
        reason="reconciliation_only_resume_no_new_entries",
        origin_run_id="run-a",
        terminal_recovery=True,
        latest_snapshot_id="snapshot-b",
        objective_status="deadline_reached",
    )
    assert resolution.terminalized is True
    assert resolution.request is not None
    assert resolution.request.status == PortfolioReoptimizationStatus.COMPLETED
    after = await registry.portfolio_executions.get("tuple-b")
    assert after is not None
    assert after.status == PortfolioComponentStatus.REOPTIMIZATION_REQUIRED
    assert after.resolution_status == PortfolioComponentResolutionStatus.OPERATOR_RESOLVED
