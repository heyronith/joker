"""Crash/atomicity and interrupted-state repair acceptance for portfolio recovery."""

from __future__ import annotations

import sqlite3
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
    session_id="cog:paper:acct:2026-08-05",
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
        submitted_quantity=filled_quantity,
        filled_quantity=filled_quantity,
        original_decision_snapshot_id="snapshot-a",
        evaluated_objective_version=1,
        evaluated_timestamp=NOW.isoformat(),
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        session_id=OWNER.session_id,
        project_session=AsyncMock(return_value=SimpleNamespace(orders={}, positions={})),
        poll_order_status=AsyncMock(),
        unresolved_reconciliation=None,
        kill_switch=False,
        entry_permission=None,
        _broker=None,
        _bridge_poll_order_status=None,
    )


def _clock() -> FrozenExchangeClock:
    return FrozenExchangeClock(NOW, calendar=MarketCalendar())


class _Obj:
    version = 2
    status = "deadline_reached"

    def as_dict(self):
        return {"version": 2, "status": "deadline_reached"}


@pytest.mark.asyncio
async def test_crash_before_terminal_request_insert_is_repaired(tmp_path) -> None:
    """Simulate interrupted pre-request transition, then repair on recovery."""
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1))
    # Crash window: suffix already lost AUTHORIZED authority, request absent.
    await registry.portfolio_executions.transition(
        "tuple-b",
        owner=OWNER,
        status=PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
        failure_reoptimization_reason="crash_before_request",
    )
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=_runtime(),  # type: ignore[arg-type]
        provenance_registry=registry,
        stable_owner=OWNER,
        clock=_clock(),
        objective_service=SimpleNamespace(get_state=AsyncMock(return_value=_Obj())),
        recovery_mode=RecoveryMode.RECONCILIATION_ONLY,
    )
    repaired = await coordinator.repair_interrupted_suffix_states(
        decision_id="decision-a",
        origin_run_id="run-b",
        latest_snapshot_id="snapshot-b",
        objective_status="deadline_reached",
        terminal_recovery_reason="reconciliation_only_resume_no_new_entries",
    )
    assert repaired.request is not None
    assert repaired.terminalized is True
    assert repaired.request.status == PortfolioReoptimizationStatus.COMPLETED
    component = await registry.portfolio_executions.get("tuple-b")
    assert component is not None
    assert component.resolution_status == PortfolioComponentResolutionStatus.OPERATOR_RESOLVED


@pytest.mark.asyncio
async def test_crash_between_suffix_components_is_repaired(tmp_path) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1, count=3))
    await registry.portfolio_executions.authorize(_component("tuple-c", index=2, count=3))
    await registry.portfolio_executions.transition(
        "tuple-b",
        owner=OWNER,
        status=PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
        failure_reoptimization_reason="crash_mid_suffix",
    )
    # tuple-c still AUTHORIZED — interrupted mid-suffix.
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=_runtime(),  # type: ignore[arg-type]
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
    # Repair reconstructs request for orphaned REOPTIMIZATION_REQUIRED component.
    assert "tuple-b" in repaired.request.remaining_authorized_tuple_ids
    # Completing the remaining AUTHORIZED suffix is a separate atomic call.
    completed = await coordinator.request_suffix_reoptimization(
        decision_id="decision-a",
        reason="material_truth_changed",
        origin_run_id="run-b",
        latest_snapshot_id="snapshot-b",
    )
    assert completed.request is not None
    tuple_c = await registry.portfolio_executions.get("tuple-c")
    assert tuple_c is not None
    assert tuple_c.status == PortfolioComponentStatus.REOPTIMIZATION_REQUIRED


@pytest.mark.asyncio
async def test_terminal_request_and_components_commit_atomically(tmp_path) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1))
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=_runtime(),  # type: ignore[arg-type]
        provenance_registry=registry,
        stable_owner=OWNER,
        clock=_clock(),
        objective_service=SimpleNamespace(get_state=AsyncMock(return_value=_Obj())),
        recovery_mode=RecoveryMode.RECONCILIATION_ONLY,
    )
    resolution = await coordinator.request_suffix_reoptimization(
        decision_id="decision-a",
        reason="reconciliation_only_resume_no_new_entries",
        origin_run_id="run-a",
        terminal_recovery=True,
        latest_snapshot_id="snapshot-b",
        objective_status="deadline_reached",
    )
    assert resolution.terminalized is True
    component = await registry.portfolio_executions.get("tuple-b")
    assert component is not None
    assert component.status == PortfolioComponentStatus.REOPTIMIZATION_REQUIRED
    assert component.resolution_status == PortfolioComponentResolutionStatus.OPERATOR_RESOLVED
    assert resolution.request is not None
    assert resolution.request.status == PortfolioReoptimizationStatus.COMPLETED
    assert not await registry.portfolio_reoptimizations.list_pending(
        session_id=OWNER.session_id,
        broker_account_identity=OWNER.broker_account_identity,
        trading_date=OWNER.trading_date,
    )


@pytest.mark.asyncio
async def test_normal_reoptimization_request_and_components_commit_atomically(
    tmp_path,
) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1))
    db_path = tmp_path / "prov.db"

    # Prove there is no committed REOPTIMIZATION_REQUIRED without a request mid-flight
    # by inspecting SQLite after the atomic operation.
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=_runtime(),  # type: ignore[arg-type]
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
    with sqlite3.connect(db_path) as db:
        orphan = db.execute(
            """
            SELECT c.authorized_position_tuple_id
            FROM portfolio_execution_components c
            WHERE c.status = 'REOPTIMIZATION_REQUIRED'
              AND COALESCE(c.resolution_status, 'UNRESOLVED') = 'UNRESOLVED'
              AND NOT EXISTS (
                SELECT 1 FROM portfolio_reoptimization_requests r
                WHERE r.session_id = c.session_id
                  AND r.broker_account_id = c.broker_account_id
                  AND r.trading_date = c.trading_date
                  AND r.remaining_authorized_tuple_ids_json LIKE
                      '%' || c.authorized_position_tuple_id || '%'
              )
            """
        ).fetchall()
    assert orphan == []


@pytest.mark.asyncio
async def test_repeated_terminal_recovery_is_idempotent(tmp_path) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1))
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=_runtime(),  # type: ignore[arg-type]
        provenance_registry=registry,
        stable_owner=OWNER,
        clock=_clock(),
        objective_service=SimpleNamespace(get_state=AsyncMock(return_value=_Obj())),
        recovery_mode=RecoveryMode.RECONCILIATION_ONLY,
    )
    first = await coordinator.request_suffix_reoptimization(
        decision_id="decision-a",
        reason="reconciliation_only_resume_no_new_entries",
        origin_run_id="run-a",
        terminal_recovery=True,
        latest_snapshot_id="snapshot-b",
        objective_status="deadline_reached",
    )
    second = await coordinator.request_suffix_reoptimization(
        decision_id="decision-a",
        reason="reconciliation_only_resume_no_new_entries",
        origin_run_id="run-b",
        terminal_recovery=True,
        latest_snapshot_id="snapshot-b",
        objective_status="deadline_reached",
    )
    assert first.request is not None
    # No remaining AUTHORIZED/READY suffix after first terminalization.
    assert second.request is None
    repaired = await coordinator.repair_interrupted_suffix_states(
        decision_id="decision-a",
        origin_run_id="run-c",
        latest_snapshot_id="snapshot-b",
        objective_status="deadline_reached",
        terminal_recovery_reason="reconciliation_only_resume_no_new_entries",
    )
    # Already resolved components are not treated as orphans.
    assert repaired.request is None or repaired.request.request_id == first.request.request_id


@pytest.mark.asyncio
async def test_no_unresolved_component_without_corresponding_request(tmp_path) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    await registry.portfolio_executions.authorize(_component("tuple-b", index=1))
    await registry.portfolio_executions.transition(
        "tuple-b",
        owner=OWNER,
        status=PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
        failure_reoptimization_reason="orphaned",
    )
    coordinator = PortfolioRecoveryCoordinator(
        execution_runtime=_runtime(),  # type: ignore[arg-type]
        provenance_registry=registry,
        stable_owner=OWNER,
        clock=_clock(),
        objective_service=None,
        recovery_mode=RecoveryMode.NORMAL,
    )
    await coordinator.repair_interrupted_suffix_states(
        decision_id="decision-a",
        origin_run_id="run-b",
        latest_snapshot_id="snapshot-b",
    )
    component = await registry.portfolio_executions.get("tuple-b")
    assert component is not None
    assert component.status == PortfolioComponentStatus.REOPTIMIZATION_REQUIRED
    assert component.resolution_status == PortfolioComponentResolutionStatus.UNRESOLVED
    remaining = (component.authorized_position_tuple_id,)
    request = await registry.portfolio_reoptimizations.get_by_remaining(
        session_id=OWNER.session_id,
        broker_account_identity=OWNER.broker_account_identity,
        trading_date=OWNER.trading_date,
        original_portfolio_decision_id="decision-a",
        remaining_authorized_tuple_ids=remaining,
    )
    assert request is not None


@pytest.mark.asyncio
async def test_newly_polled_fill_is_recorded_in_terminal_request(tmp_path) -> None:
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
        status=PortfolioComponentStatus.WORKING,
        submitted_quantity=1,
        expected_state_version=submitted.state_version,
    )

    class Obj:
        version = 5
        status = "deadline_reached"
        objective_id = "objective-a"
        available_capital_usd = Decimal("900")
        reserved_capital_usd = Decimal("100")
        working_order_reservation_usd = Decimal("0")
        filled_position_exposure_usd = Decimal("100")
        required_profit_remaining_usd = Decimal("50")
        realised_pnl_usd = Decimal("0")
        deadline_exchange_time = NOW
        time_remaining_seconds = 0
        entries_paused = True
        truth_degraded = False
        open_position_count = 1
        max_concurrent_positions = 3

        def as_dict(self):
            return {"version": 5, "status": "deadline_reached"}

    projection_after_fill = SimpleNamespace(
        orders={
            "client-tuple-a": SimpleNamespace(
                client_order_id="client-tuple-a",
                status="filled",
                filled_qty=1,
                order_id="broker-a",
            )
        },
        positions={
            "contract-tuple-a": {
                "contract_id": "contract-tuple-a",
                "quantity": 1,
            }
        },
    )
    runtime = SimpleNamespace(
        session_id=OWNER.session_id,
        project_session=AsyncMock(return_value=projection_after_fill),
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
        objective_service=SimpleNamespace(
            get_state=AsyncMock(return_value=Obj()),
            recompute_from_truth=AsyncMock(return_value=Obj()),
        ),
        recovery_mode=RecoveryMode.RECONCILIATION_ONLY,
    )

    updated = await coordinator.reconcile_owner_components(
        projection=projection_after_fill,
        latest_snapshot_id="snapshot-post-fill",
        terminal_recovery_reason="reconciliation_only_resume_no_new_entries",
        objective_status="deadline_reached",
        origin_run_id="run-a",
    )
    filled = await registry.portfolio_executions.get("tuple-a")
    suffix = await registry.portfolio_executions.get("tuple-b")
    assert filled is not None
    assert filled.status == PortfolioComponentStatus.FILLED
    assert filled.continuation_ready is True
    assert filled.post_fill_snapshot_id == "snapshot-post-fill"
    assert suffix is not None
    assert suffix.status == PortfolioComponentStatus.REOPTIMIZATION_REQUIRED
    assert suffix.resolution_status == PortfolioComponentResolutionStatus.OPERATOR_RESOLVED

    request = next(
        (
            item
            for item in updated
            if getattr(item, "request_id", None)
        ),
        None,
    )
    if request is None:
        remaining = ("tuple-b",)
        request = await registry.portfolio_reoptimizations.get_by_remaining(
            session_id=OWNER.session_id,
            broker_account_identity=OWNER.broker_account_identity,
            trading_date=OWNER.trading_date,
            original_portfolio_decision_id="decision-a",
            remaining_authorized_tuple_ids=remaining,
        )
    assert request is not None
    assert request.status == PortfolioReoptimizationStatus.COMPLETED
    assert request.replacement_action == "WAIT"
    assert "tuple-a" in request.already_filled_tuple_ids
    assert request.remaining_authorized_tuple_ids == ("tuple-b",)
    assert not await registry.portfolio_reoptimizations.list_pending(
        session_id=OWNER.session_id,
        broker_account_identity=OWNER.broker_account_identity,
        trading_date=OWNER.trading_date,
    )
    assert not await registry.portfolio_executions.has_unresolved(
        session_id=OWNER.session_id,
        broker_account_identity=OWNER.broker_account_identity,
        trading_date=OWNER.trading_date,
    )


def test_production_runner_post_poll_fill_is_recorded_in_terminal_request(
    tmp_path, monkeypatch
) -> None:
    """LivePaperRunner must reconcile from a post-poll projection in the same pass."""
    import asyncio
    from pathlib import Path

    from joker.broker.webull_trade_api import MockWebullTradeApi
    from joker.config.settings import AppSettings, EnvSettings
    from joker.runtime.cognitive_session import paper_account_identity
    from joker.runtime.compatibility import CompatibilityLivePaperBridge
    from joker.runtime.live_paper_runner import LivePaperRunConfig, LivePaperRunner
    from joker.runtime.portfolio_recovery import PortfolioRecoveryCoordinator

    prior_date = "2026-08-05"
    app = AppSettings(db_path=str(tmp_path / "app.db"))
    app = app.model_copy(
        update={
            "broker": app.broker.model_copy(update={"provider": "webull_paper"}),
            "agents": app.agents.model_copy(update={"runtime": "cognitive_graph"}),
        }
    )
    env = EnvSettings(
        _env_file=None,
        OPENAI_API_KEY="test-ci-key-not-real",
        WEBULL_MARKET_DATA_ENABLED=False,
        WEBULL_LIVE_TRADING_ENABLED=False,
        WEBULL_PAPER_TRADING_ENABLED=True,
        WEBULL_PAPER_ACCOUNT_ID="PAPER_ACCT_1",
        WEBULL_APP_KEY="paper-key",
        WEBULL_APP_SECRET="paper-secret",
        WEBULL_ACCESS_TOKEN="paper-token",
    )
    runner = LivePaperRunner(app, env)
    api = MockWebullTradeApi(account_id="PAPER_ACCT_1")
    account_identity = paper_account_identity(broker_kind="webull_paper", env=env)
    session_id = f"cog:paper:{account_identity}:{prior_date}"
    client_filled = "client-tuple-a"
    client_suffix = "client-tuple-b"
    ordering: list[str] = []

    async def _seed() -> CognitiveExecutionProvenanceRegistry:
        task1_db = Path(app.db_path).parent / "joker_task1.db"
        registry = CognitiveExecutionProvenanceRegistry(
            task1_db.with_name(task1_db.stem + "_cognitive_provenance.db")
        )
        await registry.initialize()
        owner = PortfolioExecutionOwner(
            session_id=session_id,
            broker_account_identity=account_identity,
            trading_date=prior_date,
        )
        working = PortfolioExecutionComponentRecord(
            session_id=owner.session_id,
            origin_run_id="run-a",
            broker_account_identity=owner.broker_account_identity,
            trading_date=owner.trading_date,
            target_portfolio_decision_id="decision-a",
            selected_portfolio_id="portfolio-a",
            authorized_position_tuple_id="tuple-a",
            component_index=0,
            component_count=2,
            strategy_id="strategy-tuple-a",
            contract_id="contract-tuple-a",
            authorized_quantity=1,
            capital_allocation=Decimal("100"),
            client_order_id=client_filled,
            status=PortfolioComponentStatus.WORKING,
            remaining_quantity=1,
            submitted_quantity=1,
            filled_quantity=0,
            original_decision_snapshot_id="snapshot-a",
            evaluated_objective_version=1,
            evaluated_timestamp=NOW.isoformat(),
        )
        await registry.portfolio_executions.authorize(working)
        authorized = await registry.portfolio_executions.authorize(
            PortfolioExecutionComponentRecord(
                session_id=owner.session_id,
                origin_run_id="run-a",
                broker_account_identity=owner.broker_account_identity,
                trading_date=owner.trading_date,
                target_portfolio_decision_id="decision-a",
                selected_portfolio_id="portfolio-a",
                authorized_position_tuple_id="tuple-b",
                component_index=1,
                component_count=2,
                strategy_id="strategy-tuple-b",
                contract_id="contract-tuple-b",
                authorized_quantity=1,
                capital_allocation=Decimal("100"),
                client_order_id=client_suffix,
                status=PortfolioComponentStatus.AUTHORIZED,
                remaining_quantity=1,
                original_decision_snapshot_id="snapshot-a",
                evaluated_objective_version=1,
                evaluated_timestamp=NOW.isoformat(),
            )
        )
        assert authorized.status == PortfolioComponentStatus.AUTHORIZED
        return registry

    registry = asyncio.run(_seed())

    class Obj:
        version = 5
        status = "deadline_reached"
        objective_id = "objective-a"
        available_capital_usd = Decimal("900")
        reserved_capital_usd = Decimal("100")
        working_order_reservation_usd = Decimal("0")
        filled_position_exposure_usd = Decimal("100")
        required_profit_remaining_usd = Decimal("50")
        realised_pnl_usd = Decimal("0")
        deadline_exchange_time = NOW
        time_remaining_seconds = 0
        entries_paused = True
        truth_degraded = False
        open_position_count = 1
        max_concurrent_positions = 3

        def as_dict(self):
            return {"version": 5, "status": "deadline_reached"}

    objective_service = SimpleNamespace(
        get_state=AsyncMock(return_value=Obj()),
        recompute_from_truth=AsyncMock(return_value=Obj()),
        repository=None,
    )
    real_bridge = CompatibilityLivePaperBridge

    class TrackingBridge(real_bridge):
        def start(self, *, start_agent: bool = True) -> None:
            super().start(start_agent=start_agent)
            self._fill_written = False
            self._project_count = 0
            if self.supervisor.snapshot_repository is not None:
                self.supervisor.snapshot_repository.get_latest = AsyncMock(  # type: ignore[method-assign]
                    return_value=SimpleNamespace(snapshot_id="snapshot-post-fill")
                )

        def project_session(self):
            self._project_count += 1
            if not self._fill_written:
                ordering.append("project:WORKING")
                return SimpleNamespace(
                    orders={
                        client_filled: SimpleNamespace(
                            client_order_id=client_filled,
                            status="working",
                            filled_qty=0,
                            order_id="broker-a",
                        )
                    },
                    positions={},
                )
            ordering.append("project:FILLED")
            return SimpleNamespace(
                orders={
                    client_filled: SimpleNamespace(
                        client_order_id=client_filled,
                        status="filled",
                        filled_qty=1,
                        order_id="broker-a",
                    )
                },
                positions={
                    "contract-tuple-a": {
                        "contract_id": "contract-tuple-a",
                        "quantity": 1,
                    }
                },
            )

        def poll_order_status(self, client_order_id: str):
            assert client_order_id == client_filled
            self._fill_written = True
            ordering.append("poll:FINAL_FILL")
            return SimpleNamespace(
                client_order_id=client_filled,
                status="filled",
                filled_qty=1,
                order_id="broker-a",
            )

    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.CompatibilityLivePaperBridge",
        TrackingBridge,
    )
    monkeypatch.setattr(
        "joker.runtime.objective_recovery.recover_session_objective",
        AsyncMock(return_value=None),
    )

    original_reconcile = PortfolioRecoveryCoordinator.reconcile_owner_components

    async def _tracked_reconcile(self, *args, **kwargs):
        projection = kwargs.get("projection")
        orders = getattr(projection, "orders", {}) or {}
        order = orders.get(client_filled) if isinstance(orders, dict) else None
        status = str(getattr(order, "status", "missing") or "missing").lower()
        ordering.append(f"coordinator:{status}")
        return await original_reconcile(self, *args, **kwargs)

    monkeypatch.setattr(
        PortfolioRecoveryCoordinator,
        "reconcile_owner_components",
        _tracked_reconcile,
    )

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=0.1,
            mock_agents=True,
            require_options=False,
            recovery_mode="reconciliation_only",
            recovery_owner_trading_date=prior_date,
            reconciliation_only_recovery=True,
            trade_api=api,
            cognitive_session_id_override=session_id,
            objective_service=objective_service,
        )
    )
    assert result.feed_health == "RECOVERY_ONLY"

    # Same-iteration ordering: initial project → poll → fresh project → coordinator.
    first_pass = ordering[:4]
    assert first_pass == [
        "project:WORKING",
        "poll:FINAL_FILL",
        "project:FILLED",
        "coordinator:filled",
    ], f"stale pre-poll projection ordering observed: {ordering}"

    filled = asyncio.run(registry.portfolio_executions.get("tuple-a"))
    suffix = asyncio.run(registry.portfolio_executions.get("tuple-b"))
    assert filled is not None
    assert filled.status == PortfolioComponentStatus.FILLED
    assert filled.continuation_ready is True
    assert suffix is not None
    assert suffix.status == PortfolioComponentStatus.REOPTIMIZATION_REQUIRED
    assert suffix.resolution_status == PortfolioComponentResolutionStatus.OPERATOR_RESOLVED
    assert suffix.submitted_quantity == 0

    request = asyncio.run(
        registry.portfolio_reoptimizations.get_by_remaining(
            session_id=session_id,
            broker_account_identity=account_identity,
            trading_date=prior_date,
            original_portfolio_decision_id="decision-a",
            remaining_authorized_tuple_ids=("tuple-b",),
        )
    )
    assert request is not None
    assert request.status == PortfolioReoptimizationStatus.COMPLETED
    assert request.replacement_action == "WAIT"
    assert "tuple-a" in request.already_filled_tuple_ids
    assert request.remaining_authorized_tuple_ids == ("tuple-b",)
    assert not asyncio.run(
        registry.portfolio_reoptimizations.list_pending(
            session_id=session_id,
            broker_account_identity=account_identity,
            trading_date=prior_date,
        )
    )
    assert not asyncio.run(
        registry.portfolio_executions.has_unresolved(
            session_id=session_id,
            broker_account_identity=account_identity,
            trading_date=prior_date,
        )
    )
