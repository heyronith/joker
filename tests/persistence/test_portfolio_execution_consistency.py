from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.config.settings import EnvSettings
from joker.persistence.cognitive_execution_provenance import (
    PortfolioComponentStatus,
    PortfolioAttemptLeaseActive,
    PortfolioExecutionComponentRecord,
    PortfolioExecutionOwner,
    PortfolioExecutionRepository,
    PortfolioReoptimizationRepository,
    PortfolioReoptimizationRequestRecord,
    PortfolioReoptimizationResolutionStatus,
    PortfolioReoptimizationStatus,
    PortfolioTransitionConflict,
    apply_portfolio_execution_migration,
    stable_reoptimization_request_id,
)
from joker.runtime.cognitive_session import paper_account_identity


NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc).isoformat()


def _owner(
    *,
    session_id: str = "session-a",
    broker_account_id: str = "paper-a",
    trading_date: str = "2026-08-05",
) -> PortfolioExecutionOwner:
    return PortfolioExecutionOwner(
        session_id=session_id,
        broker_account_identity=broker_account_id,
        trading_date=trading_date,
    )


def _record(
    tuple_id: str,
    *,
    owner: PortfolioExecutionOwner | None = None,
    decision_id: str = "decision-a",
    component_index: int = 0,
    component_count: int = 1,
    quantity: int = 2,
    origin_run_id: str = "run-a",
) -> PortfolioExecutionComponentRecord:
    scoped = owner or _owner()
    return PortfolioExecutionComponentRecord(
        session_id=scoped.session_id,
        origin_run_id=origin_run_id,
        broker_account_identity=scoped.broker_account_identity,
        trading_date=scoped.trading_date,
        target_portfolio_decision_id=decision_id,
        selected_portfolio_id="portfolio-a",
        authorized_position_tuple_id=tuple_id,
        component_index=component_index,
        component_count=component_count,
        strategy_id=f"strategy-{tuple_id}",
        contract_id=f"contract-{tuple_id}",
        authorized_quantity=quantity,
        capital_allocation=Decimal("200"),
        client_order_id=f"client-{tuple_id}",
        status=PortfolioComponentStatus.AUTHORIZED,
        remaining_quantity=quantity,
        original_decision_snapshot_id="snapshot-a",
        evaluated_objective_version=1,
        evaluated_timestamp=NOW,
    )


@pytest.mark.asyncio
async def test_different_session_pending_portfolio_is_not_resumed(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", owner=_owner(session_id="other")))

    assert (
        await repo.list_resumable(
            session_id="session-a",
            broker_account_identity="paper-a",
            trading_date="2026-08-05",
        )
        == []
    )


@pytest.mark.asyncio
async def test_different_broker_account_pending_portfolio_is_not_resumed(
    tmp_path,
) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", owner=_owner(broker_account_id="paper-b")))

    assert (
        await repo.list_resumable(
            session_id="session-a",
            broker_account_identity="paper-a",
            trading_date="2026-08-05",
        )
        == []
    )


@pytest.mark.asyncio
async def test_matching_session_and_account_resume_normally(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a"))

    rows = await repo.list_resumable(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
    )
    assert [row.authorized_position_tuple_id for row in rows] == ["tuple-a"]


@pytest.mark.asyncio
async def test_same_session_account_date_new_run_resumes_component(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", origin_run_id=str(uuid4())))
    current_run_id = str(uuid4())

    rows = await repo.list_resumable(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
    )
    resumed = await repo.record_resume(
        rows[0].authorized_position_tuple_id,
        owner=_owner(),
        current_run_id=current_run_id,
        resumed_at=NOW,
    )

    assert resumed.origin_run_id != current_run_id
    assert resumed.last_resumed_run_id == current_run_id
    assert resumed.resume_count == 1


@pytest.mark.asyncio
async def test_origin_run_id_remains_immutable_after_resume(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    origin_run_id = str(uuid4())
    await repo.authorize(_record("tuple-a", origin_run_id=origin_run_id))
    await repo.record_resume(
        "tuple-a",
        owner=_owner(),
        current_run_id=str(uuid4()),
        resumed_at=NOW,
    )
    stored = await repo.get("tuple-a")
    assert stored is not None
    assert stored.origin_run_id == origin_run_id


@pytest.mark.asyncio
async def test_different_trading_date_still_cannot_resume(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", owner=_owner(trading_date="2026-08-04")))
    assert await repo.list_resumable(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
    ) == []


def test_provider_kind_alone_is_not_account_identity() -> None:
    with pytest.raises(ValueError, match="provider kind"):
        _owner(broker_account_id="webull_paper")


@pytest.mark.asyncio
async def test_raw_webull_account_id_is_never_persisted(tmp_path) -> None:
    raw_account_id = "RAW_WEBULL_ACCOUNT_123456"
    identity = paper_account_identity(
        broker_kind="webull_paper",
        env=EnvSettings.model_construct(webull_paper_account_id=raw_account_id),
    )
    db_path = tmp_path / "state.db"
    repo = PortfolioExecutionRepository(db_path)
    await repo.authorize(
        _record("tuple-a", owner=_owner(broker_account_id=identity))
    )

    assert raw_account_id.encode() not in db_path.read_bytes()
    stored = await repo.get("tuple-a")
    assert stored is not None and stored.broker_account_id == identity


@pytest.mark.asyncio
async def test_matching_hashed_account_resumes_normally(tmp_path) -> None:
    identity = paper_account_identity(
        broker_kind="webull_paper",
        env=EnvSettings.model_construct(webull_paper_account_id="ACCOUNT_A"),
    )
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", owner=_owner(broker_account_id=identity)))
    rows = await repo.list_resumable(
        session_id="session-a",
        broker_account_identity=identity,
        trading_date="2026-08-05",
    )
    assert [row.authorized_position_tuple_id for row in rows] == ["tuple-a"]


@pytest.mark.asyncio
async def test_different_hashed_account_cannot_resume(tmp_path) -> None:
    identity_a = paper_account_identity(
        broker_kind="webull_paper",
        env=EnvSettings.model_construct(webull_paper_account_id="ACCOUNT_A"),
    )
    identity_b = paper_account_identity(
        broker_kind="webull_paper",
        env=EnvSettings.model_construct(webull_paper_account_id="ACCOUNT_B"),
    )
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", owner=_owner(broker_account_id=identity_a)))
    assert await repo.list_resumable(
        session_id="session-a",
        broker_account_identity=identity_b,
        trading_date="2026-08-05",
    ) == []


@pytest.mark.asyncio
async def test_two_sessions_in_one_database_do_not_cross_resume(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a"))
    await repo.authorize(
        _record(
            "tuple-b",
            owner=_owner(session_id="session-b"),
            origin_run_id="run-b",
            decision_id="decision-b",
        )
    )

    rows = await repo.list_resumable(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
    )
    assert [row.authorized_position_tuple_id for row in rows] == ["tuple-a"]


def test_legacy_unscoped_record_fails_closed(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as db:
        db.execute(
            """
            CREATE TABLE portfolio_execution_components (
                target_portfolio_decision_id TEXT NOT NULL,
                selected_portfolio_id TEXT,
                authorized_position_tuple_id TEXT PRIMARY KEY NOT NULL,
                component_index INTEGER NOT NULL,
                component_count INTEGER NOT NULL,
                strategy_id TEXT NOT NULL,
                contract_id TEXT NOT NULL,
                authorized_quantity INTEGER NOT NULL,
                capital_allocation TEXT NOT NULL,
                client_order_id TEXT NOT NULL UNIQUE,
                broker_order_id TEXT,
                status TEXT NOT NULL,
                submitted_quantity INTEGER NOT NULL,
                filled_quantity INTEGER NOT NULL,
                remaining_quantity INTEGER NOT NULL,
                original_decision_snapshot_id TEXT NOT NULL,
                latest_validation_snapshot_id TEXT,
                evaluated_objective_version INTEGER NOT NULL,
                submission_objective_version INTEGER,
                evaluated_timestamp TEXT NOT NULL,
                last_validation_timestamp TEXT,
                last_reconciliation_timestamp TEXT,
                failure_reoptimization_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            INSERT INTO portfolio_execution_components VALUES (
                'decision', NULL, 'tuple', 0, 1, 'strategy', 'contract', 1,
                '100', 'client', NULL, 'WORKING', 1, 0, 1, 'snapshot', NULL,
                1, 1, ?, ?, ?, NULL, ?, ?, '{}'
            )
            """,
            (NOW, NOW, NOW, NOW, NOW),
        )
        db.commit()

    apply_portfolio_execution_migration(db_path)
    with sqlite3.connect(db_path) as db:
        status, reason, session_id = db.execute(
            """SELECT status, failure_reoptimization_reason, session_id
            FROM portfolio_execution_components"""
        ).fetchone()
    assert status == "REOPTIMIZATION_REQUIRED"
    assert reason == "legacy_unscoped_portfolio_component"
    assert session_id is None


@pytest.mark.asyncio
async def test_stale_reconciliation_cannot_reduce_filled_quantity(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a"))
    await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.SUBMITTED,
        submitted_quantity=2,
    )
    partial = await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.PARTIALLY_FILLED,
        submitted_quantity=2,
        filled_quantity=1,
    )

    with pytest.raises(ValueError, match="cannot decrease"):
        await repo.transition(
            "tuple-a",
            owner=_owner(),
            status=PortfolioComponentStatus.PARTIALLY_FILLED,
            submitted_quantity=2,
            filled_quantity=0,
            expected_state_version=partial.state_version,
        )


@pytest.mark.asyncio
async def test_duplicate_fill_is_idempotent(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a"))
    await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.SUBMITTED,
        submitted_quantity=2,
    )
    filled = await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.FILLED,
        submitted_quantity=2,
        filled_quantity=2,
    )
    duplicate = await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.FILLED,
        submitted_quantity=2,
        filled_quantity=2,
    )
    assert duplicate.state_version == filled.state_version


@pytest.mark.asyncio
async def test_out_of_order_working_event_cannot_regress_filled(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", quantity=1))
    await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.SUBMITTED,
        submitted_quantity=1,
    )
    await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.FILLED,
        submitted_quantity=1,
        filled_quantity=1,
    )

    with pytest.raises(ValueError, match="invalid portfolio component transition"):
        await repo.transition(
            "tuple-a",
            owner=_owner(),
            status=PortfolioComponentStatus.WORKING,
            submitted_quantity=1,
            filled_quantity=1,
        )


@pytest.mark.asyncio
async def test_filled_with_incomplete_quantity_is_rejected(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a"))
    await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.READY,
    )

    with pytest.raises(ValueError, match="FILLED requires"):
        await repo.transition(
            "tuple-a",
            owner=_owner(),
            status=PortfolioComponentStatus.FILLED,
            submitted_quantity=2,
            filled_quantity=1,
        )


@pytest.mark.asyncio
async def test_concurrent_transition_has_one_winner(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    initial = await repo.authorize(_record("tuple-a"))

    async def submit() -> PortfolioExecutionComponentRecord | BaseException:
        try:
            return await repo.transition(
                "tuple-a",
                owner=_owner(),
                status=PortfolioComponentStatus.SUBMITTED,
                submitted_quantity=2,
                expected_state_version=initial.state_version,
                expected_status=PortfolioComponentStatus.AUTHORIZED,
            )
        except BaseException as exc:  # test records the losing CAS result
            return exc

    results = await asyncio.gather(submit(), submit())
    assert sum(isinstance(result, PortfolioTransitionConflict) for result in results) == 1
    stored = await repo.get("tuple-a")
    assert stored is not None and stored.state_version == 1


@pytest.mark.asyncio
async def test_terminal_component_cannot_return_to_working(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", quantity=1))
    await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
    )
    with pytest.raises(ValueError, match="invalid portfolio component transition"):
        await repo.transition(
            "tuple-a",
            owner=_owner(),
            status=PortfolioComponentStatus.WORKING,
            submitted_quantity=1,
        )


@pytest.mark.asyncio
async def test_post_fill_checkpoint_is_atomic_and_authoritative(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", quantity=1))
    submitted = await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.SUBMITTED,
        submitted_quantity=1,
    )
    filled = await repo.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.FILLED,
        submitted_quantity=1,
        filled_quantity=1,
        post_fill_objective_version=3,
        post_fill_objective_fingerprint='{"objective_id":"objective"}',
        post_fill_snapshot_id="snapshot-post-fill",
        post_fill_exchange_time=NOW,
        reconciled_filled_quantity=1,
        continuation_ready=True,
        expected_state_version=submitted.state_version,
    )
    assert filled.continuation_ready is True
    assert filled.post_fill_snapshot_id == "snapshot-post-fill"
    assert filled.reconciled_filled_quantity == 1
    assert filled.state_version == submitted.state_version + 1


def _reoptimization_request(*, origin_run_id: str) -> PortfolioReoptimizationRequestRecord:
    request_id = stable_reoptimization_request_id(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
        original_portfolio_decision_id="decision-a",
        remaining_authorized_tuple_ids=("tuple-b",),
    )
    return PortfolioReoptimizationRequestRecord(
        request_id=request_id,
        session_id="session-a",
        origin_run_id=origin_run_id,
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
        original_portfolio_decision_id="decision-a",
        already_filled_tuple_ids=("tuple-a",),
        open_positions=({"contract_id": "contract-a", "quantity": 1},),
        remaining_authorized_tuple_ids=("tuple-b",),
        reason_codes=("capital_changed",),
        latest_objective_state={"version": 2},
        latest_objective_version=2,
        latest_snapshot_id="snapshot-b",
        created_exchange_time=NOW,
    )


@pytest.mark.asyncio
async def test_same_session_account_date_new_run_resumes_reoptimization(tmp_path) -> None:
    repo = PortfolioReoptimizationRepository(tmp_path / "state.db")
    origin_run_id = str(uuid4())
    request = await repo.enqueue(_reoptimization_request(origin_run_id=origin_run_id))
    current_run_id = str(uuid4())

    pending = await repo.list_pending(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
    )
    attempted = await repo.begin_attempt(
        pending[0].request_id,
        owner=_owner(),
        current_run_id=current_run_id,
        attempt_exchange_time=NOW,
    )

    assert attempted.request_id == request.request_id
    assert attempted.origin_run_id == origin_run_id
    assert attempted.last_resumed_run_id == current_run_id
    assert attempted.last_attempt_run_id == current_run_id
    assert attempted.resume_count == 1
    assert attempted.attempt_count == 1


def test_different_run_alone_does_not_change_reoptimization_identity() -> None:
    kwargs = {
        "session_id": "session-a",
        "broker_account_identity": "paper-a",
        "trading_date": "2026-08-05",
        "original_portfolio_decision_id": "decision-a",
        "remaining_authorized_tuple_ids": ("tuple-b",),
    }
    assert stable_reoptimization_request_id(**kwargs) == stable_reoptimization_request_id(
        **kwargs
    )


@pytest.mark.asyncio
async def test_duplicate_component_index_is_rejected(tmp_path) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(
        _record("tuple-a", component_index=0, component_count=2)
    )
    with pytest.raises(ValueError, match="duplicate portfolio component index"):
        await repo.authorize(
            _record("tuple-b", component_index=0, component_count=2)
        )


@pytest.mark.asyncio
async def test_same_component_index_is_allowed_for_different_decisions_and_owners(
    tmp_path,
) -> None:
    repo = PortfolioExecutionRepository(tmp_path / "state.db")
    await repo.authorize(_record("tuple-a", decision_id="decision-a"))
    await repo.authorize(_record("tuple-b", decision_id="decision-b"))
    await repo.authorize(
        _record(
            "tuple-c",
            decision_id="decision-a",
            owner=_owner(session_id="session-b"),
        )
    )
    assert await repo.get("tuple-b") is not None
    assert await repo.get("tuple-c") is not None


def test_migration_duplicate_component_order_fails_closed(tmp_path) -> None:
    db_path = tmp_path / "duplicates.db"
    apply_portfolio_execution_migration(db_path)
    with sqlite3.connect(db_path) as db:
        db.execute("DROP INDEX idx_portfolio_component_order")
        base = _record("tuple-a", component_index=0, component_count=2)
        columns = [
            row[1]
            for row in db.execute("PRAGMA table_info(portfolio_execution_components)")
        ]
        values = {
            "session_id": base.session_id,
            "run_id": base.origin_run_id,
            "origin_run_id": base.origin_run_id,
            "broker_account_id": base.broker_account_id,
            "trading_date": base.trading_date,
            "target_portfolio_decision_id": base.target_portfolio_decision_id,
            "selected_portfolio_id": base.selected_portfolio_id,
            "component_index": 0,
            "component_count": 2,
            "strategy_id": "strategy",
            "contract_id": "contract",
            "authorized_quantity": 1,
            "capital_allocation": "100",
            "status": "AUTHORIZED",
            "submitted_quantity": 0,
            "filled_quantity": 0,
            "remaining_quantity": 1,
            "original_decision_snapshot_id": "snapshot",
            "evaluated_objective_version": 1,
            "evaluated_timestamp": NOW,
            "continuation_ready": 0,
            "state_version": 0,
            "component_order_conflicted": 0,
            "created_at": NOW,
            "updated_at": NOW,
            "payload_json": "{}",
        }
        for suffix in ("a", "b"):
            row = dict(values)
            row["authorized_position_tuple_id"] = f"tuple-{suffix}"
            row["client_order_id"] = f"client-{suffix}"
            used = [column for column in columns if column in row]
            db.execute(
                f"INSERT INTO portfolio_execution_components ({','.join(used)}) "
                f"VALUES ({','.join('?' for _ in used)})",
                tuple(row[column] for column in used),
            )
        db.commit()

    apply_portfolio_execution_migration(db_path)
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            """SELECT status, failure_reoptimization_reason,
                      component_order_conflicted
               FROM portfolio_execution_components ORDER BY authorized_position_tuple_id"""
        ).fetchall()
        index_sql = db.execute(
            "SELECT sql FROM sqlite_master WHERE name='idx_portfolio_component_order'"
        ).fetchone()[0]
    assert rows == [
        ("REOPTIMIZATION_REQUIRED", "duplicate_component_index_migration", 1),
        ("REOPTIMIZATION_REQUIRED", "duplicate_component_index_migration", 1),
    ]
    assert "UNIQUE INDEX" in index_sql


@pytest.mark.asyncio
async def test_unresolved_component_and_failed_request_block_until_resolution(tmp_path) -> None:
    db = tmp_path / "state.db"
    components = PortfolioExecutionRepository(db)
    requests = PortfolioReoptimizationRepository(db)
    await components.authorize(_record("tuple-a"))
    assert await components.has_unresolved(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
    )
    await components.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.SUBMITTED,
        submitted_quantity=2,
    )
    await components.transition(
        "tuple-a",
        owner=_owner(),
        status=PortfolioComponentStatus.FILLED,
        submitted_quantity=2,
        filled_quantity=2,
    )
    assert not await components.has_unresolved(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
    )
    request = await requests.enqueue(_reoptimization_request(origin_run_id="run-a"))
    running = await requests.begin_attempt(
        request.request_id,
        owner=_owner(),
        current_run_id="run-a",
        attempt_exchange_time=NOW,
    )
    failed = await requests.transition(
        request.request_id,
        status=PortfolioReoptimizationStatus.FAILED,
        failure_reason="manual_review_required",
        expected_state_version=running.state_version,
    )
    assert failed.resolution_status == PortfolioReoptimizationResolutionStatus.UNRESOLVED
    assert await requests.has_unresolved(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
    )
    await requests.resolve_failed(
        request.request_id,
        resolved_at=NOW,
        resolved_by="operator:test",
        resolution_reason="broker truth reconciled",
    )
    assert not await requests.has_unresolved(
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
    )


@pytest.mark.asyncio
async def test_reoptimization_attempt_lease_claim_and_takeover(tmp_path) -> None:
    repo = PortfolioReoptimizationRepository(tmp_path / "state.db")
    request = await repo.enqueue(_reoptimization_request(origin_run_id="origin"))
    first = await repo.begin_attempt(
        request.request_id,
        owner=_owner(),
        current_run_id="run-one",
        attempt_exchange_time=NOW,
        lease_seconds=30,
    )
    same = await repo.begin_attempt(
        request.request_id,
        owner=_owner(),
        current_run_id="run-one",
        attempt_exchange_time=(datetime.fromisoformat(NOW) + timedelta(seconds=1)).isoformat(),
        lease_seconds=30,
    )
    assert same.state_version == first.state_version
    assert same.attempt_count == 1
    with pytest.raises(PortfolioAttemptLeaseActive):
        await repo.begin_attempt(
            request.request_id,
            owner=_owner(),
            current_run_id="run-two",
            attempt_exchange_time=(
                datetime.fromisoformat(NOW) + timedelta(seconds=2)
            ).isoformat(),
            lease_seconds=30,
        )
    takeover = await repo.begin_attempt(
        request.request_id,
        owner=_owner(),
        current_run_id="run-two",
        attempt_exchange_time=(datetime.fromisoformat(NOW) + timedelta(seconds=31)).isoformat(),
        lease_seconds=30,
    )
    assert takeover.attempt_owner_run_id == "run-two"
    assert takeover.attempt_count == 2


@pytest.mark.asyncio
async def test_concurrent_reoptimization_claim_has_one_winner(tmp_path) -> None:
    repo = PortfolioReoptimizationRepository(tmp_path / "state.db")
    request = await repo.enqueue(_reoptimization_request(origin_run_id="origin"))

    async def claim(run_id):
        try:
            return await repo.begin_attempt(
                request.request_id,
                owner=_owner(),
                current_run_id=run_id,
                attempt_exchange_time=NOW,
            )
        except BaseException as exc:
            return exc

    results = await asyncio.gather(claim("run-one"), claim("run-two"))
    assert sum(isinstance(result, PortfolioReoptimizationRequestRecord) for result in results) == 1
    assert sum(isinstance(result, PortfolioTransitionConflict) for result in results) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_status",
    [PortfolioReoptimizationStatus.COMPLETED, PortfolioReoptimizationStatus.FAILED],
)
async def test_terminal_reoptimization_cannot_be_reclaimed(
    tmp_path, terminal_status
) -> None:
    repo = PortfolioReoptimizationRepository(tmp_path / "state.db")
    request = await repo.enqueue(_reoptimization_request(origin_run_id="origin"))
    running = await repo.begin_attempt(
        request.request_id,
        owner=_owner(),
        current_run_id="run-one",
        attempt_exchange_time=NOW,
    )
    await repo.transition(
        request.request_id,
        status=terminal_status,
        replacement_decision_id=("decision-new" if terminal_status.value == "COMPLETED" else None),
        replacement_action=("WAIT" if terminal_status.value == "COMPLETED" else None),
        failure_reason=("manual resolution required" if terminal_status.value == "FAILED" else None),
        expected_state_version=running.state_version,
    )
    with pytest.raises(ValueError, match="terminal"):
        await repo.begin_attempt(
            request.request_id,
            owner=_owner(),
            current_run_id="run-two",
            attempt_exchange_time=(datetime.fromisoformat(NOW) + timedelta(minutes=10)).isoformat(),
        )
