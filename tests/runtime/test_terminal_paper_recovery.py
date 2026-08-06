from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from joker.persistence.cognitive_execution_provenance import (
    PortfolioExecutionOwner,
    PortfolioReoptimizationRepository,
    PortfolioReoptimizationRequestRecord,
    stable_reoptimization_request_id,
)
from joker.runtime.cognitive_agent_runtime import CognitiveAgentRuntime


class _RealtimeClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def trading_date(self) -> date:
        return date(2026, 8, 5)


def _request(now: datetime) -> PortfolioReoptimizationRequestRecord:
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
        origin_run_id="run-origin",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
        original_portfolio_decision_id="decision-a",
        already_filled_tuple_ids=("tuple-a",),
        open_positions=({"contract_id": "contract-a", "quantity": 1},),
        remaining_authorized_tuple_ids=("tuple-b",),
        reason_codes=("capital_changed",),
        latest_objective_state={"version": 2},
        latest_objective_version=2,
        latest_snapshot_id="snapshot-a",
        created_exchange_time=now.isoformat(),
    )


async def _runtime_with_retry(tmp_path):
    db = tmp_path / "state.db"
    repo = PortfolioReoptimizationRepository(db)
    now = datetime.now(timezone.utc)
    pending = await repo.enqueue(_request(now))
    running = await repo.begin_attempt(
        pending.request_id,
        owner=PortfolioExecutionOwner("session-a", "paper-a", "2026-08-05"),
        current_run_id="run-other",
        attempt_exchange_time=now.isoformat(),
        lease_seconds=0.2,
    )
    runtime = CognitiveAgentRuntime.__new__(CognitiveAgentRuntime)
    runtime._session_id = "session-a"
    runtime._run_id = "run-self"
    runtime._shutdown = False
    runtime._reconciliation_only_recovery = False
    runtime._reoptimization_retry_tasks = {}
    runtime._active_decision_tasks = set()
    runtime._active_position_tasks = set()
    runtime._decision_worker = None
    runtime._position_worker = None
    runtime._status = "healthy"
    runtime._deps = SimpleNamespace(
        provenance_registry=SimpleNamespace(portfolio_reoptimizations=repo),
        clock=_RealtimeClock(),
        execution_runtime=SimpleNamespace(broker_account_identity="paper-a"),
        broker_account_identity="paper-a",
    )
    calls: list[str] = []

    async def _resume():
        calls.append("resume")

    runtime._resume_pending_portfolio_reoptimizations = _resume  # type: ignore[method-assign]
    return runtime, running, calls


@pytest.mark.asyncio
async def test_restart_before_lease_expiry_retries_after_expiry(tmp_path) -> None:
    runtime, running, calls = await _runtime_with_retry(tmp_path)
    await runtime._schedule_reoptimization_retry(
        running,
        lease_expiry=running.attempt_lease_expires_at,
    )
    await asyncio.sleep(0.35)
    assert calls == ["resume"]


@pytest.mark.asyncio
async def test_new_generation_executes_once(tmp_path) -> None:
    runtime, running, calls = await _runtime_with_retry(tmp_path)
    await runtime._schedule_reoptimization_retry(
        running,
        lease_expiry=running.attempt_lease_expires_at,
    )
    await runtime._schedule_reoptimization_retry(
        running,
        lease_expiry=running.attempt_lease_expires_at,
    )
    await asyncio.sleep(0.35)
    assert calls == ["resume"]


@pytest.mark.asyncio
async def test_retry_task_stops_on_shutdown(tmp_path) -> None:
    runtime, running, calls = await _runtime_with_retry(tmp_path)
    await runtime._schedule_reoptimization_retry(
        running,
        lease_expiry=running.attempt_lease_expires_at,
    )
    await runtime.pause_event_workers()
    await asyncio.sleep(0.25)
    assert calls == []
