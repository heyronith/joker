from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
    PortfolioComponentStatus,
    PortfolioExecutionComponentRecord,
    PortfolioExecutionOwner,
    stable_portfolio_client_order_id,
)
from joker.runtime.cognitive_agent_runtime import CognitiveAgentRuntime


NOW = "2026-08-05T15:00:00+00:00"
OWNER = PortfolioExecutionOwner(
    session_id="session-a",
    broker_account_identity="paper-a",
    trading_date="2026-08-05",
)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="request-a",
        session_id=OWNER.session_id,
        broker_account_identity=OWNER.broker_account_identity,
        trading_date=OWNER.trading_date,
        owner=OWNER,
        original_portfolio_decision_id="decision-old",
        already_filled_tuple_ids=("tuple-filled",),
        remaining_authorized_tuple_ids=("tuple-old",),
        open_positions=({"contract_id": "SPY:2026-08-05:499:call", "quantity": 1},),
    )


def _position(tuple_id: str, contract_id: str) -> dict:
    return {
        "position_tuple_id": tuple_id,
        "contract_id": contract_id,
        "quantity": 1,
        "strategy_id": f"strategy-{tuple_id}",
        "capital_allocation": "100",
        "decision_id": "decision-new",
        "objective_version": 4,
        "snapshot_id": "snapshot-new",
    }


def _record(
    tuple_id: str,
    contract_id: str,
    *,
    component_index: int,
    component_count: int = 2,
    status: PortfolioComponentStatus,
) -> PortfolioExecutionComponentRecord:
    submitted = 1 if status in {
        PortfolioComponentStatus.SUBMITTED,
        PortfolioComponentStatus.WORKING,
        PortfolioComponentStatus.PARTIALLY_FILLED,
        PortfolioComponentStatus.FILLED,
    } else 0
    filled = 1 if status == PortfolioComponentStatus.FILLED else 0
    return PortfolioExecutionComponentRecord(
        session_id=OWNER.session_id,
        origin_run_id="run-one",
        broker_account_identity=OWNER.broker_account_identity,
        trading_date=OWNER.trading_date,
        target_portfolio_decision_id="decision-new",
        selected_portfolio_id="portfolio-new",
        authorized_position_tuple_id=tuple_id,
        component_index=component_index,
        component_count=component_count,
        strategy_id=f"strategy-{tuple_id}",
        contract_id=contract_id,
        authorized_quantity=1,
        capital_allocation=Decimal("100"),
        client_order_id=stable_portfolio_client_order_id("decision-new", tuple_id),
        broker_order_id=(f"broker-{tuple_id}" if submitted else None),
        status=status,
        submitted_quantity=submitted,
        filled_quantity=filled,
        remaining_quantity=1 - filled,
        original_decision_snapshot_id="snapshot-new",
        latest_validation_snapshot_id=("snapshot-new" if submitted else None),
        evaluated_objective_version=4,
        submission_objective_version=(4 if submitted else None),
        evaluated_timestamp=NOW,
    )


def _result(positions: list[dict]) -> dict:
    return {
        "_portfolio_reoptimization_request_id": "request-a",
        "_portfolio_execution_owner": {
            "session_id": OWNER.session_id,
            "broker_account_identity": OWNER.broker_account_identity,
            "trading_date": OWNER.trading_date,
        },
        "_target_portfolio_decision": {
            "decision_id": "decision-new",
            "action": "ENTER",
            "selected_portfolio_id": "portfolio-new",
            "authorized_positions": positions,
            "objective_version": 4,
            "snapshot_id": "snapshot-new",
        },
        "_target_authorized_positions": positions,
        "_reoptimization_excluded_contract_ids": ["SPY:2026-08-05:499:call"],
        "_reoptimization_expected_objective_version": 4,
        "_reoptimization_expected_snapshot_id": "snapshot-new",
        "execution_proposal": object(),
        "_execution_command_ids": ["command-one"],
        "errors": [],
        "node_trace": [
            {"node_name": "validate_execution_proposal", "status": "completed"},
            {"node_name": "submit_execution_command", "status": "completed"},
            {"node_name": "persist_cycle", "status": "completed"},
        ],
    }


async def _runtime(tmp_path, records):
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "provenance.db")
    await registry.initialize()
    for record in records:
        await registry.portfolio_executions.authorize(record)
    runtime = CognitiveAgentRuntime.__new__(CognitiveAgentRuntime)
    runtime._deps = SimpleNamespace(provenance_registry=registry)
    runtime._session_id = OWNER.session_id
    return runtime


@pytest.mark.asyncio
async def test_multi_component_reoptimization_working_prefix_is_valid(tmp_path) -> None:
    positions = [
        _position("tuple-new-a", "SPY:2026-08-05:500:call"),
        _position("tuple-new-b", "SPY:2026-08-05:501:call"),
    ]
    runtime = await _runtime(
        tmp_path,
        [
            _record(
                "tuple-new-a",
                positions[0]["contract_id"],
                component_index=0,
                status=PortfolioComponentStatus.WORKING,
            ),
            _record(
                "tuple-new-b",
                positions[1]["contract_id"],
                component_index=1,
                status=PortfolioComponentStatus.AUTHORIZED,
            ),
        ],
    )
    valid, reason, decision_id, action = await runtime._validate_reoptimization_result(
        _request(), _result(positions)
    )
    assert (valid, reason, decision_id, action) == (
        True,
        "",
        "decision-new",
        "ENTER",
    )


@pytest.mark.asyncio
async def test_authorized_suffix_does_not_require_submission_provenance(tmp_path) -> None:
    positions = [
        _position("tuple-new-a", "SPY:2026-08-05:500:call"),
        _position("tuple-new-b", "SPY:2026-08-05:501:call"),
    ]
    suffix = _record(
        "tuple-new-b",
        positions[1]["contract_id"],
        component_index=1,
        status=PortfolioComponentStatus.AUTHORIZED,
    )
    assert suffix.submission_objective_version is None
    assert suffix.latest_validation_snapshot_id is None
    runtime = await _runtime(
        tmp_path,
        [
            _record(
                "tuple-new-a",
                positions[0]["contract_id"],
                component_index=0,
                status=PortfolioComponentStatus.WORKING,
            ),
            suffix,
        ],
    )
    valid, reason, *_ = await runtime._validate_reoptimization_result(
        _request(), _result(positions)
    )
    assert valid is True, reason


@pytest.mark.asyncio
async def test_two_active_working_replacement_components_are_rejected(tmp_path) -> None:
    positions = [
        _position("tuple-new-a", "SPY:2026-08-05:500:call"),
        _position("tuple-new-b", "SPY:2026-08-05:501:call"),
    ]
    runtime = await _runtime(
        tmp_path,
        [
            _record(
                "tuple-new-a",
                positions[0]["contract_id"],
                component_index=0,
                status=PortfolioComponentStatus.WORKING,
            ),
            _record(
                "tuple-new-b",
                positions[1]["contract_id"],
                component_index=1,
                status=PortfolioComponentStatus.WORKING,
            ),
        ],
    )
    valid, reason, *_ = await runtime._validate_reoptimization_result(
        _request(), _result(positions)
    )
    assert valid is False
    assert reason == "replacement_component_sequence_invalid"


@pytest.mark.asyncio
async def test_noncontiguous_replacement_components_are_rejected(tmp_path) -> None:
    positions = [
        _position("tuple-new-a", "SPY:2026-08-05:500:call"),
        _position("tuple-new-b", "SPY:2026-08-05:501:call"),
    ]
    runtime = await _runtime(
        tmp_path,
        [
            _record(
                "tuple-new-a",
                positions[0]["contract_id"],
                component_index=0,
                component_count=3,
                status=PortfolioComponentStatus.WORKING,
            ),
            _record(
                "tuple-new-b",
                positions[1]["contract_id"],
                component_index=2,
                component_count=3,
                status=PortfolioComponentStatus.AUTHORIZED,
            ),
        ],
    )
    valid, reason, *_ = await runtime._validate_reoptimization_result(
        _request(), _result(positions)
    )
    assert valid is False
    assert reason == "replacement_component_order_invalid"


@pytest.mark.asyncio
async def test_replacement_strategy_mismatch_is_rejected(tmp_path) -> None:
    position = _position("tuple-new-a", "SPY:2026-08-05:500:call")
    bad = position | {"strategy_id": "different-strategy"}
    runtime = await _runtime(
        tmp_path,
        [
            _record(
                "tuple-new-a",
                position["contract_id"],
                component_index=0,
                component_count=1,
                status=PortfolioComponentStatus.WORKING,
            )
        ],
    )
    valid, reason, *_ = await runtime._validate_reoptimization_result(
        _request(), _result([bad])
    )
    assert valid is False
    assert reason == "replacement_strategy_mismatch"


@pytest.mark.asyncio
async def test_replacement_capital_allocation_mismatch_is_rejected(tmp_path) -> None:
    position = _position("tuple-new-a", "SPY:2026-08-05:500:call")
    bad = position | {"capital_allocation": "125"}
    runtime = await _runtime(
        tmp_path,
        [
            _record(
                "tuple-new-a",
                position["contract_id"],
                component_index=0,
                component_count=1,
                status=PortfolioComponentStatus.WORKING,
            )
        ],
    )
    valid, reason, *_ = await runtime._validate_reoptimization_result(
        _request(), _result([bad])
    )
    assert valid is False
    assert reason == "replacement_capital_allocation_mismatch"


@pytest.mark.asyncio
async def test_replacement_portfolio_id_mismatch_is_rejected(tmp_path) -> None:
    position = _position("tuple-new-a", "SPY:2026-08-05:500:call")
    result = _result([position])
    result["_target_portfolio_decision"]["selected_portfolio_id"] = "portfolio-other"
    runtime = await _runtime(
        tmp_path,
        [
            _record(
                "tuple-new-a",
                position["contract_id"],
                component_index=0,
                component_count=1,
                status=PortfolioComponentStatus.WORKING,
            )
        ],
    )
    valid, reason, *_ = await runtime._validate_reoptimization_result(
        _request(), result
    )
    assert valid is False
    assert reason == "replacement_portfolio_id_mismatch"


@pytest.mark.asyncio
async def test_filled_replacement_with_stale_submission_version_is_rejected(tmp_path) -> None:
    position = _position("tuple-new-a", "SPY:2026-08-05:500:call")
    filled = _record(
        "tuple-new-a",
        position["contract_id"],
        component_index=0,
        component_count=1,
        status=PortfolioComponentStatus.FILLED,
    )
    filled = replace(
        filled,
        submission_objective_version=3,
        continuation_ready=True,
        post_fill_objective_version=4,
        post_fill_objective_fingerprint='{"objective_id":"objective"}',
        post_fill_snapshot_id="snapshot-new",
        post_fill_exchange_time=NOW,
        reconciled_filled_quantity=1,
    )
    runtime = await _runtime(tmp_path, [filled])
    valid, reason, *_ = await runtime._validate_reoptimization_result(
        _request(), _result([position])
    )
    assert valid is False
    assert reason == "replacement_submission_provenance_invalid"
