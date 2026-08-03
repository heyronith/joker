"""Entry order causation event ID must persist as factual horizon-start provenance."""

from __future__ import annotations

from uuid import uuid4

import pytest

from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
    ExecutionProvenanceRecord,
)
from joker.runtime.order_action_gateway import OrderActionKind, OrderActionRequest


@pytest.mark.asyncio
async def test_entry_provenance_persists_causation_event_id(tmp_path) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    causation = str(uuid4())
    client_order_id = f"entry-{uuid4()}"
    request = OrderActionRequest(
        action=OrderActionKind.ENTRY,
        client_order_id=client_order_id,
        contract_id="SPY:2026-07-01:500:call",
        side="buy",
        quantity=1,
        order_type="limit",
        limit_price=1.10,
        causation_event_id=causation,
        cycle_id="cycle-1",
        strategy_id=str(uuid4()),
        proposal_id=str(uuid4()),
        decision_id=str(uuid4()),
        snapshot_id=str(uuid4()),
    )
    assert request.causation_event_id == causation

    await registry.record(
        ExecutionProvenanceRecord(
            client_order_id=request.client_order_id,
            proposal_id=request.proposal_id,
            decision_id=request.decision_id,
            strategy_id=request.strategy_id,
            cycle_id=request.cycle_id,
            snapshot_id=request.snapshot_id,
            contract_id=request.contract_id,
            session_id="s",
            kind=request.action.value,
            causation_event_id=request.causation_event_id,
            extra={"causation_event_id": request.causation_event_id},
        )
    )
    stored = await registry.get_by_client_order_id(client_order_id)
    assert stored is not None
    assert stored.causation_event_id == causation
    assert (stored.extra or {}).get("causation_event_id") == causation
    # Fill events are never the cognitive entry anchor.
    assert stored.kind == "entry"
