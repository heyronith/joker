"""Authoritative portfolio sizing integrity — no silent mutate/resize."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from joker.cognition.schemas import (
    ExecutionLeg,
    ExecutionProposal,
    MetaDecisionAction,
)
from joker.config.settings import CognitiveGraphSettings
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.objective_nodes import (
    apply_objective_sizing_to_proposal,
    deterministic_sizing_node,
)
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.objectives.schemas import SessionObjectiveState
from joker.objectives.sizing import DeterministicObjectiveSizer


def _router() -> ModelRouter:
    profiles = {
        n: p.model_copy(update={"provider": "fake", "model": "fake"})
        for n, p in default_model_profiles().items()
    }
    return ModelRouter(
        ModelRegistry(
            ModelsConfig(profiles=profiles), providers={"fake": FakeModelProvider()}
        ),
        session_id="portfolio-sizing",
    )


def _obj_state(**overrides: object) -> SessionObjectiveState:
    data = {
        "objective_id": uuid4(),
        "session_id": "s",
        "status": "active",
        "authorised_capital_usd": Decimal("500"),
        "target_profit_usd": Decimal("50"),
        "target_ending_equity_usd": Decimal("550"),
        "working_order_reservation_usd": Decimal("0"),
        "filled_position_exposure_usd": Decimal("0"),
        "reserved_capital_usd": Decimal("0"),
        "available_capital_usd": Decimal("500"),
        "realised_pnl_usd": Decimal("0"),
        "unrealised_pnl_usd": Decimal("0"),
        "progress_to_goal_pct": Decimal("0"),
        "required_profit_remaining_usd": Decimal("50"),
        "time_remaining_seconds": 1800,
        "objective_duration_seconds": 3600,
        "version": 4,
        "max_concurrent_positions": 3,
        "open_position_count": 0,
        "deadline_exchange_time": datetime(2026, 8, 5, 16, 0, tzinfo=timezone.utc),
    }
    data.update(overrides)
    return SessionObjectiveState.model_validate(data)


class _Svc:
    def __init__(self, state):
        self._state = state

    async def get_state(self):
        return self._state


def _authorized(
    *,
    strategy_id,
    contract_id: str,
    quantity: int,
    premium: str,
    decision_id,
    snapshot_id,
    portfolio_id,
    position_tuple_id=None,
    objective_version: int = 4,
):
    capital = (Decimal(premium) * Decimal(quantity) * Decimal("100")).quantize(
        Decimal("0.01")
    )
    return {
        "position_tuple_id": str(position_tuple_id or uuid4()),
        "strategy_id": str(strategy_id),
        "contract_id": contract_id,
        "quantity": quantity,
        "evaluation_premium": premium,
        "capital_allocation": str(capital),
        "maximum_loss": str(capital),
        "snapshot_id": str(snapshot_id),
        "objective_version": objective_version,
        "decision_id": str(decision_id),
        "evaluated_objective_fingerprint": None,
    }


def _leg(
    *,
    strategy_id,
    contract_id: str,
    quantity: int,
    premium: Decimal,
    capital: Decimal,
    decision_id,
    portfolio_id,
    position_tuple_id,
    snapshot_id,
    component_index: int,
    component_count: int,
    objective_version: int = 4,
) -> ExecutionLeg:
    return ExecutionLeg(
        strategy_id=strategy_id,
        contract_id=contract_id,
        side="buy",
        quantity=quantity,
        limit_price=premium,
        evaluation_premium=premium,
        capital_allocation=capital,
        authorized_position_tuple_id=position_tuple_id,
        target_portfolio_decision_id=decision_id,
        selected_portfolio_id=portfolio_id,
        component_index=component_index,
        component_count=component_count,
        evaluated_objective_version=objective_version,
        original_decision_snapshot_id=snapshot_id,
        sequence_order=component_index + 1,
        max_quote_age_seconds=30,
        replacement_policy="reject",
        partial_fill_policy="reject",
    )


def _proposal(*, strategy_id, snapshot_id, decision_id, legs) -> ExecutionProposal:
    return ExecutionProposal(
        decision_id=decision_id,
        strategy_id=strategy_id,
        session_id="s",
        cycle_id="c1",
        snapshot_id=snapshot_id,
        action="execute",
        legs=tuple(legs),
        order_type="limit",
        time_in_force="DAY",
        entry_rationale="test",
        prompt_version="test",
        model_call_id=uuid4(),
    )


def _portfolio_state(
    *,
    proposal: ExecutionProposal,
    positions: list[dict],
    decision_id,
    snapshot_id,
    portfolio_id,
    available: Decimal = Decimal("500"),
) -> tuple[CognitiveGraphDeps, CognitiveGraphState]:
    obj = _obj_state(available_capital_usd=available)
    deps = CognitiveGraphDeps(
        router=_router(),
        config=CognitiveGraphSettings(),
        session_id="s",
        run_id="s",
        objective_service=_Svc(obj),  # type: ignore[arg-type]
        capital_sizer=DeterministicObjectiveSizer(
            maximum_authorised_contracts=20,
            require_positive_expected_value=False,
        ),
        objective_policy="target_attainment",
    )
    state: CognitiveGraphState = {
        "execution_proposal": proposal,
        "meta_decision": SimpleNamespace(
            action=MetaDecisionAction.EXECUTE,
            selected_strategy_id=proposal.strategy_id,
        ),  # type: ignore[typeddict-item]
        "snapshot_id": str(snapshot_id),
        "_target_portfolio_decision": {
            "action": "enter",
            "decision_id": str(decision_id),
            "snapshot_id": str(snapshot_id),
            "objective_version": 4,
            "selected_portfolio_id": str(portfolio_id),
        },
        "_target_authorized_positions": positions,
        "_strategy_estimates": [
            {
                "strategy_id": str(proposal.strategy_id),
                "valid": True,
                "estimate_id": str(uuid4()),
                "quote_inputs": {"premium_per_contract": "1.00"},
            }
        ],
        "errors": [],
    }
    return deps, state


def _two_component_bundle(*, qty_a: int = 2, qty_b: int = 5):
    sid_a, sid_b = uuid4(), uuid4()
    decision_id = uuid4()
    snapshot_id = uuid4()
    portfolio_id = uuid4()
    tuple_a, tuple_b = uuid4(), uuid4()
    prem_a, prem_b = Decimal("1.00"), Decimal("0.50")
    positions = [
        _authorized(
            strategy_id=sid_a,
            contract_id="SPY:A",
            quantity=qty_a,
            premium=str(prem_a),
            decision_id=decision_id,
            snapshot_id=snapshot_id,
            portfolio_id=portfolio_id,
            position_tuple_id=tuple_a,
        ),
        _authorized(
            strategy_id=sid_b,
            contract_id="SPY:B",
            quantity=qty_b,
            premium=str(prem_b),
            decision_id=decision_id,
            snapshot_id=snapshot_id,
            portfolio_id=portfolio_id,
            position_tuple_id=tuple_b,
        ),
    ]
    legs = [
        _leg(
            strategy_id=sid_a,
            contract_id="SPY:A",
            quantity=qty_a,
            premium=prem_a,
            capital=Decimal(positions[0]["capital_allocation"]),
            decision_id=decision_id,
            portfolio_id=portfolio_id,
            position_tuple_id=tuple_a,
            snapshot_id=snapshot_id,
            component_index=0,
            component_count=2,
        ),
        _leg(
            strategy_id=sid_b,
            contract_id="SPY:B",
            quantity=qty_b,
            premium=prem_b,
            capital=Decimal(positions[1]["capital_allocation"]),
            decision_id=decision_id,
            portfolio_id=portfolio_id,
            position_tuple_id=tuple_b,
            snapshot_id=snapshot_id,
            component_index=1,
            component_count=2,
        ),
    ]
    proposal = _proposal(
        strategy_id=sid_a,
        snapshot_id=snapshot_id,
        decision_id=decision_id,
        legs=legs,
    )
    return proposal, positions, decision_id, snapshot_id, portfolio_id


@pytest.mark.asyncio
async def test_compiled_graph_rejects_component_count_change() -> None:
    proposal, positions, decision_id, snapshot_id, portfolio_id = _two_component_bundle()
    # Drop one authorized component without changing proposal legs.
    deps, state = _portfolio_state(
        proposal=proposal,
        positions=positions[:1],
        decision_id=decision_id,
        snapshot_id=snapshot_id,
        portfolio_id=portfolio_id,
    )
    out = await apply_objective_sizing_to_proposal(deps, state)
    assert "execution_proposal" not in out or out.get("_block_new_entries")
    assert any(
        getattr(e, "error_code", None) == "target_attainment_recalculation_required"
        for e in (out.get("errors") or [])
    )
    codes = (out.get("_sizing_decision") or {}).get("reason_codes") or []
    assert "component_count_changed" in codes


@pytest.mark.asyncio
async def test_compiled_graph_rejects_contract_or_quantity_mutation() -> None:
    proposal, positions, decision_id, snapshot_id, portfolio_id = _two_component_bundle()
    mutated = proposal.model_copy(
        update={
            "legs": (
                proposal.legs[0].model_copy(
                    update={"contract_id": "SPY:MUTATED", "quantity": 9}
                ),
                proposal.legs[1],
            )
        }
    )
    deps, state = _portfolio_state(
        proposal=mutated,
        positions=positions,
        decision_id=decision_id,
        snapshot_id=snapshot_id,
        portfolio_id=portfolio_id,
    )
    out = await apply_objective_sizing_to_proposal(deps, state)
    assert out.get("_block_new_entries") is True
    codes = (out.get("_sizing_decision") or {}).get("reason_codes") or []
    assert "contract_changed:0" in codes
    assert "quantity_changed:0" in codes


@pytest.mark.asyncio
async def test_target_portfolio_quantities_are_never_normalized_to_first_leg() -> None:
    proposal, positions, decision_id, snapshot_id, portfolio_id = _two_component_bundle(
        qty_a=2, qty_b=5
    )
    deps, state = _portfolio_state(
        proposal=proposal,
        positions=positions,
        decision_id=decision_id,
        snapshot_id=snapshot_id,
        portfolio_id=portfolio_id,
    )
    out = await apply_objective_sizing_to_proposal(deps, state)
    sized = out["execution_proposal"]
    assert [leg.quantity for leg in sized.legs] == [2, 5]
    assert [leg.quantity for leg in sized.legs] != [2, 2]


@pytest.mark.asyncio
async def test_generic_sizer_cannot_shrink_one_portfolio_component() -> None:
    # Capital-sizer alone would shrink qty 5 @ $2 → but portfolio path must reject.
    proposal, positions, decision_id, snapshot_id, portfolio_id = _two_component_bundle(
        qty_a=1, qty_b=5
    )
    # Make component B unaffordable if resized individually under tight capital —
    # mismatch by lowering available so total allocation fails, not silent shrink.
    positions[1]["quantity"] = 5
    positions[1]["capital_allocation"] = "500.00"
    proposal = proposal.model_copy(
        update={
            "legs": (
                proposal.legs[0],
                proposal.legs[1].model_copy(
                    update={
                        "quantity": 5,
                        "limit_price": Decimal("1.00"),
                        "evaluation_premium": Decimal("1.00"),
                        "capital_allocation": Decimal("500.00"),
                    }
                ),
            )
        }
    )
    # Proposal still matches authorized; available capital cannot fund total.
    deps, state = _portfolio_state(
        proposal=proposal,
        positions=positions,
        decision_id=decision_id,
        snapshot_id=snapshot_id,
        portfolio_id=portfolio_id,
        available=Decimal("400"),
    )
    out = await apply_objective_sizing_to_proposal(deps, state)
    assert out.get("_block_new_entries") is True
    assert "execution_proposal" not in out or out.get("execution_proposal") is proposal
    codes = (out.get("_sizing_decision") or {}).get("reason_codes") or []
    assert "portfolio_allocation_unavailable" in codes
    # Authorized quantities untouched in state.
    assert [int(p["quantity"]) for p in positions] == [1, 5]


@pytest.mark.asyncio
async def test_generic_sizer_cannot_enlarge_one_portfolio_component() -> None:
    proposal, positions, decision_id, snapshot_id, portfolio_id = _two_component_bundle(
        qty_a=1, qty_b=1
    )
    enlarged = proposal.model_copy(
        update={
            "legs": (
                proposal.legs[0],
                proposal.legs[1].model_copy(update={"quantity": 4}),
            )
        }
    )
    deps, state = _portfolio_state(
        proposal=enlarged,
        positions=positions,
        decision_id=decision_id,
        snapshot_id=snapshot_id,
        portfolio_id=portfolio_id,
    )
    out = await apply_objective_sizing_to_proposal(deps, state)
    assert out.get("_block_new_entries") is True
    codes = (out.get("_sizing_decision") or {}).get("reason_codes") or []
    assert "quantity_changed:1" in codes


@pytest.mark.asyncio
async def test_total_portfolio_allocation_is_validated_without_mutation() -> None:
    proposal, positions, decision_id, snapshot_id, portfolio_id = _two_component_bundle()
    original_qtys = [leg.quantity for leg in proposal.legs]
    original_caps = [leg.capital_allocation for leg in proposal.legs]
    deps, state = _portfolio_state(
        proposal=proposal,
        positions=positions,
        decision_id=decision_id,
        snapshot_id=snapshot_id,
        portfolio_id=portfolio_id,
        available=Decimal("50"),
    )
    out = await apply_objective_sizing_to_proposal(deps, state)
    assert out.get("_block_new_entries") is True
    codes = (out.get("_sizing_decision") or {}).get("reason_codes") or []
    assert "portfolio_allocation_unavailable" in codes
    assert [leg.quantity for leg in proposal.legs] == original_qtys
    assert [leg.capital_allocation for leg in proposal.legs] == original_caps


@pytest.mark.asyncio
async def test_deterministic_sizing_preserves_exact_portfolio_allocation() -> None:
    proposal, positions, decision_id, snapshot_id, portfolio_id = _two_component_bundle()
    deps, state = _portfolio_state(
        proposal=proposal,
        positions=positions,
        decision_id=decision_id,
        snapshot_id=snapshot_id,
        portfolio_id=portfolio_id,
    )
    out = await deterministic_sizing_node(deps, state)
    decision = out["_sizing_decision"]
    assert decision["approved"] is True
    assert decision["approved_quantity"] == 7
    assert "authoritative_portfolio_exact_allocation" in decision["reason_codes"]
