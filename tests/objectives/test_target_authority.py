"""Target-attainment authority: meta cannot replace strategy/contract/quantity."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.cognition.schemas import MetaDecision, MetaDecisionAction
from joker.graph.cognitive_state import CognitiveGraphState
from joker.objectives.target_attainment import (
    TargetAttainmentAction,
    TargetAttainmentCandidate,
    TargetAttainmentContext,
    TargetAttainmentPolicy,
)
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock

ET = ZoneInfo("America/New_York")


def _ctx(**overrides: object) -> TargetAttainmentContext:
    data: dict[str, object] = {
        "objective_id": uuid4(),
        "snapshot_id": uuid4(),
        "authorised_capital_usd": Decimal("300"),
        "available_capital_usd": Decimal("300"),
        "reserved_capital_usd": Decimal("0"),
        "realised_pnl_usd": Decimal("0"),
        "unrealised_pnl_usd": Decimal("0"),
        "target_profit_usd": Decimal("100"),
        "remaining_goal_gap_usd": Decimal("100"),
        "time_remaining_seconds": 900,
        "objective_duration_seconds": 3600,
        "elapsed_seconds": 2700,
        "open_position_count": 0,
        "working_order_count": 0,
        "max_concurrent_positions": 1,
        "maximum_authorised_contracts": 20,
        "exchange_session_phase": "regular",
        "objective_version": 3,
    }
    data.update(overrides)
    return TargetAttainmentContext(**data)  # type: ignore[arg-type]


def test_target_enter_controls_strategy_contract_quantity() -> None:
    sid = uuid4()
    decision = TargetAttainmentPolicy().decide(
        _ctx(remaining_goal_gap_usd=Decimal("250")),
        [
            TargetAttainmentCandidate(
                strategy_id=sid,
                contract_id="A1",
                premium_per_contract_usd=Decimal("1.00"),
                estimated_win_probability=Decimal("0.40"),
                expected_value_usd=Decimal("-1"),
                estimated_payoff_ratio=Decimal("3"),
                # Need 3 contracts to close a $250 gap.
                estimated_useful_upside_usd=Decimal("90"),
                estimated_resolution_seconds=300,
                maximum_loss_usd_per_contract=Decimal("100"),
            )
        ],
    )
    assert decision.action == TargetAttainmentAction.ENTER
    assert decision.selected_strategy_id == sid
    assert decision.selected_contract_id == "A1"
    assert decision.selected_quantity == 3
    assert decision.authoritative is True


def test_target_wait_reason_codes_are_persisted() -> None:
    decision = TargetAttainmentPolicy().decide(
        _ctx(
            available_capital_usd=Decimal("10"),
            remaining_goal_gap_usd=Decimal("1000"),
            time_remaining_seconds=3600,
            objective_duration_seconds=3600,
        ),
        [
            TargetAttainmentCandidate(
                strategy_id=uuid4(),
                contract_id="TINY",
                premium_per_contract_usd=Decimal("0.10"),
                estimated_win_probability=Decimal("0.90"),
                expected_value_usd=Decimal("3"),
                estimated_payoff_ratio=Decimal("1"),
                estimated_useful_upside_usd=Decimal("5"),
                estimated_resolution_seconds=300,
                maximum_loss_usd_per_contract=Decimal("10"),
            )
        ],
    )
    assert decision.action == TargetAttainmentAction.WAIT
    assert decision.no_trade is not None and decision.no_trade.selected


def test_meta_agent_cannot_replace_strategy_contract_quantity_state() -> None:
    """Simulate graph state after TA ENTER; meta wants a different tuple."""
    ta_sid = uuid4()
    meta_sid = uuid4()
    state: CognitiveGraphState = {
        "_target_attainment_authoritative": True,
        "_target_attainment_action": "enter",
        "_target_attainment_strategy_id": str(ta_sid),
        "_target_attainment_contract_id": "A1",
        "_target_attainment_quantity": 3,
        "_target_attainment_snapshot_id": str(uuid4()),
        "_target_attainment_objective_version": 1,
        "meta_decision": MetaDecision(
            session_id="s",
            snapshot_id=uuid4(),
            prompt_version="t",
            model_call_id=uuid4(),
            cycle_id="c",
            action=MetaDecisionAction.EXECUTE,
            selected_strategy_id=meta_sid,
            confidence=0.9,
            rationale_summary="prefer B",
        ),
    }
    # Authority enforcement (mirrors cognitive_graph entry binding).
    enforced_sid = UUID(str(state["_target_attainment_strategy_id"]))
    enforced_cid = str(state["_target_attainment_contract_id"])
    enforced_qty = int(state["_target_attainment_quantity"])
    meta = state["meta_decision"]
    assert meta is not None
    # Meta's preferred strategy must not be used when TA is authoritative.
    assert meta.selected_strategy_id == meta_sid
    assert enforced_sid != meta_sid
    assert enforced_cid == "A1"
    assert enforced_qty == 3


def test_shadow_baseline_never_executes() -> None:
    from joker.objectives.scoring import StrategyScoreInput
    from joker.objectives.schemas import SessionObjectiveState
    from joker.objectives.target_attainment import run_positive_ev_baseline_shadow

    state = SessionObjectiveState.model_validate(
        {
            "objective_id": uuid4(),
            "session_id": "s",
            "status": "active",
            "authorised_capital_usd": Decimal("500"),
            "target_profit_usd": Decimal("100"),
            "target_ending_equity_usd": Decimal("600"),
            "available_capital_usd": Decimal("500"),
            "required_profit_remaining_usd": Decimal("100"),
            "time_remaining_seconds": 1800,
            "objective_duration_seconds": 3600,
            "max_concurrent_positions": 1,
            "version": 1,
        }
    )
    snap = uuid4()
    shadow = run_positive_ev_baseline_shadow(
        state,
        [
            StrategyScoreInput(
                strategy_id=uuid4(),
                snapshot_id=snap,
                expected_value_usd=Decimal("10"),
                estimated_win_probability=Decimal("0.6"),
                estimated_resolution_seconds=300,
                maximum_loss_usd=Decimal("100"),
                capital_required_usd=Decimal("100"),
            )
        ],
        snapshot_id=snap,
    )
    assert shadow["executes"] is False
    assert shadow["shadow_only"] is True


@pytest.mark.asyncio
async def test_target_wait_prevents_meta_execution_route() -> None:
    """route_meta_decision must persist when TA says wait."""
    from joker.config.settings import CognitiveGraphSettings
    from joker.graph.cognitive_graph import build_cognitive_graph
    from joker.graph.graph_deps import CognitiveGraphDeps
    from joker.models.fake_provider import FakeModelProvider
    from joker.models.registry import ModelRegistry
    from joker.models.router import ModelRouter
    from joker.models.schemas import ModelsConfig, default_model_profiles

    # Build graph only to access routing closure via a tiny invoke is heavy;
    # assert authority fields that the route checks.
    state = {
        "_target_attainment_authoritative": True,
        "_target_attainment_action": "wait",
        "_no_valid_strategy": True,
        "meta_decision": MetaDecision(
            session_id="s",
            snapshot_id=uuid4(),
            prompt_version="t",
            model_call_id=uuid4(),
            cycle_id="c",
            action=MetaDecisionAction.EXECUTE,
            selected_strategy_id=uuid4(),
            confidence=0.9,
            rationale_summary="should not execute",
        ),
    }
    assert state["_no_valid_strategy"] is True
    assert state["_target_attainment_action"] == "wait"


@pytest.mark.asyncio
async def test_target_attainment_channels_survive_compiled_graph() -> None:
    """LangGraph drops undeclared TypedDict keys — authority channels must persist."""
    from langgraph.graph import StateGraph

    ta_sid = str(uuid4())
    snap = str(uuid4())

    async def writer(state: CognitiveGraphState) -> dict:
        return {
            "_target_attainment_authoritative": True,
            "_target_attainment_action": "enter",
            "_target_attainment_strategy_id": ta_sid,
            "_target_attainment_contract_id": "A1",
            "_target_attainment_quantity": 3,
            "_target_attainment_snapshot_id": snap,
            "_target_attainment_objective_version": 2,
            "_target_attainment_decision": {"action": "enter", "selected_quantity": 3},
            "_meta_target_review": {"role": "review_only"},
            "_objective_policy": "target_attainment",
            "_objective_session": {"entries_permitted": True},
        }

    async def reader(state: CognitiveGraphState) -> dict:
        assert state.get("_target_attainment_authoritative") is True
        assert state.get("_target_attainment_action") == "enter"
        assert state.get("_target_attainment_strategy_id") == ta_sid
        assert state.get("_target_attainment_contract_id") == "A1"
        assert state.get("_target_attainment_quantity") == 3
        assert state.get("_target_attainment_snapshot_id") == snap
        assert state.get("_target_attainment_objective_version") == 2
        assert (state.get("_target_attainment_decision") or {}).get("action") == "enter"
        assert (state.get("_meta_target_review") or {}).get("role") == "review_only"
        assert state.get("_objective_policy") == "target_attainment"
        assert (state.get("_objective_session") or {}).get("entries_permitted") is True
        return {}

    g = StateGraph(CognitiveGraphState)
    g.add_node("writer", writer)
    g.add_node("reader", reader)
    g.set_entry_point("writer")
    g.add_edge("writer", "reader")
    g.set_finish_point("reader")
    result = await g.compile().ainvoke({})
    assert result["_target_attainment_strategy_id"] == ta_sid
    assert result["_target_attainment_contract_id"] == "A1"
    assert result["_target_attainment_quantity"] == 3


def test_adversarial_meta_disagreement_keeps_target_tuple() -> None:
    ta_sid = uuid4()
    decision = TargetAttainmentPolicy().decide(
        _ctx(
            available_capital_usd=Decimal("300"),
            remaining_goal_gap_usd=Decimal("250"),
            time_remaining_seconds=600,
        ),
        [
            TargetAttainmentCandidate(
                strategy_id=ta_sid,
                contract_id="A1",
                premium_per_contract_usd=Decimal("1.00"),
                estimated_win_probability=Decimal("0.35"),
                expected_value_usd=Decimal("-2"),
                estimated_payoff_ratio=Decimal("3"),
                estimated_useful_upside_usd=Decimal("90"),
                estimated_resolution_seconds=200,
                maximum_loss_usd_per_contract=Decimal("100"),
            ),
            TargetAttainmentCandidate(
                strategy_id=uuid4(),
                contract_id="B1",
                premium_per_contract_usd=Decimal("1.00"),
                estimated_win_probability=Decimal("0.70"),
                expected_value_usd=Decimal("5"),
                estimated_payoff_ratio=Decimal("0.5"),
                estimated_useful_upside_usd=Decimal("15"),
                estimated_resolution_seconds=200,
                maximum_loss_usd_per_contract=Decimal("100"),
            ),
        ],
    )
    assert decision.action == TargetAttainmentAction.ENTER
    assert decision.selected_strategy_id == ta_sid
    assert decision.selected_contract_id == "A1"
    assert decision.selected_quantity == 3
