"""Full-graph missing/negative EV cannot reach ExecutionRuntime."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.cognition.schemas import MetaDecisionAction
from joker.graph.objective_nodes import apply_objective_sizing_to_proposal
from joker.objectives.schemas import SessionObjectiveState
from joker.objectives.sizing import DeterministicObjectiveSizer


class _Deps:
    def __init__(self, sizer, service):
        self.capital_sizer = sizer
        self.objective_service = service


class _Svc:
    def __init__(self, state):
        self._state = state
        self.require_positive_expected_value = True

    async def get_state(self):
        return self._state


def _state() -> SessionObjectiveState:
    return SessionObjectiveState.model_validate(
        {
            "objective_id": uuid4(),
            "session_id": "g",
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
            "time_remaining_seconds": 3600,
            "version": 1,
            "max_concurrent_positions": 1,
            "deadline_exchange_time": datetime.now(tz=ZoneInfo("America/New_York"))
            + timedelta(hours=1),
        }
    )


class _Leg:
    quantity = 2
    limit_price = Decimal("1.00")

    def model_copy(self, *, update):
        out = _Leg()
        for k, v in update.items():
            setattr(out, k, v)
        return out


class _Proposal:
    legs = (_Leg(),)
    action = "entry"
    strategy_id = uuid4()

    def model_copy(self, *, update):
        out = _Proposal()
        for k, v in update.items():
            setattr(out, k, v)
        return out


class _Meta:
    action = MetaDecisionAction.EXECUTE
    selected_strategy_id = uuid4()


@pytest.mark.asyncio
async def test_apply_sizing_blocks_missing_estimate() -> None:
    deps = _Deps(DeterministicObjectiveSizer(), _Svc(_state()))
    state = {
        "execution_proposal": _Proposal(),
        "meta_decision": _Meta(),
        "_strategy_estimates": [],
        "errors": [],
    }
    out = await apply_objective_sizing_to_proposal(deps, state)  # type: ignore[arg-type]
    assert any(e.error_code == "estimate_invalid" for e in out.get("errors") or [])


@pytest.mark.asyncio
async def test_apply_sizing_blocks_negative_ev_estimate() -> None:
    sid = uuid4()
    meta = _Meta()
    meta.selected_strategy_id = sid
    deps = _Deps(DeterministicObjectiveSizer(), _Svc(_state()))
    state = {
        "execution_proposal": _Proposal(),
        "meta_decision": meta,
        "_strategy_estimates": [
            {
                "estimate_id": str(uuid4()),
                "strategy_id": str(sid),
                "valid": True,
                "expected_value_usd": "-5.00",
                "estimated_win_probability": "0.55",
                "estimated_payoff_ratio": None,
                "quote_inputs": {"premium_per_contract": "1.00"},
            }
        ],
        "errors": [],
    }
    out = await apply_objective_sizing_to_proposal(deps, state)  # type: ignore[arg-type]
    assert any(e.error_code == "sizing_rejected" for e in out.get("errors") or [])
    assert out.get("_sizing_decision", {}).get("approved") is False
