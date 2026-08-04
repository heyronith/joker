"""Strategy × contract × quantity evaluation tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from joker.objectives.contract_candidates import build_contract_candidates_for_strategies
from joker.objectives.target_attainment import (
    TargetAttainmentAction,
    TargetAttainmentCandidate,
    TargetAttainmentContext,
    TargetAttainmentPolicy,
)


def _ctx(**overrides: object) -> TargetAttainmentContext:
    data: dict[str, object] = {
        "objective_id": uuid4(),
        "snapshot_id": uuid4(),
        "authorised_capital_usd": Decimal("500.00"),
        "available_capital_usd": Decimal("500.00"),
        "reserved_capital_usd": Decimal("0.00"),
        "realised_pnl_usd": Decimal("0.00"),
        "unrealised_pnl_usd": Decimal("0.00"),
        "target_profit_usd": Decimal("100.00"),
        "remaining_goal_gap_usd": Decimal("100.00"),
        "time_remaining_seconds": 1800,
        "objective_duration_seconds": 3600,
        "elapsed_seconds": 1800,
        "open_position_count": 0,
        "working_order_count": 0,
        "max_concurrent_positions": 1,
        "maximum_authorised_contracts": 20,
        "exchange_session_phase": "regular",
    }
    data.update(overrides)
    return TargetAttainmentContext(**data)  # type: ignore[arg-type]


def test_evaluates_each_strategy_contract_quantity_tuple() -> None:
    sid_a = uuid4()
    sid_b = uuid4()
    cands = [
        TargetAttainmentCandidate(
            strategy_id=sid_a,
            contract_id="A1",
            premium_per_contract_usd=Decimal("1.00"),
            estimated_win_probability=Decimal("0.40"),
            expected_value_usd=Decimal("1"),
            estimated_payoff_ratio=Decimal("2"),
            estimated_useful_upside_usd=Decimal("120"),
            estimated_resolution_seconds=300,
            maximum_loss_usd_per_contract=Decimal("100"),
        ),
        TargetAttainmentCandidate(
            strategy_id=sid_b,
            contract_id="B1",
            premium_per_contract_usd=Decimal("0.50"),
            estimated_win_probability=Decimal("0.70"),
            expected_value_usd=Decimal("5"),
            estimated_payoff_ratio=Decimal("1"),
            # Tiny upside even at max affordable size — cannot close $100 gap.
            estimated_useful_upside_usd=Decimal("5"),
            estimated_resolution_seconds=300,
            maximum_loss_usd_per_contract=Decimal("50"),
        ),
    ]
    decision = TargetAttainmentPolicy().decide(
        _ctx(
            available_capital_usd=Decimal("200"),
            remaining_goal_gap_usd=Decimal("100"),
            time_remaining_seconds=600,
            objective_duration_seconds=3600,
        ),
        cands,
    )
    pairs = {
        (str(e.strategy_id), e.contract_id, e.quantity)
        for e in decision.quantity_evaluations
        if e.quantity > 0
    }
    assert any(p[1] == "A1" for p in pairs)
    assert any(p[1] == "B1" for p in pairs)
    # A can close gap; B cannot at any affordable qty — prefer A
    assert decision.action == TargetAttainmentAction.ENTER
    assert decision.selected_contract_id == "A1"
    assert decision.selected_strategy_id == sid_a


def test_uses_exact_contract_ask_for_affordability() -> None:
    sid = uuid4()
    strategies = [
        SimpleNamespace(
            strategy_id=sid,
            direction=SimpleNamespace(value="bullish"),
            expected_horizon_seconds=300,
            candidate_legs=(
                SimpleNamespace(
                    contract_id="SPY:2026-08-04:500.0:call",
                    option_type="call",
                    strike=Decimal("500"),
                ),
            ),
        )
    ]
    surface = [
        SimpleNamespace(
            contract_id="SPY:2026-08-04:500.0:call",
            symbol="SPY",
            expiry=date(2026, 8, 4),
            strike=Decimal("500"),
            option_type="call",
            bid=Decimal("1.90"),
            ask=Decimal("2.00"),  # affordability must use ask
        )
    ]
    built = build_contract_candidates_for_strategies(
        strategies=strategies,
        surface_slice=surface,
        trading_date=date(2026, 8, 4),
    )
    assert len(built) == 1
    assert built[0].premium_per_contract_usd == Decimal("2.00")
    assert built[0].ask == Decimal("2.00")
    # $500 capital / ($2*100) = 2 contracts max
    decision = TargetAttainmentPolicy().decide(_ctx(), [built[0].as_candidate()])
    qtys = [e.quantity for e in decision.quantity_evaluations if e.quantity > 0]
    assert max(qtys) == 2


def test_does_not_use_first_surface_contract_for_all_strategies() -> None:
    sid_a = uuid4()
    sid_b = uuid4()
    strategies = [
        SimpleNamespace(
            strategy_id=sid_a,
            direction=SimpleNamespace(value="bullish"),
            expected_horizon_seconds=300,
            candidate_legs=(
                SimpleNamespace(
                    contract_id="CALL_A", option_type="call", strike=Decimal("500")
                ),
            ),
        ),
        SimpleNamespace(
            strategy_id=sid_b,
            direction=SimpleNamespace(value="bearish"),
            expected_horizon_seconds=300,
            candidate_legs=(
                SimpleNamespace(
                    contract_id="PUT_B", option_type="put", strike=Decimal("500")
                ),
            ),
        ),
    ]
    surface = [
        SimpleNamespace(
            contract_id="CALL_A",
            symbol="SPY",
            expiry=date(2026, 8, 4),
            strike=Decimal("500"),
            option_type="call",
            bid=Decimal("0.90"),
            ask=Decimal("1.00"),
        ),
        SimpleNamespace(
            contract_id="PUT_B",
            symbol="SPY",
            expiry=date(2026, 8, 4),
            strike=Decimal("500"),
            option_type="put",
            bid=Decimal("1.80"),
            ask=Decimal("2.00"),
        ),
    ]
    built = build_contract_candidates_for_strategies(
        strategies=strategies,
        surface_slice=surface,
        trading_date=date(2026, 8, 4),
    )
    by_sid = {str(c.strategy_id): c for c in built}
    assert by_sid[str(sid_a)].contract_id == "CALL_A"
    assert by_sid[str(sid_a)].premium_per_contract_usd == Decimal("1.00")
    assert by_sid[str(sid_b)].contract_id == "PUT_B"
    assert by_sid[str(sid_b)].premium_per_contract_usd == Decimal("2.00")


def test_rejects_contract_not_in_linked_surface() -> None:
    sid = uuid4()
    strategies = [
        SimpleNamespace(
            strategy_id=sid,
            direction=SimpleNamespace(value="bullish"),
            expected_horizon_seconds=300,
            candidate_legs=(
                SimpleNamespace(
                    contract_id="MISSING", option_type="call", strike=Decimal("500")
                ),
            ),
        )
    ]
    built = build_contract_candidates_for_strategies(
        strategies=strategies,
        surface_slice=[],
        trading_date=date(2026, 8, 4),
    )
    assert built == []


def test_rejects_non_0dte_contract() -> None:
    sid = uuid4()
    strategies = [
        SimpleNamespace(
            strategy_id=sid,
            direction=SimpleNamespace(value="bullish"),
            expected_horizon_seconds=300,
            candidate_legs=(
                SimpleNamespace(
                    contract_id="FUTURE", option_type="call", strike=Decimal("500")
                ),
            ),
        )
    ]
    surface = [
        SimpleNamespace(
            contract_id="FUTURE",
            symbol="SPY",
            expiry=date(2026, 8, 11),
            strike=Decimal("500"),
            option_type="call",
            bid=Decimal("1"),
            ask=Decimal("1.1"),
        )
    ]
    built = build_contract_candidates_for_strategies(
        strategies=strategies,
        surface_slice=surface,
        trading_date=date(2026, 8, 4),
    )
    assert built == []


def test_rejects_wrong_option_type_for_strategy() -> None:
    sid = uuid4()
    strategies = [
        SimpleNamespace(
            strategy_id=sid,
            direction=SimpleNamespace(value="bullish"),
            expected_horizon_seconds=300,
            candidate_legs=(
                SimpleNamespace(
                    contract_id="PUT_X", option_type="put", strike=Decimal("500")
                ),
            ),
        )
    ]
    surface = [
        SimpleNamespace(
            contract_id="PUT_X",
            symbol="SPY",
            expiry=date(2026, 8, 4),
            strike=Decimal("500"),
            option_type="put",
            bid=Decimal("1"),
            ask=Decimal("1.1"),
        )
    ]
    built = build_contract_candidates_for_strategies(
        strategies=strategies,
        surface_slice=surface,
        trading_date=date(2026, 8, 4),
    )
    assert built == []


def test_rejects_stale_contract() -> None:
    sid = uuid4()
    strategies = [
        SimpleNamespace(
            strategy_id=sid,
            direction=SimpleNamespace(value="bullish"),
            expected_horizon_seconds=300,
            candidate_legs=(
                SimpleNamespace(
                    contract_id="STALE", option_type="call", strike=Decimal("500")
                ),
            ),
        )
    ]
    surface = [
        SimpleNamespace(
            contract_id="STALE",
            symbol="SPY",
            expiry=date(2026, 8, 4),
            strike=Decimal("500"),
            option_type="call",
            bid=Decimal("1.00"),
            ask=Decimal("1.10"),
            quote_age_seconds=120,
            stale=True,
        )
    ]
    built = build_contract_candidates_for_strategies(
        strategies=strategies,
        surface_slice=surface,
        trading_date=date(2026, 8, 4),
        max_quote_age_seconds=30.0,
    )
    assert built == []


def test_rejects_unusable_spread() -> None:
    sid = uuid4()
    strategies = [
        SimpleNamespace(
            strategy_id=sid,
            direction=SimpleNamespace(value="bullish"),
            expected_horizon_seconds=300,
            candidate_legs=(
                SimpleNamespace(
                    contract_id="WIDE", option_type="call", strike=Decimal("500")
                ),
            ),
        )
    ]
    surface = [
        SimpleNamespace(
            contract_id="WIDE",
            symbol="SPY",
            expiry=date(2026, 8, 4),
            strike=Decimal("500"),
            option_type="call",
            bid=Decimal("0.10"),
            ask=Decimal("1.00"),  # 90% spread
        )
    ]
    built = build_contract_candidates_for_strategies(
        strategies=strategies,
        surface_slice=surface,
        trading_date=date(2026, 8, 4),
        max_relative_spread=0.35,
    )
    assert built == []


def test_may_select_full_capital_for_best_tuple() -> None:
    cand = TargetAttainmentCandidate(
        strategy_id=uuid4(),
        contract_id="C1",
        premium_per_contract_usd=Decimal("1.00"),
        estimated_win_probability=Decimal("0.55"),
        expected_value_usd=Decimal("10"),
        estimated_payoff_ratio=Decimal("2"),
        estimated_useful_upside_usd=Decimal("100"),
        estimated_resolution_seconds=120,
        maximum_loss_usd_per_contract=Decimal("100"),
    )
    decision = TargetAttainmentPolicy().decide(
        _ctx(
            available_capital_usd=Decimal("500"),
            remaining_goal_gap_usd=Decimal("1000"),
            time_remaining_seconds=120,
            objective_duration_seconds=3600,
        ),
        [cand],
    )
    assert decision.action == TargetAttainmentAction.ENTER
    assert decision.selected_quantity == 5
    assert decision.selected_contract_id == "C1"


def test_does_not_always_select_full_capital() -> None:
    cand = TargetAttainmentCandidate(
        strategy_id=uuid4(),
        contract_id="C1",
        premium_per_contract_usd=Decimal("1.00"),
        estimated_win_probability=Decimal("0.60"),
        expected_value_usd=Decimal("10"),
        estimated_payoff_ratio=Decimal("1"),
        estimated_useful_upside_usd=Decimal("60"),
        estimated_resolution_seconds=300,
        maximum_loss_usd_per_contract=Decimal("100"),
    )
    decision = TargetAttainmentPolicy().decide(
        _ctx(available_capital_usd=Decimal("500"), remaining_goal_gap_usd=Decimal("50")),
        [cand],
    )
    assert decision.action == TargetAttainmentAction.ENTER
    assert decision.selected_quantity == 1
    assert decision.selected_contract_id == "C1"
