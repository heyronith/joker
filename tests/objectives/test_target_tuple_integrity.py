"""Timing refresh, quantity integrity, historical mapping, no-contract WAIT."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.cognition.schemas import MetaDecisionAction
from joker.config.settings import CognitiveGraphSettings
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.objective_nodes import (
    apply_objective_sizing_to_proposal,
    gate_objective_confirmed,
    refresh_objective_timing_for_cycle,
)
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
from joker.objectives.schemas import SessionObjectiveState
from joker.objectives.service import SessionObjectiveService
from joker.objectives.sizing import DeterministicObjectiveSizer
from joker.objectives.target_attainment import (
    TargetAttainmentAction,
    TargetAttainmentCandidate,
    TargetAttainmentContext,
    TargetAttainmentPolicy,
    estimate_target_hit_probability,
)
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock

ET = ZoneInfo("America/New_York")


def _router(session_id: str = "t") -> ModelRouter:
    profiles = {
        n: p.model_copy(update={"provider": "fake", "model": "fake"})
        for n, p in default_model_profiles().items()
    }
    return ModelRouter(
        ModelRegistry(
            ModelsConfig(profiles=profiles), providers={"fake": FakeModelProvider()}
        ),
        session_id=session_id,
    )


async def _armed_svc(tmp_path, *, clock: FrozenExchangeClock, minutes: int = 60):
    db = tmp_path / "obj.db"
    apply_objective_migrations(db)
    svc = SessionObjectiveService(
        ObjectiveRepository(db),
        exchange_tz="America/New_York",
        objective_policy="target_attainment",
        require_positive_expected_value=False,
    )
    start = clock.now()
    definition = await svc.create_objective(
        session_id="sess",
        authorised_capital_usd=500,
        target_profit_pct=20,
        deadline_exchange_time=start + timedelta(minutes=minutes),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(
        definition.objective_id, confirmed_at_exchange_time=start
    )
    return svc


@pytest.mark.asyncio
async def test_compiled_graph_refreshes_objective_time_each_cycle(tmp_path) -> None:
    """Gate refresh advances remaining time via FrozenExchangeClock — no manual recompute."""
    start = datetime(2026, 8, 4, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    svc = await _armed_svc(tmp_path, clock=clock, minutes=60)
    deps = CognitiveGraphDeps(
        router=_router(),
        config=CognitiveGraphSettings(),
        session_id="sess",
        run_id="sess",
        objective_service=svc,
        objective_state_loader=svc.get_state,
        clock=clock,
    )
    # First cycle at t0 — remaining ≈ 3600.
    out0 = await gate_objective_confirmed(deps, {})
    assert out0 is None
    # Do NOT call recompute_from_truth from the test — only advance the clock.
    clock.set_now(start + timedelta(minutes=30))
    out1 = await gate_objective_confirmed(deps, {})
    assert out1 is None
    state = await svc.get_state()
    assert state.objective_duration_seconds == 3600
    assert state.time_remaining_seconds == 1800
    assert state.elapsed_seconds == 1800
    assert float(state.fraction_remaining) == pytest.approx(0.5)

    clock.set_now(start + timedelta(minutes=57))
    assert await gate_objective_confirmed(deps, {}) is None
    state2 = await svc.get_state()
    assert state2.time_remaining_seconds == 180
    assert float(state2.fraction_remaining) == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_live_no_trade_probability_decays_with_clock(tmp_path) -> None:
    start = datetime(2026, 8, 4, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    svc = await _armed_svc(tmp_path, clock=clock, minutes=60)
    deps = CognitiveGraphDeps(
        router=_router(),
        config=CognitiveGraphSettings(),
        session_id="sess",
        run_id="sess",
        objective_service=svc,
        objective_state_loader=svc.get_state,
        clock=clock,
    )

    async def _p_after_refresh() -> Decimal:
        await refresh_objective_timing_for_cycle(deps)
        st = await svc.get_state()
        ctx = TargetAttainmentContext.from_state(st, snapshot_id=uuid4())
        est = estimate_target_hit_probability(
            ctx=ctx,
            win_p=None,
            useful_upside_usd=Decimal("0"),
            capital_required_usd=Decimal("0"),
            sample_count=0,
            historical_hit_rate=None,
            resolution_seconds=None,
            is_no_trade=True,
        )
        assert est.p_goal is not None
        return est.p_goal

    p0 = await _p_after_refresh()
    clock.set_now(start + timedelta(minutes=30))
    p_half = await _p_after_refresh()
    clock.set_now(start + timedelta(minutes=57))
    p_near = await _p_after_refresh()
    assert p0 > p_half > p_near
    assert float(p0) == pytest.approx(0.9, abs=0.05)
    assert float(p_half) == pytest.approx(0.45, abs=0.05)
    assert float(p_near) == pytest.approx(0.045, abs=0.02)


def _obj_state(**overrides: object) -> SessionObjectiveState:
    data = {
        "objective_id": uuid4(),
        "session_id": "s",
        "status": "active",
        "authorised_capital_usd": Decimal("500"),
        "target_profit_usd": Decimal("100"),
        "target_ending_equity_usd": Decimal("600"),
        "working_order_reservation_usd": Decimal("0"),
        "filled_position_exposure_usd": Decimal("0"),
        "reserved_capital_usd": Decimal("0"),
        "available_capital_usd": Decimal("500"),
        "realised_pnl_usd": Decimal("0"),
        "unrealised_pnl_usd": Decimal("0"),
        "progress_to_goal_pct": Decimal("0"),
        "required_profit_remaining_usd": Decimal("100"),
        "time_remaining_seconds": 1800,
        "objective_duration_seconds": 3600,
        "version": 1,
        "max_concurrent_positions": 1,
        "deadline_exchange_time": datetime(2026, 8, 4, 15, 0, tzinfo=ET),
    }
    data.update(overrides)
    return SessionObjectiveState.model_validate(data)


class _Leg:
    def __init__(self, *, contract_id: str, quantity: int, limit_price: Decimal):
        self.contract_id = contract_id
        self.quantity = quantity
        self.limit_price = limit_price

    def model_copy(self, *, update):
        out = _Leg(
            contract_id=self.contract_id,
            quantity=self.quantity,
            limit_price=self.limit_price,
        )
        for k, v in update.items():
            setattr(out, k, v)
        return out


class _Proposal:
    def __init__(self, *, strategy_id, legs):
        self.strategy_id = strategy_id
        self.legs = legs
        self.action = "entry"

    def model_copy(self, *, update):
        out = _Proposal(strategy_id=self.strategy_id, legs=self.legs)
        for k, v in update.items():
            setattr(out, k, v)
        return out


class _Svc:
    def __init__(self, state):
        self._state = state

    async def get_state(self):
        return self._state


@pytest.mark.asyncio
async def test_target_quantity_change_requires_recalculation() -> None:
    """Qty 3 affordable at ask; limit makes 3 unaffordable → reject, not shrink to 2."""
    sid = uuid4()
    obj_state = _obj_state()
    sizer = DeterministicObjectiveSizer(
        maximum_authorised_contracts=20,
        require_positive_expected_value=False,
    )
    # Sizer alone would shrink 3 → 2 at $2.00 limit ($600 > $500).
    alone = sizer.size(
        obj_state,
        strategy_id=sid,
        premium_per_contract_usd=Decimal("2.00"),
        requested_quantity=3,
        expected_value_usd=None,
        estimated_win_probability=None,
        expected_r=None,
    )
    assert alone.approved is True
    assert alone.approved_quantity == 2

    deps = CognitiveGraphDeps(
        router=_router(),
        config=CognitiveGraphSettings(),
        session_id="s",
        run_id="s",
        objective_service=_Svc(obj_state),  # type: ignore[arg-type]
        capital_sizer=sizer,
        objective_policy="target_attainment",
    )
    proposal = _Proposal(
        strategy_id=sid,
        legs=(_Leg(contract_id="A1", quantity=3, limit_price=Decimal("2.00")),),
    )
    state: CognitiveGraphState = {
        "execution_proposal": proposal,  # type: ignore[typeddict-item]
        "meta_decision": SimpleNamespace(
            action=MetaDecisionAction.EXECUTE, selected_strategy_id=sid
        ),  # type: ignore[typeddict-item]
        "_target_attainment_authoritative": True,
        "_target_attainment_quantity": 3,
        "_target_attainment_contract_id": "A1",
        "_strategy_estimates": [
            {
                "strategy_id": str(sid),
                "valid": True,
                "estimate_id": str(uuid4()),
                "quote_inputs": {"premium_per_contract": "1.00"},
            }
        ],
        "errors": [],
    }
    out = await apply_objective_sizing_to_proposal(deps, state)
    assert "execution_proposal" not in out
    assert any(
        getattr(e, "error_code", None) == "target_attainment_recalculation_required"
        for e in (out.get("errors") or [])
    )


@pytest.mark.asyncio
async def test_target_quantity_is_never_silently_shrunk() -> None:
    sid = uuid4()
    obj_state = _obj_state()
    sizer = DeterministicObjectiveSizer(
        maximum_authorised_contracts=20,
        require_positive_expected_value=False,
    )
    shrink = sizer.size(
        obj_state,
        strategy_id=sid,
        premium_per_contract_usd=Decimal("2.00"),
        requested_quantity=3,
        expected_value_usd=None,
        estimated_win_probability=None,
        expected_r=None,
    )
    assert shrink.approved is True
    assert shrink.approved_quantity == 2
    assert shrink.approved_quantity != 3

    deps = CognitiveGraphDeps(
        router=_router(),
        config=CognitiveGraphSettings(),
        session_id="s",
        run_id="s",
        objective_service=_Svc(obj_state),  # type: ignore[arg-type]
        capital_sizer=sizer,
        objective_policy="target_attainment",
    )
    state: CognitiveGraphState = {
        "execution_proposal": _Proposal(  # type: ignore[typeddict-item]
            strategy_id=sid,
            legs=(_Leg(contract_id="A1", quantity=3, limit_price=Decimal("2.00")),),
        ),
        "meta_decision": SimpleNamespace(  # type: ignore[typeddict-item]
            action=MetaDecisionAction.EXECUTE, selected_strategy_id=sid
        ),
        "_target_attainment_authoritative": True,
        "_target_attainment_quantity": 3,
        "_target_attainment_contract_id": "A1",
        "_strategy_estimates": [
            {
                "strategy_id": str(sid),
                "valid": True,
                "estimate_id": str(uuid4()),
                "quote_inputs": {"premium_per_contract": "1.00"},
            }
        ],
        "errors": [],
    }
    out = await apply_objective_sizing_to_proposal(deps, state)
    assert "execution_proposal" not in out
    assert any(
        getattr(e, "error_code", None) == "target_attainment_recalculation_required"
        for e in (out.get("errors") or [])
    )


def test_historical_summary_maps_to_all_contracts_for_strategy() -> None:
    from dataclasses import replace

    from joker.objectives.target_attainment import TargetAttainmentCandidate

    sid_a = uuid4()
    sid_b = uuid4()
    cands = [
        TargetAttainmentCandidate(
            strategy_id=sid_a,
            contract_id="A1",
            premium_per_contract_usd=Decimal("1.00"),
            estimated_win_probability=Decimal("0.5"),
            expected_value_usd=Decimal("5"),
            estimated_payoff_ratio=Decimal("1"),
            estimated_useful_upside_usd=Decimal("80"),
            estimated_resolution_seconds=300,
            maximum_loss_usd_per_contract=Decimal("100"),
            sample_count=0,
        ),
        TargetAttainmentCandidate(
            strategy_id=sid_a,
            contract_id="A2",
            premium_per_contract_usd=Decimal("1.10"),
            estimated_win_probability=Decimal("0.5"),
            expected_value_usd=Decimal("5"),
            estimated_payoff_ratio=Decimal("1"),
            estimated_useful_upside_usd=Decimal("90"),
            estimated_resolution_seconds=300,
            maximum_loss_usd_per_contract=Decimal("110"),
            sample_count=0,
        ),
        TargetAttainmentCandidate(
            strategy_id=sid_b,
            contract_id="B1",
            premium_per_contract_usd=Decimal("1.00"),
            estimated_win_probability=Decimal("0.4"),
            expected_value_usd=Decimal("2"),
            estimated_payoff_ratio=Decimal("1"),
            estimated_useful_upside_usd=Decimal("70"),
            estimated_resolution_seconds=300,
            maximum_loss_usd_per_contract=Decimal("100"),
            sample_count=0,
        ),
    ]
    summaries = [
        {"strategy_id": str(sid_a), "hit_rate": "0.55", "sample_count": 40},
        {"strategy_id": str(sid_b), "hit_rate": "0.30", "sample_count": 25},
    ]
    by_sid = {str(s["strategy_id"]): s for s in summaries}
    mapped = []
    for c0 in cands:
        summary = by_sid.get(str(c0.strategy_id))
        assert summary is not None
        mapped.append(
            replace(
                c0,
                historical_hit_rate=Decimal(str(summary["hit_rate"])),
                sample_count=int(summary["sample_count"]),
            )
        )
    assert mapped[0].contract_id == "A1"
    assert mapped[1].contract_id == "A2"
    assert mapped[0].historical_hit_rate == Decimal("0.55")
    assert mapped[1].historical_hit_rate == Decimal("0.55")
    assert mapped[0].sample_count == 40
    assert mapped[1].sample_count == 40
    assert mapped[2].historical_hit_rate == Decimal("0.30")
    assert mapped[2].sample_count == 25


def test_historical_summary_never_crosses_strategy() -> None:
    """Positional zip would wrongly attach summary B to contract A2 — must not."""
    sid_a = uuid4()
    sid_b = uuid4()
    # Two contracts for A, then one for B — positional index 1 would steal B's summary.
    ta_cands = [
        SimpleNamespace(strategy_id=sid_a, contract_id="A1", sample_count=0),
        SimpleNamespace(strategy_id=sid_a, contract_id="A2", sample_count=0),
        SimpleNamespace(strategy_id=sid_b, contract_id="B1", sample_count=0),
    ]
    historical_summaries = [
        {"strategy_id": str(sid_a), "hit_rate": "0.70", "sample_count": 50},
        {"strategy_id": str(sid_b), "hit_rate": "0.20", "sample_count": 10},
    ]
    # Wrong (old) positional mapping:
    positional = []
    for i, summary in enumerate(historical_summaries):
        positional.append((ta_cands[i].contract_id, summary["hit_rate"]))
    assert positional == [("A1", "0.70"), ("A2", "0.20")]  # corruption

    # Correct strategy_id mapping:
    by_sid = {str(s["strategy_id"]): s for s in historical_summaries}
    correct = [
        (c.contract_id, by_sid[str(c.strategy_id)]["hit_rate"]) for c in ta_cands
    ]
    assert correct == [("A1", "0.70"), ("A2", "0.70"), ("B1", "0.20")]


def test_no_exact_contract_candidate_returns_wait_from_policy() -> None:
    decision = TargetAttainmentPolicy().decide(
        TargetAttainmentContext(
            objective_id=uuid4(),
            snapshot_id=uuid4(),
            authorised_capital_usd=Decimal("500"),
            available_capital_usd=Decimal("500"),
            reserved_capital_usd=Decimal("0"),
            realised_pnl_usd=Decimal("0"),
            unrealised_pnl_usd=Decimal("0"),
            target_profit_usd=Decimal("100"),
            remaining_goal_gap_usd=Decimal("100"),
            time_remaining_seconds=1800,
            objective_duration_seconds=3600,
            elapsed_seconds=1800,
            open_position_count=0,
            working_order_count=0,
            max_concurrent_positions=1,
            maximum_authorised_contracts=20,
            exchange_session_phase="regular",
        ),
        [],
    )
    assert decision.action == TargetAttainmentAction.WAIT
    assert "no_valid_contract_candidates" in decision.reason_codes
    assert decision.selected_contract_id is None
    assert decision.selected_quantity == 0


@pytest.mark.asyncio
async def test_objective_node_no_valid_contract_candidates_waits_without_exception(
    tmp_path, monkeypatch
) -> None:
    """Strategies present but linked surface has no match → WAIT, no TypeError."""
    from datetime import date

    from joker.graph import objective_nodes as on
    from joker.objectives.feasibility import GoalFeasibilityEngine
    from joker.objectives.scoring import ObjectiveStrategyScorer

    start = datetime(2026, 8, 4, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    svc = await _armed_svc(tmp_path, clock=clock, minutes=60)
    snap_id = uuid4()
    sid = uuid4()

    async def _truth(deps_arg, sid_arg):
        # Surface has a contract, but it does not match the strategy leg id.
        surface = (
            SimpleNamespace(
                contract_id="OTHER_CONTRACT",
                symbol="SPY",
                expiry=date(2026, 8, 4),
                strike=Decimal("500"),
                option_type="call",
                bid=Decimal("1.00"),
                ask=Decimal("1.10"),
                quote_age_seconds=1,
            ),
        )
        return (
            SimpleNamespace(
                snapshot_id=sid_arg,
                exchange_time=clock.now(),
                option_surface_id=uuid4(),
            ),
            SimpleNamespace(findings=(), codes=(), usable_for_execution=True),
            None,
            surface,
        )

    monkeypatch.setattr(on, "load_snapshot_truth", _truth)
    deps = CognitiveGraphDeps(
        router=_router("no-contract"),
        config=CognitiveGraphSettings(),
        session_id="sess",
        run_id="sess",
        snapshot_repo=SimpleNamespace(  # type: ignore[arg-type]
            get_by_id=lambda _sid: SimpleNamespace(
                snapshot_id=_sid,
                exchange_time=clock.now(),
                option_surface_id=uuid4(),
            )
        ),
        objective_service=svc,
        objective_state_loader=svc.get_state,
        feasibility_engine=GoalFeasibilityEngine(policy="target_attainment"),
        objective_strategy_scorer=ObjectiveStrategyScorer(),
        capital_sizer=DeterministicObjectiveSizer(
            require_positive_expected_value=False
        ),
        target_attainment_policy=TargetAttainmentPolicy(),
        objective_policy="target_attainment",
        clock=clock,
    )
    from tests.objectives.test_strategy_family_required import _strategy

    strategy = _strategy(family="breakout_continuation")
    # Point the only leg at a contract_id absent from the linked surface.
    strategy = strategy.model_copy(
        update={
            "strategy_id": sid,
            "candidate_legs": (
                strategy.candidate_legs[0].model_copy(
                    update={"contract_id": "MISSING_FROM_SURFACE"}
                ),
            ),
        }
    )
    state = {
        "strategies": [strategy],
        "snapshot_id": str(snap_id),
        "session_id": "sess",
        "cycle_id": "c-no-contract",
        "world_model": SimpleNamespace(
            market_structure=None,
            volatility_state=None,
            options_state=None,
            temporal_state=SimpleNamespace(session_phase="regular"),
        ),
        "trace": [],
        "errors": [],
    }
    out = await on.score_strategies_against_objective_node(deps, state)  # type: ignore[arg-type]
    decision = out.get("_target_attainment_decision") or {}
    assert out.get("_target_attainment_authoritative") is True
    assert out.get("_target_attainment_action") == "wait"
    assert decision.get("action") == "wait"
    assert "no_valid_contract_candidates" in (decision.get("reason_codes") or [])
    assert decision.get("selected_contract_id") in {None, ""}
    assert int(decision.get("selected_quantity") or 0) == 0
    # No entry tuple for gateway / tactician.
    assert not out.get("_target_attainment_contract_id")
    assert int(out.get("_target_attainment_quantity") or 0) == 0
