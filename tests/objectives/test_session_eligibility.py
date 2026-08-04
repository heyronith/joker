"""Session-phase eligibility for target-attainment physical gates."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.objectives.session_eligibility import (
    ObjectiveSessionEligibility,
    resolve_objective_session_state,
)
from joker.objectives.target_attainment import (
    TargetAttainmentAction,
    TargetAttainmentCandidate,
    TargetAttainmentContext,
    TargetAttainmentPolicy,
    classify_physical_impossibility,
)
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock, SessionPhase

ET = ZoneInfo("America/New_York")


def _regular_clock(hour: int = 10) -> FrozenExchangeClock:
    return FrozenExchangeClock(
        datetime(2026, 8, 4, hour, 0, tzinfo=ET),
        calendar=MarketCalendar(),
    )


def _ctx(**overrides: object) -> TargetAttainmentContext:
    data: dict[str, object] = {
        "objective_id": uuid4(),
        "snapshot_id": uuid4(),
        "authorised_capital_usd": Decimal("200.00"),
        "available_capital_usd": Decimal("200.00"),
        "reserved_capital_usd": Decimal("0.00"),
        "realised_pnl_usd": Decimal("0.00"),
        "unrealised_pnl_usd": Decimal("0.00"),
        "target_profit_usd": Decimal("50.00"),
        "remaining_goal_gap_usd": Decimal("50.00"),
        "time_remaining_seconds": 1800,
        "objective_duration_seconds": 3600,
        "elapsed_seconds": 1800,
        "open_position_count": 0,
        "working_order_count": 0,
        "max_concurrent_positions": 1,
        "maximum_authorised_contracts": 20,
        "exchange_session_phase": "regular",
        "session_similarity_bucket": "midday",
        "session_phase": "regular",
    }
    data.update(overrides)
    return TargetAttainmentContext(**data)  # type: ignore[arg-type]


def _cand() -> TargetAttainmentCandidate:
    return TargetAttainmentCandidate(
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        estimated_win_probability=Decimal("0.55"),
        expected_value_usd=Decimal("5.00"),
        estimated_payoff_ratio=Decimal("1.0"),
        estimated_useful_upside_usd=Decimal("80.00"),
        estimated_resolution_seconds=300,
        maximum_loss_usd_per_contract=Decimal("100.00"),
        contract_id="SPY:2026-08-04:500.0:call",
    )


def test_target_policy_accepts_regular_phase() -> None:
    session = resolve_objective_session_state(clock=_regular_clock(10), similarity_bucket="regular")
    assert session.eligibility is ObjectiveSessionEligibility.REGULAR
    assert session.entries_permitted is True
    decision = TargetAttainmentPolicy().decide(
        _ctx(), [_cand()], session_state=session
    )
    assert decision.action != TargetAttainmentAction.BLOCK


def test_target_policy_accepts_open_bucket_when_exchange_phase_regular() -> None:
    session = resolve_objective_session_state(clock=_regular_clock(10), similarity_bucket="open")
    assert session.exchange_phase is SessionPhase.REGULAR
    impossible, codes = classify_physical_impossibility(_ctx(), session_state=session)
    assert impossible is False
    assert "market_not_regular" not in codes
    decision = TargetAttainmentPolicy().decide(
        _ctx(session_similarity_bucket="open"), [_cand()], session_state=session
    )
    assert decision.action != TargetAttainmentAction.BLOCK
    assert "market_not_regular" not in decision.reason_codes


def test_target_policy_accepts_midday_bucket_when_exchange_phase_regular() -> None:
    session = resolve_objective_session_state(clock=_regular_clock(12), similarity_bucket="midday")
    decision = TargetAttainmentPolicy().decide(
        _ctx(session_similarity_bucket="midday"), [_cand()], session_state=session
    )
    assert decision.action != TargetAttainmentAction.BLOCK


def test_target_policy_accepts_close_bucket_when_exchange_phase_regular() -> None:
    session = resolve_objective_session_state(clock=_regular_clock(15), similarity_bucket="close")
    decision = TargetAttainmentPolicy().decide(
        _ctx(session_similarity_bucket="close"), [_cand()], session_state=session
    )
    assert decision.action != TargetAttainmentAction.BLOCK


def test_target_policy_rejects_premarket() -> None:
    clock = FrozenExchangeClock(
        datetime(2026, 8, 4, 8, 0, tzinfo=ET), calendar=MarketCalendar()
    )
    session = resolve_objective_session_state(clock=clock, similarity_bucket="premarket")
    assert session.entries_permitted is False
    decision = TargetAttainmentPolicy().decide(
        _ctx(exchange_session_phase="premarket"), [_cand()], session_state=session
    )
    assert decision.action == TargetAttainmentAction.BLOCK
    assert "market_not_regular" in decision.reason_codes


def test_target_policy_rejects_post_market() -> None:
    clock = FrozenExchangeClock(
        datetime(2026, 8, 4, 16, 30, tzinfo=ET), calendar=MarketCalendar()
    )
    session = resolve_objective_session_state(clock=clock, similarity_bucket="post_market")
    decision = TargetAttainmentPolicy().decide(
        _ctx(exchange_session_phase="post_market"), [_cand()], session_state=session
    )
    assert decision.action == TargetAttainmentAction.BLOCK


def test_target_policy_rejects_closed() -> None:
    clock = FrozenExchangeClock(
        datetime(2026, 8, 4, 20, 0, tzinfo=ET), calendar=MarketCalendar()
    )
    session = resolve_objective_session_state(clock=clock, similarity_bucket="closed")
    decision = TargetAttainmentPolicy().decide(
        _ctx(exchange_session_phase="closed"), [_cand()], session_state=session
    )
    assert decision.action == TargetAttainmentAction.BLOCK


def test_target_policy_fails_when_exchange_phase_unknown() -> None:
    session = resolve_objective_session_state(clock=None, similarity_bucket="open")
    assert session.eligibility is ObjectiveSessionEligibility.UNKNOWN
    decision = TargetAttainmentPolicy().decide(
        _ctx(exchange_session_phase=None, session_phase="unknown"),
        [_cand()],
        session_state=session,
    )
    assert decision.action == TargetAttainmentAction.BLOCK
    assert "exchange_session_truth_unavailable" in decision.reason_codes


@pytest.mark.asyncio
async def test_objective_node_context_uses_exchange_phase_not_open_bucket(
    tmp_path, monkeypatch
) -> None:
    """Score node must gate on exchange clock, not agent similarity bucket 'open'."""
    from joker.config.settings import CognitiveGraphSettings
    from joker.graph.graph_deps import CognitiveGraphDeps
    from joker.graph import objective_nodes as on
    from joker.models.fake_provider import FakeModelProvider
    from joker.models.registry import ModelRegistry
    from joker.models.router import ModelRouter
    from joker.models.schemas import ModelsConfig, default_model_profiles
    from joker.objectives.feasibility import GoalFeasibilityEngine
    from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
    from joker.objectives.scoring import ObjectiveStrategyScorer
    from joker.objectives.service import SessionObjectiveService
    from joker.objectives.sizing import DeterministicObjectiveSizer

    db = tmp_path / "sess.db"
    apply_objective_migrations(db)
    clock = _regular_clock(10)
    svc = SessionObjectiveService(
        ObjectiveRepository(db),
        exchange_tz="America/New_York",
        objective_policy="target_attainment",
        require_positive_expected_value=False,
    )
    definition = await svc.create_objective(
        session_id="sess-open",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=clock.now().replace(hour=15),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(
        definition.objective_id, confirmed_at_exchange_time=clock.now()
    )
    profiles = {
        n: p.model_copy(update={"provider": "fake", "model": "fake"})
        for n, p in default_model_profiles().items()
    }
    router = ModelRouter(
        ModelRegistry(
            ModelsConfig(profiles=profiles), providers={"fake": FakeModelProvider()}
        ),
        session_id="sess-open",
    )
    snap_id = uuid4()

    class _SnapRepo:
        async def get_by_id(self, sid):
            return SimpleNamespace(
                snapshot_id=sid,
                exchange_time=clock.now(),
                option_surface_id=None,
                data_quality_id=uuid4(),
            )

    async def _truth(deps_arg, sid):
        return (
            SimpleNamespace(
                snapshot_id=sid,
                exchange_time=clock.now(),
                option_surface_id=None,
            ),
            SimpleNamespace(findings=(), codes=(), usable_for_execution=True),
            None,
            (),
        )

    monkeypatch.setattr(on, "load_snapshot_truth", _truth)
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="sess-open",
        run_id="sess-open",
        snapshot_repo=_SnapRepo(),  # type: ignore[arg-type]
        objective_service=svc,
        objective_state_loader=svc.get_state,
        feasibility_engine=GoalFeasibilityEngine(policy="target_attainment"),
        objective_strategy_scorer=ObjectiveStrategyScorer(),
        capital_sizer=DeterministicObjectiveSizer(),
        target_attainment_policy=TargetAttainmentPolicy(),
        objective_policy="target_attainment",
        clock=clock,
    )
    state = {
        "strategies": [],
        "snapshot_id": str(snap_id),
        "session_id": "sess-open",
        "cycle_id": "c1",
        "world_model": SimpleNamespace(
            market_structure=None,
            volatility_state=None,
            options_state=None,
            temporal_state=SimpleNamespace(session_phase="open"),
        ),
        "trace": [],
    }
    out = await on.score_strategies_against_objective_node(deps, state)  # type: ignore[arg-type]
    session = out.get("_objective_session") or {}
    assert session.get("exchange_phase") == "regular"
    assert session.get("similarity_bucket") == "open"
    assert session.get("entries_permitted") is True
    decision = out.get("_target_attainment_decision") or {}
    assert "market_not_regular" not in (decision.get("reason_codes") or [])
    assert out.get("_target_attainment_authoritative") is True
