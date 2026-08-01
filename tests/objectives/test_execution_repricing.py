"""Execution-time EV repricing helper and real OrderActionGateway tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from joker.objectives.estimate import StrategyEstimateBuilder
from joker.objectives.execution_quote import CurrentExecutionQuote
from joker.objectives.repricing import reprice_long_option_estimate
from joker.objectives.schemas import SessionObjectiveState, StrategyObjectiveEstimate
from joker.objectives.service import SessionObjectiveService
from joker.runtime.order_action_gateway import (
    OrderActionGateway,
    OrderActionKind,
    OrderActionRequest,
)
from tests.objectives.historical_fixtures import (
    make_repo_backed_hist_service,
    persist_positive_history,
)
from tests.objectives.test_ev_feasibility_estimates import _strategy
from tests.cognitive.task2_canned import CONTRACT_ID


def _state(**kw) -> SessionObjectiveState:
    base = {
        "objective_id": uuid4(),
        "session_id": "s",
        "status": "active",
        "authorised_capital_usd": Decimal("500"),
        "target_profit_usd": Decimal("50"),
        "target_ending_equity_usd": Decimal("550"),
        "available_capital_usd": Decimal("500"),
        "required_profit_remaining_usd": Decimal("50"),
        "time_remaining_seconds": 3600,
        "version": 1,
        "max_concurrent_positions": 1,
        "deadline_exchange_time": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    base.update(kw)
    return SessionObjectiveState.model_validate(base)


def test_repricing_helper_adjusts_ev_for_entry_cost() -> None:
    est = StrategyObjectiveEstimate(
        strategy_id=uuid4(),
        objective_id=uuid4(),
        snapshot_id=uuid4(),
        expected_value_usd=Decimal("20.00"),
        capital_required_usd=Decimal("100.00"),
        maximum_loss_usd=Decimal("100.00"),
        calculation_method="calibrated_episode_average",
        quote_inputs={
            "premium_per_contract": "1.00",
            "quantity": 1,
            "slippage_per_contract": "0.00",
        },
        valid=True,
    )
    repriced = reprice_long_option_estimate(
        est,
        current_premium_per_contract_usd=Decimal("1.05"),
        quantity=1,
        request_snapshot_id=uuid4(),
        current_slippage_per_contract_usd=Decimal("0.00"),
    )
    assert repriced.repricing_method == "long_option_entry_cost_adjust_v1"
    assert repriced.repriced_expected_value_usd == Decimal("15.00")


def test_repricing_helper_rejects_expired_estimate() -> None:
    est = StrategyObjectiveEstimate(
        strategy_id=uuid4(),
        objective_id=uuid4(),
        snapshot_id=uuid4(),
        expected_value_usd=Decimal("10.00"),
        capital_required_usd=Decimal("100.00"),
        maximum_loss_usd=Decimal("100.00"),
        calculation_method="calibrated_episode_average",
        quote_inputs={"premium_per_contract": "1.00", "quantity": 1},
        valid=True,
        valid_until=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    repriced = reprice_long_option_estimate(
        est,
        current_premium_per_contract_usd=Decimal("1.00"),
        quantity=1,
        request_snapshot_id=uuid4(),
    )
    assert repriced.valid is False
    assert "estimate_expired" in repriced.invalidation_reasons


def _quote(
    *,
    ask: str = "1.10",
    bid: str = "1.00",
    age: int = 1,
    usable: bool = True,
    reasons: tuple[str, ...] = (),
    spread: str = "0.095",
) -> CurrentExecutionQuote:
    return CurrentExecutionQuote(
        snapshot_id=uuid4(),
        option_surface_id=uuid4(),
        contract_id=CONTRACT_ID,
        bid=Decimal(bid),
        ask=Decimal(ask),
        mid=((Decimal(bid) + Decimal(ask)) / 2),
        relative_spread=Decimal(spread),
        quote_timestamp=datetime.now(timezone.utc),
        quote_age_seconds=age,
        data_quality_id=uuid4(),
        usable_for_execution=usable,
        invalidation_reasons=reasons,
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.submissions: list = []

    async def submit_execution_command(self, command):
        self.submissions.append(command)
        return SimpleNamespace(order_id="paper-1", client_order_id=command.client_order_id)

    async def cancel_order(self, **kwargs):
        return None


async def _gateway_stack(tmp_path, *, quote: CurrentExecutionQuote, pnl: Decimal = Decimal("20")):
    from joker.graph.graph_deps import CognitiveGraphDeps
    from joker.config.settings import CognitiveGraphSettings
    from joker.objectives.repository import apply_objective_migrations, ObjectiveRepository
    from joker.models.fake_provider import FakeModelProvider
    from joker.models.registry import ModelRegistry
    from joker.models.router import ModelRouter
    from joker.models.schemas import ModelsConfig, default_model_profiles

    hist, obj_repo, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(tmp_path)
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo,
        evaluation_repo=ev_repo,
        as_of=as_of,
        n=20,
        pnl=pnl,
    )
    svc = SessionObjectiveService(obj_repo, require_positive_expected_value=True)
    definition = await svc.create_objective(
        session_id="gw",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(timezone.utc) + timedelta(hours=2),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    strategy = _strategy()
    summary = await hist.summarize_for_strategy(
        objective_id=definition.objective_id,
        strategy_id=strategy.strategy_id,
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    est = StrategyEstimateBuilder(
        minimum_samples_for_calibrated_ev=20,
        require_lower_confidence_bound_positive=False,
    ).build(
        strategy=strategy,
        objective_state=await svc.get_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        historical_summary=summary,
    )
    # Force valid estimate for gateway path even if LCB strict
    if not est.valid and summary.average_pnl_usd and summary.average_pnl_usd > 0:
        est = est.model_copy(
            update={
                "valid": True,
                "expected_value_usd": summary.average_pnl_usd,
                "uncertainty_reasons": (),
            }
        )
    svc.save_strategy_estimate(est)

    calls: list[str] = []

    async def _loader(contract_id: str):
        calls.append(contract_id)
        return quote

    profiles = {
        n: p.model_copy(update={"provider": "fake", "model": "x"})
        for n, p in default_model_profiles().items()
    }
    router = ModelRouter(
        ModelRegistry(ModelsConfig(profiles=profiles), providers={"fake": FakeModelProvider()}),
        session_id="gw",
    )
    runtime = _FakeRuntime()
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="gw",
        run_id="gw",
        execution_runtime=runtime,  # type: ignore[arg-type]
        objective_service=svc,
        historical_outcome_service=hist,
        historical_outcome_settings=hist._settings,
        current_option_quote_loader=_loader,
        max_quote_age_seconds=30,
        max_relative_spread=0.25,
        kill_switch=False,
    )
    # Bypass snapshot truth validation by stubbing load path via monkeypatch in tests
    gateway = OrderActionGateway(deps)
    return gateway, svc, est, strategy, runtime, calls, definition


def _entry_request(est, strategy, *, limit: float = 9.99) -> OrderActionRequest:
    return OrderActionRequest(
        action=OrderActionKind.ENTRY,
        client_order_id=f"cloid-{uuid4()}",
        contract_id=CONTRACT_ID,
        side="buy",
        quantity=1,
        order_type="limit",
        limit_price=limit,
        snapshot_id=est.snapshot_id,
        strategy_id=strategy.strategy_id,
        estimate_id=est.estimate_id,
        proposal_id=uuid4(),
        decision_id=uuid4(),
        cycle_id="c",
    )


def _patch_validate(monkeypatch, *, limit_price: float = 1.10):
    from datetime import date

    from joker.runtime import order_action_gateway as gw_mod
    from joker.runtime.execution_runtime import ExecutionCommand
    from joker.schemas.domain import OrderIntent, OptionContract

    async def _fake_load_snapshot_truth(deps, snapshot_id):
        return (
            SimpleNamespace(snapshot_id=snapshot_id, trading_date=date(2026, 7, 1)),
            SimpleNamespace(usable_for_execution=True),
            SimpleNamespace(surface_id=uuid4(), contracts=()),
            (),
        )

    def _fake_validate(self, request, **kwargs):
        # Sync — matches OrderActionGateway._validate_and_compile.
        intent = OrderIntent(
            candidate_id=str(request.proposal_id or uuid4()),
            side="buy",
            quantity=1,
            order_type="limit",
            limit_price=float(
                limit_price if request.limit_price is None else request.limit_price
            ),
            contract=OptionContract(
                symbol="SPY",
                expiration=date(2026, 7, 1),
                strike=500.0,
                option_type="call",
            ),
        )
        return ExecutionCommand(
            client_order_id=request.client_order_id,
            intent=intent,
        )

    monkeypatch.setattr(gw_mod, "load_snapshot_truth", _fake_load_snapshot_truth)
    monkeypatch.setattr(OrderActionGateway, "_validate_and_compile", _fake_validate)


@pytest.mark.asyncio
async def test_gateway_loads_current_task1_quote(tmp_path, monkeypatch) -> None:
    quote = _quote(ask="1.10")
    gateway, svc, est, strategy, runtime, calls, _ = await _gateway_stack(
        tmp_path, quote=quote, pnl=Decimal("20")
    )
    _patch_validate(monkeypatch)
    est2 = est.model_copy(
        update={
            "valid": True,
            "expected_value_usd": Decimal("20.00"),
            "quote_inputs": {
                "premium_per_contract": "1.00",
                "quantity": 1,
                "slippage_per_contract": "0.00",
            },
        }
    )
    svc.save_strategy_estimate(est2)
    result = await gateway.submit(_entry_request(est2, strategy, limit=9.99))
    assert calls == [CONTRACT_ID]
    assert result.submitted is True


@pytest.mark.asyncio
async def test_gateway_does_not_use_limit_price_as_quote_truth(tmp_path, monkeypatch) -> None:
    quote = _quote(ask="1.10", bid="1.00")
    gateway, svc, est, strategy, runtime, calls, _ = await _gateway_stack(
        tmp_path, quote=quote, pnl=Decimal("25")
    )
    _patch_validate(monkeypatch)
    est2 = est.model_copy(
        update={
            "valid": True,
            "expected_value_usd": Decimal("25.00"),
            "quote_inputs": {
                "premium_per_contract": "1.00",
                "quantity": 1,
                "slippage_per_contract": "0.00",
            },
        }
    )
    svc.save_strategy_estimate(est2)
    result = await gateway.submit(_entry_request(est2, strategy, limit=99.0))
    assert calls == [CONTRACT_ID]
    assert result.submitted is True
    state = await svc.get_state()
    # Reservation uses ask $1.10, not limit $99
    assert state.working_order_reservation_usd == Decimal("110.00")


@pytest.mark.asyncio
async def test_gateway_rejects_stale_current_quote(tmp_path, monkeypatch) -> None:
    quote = _quote(age=999, usable=False, reasons=("quote_stale",))
    gateway, _, est, strategy, runtime, _, _ = await _gateway_stack(tmp_path, quote=quote)
    _patch_validate(monkeypatch)
    result = await gateway.submit(_entry_request(est, strategy))
    assert result.submitted is False
    assert "current_quote_unusable" in (result.blocked_reason or "")
    assert runtime.submissions == []


@pytest.mark.asyncio
async def test_gateway_rejects_wide_current_spread(tmp_path, monkeypatch) -> None:
    quote = _quote(
        bid="1.00",
        ask="2.00",
        spread="0.50",
        usable=False,
        reasons=("spread_unacceptable",),
    )
    gateway, _, est, strategy, runtime, _, _ = await _gateway_stack(tmp_path, quote=quote)
    _patch_validate(monkeypatch)
    result = await gateway.submit(_entry_request(est, strategy))
    assert result.submitted is False
    assert runtime.submissions == []


@pytest.mark.asyncio
async def test_gateway_rejects_missing_current_contract(tmp_path, monkeypatch) -> None:
    quote = _quote(usable=False, reasons=("contract_absent_from_latest_surface",))
    gateway, _, est, strategy, runtime, _, _ = await _gateway_stack(tmp_path, quote=quote)
    _patch_validate(monkeypatch)
    result = await gateway.submit(_entry_request(est, strategy))
    assert result.submitted is False
    assert runtime.submissions == []


@pytest.mark.asyncio
async def test_gateway_rejects_non_positive_current_quote_ev(tmp_path, monkeypatch) -> None:
    quote = _quote(ask="1.50", bid="1.40", spread="0.07")
    gateway, svc, est, strategy, runtime, _, _ = await _gateway_stack(
        tmp_path, quote=quote, pnl=Decimal("8")
    )
    _patch_validate(monkeypatch)
    est2 = est.model_copy(
        update={
            "valid": True,
            "expected_value_usd": Decimal("8.00"),
            "quote_inputs": {
                "premium_per_contract": "1.00",
                "quantity": 1,
                "slippage_per_contract": "0.00",
            },
        }
    )
    svc.save_strategy_estimate(est2)
    result = await gateway.submit(_entry_request(est2, strategy))
    assert result.submitted is False
    assert "objective_repriced_ev_not_positive" in (result.blocked_reason or "")
    assert runtime.submissions == []
    state = await svc.get_state()
    assert state.working_order_reservation_usd == 0


@pytest.mark.asyncio
async def test_gateway_reserves_once_after_positive_current_quote_ev(
    tmp_path, monkeypatch
) -> None:
    quote = _quote(ask="1.05", bid="1.00", spread="0.05")
    gateway, svc, est, strategy, runtime, _, _ = await _gateway_stack(
        tmp_path, quote=quote, pnl=Decimal("20")
    )
    _patch_validate(monkeypatch)
    est2 = est.model_copy(
        update={
            "valid": True,
            "expected_value_usd": Decimal("20.00"),
            "quote_inputs": {
                "premium_per_contract": "1.00",
                "quantity": 1,
                "slippage_per_contract": "0.00",
            },
        }
    )
    svc.save_strategy_estimate(est2)
    result = await gateway.submit(_entry_request(est2, strategy))
    assert result.submitted is True
    assert len(runtime.submissions) == 1
    state = await svc.get_state()
    assert state.working_order_reservation_usd > 0


@pytest.mark.asyncio
async def test_incremental_add_requires_positive_ev(tmp_path) -> None:
    hist, obj_repo, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(tmp_path)
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo,
        evaluation_repo=ev_repo,
        as_of=as_of,
        n=20,
        pnl=Decimal("-3.00"),
    )
    summary = await hist.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    bad = StrategyObjectiveEstimate(
        strategy_id=uuid4(),
        objective_id=uuid4(),
        snapshot_id=uuid4(),
        expected_value_usd=Decimal("-3.00"),
        capital_required_usd=Decimal("100"),
        maximum_loss_usd=Decimal("100"),
        calculation_method="calibrated_episode_average",
        quote_inputs={"premium_per_contract": "1.00", "quantity": 1},
        valid=True,
        historical_summary_id=summary.summary_id,
    )
    add_reprice = reprice_long_option_estimate(
        bad,
        current_premium_per_contract_usd=Decimal("1.00"),
        quantity=1,
        request_snapshot_id=uuid4(),
    )
    assert add_reprice.valid is False
    assert OrderActionKind.ADD.value == "add"


@pytest.mark.asyncio
async def test_historical_ev_restart_reuses_persisted_artifacts(tmp_path) -> None:
    hist, obj_repo, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(tmp_path)
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=20
    )
    sid = uuid4()
    snap = uuid4()
    summary = await hist.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=sid,
        snapshot_id=snap,
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    est = StrategyEstimateBuilder(minimum_samples_for_calibrated_ev=20).build(
        strategy=_strategy(),
        objective_state=_state(),
        snapshot_id=snap,
        premium_per_contract_usd=Decimal("1.00"),
        historical_summary=summary,
    )
    obj_repo.save_strategy_estimate(est)
    from joker.objectives.repository import ObjectiveRepository

    repo2 = ObjectiveRepository(tmp_path / "hist_evo.db")
    loaded_summary = repo2.get_historical_summary(summary.summary_id)
    loaded_est = repo2.get_strategy_estimate(est.estimate_id)
    assert loaded_summary is not None
    assert loaded_est is not None
    assert loaded_est.historical_summary_id == summary.summary_id
