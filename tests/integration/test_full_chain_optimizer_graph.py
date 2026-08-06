from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from joker.broker.interface import PaperBroker
from joker.graph.cognitive_graph import build_cognitive_graph
from joker.graph.cognitive_state import CognitiveGraphState
from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.objectives.config import FullChainOptimizerSettings
from joker.objectives.full_chain_optimizer import (
    FullChainOptimizationResult,
    optimize_full_chain,
)
from joker.objectives.portfolio_search import (
    AuthorizedPositionTuple,
    PortfolioAction,
    TargetPortfolioDecision,
)
from joker.objectives.target_attainment import (
    TargetAttainmentContext,
    TargetAttainmentPolicy,
)
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned
from tests.integration.test_goal_driven_full_graph import (
    _prepare_stack,
    _teardown_stack,
)


class ControllablePaperBroker(PaperBroker):
    """Webull-paper-like deterministic broker with controllable order outcomes."""

    def __init__(self, submission_statuses: list[str]) -> None:
        super().__init__(slippage_pct=2.0)
        self.submission_statuses = list(submission_statuses)
        self.external_submission_count = 0

    def submit_order(self, intent):
        self.external_submission_count += 1
        order = super().submit_order(intent)
        status = self.submission_statuses.pop(0) if self.submission_statuses else "open"
        if status == "filled":
            self._apply_fill(order, float(intent.limit_price or 1.0))
        elif status == "partially_filled":
            order.status = "partially_filled"
            order.filled_quantity = max(1, order.quantity // 2)
            order.remaining_quantity = order.quantity - order.filled_quantity
            order.average_fill_price = float(intent.limit_price or 1.0)
        elif status in {
            "accepted",
            "rejected",
            "cancelled",
            "pending",
            "open",
        }:
            order.status = status
        else:
            raise ValueError(f"unsupported controllable broker status: {status}")
        return order

    def fill_order(self, order_id: str) -> None:
        order = self._orders[order_id]
        order.status = "open"
        order.filled_quantity = 0
        order.remaining_quantity = order.quantity
        self._apply_fill(order, float(order.limit_price or 1.0))

    def partially_fill_order(self, order_id: str, quantity: int = 1) -> None:
        order = self._orders[order_id]
        order.status = "partially_filled"
        order.filled_quantity = min(quantity, order.quantity)
        order.remaining_quantity = order.quantity - order.filled_quantity
        order.average_fill_price = float(order.limit_price or 1.0)

    def all_orders(self):
        return list(self._orders.values())


TRADING_DATE = date(2026, 8, 5)
NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)
SECOND_CONTRACT_ID = "SPY:2026-07-01:501.0:call"


def _strategy(*, expensive_hint: bool = True):
    return SimpleNamespace(
        strategy_id=uuid4(),
        name="bullish continuation",
        strategy_family="directional_momentum",
        direction=SimpleNamespace(value="bullish"),
        expected_horizon_seconds=300,
        confidence=0.75,
        supporting_evidence_ids=(uuid4(),),
        contract_selection_preferences=(),
        candidate_legs=(
            (
                SimpleNamespace(
                    contract_id="SPY:2026-08-05:500:call",
                    quantity=1,
                    limit_price=Decimal("2.25"),
                ),
            )
            if expensive_hint
            else ()
        ),
    )


def _contract(
    strike: str,
    ask: str,
    *,
    bid: str,
    option_type: str = "call",
) -> OptionContractSnapshot:
    ask_d = Decimal(ask)
    bid_d = Decimal(bid)
    mid = (ask_d + bid_d) / Decimal("2")
    return OptionContractSnapshot(
        contract_id=f"SPY:{TRADING_DATE.isoformat()}:{strike}:{option_type}",
        symbol="SPY",
        expiry=TRADING_DATE,
        strike=Decimal(strike),
        option_type=option_type,  # type: ignore[arg-type]
        bid=bid_d,
        ask=ask_d,
        mid=mid,
        implied_volatility=Decimal("0.80"),
        delta=Decimal("0.70") if option_type == "call" else Decimal("-0.70"),
        gamma=Decimal("0.03"),
        theta=Decimal("-0.05"),
        quote_timestamp=NOW,
        quote_age_ms=0,
        relative_spread=(ask_d - bid_d) / mid,
        liquidity_score=0.8,
    )


def _surface(contracts) -> OptionSurfaceSnapshot:
    return OptionSurfaceSnapshot(
        surface_id=uuid4(),
        exchange_time=NOW,
        trading_date=TRADING_DATE,
        underlying_symbol="SPY",
        underlying_price=Decimal("500"),
        contracts=tuple(contracts),
    )


def _ctx(
    *,
    snapshot_id=None,
    open_position_count: int = 0,
    max_concurrent_positions: int = 1,
    available_capital_usd: Decimal = Decimal("200"),
) -> TargetAttainmentContext:
    return TargetAttainmentContext(
        objective_id=uuid4(),
        snapshot_id=snapshot_id or uuid4(),
        authorised_capital_usd=Decimal("200"),
        available_capital_usd=available_capital_usd,
        reserved_capital_usd=Decimal("0"),
        realised_pnl_usd=Decimal("0"),
        unrealised_pnl_usd=Decimal("0"),
        target_profit_usd=Decimal("20"),
        remaining_goal_gap_usd=Decimal("10"),
        time_remaining_seconds=300,
        objective_duration_seconds=3600,
        elapsed_seconds=3300,
        open_position_count=open_position_count,
        working_order_count=0,
        max_concurrent_positions=max_concurrent_positions,
        maximum_authorised_contracts=20,
        exchange_session_phase="regular",
        objective_version=4,
    )


def test_expensive_agent_leg_does_not_block_affordable_full_chain_contract() -> None:
    snapshot_id = uuid4()
    result = optimize_full_chain(
        strategies=[_strategy(expensive_hint=True)],
        surface=_surface(
            [
                _contract("500", "2.25", bid="2.20"),
                _contract("501", "0.10", bid="0.09"),
                _contract("502", "0.20", bid="0.18"),
            ]
        ),
        ctx=_ctx(snapshot_id=snapshot_id),
        settings=FullChainOptimizerSettings(
            enabled=True,
            minimum_probability_improvement_over_wait=0,
        ),
        maximum_authorised_contracts=20,
        current_exchange_time=NOW,
        current_trading_date=TRADING_DATE,
    )
    evaluated = {row.contract_id for row in result.decision.quantity_grid}
    assert "SPY:2026-08-05:501:call" in evaluated
    assert "SPY:2026-08-05:502:call" in evaluated
    assert "SPY:2026-08-05:500:call" not in evaluated
    assert result.decision.action in {PortfolioAction.ENTER, PortfolioAction.WAIT}


def test_no_valid_contract_returns_wait_without_exception() -> None:
    result = optimize_full_chain(
        strategies=[_strategy()],
        surface=_surface([_contract("500", "1.00", bid="0.10")]),
        ctx=_ctx(),
        settings=FullChainOptimizerSettings(enabled=True),
        maximum_authorised_contracts=20,
        current_exchange_time=NOW,
        current_trading_date=TRADING_DATE,
    )
    assert result.decision.action == PortfolioAction.WAIT
    assert "no_valid_contract_candidates" in result.decision.reason_codes


def test_reoptimization_can_authorize_one_new_component_with_existing_exposure() -> None:
    open_contract_id = "SPY:2026-08-05:500:call"
    result = optimize_full_chain(
        strategies=[_strategy(expensive_hint=False)],
        # The reoptimization route removes authoritative open contracts before
        # invoking the optimizer; only replacement candidates remain here.
        surface=_surface(
            [
                _contract("501", "0.10", bid="0.09"),
                _contract("502", "0.20", bid="0.18"),
            ]
        ),
        ctx=_ctx(open_position_count=1, max_concurrent_positions=2),
        settings=FullChainOptimizerSettings(
            enabled=True,
            minimum_probability_improvement_over_wait=0,
        ),
        maximum_authorised_contracts=20,
        current_exchange_time=NOW,
        current_trading_date=TRADING_DATE,
    )
    assert result.decision.action == PortfolioAction.ENTER
    assert len(result.decision.authorized_positions) == 1
    assert all(
        position.contract_id != open_contract_id
        for position in result.decision.authorized_positions
    )


def test_portfolio_authority_channels_are_declared_for_checkpoints() -> None:
    annotations = CognitiveGraphState.__annotations__
    for field in (
        "_full_chain_universe",
        "_contract_outcomes",
        "_quantity_grid",
        "_portfolio_grid",
        "_target_portfolio_decision",
        "_target_authorized_positions",
        "_execution_command_ids",
    ):
        assert field in annotations


@pytest.mark.asyncio
async def test_compiled_graph_full_chain_enter_reaches_entry_tactician_without_exception(
    tmp_path,
) -> None:
    stack = await _prepare_stack(
        tmp_path,
        pnl=Decimal("15"),
        n=20,
        objective_duration=timedelta(minutes=4),
    )
    try:
        deps = stack["deps"]
        deps.objective_policy = "target_attainment"
        deps.objective_service.objective_policy = "target_attainment"
        deps.target_attainment_policy = TargetAttainmentPolicy()
        deps.full_chain_optimizer_settings = FullChainOptimizerSettings(
            enabled=True,
            maximum_quote_age_seconds=3600,
            maximum_surface_age_seconds=3600,
            minimum_probability_improvement_over_wait=0,
        )
        stack["graph"] = build_cognitive_graph(deps)
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        trace_names = [trace.node_name for trace in result.get("node_trace") or ()]
        assert "entry_tactician" in trace_names
        assert not any(
            error.error_code == "missing_proposal" for error in result.get("errors") or ()
        )
        decision = result.get("_target_portfolio_decision") or {}
        assert decision.get("action") == "enter"
        assert result.get("_portfolio_review_context")
        assert (result.get("_portfolio_review_finalization") or {}).get(
            "action"
        ) == "preserve_exact_tuple"
        assert len(stack["submitted"]) == len(result.get("_target_authorized_positions") or ()), [
            (error.error_code, error.message) for error in result.get("errors") or ()
        ]
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_end_to_end_reoptimization_lease_retry_executes_compiled_graph_once(
    tmp_path, monkeypatch
) -> None:
    import asyncio
    import copy
    import time as wall_time
    from datetime import date, datetime, timedelta, timezone

    from joker.persistence.cognitive_execution_provenance import (
        CognitiveExecutionProvenanceRegistry,
        PortfolioReoptimizationStatus,
        PortfolioTransitionConflict,
    )
    from joker.runtime.cognitive_agent_runtime import CognitiveAgentRuntime

    stack, _runtime, _registry, request, _original_result, _original_get_state = (
        await _prepare_pending_reoptimization(tmp_path, monkeypatch)
    )
    try:
        class _AdvancingClock:
            def __init__(self) -> None:
                self._base = datetime(2026, 7, 1, 14, 5, tzinfo=timezone.utc)
                self._started = wall_time.monotonic()

            def now(self) -> datetime:
                return self._base + timedelta(seconds=wall_time.monotonic() - self._started)

            def trading_date(self) -> date:
                return date(2026, 7, 1)

        async def _active_objective_state():
            current = await _original_get_state()
            return current.model_copy(
                update={
                    "status": "active",
                    "entries_paused": False,
                    "target_reached": False,
                }
            )

        async def _active_recompute_from_truth(**_kwargs):
            return await _active_objective_state()

        stack["objective_service"].get_state = _active_objective_state
        stack["objective_service"].recompute_from_truth = _active_recompute_from_truth
        owner = request.owner
        registry_a = CognitiveExecutionProvenanceRegistry(stack["deps"].db_path)
        registry_b = CognitiveExecutionProvenanceRegistry(stack["deps"].db_path)
        await registry_a.initialize()
        await registry_b.initialize()

        deps_a = copy.copy(stack["deps"])
        deps_a.provenance_registry = registry_a
        deps_a.clock = _AdvancingClock()
        deps_b = copy.copy(stack["deps"])
        deps_b.provenance_registry = registry_b
        deps_b.clock = _AdvancingClock()

        runtime_a = CognitiveAgentRuntime(
            session_id=deps_a.session_id,
            run_id="retry-run-a",
            router=deps_a.router,
            config=deps_a.config,
            graph_deps=deps_a,
        )
        runtime_b = CognitiveAgentRuntime(
            session_id=deps_b.session_id,
            run_id="retry-run-b",
            router=deps_b.router,
            config=deps_b.config,
            graph_deps=deps_b,
        )

        running = await registry_a.portfolio_reoptimizations.begin_attempt(
            request.request_id,
            owner=owner,
            current_run_id=runtime_a._run_id,
            attempt_exchange_time=deps_a.clock.now().isoformat(),
            lease_seconds=0.2,
        )

        register_full_path_canned(
            stack["fake_model_provider"],
            UUID(request.latest_snapshot_id),
            f"portfolio-reoptimization-{request.request_id}",
            session=deps_b.session_id,
        )
        compiled_graph = build_cognitive_graph(deps_b)
        graph_calls: list[str] = []

        class _CountingGraph:
            async def ainvoke(self, state, config=None):
                graph_calls.append(str(state.get("_portfolio_reoptimization_request_id")))
                return await compiled_graph.ainvoke(state, config=config)

        runtime_b._decision_graph = _CountingGraph()
        await runtime_b._resume_pending_portfolio_reoptimizations()
        assert graph_calls == []
        assert request.request_id in runtime_b._reoptimization_retry_tasks

        completed = None
        for _ in range(20):
            await asyncio.sleep(0.1)
            completed = await registry_b.portfolio_reoptimizations.get(request.request_id)
            if (
                completed is not None
                and completed.status
                in {
                    PortfolioReoptimizationStatus.COMPLETED,
                    PortfolioReoptimizationStatus.FAILED,
                }
            ):
                break
        assert completed is not None
        assert completed.status in {
            PortfolioReoptimizationStatus.COMPLETED,
            PortfolioReoptimizationStatus.FAILED,
        }
        assert completed.last_attempt_run_id == runtime_b._run_id
        assert completed.attempt_generation == running.attempt_generation + 1
        assert graph_calls == [request.request_id]
        assert request.request_id not in runtime_b._reoptimization_retry_tasks
        if completed.status == PortfolioReoptimizationStatus.COMPLETED:
            assert completed.replacement_action == "ENTER"
            assert stack["broker"].external_submission_count == 2
        else:
            assert completed.failure_reason

        now_iso = deps_a.clock.now().isoformat()
        with pytest.raises(
            PortfolioTransitionConflict,
            match="reoptimization completion lost attempt fencing race",
        ):
            await registry_a.portfolio_reoptimizations.complete_attempt(
                attempt=running,
                completed_at=now_iso,
                replacement_decision_id=str(uuid4()),
                replacement_action="ENTER",
            )
        with pytest.raises(
            PortfolioTransitionConflict,
            match="reoptimization failure lost attempt fencing race",
        ):
            await registry_a.portfolio_reoptimizations.fail_attempt(
                attempt=running,
                failed_at=now_iso,
                failure_reason="stale-run-a-cannot-win",
            )
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_compiled_graph_builds_authoritative_portfolio_proposal(
    tmp_path,
) -> None:
    stack = await _prepare_stack(
        tmp_path,
        pnl=Decimal("15"),
        n=20,
        objective_duration=timedelta(minutes=4),
    )
    try:
        deps = stack["deps"]
        deps.objective_policy = "target_attainment"
        deps.objective_service.objective_policy = "target_attainment"
        deps.target_attainment_policy = TargetAttainmentPolicy()
        deps.full_chain_optimizer_settings = FullChainOptimizerSettings(
            enabled=True,
            maximum_quote_age_seconds=3600,
            maximum_surface_age_seconds=3600,
            minimum_probability_improvement_over_wait=0,
        )
        result = await build_cognitive_graph(deps).ainvoke(stack["state"], config=stack["config"])
        positions = result.get("_target_authorized_positions") or []
        proposal = result["execution_proposal"]
        assert [(str(leg.strategy_id), leg.contract_id, leg.quantity) for leg in proposal.legs] == [
            (
                str(position["strategy_id"]),
                position["contract_id"],
                position["quantity"],
            )
            for position in positions
        ]
        assert all(leg.authorized_position_tuple_id for leg in proposal.legs)
    finally:
        await _teardown_stack(stack)


def _enable_full_chain(deps) -> None:
    deps.objective_policy = "target_attainment"
    deps.objective_service.objective_policy = "target_attainment"
    deps.target_attainment_policy = TargetAttainmentPolicy()
    deps.full_chain_optimizer_settings = FullChainOptimizerSettings(
        enabled=True,
        maximum_quote_age_seconds=3600,
        maximum_surface_age_seconds=3600,
        minimum_probability_improvement_over_wait=0,
    )


def _force_two_component_optimize(
    monkeypatch,
    *,
    qty_a: int = 1,
    qty_b: int = 3,
    premium_a: str = "1.20",
    premium_b: str = "0.50",
):
    """Wrap optimize_full_chain so ENTER always authorizes two distinct components."""
    from joker.objectives import full_chain_optimizer as fco_mod

    real_optimize = fco_mod.optimize_full_chain

    def _wrapped(**kwargs):
        result = real_optimize(**kwargs)
        maximum_age_seconds = int(kwargs["settings"].maximum_decision_age_seconds)
        strategies = list(kwargs.get("strategies") or [])
        strategy_id = UUID(str(strategies[0].strategy_id)) if strategies else uuid4()
        decision_id = uuid4()
        snapshot_id = kwargs["ctx"].snapshot_id
        portfolio_id = uuid4()
        fingerprint = kwargs.get("evaluated_objective_fingerprint")
        positions = (
            AuthorizedPositionTuple(
                position_tuple_id=uuid4(),
                strategy_id=strategy_id,
                contract_id=CONTRACT_ID,
                quantity=qty_a,
                evaluation_premium=Decimal(premium_a),
                capital_allocation=(Decimal(premium_a) * Decimal(qty_a) * Decimal("100")).quantize(
                    Decimal("0.01")
                ),
                maximum_loss=(Decimal(premium_a) * Decimal(qty_a) * Decimal("100")).quantize(
                    Decimal("0.01")
                ),
                snapshot_id=snapshot_id,
                objective_version=kwargs["ctx"].objective_version,
                decision_id=decision_id,
                evaluated_objective_fingerprint=fingerprint,
                evaluated_at_exchange_time=kwargs["current_exchange_time"],
                decision_valid_until_exchange_time=(
                    kwargs["current_exchange_time"] + timedelta(seconds=maximum_age_seconds)
                ),
                maximum_decision_age_seconds=maximum_age_seconds,
                required_resolution_horizon_seconds=30,
            ),
            AuthorizedPositionTuple(
                position_tuple_id=uuid4(),
                strategy_id=strategy_id,
                contract_id=SECOND_CONTRACT_ID,
                quantity=qty_b,
                evaluation_premium=Decimal(premium_b),
                capital_allocation=(Decimal(premium_b) * Decimal(qty_b) * Decimal("100")).quantize(
                    Decimal("0.01")
                ),
                maximum_loss=(Decimal(premium_b) * Decimal(qty_b) * Decimal("100")).quantize(
                    Decimal("0.01")
                ),
                snapshot_id=snapshot_id,
                objective_version=kwargs["ctx"].objective_version,
                decision_id=decision_id,
                evaluated_objective_fingerprint=fingerprint,
                evaluated_at_exchange_time=kwargs["current_exchange_time"],
                decision_valid_until_exchange_time=(
                    kwargs["current_exchange_time"] + timedelta(seconds=maximum_age_seconds)
                ),
                maximum_decision_age_seconds=maximum_age_seconds,
                required_resolution_horizon_seconds=30,
            ),
        )
        decision = TargetPortfolioDecision(
            decision_id=decision_id,
            action=PortfolioAction.ENTER,
            authorized_positions=positions,
            selected_portfolio_id=portfolio_id,
            selected_probability_goal=Decimal("0.55"),
            wait_probability_goal=Decimal("0.05"),
            probability_delta=Decimal("0.50"),
            snapshot_id=snapshot_id,
            objective_version=kwargs["ctx"].objective_version,
            time_remaining_seconds=kwargs["ctx"].time_remaining_seconds,
            reason_codes=("forced_two_component_test_portfolio",),
            quantity_grid=result.decision.quantity_grid,
            portfolio_evaluations=result.decision.portfolio_evaluations,
            evaluated_objective_fingerprint=fingerprint,
            evaluated_at_exchange_time=kwargs["current_exchange_time"],
            decision_valid_until_exchange_time=(
                kwargs["current_exchange_time"] + timedelta(seconds=maximum_age_seconds)
            ),
            maximum_decision_age_seconds=maximum_age_seconds,
            required_resolution_horizon_seconds=30,
        )
        return FullChainOptimizationResult(
            universe=result.universe,
            selection_specs=result.selection_specs,
            shared_scenario_grid=result.shared_scenario_grid,
            contract_outcomes=result.contract_outcomes,
            decision=decision,
        )

    monkeypatch.setattr(fco_mod, "optimize_full_chain", _wrapped)
    return real_optimize


async def _two_contract_stack(
    tmp_path,
    monkeypatch,
    *,
    broker=None,
    qty_a: int = 1,
    qty_b: int = 3,
    objective_duration: timedelta = timedelta(minutes=4),
):
    from zoneinfo import ZoneInfo

    from joker.persistence.cognitive_execution_provenance import (
        CognitiveExecutionProvenanceRegistry,
    )

    et = ZoneInfo("America/New_York")
    quote_ts = datetime(2026, 7, 1, 10, 3, tzinfo=et)
    stack = await _prepare_stack(
        tmp_path,
        pnl=Decimal("15"),
        n=20,
        objective_duration=objective_duration,
        max_concurrent_positions=2,
        option_quotes=[
            {
                "contract_id": CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "1.00",
                "ask": "1.20",
                "quote_timestamp": quote_ts,
            },
            {
                "contract_id": SECOND_CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "501",
                "option_type": "call",
                "bid": "0.45",
                "ask": "0.50",
                "quote_timestamp": quote_ts,
            },
        ],
        broker=broker,
    )
    _enable_full_chain(stack["deps"])
    stack["real_optimize_full_chain"] = _force_two_component_optimize(
        monkeypatch, qty_a=qty_a, qty_b=qty_b
    )
    provenance = CognitiveExecutionProvenanceRegistry(stack["deps"].db_path)
    await provenance.initialize()
    stack["deps"].provenance_registry = provenance

    async def _projection_loader():
        return await stack["deps"].execution_runtime.project_session()

    stack["deps"].projection_loader = _projection_loader
    tracked_requests: list = []
    gateway = stack["gateway"]
    # Chain through the prepare_stack tracking wrapper (gateway.submit).
    tracking_submit = gateway.submit

    async def _capture(request):
        tracked_requests.append(request)
        return await tracking_submit(request)

    gateway.submit = _capture  # type: ignore[method-assign]
    stack["tracked_requests"] = tracked_requests
    stack["tracking_submit"] = tracking_submit
    stack["graph"] = build_cognitive_graph(stack["deps"])
    return stack


def _advance_exchange_clock_during_debate(stack, *, seconds: int, after_advance=None) -> None:
    """Advance exchange time after optimization but before final review."""
    original = stack["deps"].router.route_and_complete
    advanced = False

    async def _wrapped(request, output_type, **kwargs):
        nonlocal advanced
        if not advanced and str(request.role) in {
            "strategy_advocate",
            "falsifier",
            "historical_critic",
            "execution_critic",
            "alternative_explanation",
        }:
            advanced = True
            stack["clock"].set_now(stack["clock"].now() + timedelta(seconds=seconds))
            if after_advance is not None:
                after_advance()
        return await original(request, output_type, **kwargs)

    stack["deps"].router.route_and_complete = _wrapped


async def _restart_portfolio_runtime(stack):
    """Rebuild the durable registry and graph without starting worker tasks."""
    from joker.persistence.cognitive_execution_provenance import (
        CognitiveExecutionProvenanceRegistry,
    )
    from joker.runtime.cognitive_agent_runtime import CognitiveAgentRuntime

    registry = CognitiveExecutionProvenanceRegistry(stack["deps"].db_path)
    await registry.initialize()
    stack["deps"].provenance_registry = registry
    runtime = CognitiveAgentRuntime(
        session_id=stack["deps"].session_id,
        run_id=stack["deps"].run_id,
        router=stack["deps"].router,
        config=stack["deps"].config,
        graph_deps=stack["deps"],
    )
    runtime._decision_graph = build_cognitive_graph(stack["deps"])
    return runtime, registry


@pytest.mark.asyncio
async def test_compiled_graph_preserves_distinct_portfolio_quantities(
    tmp_path, monkeypatch
) -> None:
    stack = await _two_contract_stack(tmp_path, monkeypatch)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        positions = result.get("_target_authorized_positions") or []
        proposal = result["execution_proposal"]
        assert [int(p["quantity"]) for p in positions] == [1, 3]
        assert [leg.quantity for leg in proposal.legs] == [1, 3]
        assert len({leg.quantity for leg in proposal.legs}) == 2
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_target_decision_and_position_tuple_ids_reach_submission_provenance(
    tmp_path, monkeypatch
) -> None:
    stack = await _two_contract_stack(tmp_path, monkeypatch)
    try:
        from joker.persistence.cognitive_execution_provenance import (
            ExecutionProvenanceRecord,
        )
        from joker.runtime.order_action_gateway import OrderActionResult

        gateway = stack["gateway"]

        # Auto-approve gateway submits so both components reach provenance recording.
        async def _approve(request):
            stack["tracked_requests"].append(request)
            await stack["deps"].provenance_registry.record(
                ExecutionProvenanceRecord(
                    client_order_id=request.client_order_id,
                    proposal_id=request.proposal_id,
                    decision_id=request.decision_id,
                    strategy_id=request.strategy_id,
                    cycle_id=request.cycle_id,
                    snapshot_id=request.snapshot_id,
                    contract_id=request.contract_id,
                    session_id=stack["deps"].session_id,
                    kind="entry",
                    extra={
                        "target_portfolio_decision_id": (request.target_portfolio_decision_id),
                        "selected_portfolio_id": request.selected_portfolio_id,
                        "authorized_position_tuple_id": (request.authorized_position_tuple_id),
                        "component_index": request.component_index,
                        "component_count": request.component_count,
                        "evaluated_at_exchange_time": (request.evaluated_at_exchange_time),
                        "decision_valid_until_exchange_time": (
                            request.decision_valid_until_exchange_time
                        ),
                        "maximum_decision_age_seconds": (request.maximum_decision_age_seconds),
                        "submission_exchange_time": (request.submission_exchange_time),
                        "decision_age_seconds": request.decision_age_seconds,
                        "required_resolution_horizon_seconds": (
                            request.required_resolution_horizon_seconds
                        ),
                    },
                )
            )
            return OrderActionResult(
                submitted=True,
                client_order_id=request.client_order_id,
                broker_order=SimpleNamespace(
                    order_id=request.client_order_id,
                    status="filled",
                    filled_quantity=request.quantity,
                ),
                working_orders={},
            )

        gateway.submit = _approve  # type: ignore[method-assign]
        stack["tracked_requests"].clear()
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        positions = result.get("_target_authorized_positions") or []
        decision = result.get("_target_portfolio_decision") or {}
        assert len(stack["tracked_requests"]) == len(positions) == 2
        decision_id = str(decision["decision_id"])
        for index, (request, position) in enumerate(
            zip(stack["tracked_requests"], positions, strict=True)
        ):
            assert request.target_portfolio_decision_id == decision_id
            assert request.authorized_position_tuple_id == str(position["position_tuple_id"])
            assert request.component_index == index
            assert request.selected_portfolio_id == str(decision["selected_portfolio_id"])
            assert request.evaluated_at_exchange_time
            assert request.submission_exchange_time
            assert request.maximum_decision_age_seconds == 60
            assert request.decision_age_seconds is not None
            assert request.session_id == stack["deps"].session_id
            assert request.run_id == stack["deps"].run_id
            assert request.broker_account_id == (
                stack["deps"].execution_runtime.broker_account_id
            )
            assert request.trading_date == stack["clock"].trading_date().isoformat()
        recorded = await stack["deps"].provenance_registry.list_by_target_portfolio_decision_id(
            decision_id
        )
        assert len(recorded) == 2
        assert {str((row.extra or {}).get("authorized_position_tuple_id")) for row in recorded} == {
            str(p["position_tuple_id"]) for p in positions
        }
        assert all((row.extra or {}).get("submission_exchange_time") for row in recorded)
        durable = await stack["deps"].provenance_registry.portfolio_executions.list_by_decision(
            decision_id
        )
        assert len(durable) == 2
        assert all(row.latest_validation_snapshot_id for row in durable)
        assert all(row.last_validation_timestamp for row in durable)
        assert all(row.last_reconciliation_timestamp for row in durable)
        assert all(row.session_id == stack["deps"].session_id for row in durable)
        assert all(row.run_id == stack["deps"].run_id for row in durable)
        assert all(
            row.broker_account_id
            == stack["deps"].execution_runtime.broker_account_id
            for row in durable
        )
        assert all(
            row.trading_date == stack["clock"].trading_date().isoformat()
            for row in durable
        )
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_restart_after_first_component_does_not_duplicate_or_skip(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision = result.get("_target_portfolio_decision") or {}
        decision_id = str(decision["decision_id"])
        first_client_id = stack["tracked_requests"][0].client_order_id
        runtime, registry = await _restart_portfolio_runtime(stack)

        await runtime._resume_portfolio_decision(decision_id)
        await runtime._resume_portfolio_decision(decision_id)

        records = await registry.portfolio_executions.list_by_decision(decision_id)
        assert broker.external_submission_count == 1
        assert [record.status.value for record in records] == [
            "WORKING",
            "AUTHORIZED",
        ]
        assert records[0].client_order_id == first_client_id
        assert records[1].client_order_id != first_client_id
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_same_session_account_date_genuinely_new_run_resumes_component(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision_id = str(result["_target_portfolio_decision"]["decision_id"])
        origin_run_id = stack["deps"].run_id
        new_run_id = str(uuid4())
        assert new_run_id != origin_run_id
        stack["deps"].run_id = new_run_id
        runtime, registry = await _restart_portfolio_runtime(stack)
        assert runtime._run_id == new_run_id

        await runtime._resume_portfolio_decision(decision_id)

        records = await registry.portfolio_executions.list_by_decision(decision_id)
        assert broker.external_submission_count == 1
        assert [record.status.value for record in records] == [
            "WORKING",
            "AUTHORIZED",
        ]
        assert all(record.origin_run_id == origin_run_id for record in records)
        assert all(record.last_resumed_run_id == new_run_id for record in records)
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_different_session_pending_portfolio_is_not_resumed_by_runtime(
    tmp_path, monkeypatch
) -> None:
    from joker.runtime.cognitive_agent_runtime import CognitiveAgentRuntime

    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        runtime = CognitiveAgentRuntime(
            session_id="different-session",
            run_id=stack["deps"].run_id,
            router=stack["deps"].router,
            config=stack["deps"].config,
            graph_deps=stack["deps"],
        )
        runtime._decision_graph = build_cognitive_graph(stack["deps"])
        await runtime._resume_pending_portfolio_executions()
        assert broker.external_submission_count == 1
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_different_broker_account_pending_portfolio_is_not_resumed_by_runtime(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        runtime, _registry = await _restart_portfolio_runtime(stack)
        stack["deps"].execution_runtime._broker_account_id = "other-paper-account"
        await runtime._resume_pending_portfolio_executions()
        assert broker.external_submission_count == 1
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_material_change_after_first_fill_enqueues_reoptimization(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision_id = str(result["_target_portfolio_decision"]["decision_id"])
        first_order = broker.list_open_orders()[0]
        broker.fill_order(first_order.order_id)

        svc = stack["objective_service"]
        original_get = svc.get_state

        async def _tight_capital():
            current = await original_get()
            return current.model_copy(update={"available_capital_usd": Decimal("1.00")})

        svc.get_state = _tight_capital  # type: ignore[method-assign]
        runtime, registry = await _restart_portfolio_runtime(stack)
        await runtime._resume_portfolio_decision(decision_id)

        records = await registry.portfolio_executions.list_by_decision(decision_id)
        assert broker.external_submission_count == 1
        assert [record.status.value for record in records] == [
            "FILLED",
            "REOPTIMIZATION_REQUIRED",
        ]
        assert "available_capital" in str(records[1].failure_reoptimization_reason)
        pending = await registry.portfolio_reoptimizations.list_pending(
            session_id=records[0].session_id,
            broker_account_identity=records[0].broker_account_identity,
            trading_date=records[0].trading_date,
        )
        assert len(pending) == 1
        request = pending[0]
        assert request.original_portfolio_decision_id == decision_id
        assert request.already_filled_tuple_ids == (records[0].authorized_position_tuple_id,)
        assert request.remaining_authorized_tuple_ids == (records[1].authorized_position_tuple_id,)
        assert request.latest_objective_version > 0
        assert request.latest_snapshot_id
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_restart_after_filled_first_component_submits_second_once(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision_id = str(result["_target_portfolio_decision"]["decision_id"])
        positions = result["_target_authorized_positions"]
        first_order = broker.list_open_orders()[0]
        broker.fill_order(first_order.order_id)

        runtime, registry = await _restart_portfolio_runtime(stack)
        await runtime._resume_portfolio_decision(decision_id)
        await runtime._resume_portfolio_decision(decision_id)

        records = await registry.portfolio_executions.list_by_decision(decision_id)
        assert broker.external_submission_count == 2
        assert [record.status.value for record in records] == ["FILLED", "FILLED"]
        assert [record.contract_id for record in records] == [
            position["contract_id"] for position in positions
        ]
        assert [record.authorized_quantity for record in records] == [
            int(position["quantity"]) for position in positions
        ]
        assert [record.component_index for record in records] == [0, 1]
        assert len({record.client_order_id for record in records}) == 2
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_reconciliation_only_resume_never_submits_new_component(
    tmp_path, monkeypatch
) -> None:
    from joker.persistence.cognitive_execution_provenance import (
        PortfolioComponentResolutionStatus,
        PortfolioComponentStatus,
        PortfolioExecutionOwner,
        PortfolioReoptimizationStatus,
        stable_reoptimization_request_id,
    )

    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision_id = str(result["_target_portfolio_decision"]["decision_id"])
        first_order = broker.list_open_orders()[0]
        broker.fill_order(first_order.order_id)

        runtime, registry = await _restart_portfolio_runtime(stack)
        runtime.enable_reconciliation_only_recovery(True)
        await runtime._resume_portfolio_decision(decision_id)

        resumed = await registry.portfolio_executions.list_by_decision(decision_id)
        owner = PortfolioExecutionOwner(
            session_id=resumed[0].session_id,
            broker_account_identity=resumed[0].broker_account_identity,
            trading_date=resumed[0].trading_date,
        )
        request_id = stable_reoptimization_request_id(
            session_id=owner.session_id,
            broker_account_identity=owner.broker_account_identity,
            trading_date=owner.trading_date,
            original_portfolio_decision_id=decision_id,
            remaining_authorized_tuple_ids=(resumed[1].authorized_position_tuple_id,),
        )
        terminal_request = await registry.portfolio_reoptimizations.get(request_id)
        assert broker.external_submission_count == 1
        assert [record.status.value for record in resumed] == [
            "FILLED",
            "REOPTIMIZATION_REQUIRED",
        ]
        assert resumed[1].resolution_status == PortfolioComponentResolutionStatus.OPERATOR_RESOLVED
        assert resumed[1].resolution_reason == "reconciliation_only_resume_no_new_entries"
        assert terminal_request is not None
        assert terminal_request.status == PortfolioReoptimizationStatus.COMPLETED
        assert terminal_request.replacement_action == "WAIT"
        assert not await registry.portfolio_reoptimizations.list_pending(
            session_id=owner.session_id,
            broker_account_identity=owner.broker_account_identity,
            trading_date=owner.trading_date,
        )
        assert not await registry.portfolio_executions.has_unresolved(
            session_id=owner.session_id,
            broker_account_identity=owner.broker_account_identity,
            trading_date=owner.trading_date,
        )
        assert not await registry.portfolio_reoptimizations.has_unresolved(
            session_id=owner.session_id,
            broker_account_identity=owner.broker_account_identity,
            trading_date=owner.trading_date,
        )
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_compiled_graph_five_second_latency_allows_valid_submission(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["filled", "filled"])
    stack = await _two_contract_stack(
        tmp_path,
        monkeypatch,
        broker=broker,
        objective_duration=timedelta(minutes=10),
    )
    try:
        _advance_exchange_clock_during_debate(stack, seconds=5)
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision = result["_target_portfolio_decision"]
        assert broker.external_submission_count == 2
        assert Decimal(str(decision["decision_age_seconds"])) == Decimal("5.0")
        assert decision["decision_timing_reason_codes"] == []
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_compiled_graph_real_clock_latency_does_not_fail_on_time_decay_only(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["filled", "filled"])
    stack = await _two_contract_stack(
        tmp_path,
        monkeypatch,
        broker=broker,
        objective_duration=timedelta(minutes=10),
    )
    try:
        # The injected exchange clock progresses between the optimizer and the
        # submission node; the fingerprint's time_remaining field therefore
        # decays exactly as it does during a real model-backed graph cycle.
        _advance_exchange_clock_during_debate(stack, seconds=7)
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        assert broker.external_submission_count == 2
        assert not any(
            "time_remaining_seconds" in error.message for error in result.get("errors") or []
        )
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("advance_seconds", "expected_submissions", "reason"),
    [
        (5, 2, None),
        (6, 0, "decision_age_exceeded"),
    ],
)
async def test_compiled_graph_decision_age_boundary(
    tmp_path,
    monkeypatch,
    advance_seconds: int,
    expected_submissions: int,
    reason: str | None,
) -> None:
    broker = ControllablePaperBroker(["filled", "filled"])
    stack = await _two_contract_stack(
        tmp_path,
        monkeypatch,
        broker=broker,
        objective_duration=timedelta(minutes=10),
    )
    try:
        stack["deps"].full_chain_optimizer_settings = stack[
            "deps"
        ].full_chain_optimizer_settings.model_copy(update={"maximum_decision_age_seconds": 5})
        _advance_exchange_clock_during_debate(stack, seconds=advance_seconds)
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        assert broker.external_submission_count == expected_submissions
        messages = " ".join(error.message for error in result.get("errors") or [])
        if reason is None:
            assert "decision_age_exceeded" not in messages
        else:
            assert reason in messages
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_compiled_graph_deadline_crossed_during_debate_blocks_submission(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["filled", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        stack["deps"].full_chain_optimizer_settings = stack[
            "deps"
        ].full_chain_optimizer_settings.model_copy(update={"maximum_decision_age_seconds": 120})
        _advance_exchange_clock_during_debate(stack, seconds=60)
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        assert broker.external_submission_count == 0
        assert any(
            "objective_deadline_reached" in error.message for error in result.get("errors") or []
        )
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_remaining_resolution_horizon_no_longer_fits_requires_reoptimization(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["filled", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        _advance_exchange_clock_during_debate(stack, seconds=30)
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        assert broker.external_submission_count == 0
        assert any(
            "resolution_horizon_no_longer_fits" in error.message
            for error in result.get("errors") or []
        )
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
@pytest.mark.parametrize("truth_change", ["target_reached", "capital"])
async def test_material_truth_change_during_debate_requires_reoptimization(
    tmp_path, monkeypatch, truth_change: str
) -> None:
    broker = ControllablePaperBroker(["filled", "filled"])
    stack = await _two_contract_stack(
        tmp_path,
        monkeypatch,
        broker=broker,
        objective_duration=timedelta(minutes=10),
    )
    try:
        changed = False
        service = stack["objective_service"]
        original_get_state = service.get_state

        async def _changed_state():
            state = await original_get_state()
            if not changed:
                return state
            if truth_change == "target_reached":
                return state.model_copy(
                    update={
                        "status": "target_reached",
                        "required_profit_remaining_usd": Decimal("0"),
                        "realised_pnl_usd": state.target_profit_usd,
                    }
                )
            return state.model_copy(update={"available_capital_usd": Decimal("1.00")})

        service.get_state = _changed_state  # type: ignore[method-assign]

        def _change_truth() -> None:
            nonlocal changed
            changed = True

        _advance_exchange_clock_during_debate(stack, seconds=5, after_advance=_change_truth)
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        assert broker.external_submission_count == 0
        expected = (
            "target_already_reached" if truth_change == "target_reached" else "available capital"
        )
        assert any(expected in error.message for error in result.get("errors") or []), [
            (error.error_code, error.message) for error in result.get("errors") or []
        ]
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_review_forced_wait_clears_authority_events_and_checkpoint(
    tmp_path, monkeypatch
) -> None:
    from joker.cli.graph_view import render_graph_event
    from joker.events.schemas import EventType
    from joker.objectives import portfolio_review as portfolio_review_mod

    broker = ControllablePaperBroker(["filled", "filled"])
    stack = await _two_contract_stack(
        tmp_path,
        monkeypatch,
        broker=broker,
        objective_duration=timedelta(minutes=10),
    )
    captured = []

    class _CaptureBus:
        async def publish(self, event):
            captured.append(event)
            return True

    original_review = portfolio_review_mod.portfolio_review_from_debate

    def _force_wait(review, context):
        return original_review(review, context).model_copy(
            update={"finalizer_recommendation": "wait"}
        )

    monkeypatch.setattr(portfolio_review_mod, "portfolio_review_from_debate", _force_wait)
    stack["deps"].event_bus = _CaptureBus()
    stack["graph"] = build_cognitive_graph(stack["deps"])
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision = result["_target_portfolio_decision"]
        legacy = result["_target_attainment_decision"]
        assert decision["action"] == "wait"
        assert decision["authorized_positions"] == []
        assert decision["selected_portfolio_id"] is None
        assert decision["selected_strategy_id"] is None
        assert decision["selected_contract_id"] is None
        assert decision["selected_quantity"] == 0
        assert Decimal(str(decision["selected_capital"])) == 0
        assert decision["selected_probability_goal"] == decision["wait_probability_goal"]
        assert Decimal(str(decision["probability_delta"])) == 0
        assert legacy["selected_strategy_id"] is None
        assert legacy["selected_contract_id"] is None
        assert legacy["selected_quantity"] == 0
        assert result["_target_authorized_positions"] == []
        assert result["_sizing_decision"] is None
        assert result["execution_proposal"] is None
        assert result["execution_command_id"] is None
        assert broker.external_submission_count == 0

        audit = result["_portfolio_review_rejected_decision_audit"]
        assert audit["audit_only"] is True
        assert audit["authoritative"] is False

        restored = await stack["graph"].aget_state(stack["config"])
        restored_values = restored.values
        assert restored_values["_target_authorized_positions"] == []
        assert restored_values["execution_proposal"] is None
        submit_node = stack["graph"].nodes["submit_execution_command"]
        await submit_node.ainvoke(restored_values)
        assert broker.external_submission_count == 0

        wait_event = next(
            event for event in captured if event.event_type == EventType.TARGET_WAIT_SELECTED
        )
        wait_payload = wait_event.payload
        assert wait_payload["decision"]["authorized_positions"] == []
        assert wait_payload["decision"]["selected_portfolio_id"] is None
        assert wait_payload["decision"]["selected_quantity"] == 0
        assert Decimal(str(wait_payload["probability_delta"])) == 0

        scored_events = [
            event
            for event in captured
            if event.event_type in {EventType.CONTRACT_GRID_SCORED, EventType.PORTFOLIO_GRID_SCORED}
        ]
        assert scored_events
        for event in scored_events:
            rows = event.payload.get("contracts") or event.payload.get("portfolios") or []
            assert all(row.get("selected") is False for row in rows)

        verbose = render_graph_event(wait_event.event_type.value, wait_payload, view="verbose")
        rendered_json = render_graph_event(wait_event.event_type.value, wait_payload, view="json")
        assert "selected=True" not in verbose
        assert '"selected": true' not in rendered_json.lower()
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
@pytest.mark.parametrize("working_status", ["accepted", "open", "pending"])
async def test_pending_first_component_queues_remaining_components_with_real_gateway(
    tmp_path, monkeypatch, working_status
) -> None:
    broker = ControllablePaperBroker([working_status, "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        positions = result.get("_target_authorized_positions") or []
        execution_state = result.get("_portfolio_execution_state") or []
        assert broker.external_submission_count == 1
        assert len(stack["tracked_requests"]) == 1
        assert [row["status"] for row in execution_state] == [
            "WORKING",
            "AUTHORIZED",
        ]
        assert [row["contract_id"] for row in execution_state] == [
            position["contract_id"] for position in positions
        ]
        assert [row["authorized_quantity"] for row in execution_state] == [
            position["quantity"] for position in positions
        ]
        assert not any(
            error.error_code in {"gateway_blocked", "submit_validation_failed"}
            for error in result.get("errors") or []
        )
        assert len(broker.all_orders()) == 1
        assert broker.all_orders()[0].status == working_status
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_partial_fill_preserves_remaining_component_without_second_entry(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["partially_filled", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker, qty_a=2)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        execution_state = result.get("_portfolio_execution_state") or []
        assert broker.external_submission_count == 1
        assert [row["status"] for row in execution_state] == [
            "PARTIALLY_FILLED",
            "AUTHORIZED",
        ]
        assert execution_state[0]["remaining_quantity"] > 0
        assert execution_state[1]["submitted_quantity"] == 0

        first_order = broker.all_orders()[0]
        first_client_id = stack["tracked_requests"][0].client_order_id
        broker.fill_order(first_order.order_id)
        await stack["deps"].execution_runtime.poll_order_status(first_client_id)
        runtime, registry = await _restart_portfolio_runtime(stack)
        await runtime._resume_portfolio_for_order(first_client_id)
        records = await registry.portfolio_executions.list_by_decision(
            str(result["_target_portfolio_decision"]["decision_id"])
        )
        assert broker.external_submission_count == 2
        assert [record.status.value for record in records] == ["FILLED", "FILLED"]
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_fill_event_resumes_next_component_once_with_stable_identity(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        first_order = broker.list_open_orders()[0]
        first_client_id = stack["tracked_requests"][0].client_order_id
        broker.fill_order(first_order.order_id)
        await stack["deps"].execution_runtime.poll_order_status(first_client_id)
        runtime, registry = await _restart_portfolio_runtime(stack)
        await runtime._resume_portfolio_for_order(first_client_id)
        decision_id = str(result["_target_portfolio_decision"]["decision_id"])
        execution_state = await registry.portfolio_executions.list_by_decision(decision_id)
        assert broker.external_submission_count == 2
        assert [row.status.value for row in execution_state] == ["FILLED", "FILLED"]
        assert len({row.client_order_id for row in execution_state}) == 2

        await runtime._resume_portfolio_for_order(first_client_id)
        assert broker.external_submission_count == 2
        replayed = await registry.portfolio_executions.list_by_decision(decision_id)
        assert replayed[1].client_order_id == execution_state[1].client_order_id
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", ["rejected", "cancelled"])
async def test_terminal_first_component_prevents_later_components(
    tmp_path, monkeypatch, terminal_status
) -> None:
    broker = ControllablePaperBroker([terminal_status, "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        execution_state = result.get("_portfolio_execution_state") or []
        assert broker.external_submission_count == 1
        assert execution_state[0]["status"] == terminal_status.upper()
        assert execution_state[1]["status"] == "REOPTIMIZATION_REQUIRED"
        assert execution_state[1]["failure_reoptimization_reason"] == (
            f"prior_component_{terminal_status}"
        )
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_crash_after_broker_fill_before_component_transition(tmp_path, monkeypatch) -> None:
    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision_id = str(result["_target_portfolio_decision"]["decision_id"])
        broker.fill_order(broker.list_open_orders()[0].order_id)

        runtime, registry = await _restart_portfolio_runtime(stack)
        await runtime._resume_portfolio_decision(decision_id)
        records = await registry.portfolio_executions.list_by_decision(decision_id)

        assert broker.external_submission_count == 2
        assert [record.status.value for record in records] == ["FILLED", "FILLED"]
        assert all(record.continuation_ready for record in records)
        assert all(record.post_fill_objective_fingerprint for record in records)
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_crash_after_filled_transition_before_post_fill_fingerprint(
    tmp_path, monkeypatch
) -> None:
    from joker.persistence.cognitive_execution_provenance import (
        PortfolioComponentStatus,
        PortfolioExecutionOwner,
    )

    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision_id = str(result["_target_portfolio_decision"]["decision_id"])
        records = await stack["deps"].provenance_registry.portfolio_executions.list_by_decision(
            decision_id
        )
        owner = PortfolioExecutionOwner(
            session_id=records[0].session_id,
            broker_account_identity=records[0].broker_account_identity,
            trading_date=records[0].trading_date,
        )
        await stack["deps"].provenance_registry.portfolio_executions.transition(
            records[0].authorized_position_tuple_id,
            owner=owner,
            status=PortfolioComponentStatus.FILLED,
            submitted_quantity=records[0].authorized_quantity,
            filled_quantity=records[0].authorized_quantity,
        )

        runtime, registry = await _restart_portfolio_runtime(stack)
        await runtime._resume_portfolio_decision(decision_id)
        resumed = await registry.portfolio_executions.list_by_decision(decision_id)

        assert broker.external_submission_count == 1
        assert resumed[0].status == PortfolioComponentStatus.FILLED
        assert resumed[0].continuation_ready is False
        assert resumed[1].status == PortfolioComponentStatus.REOPTIMIZATION_REQUIRED
        assert resumed[1].failure_reoptimization_reason == (
            "filled_component_missing_post_fill_checkpoint"
        )
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_crash_after_post_fill_fingerprint_before_next_submission(
    tmp_path, monkeypatch
) -> None:
    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    original_submit = stack["gateway"].submit
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision_id = str(result["_target_portfolio_decision"]["decision_id"])
        first_client_id = stack["tracked_requests"][0].client_order_id
        broker.fill_order(broker.list_open_orders()[0].order_id)
        await stack["deps"].execution_runtime.poll_order_status(first_client_id)

        async def _crash_before_second_submit(request):
            if request.component_index == 1:
                raise RuntimeError("simulated crash before second submission")
            return await original_submit(request)

        stack["gateway"].submit = _crash_before_second_submit
        runtime, registry = await _restart_portfolio_runtime(stack)
        with pytest.raises(RuntimeError, match="simulated crash"):
            await runtime._resume_portfolio_decision(decision_id)
        crashed = await registry.portfolio_executions.list_by_decision(decision_id)
        assert crashed[0].continuation_ready is True
        assert crashed[1].status.value == "READY"
        assert broker.external_submission_count == 1

        stack["gateway"].submit = original_submit
        retry_runtime, registry = await _restart_portfolio_runtime(stack)
        await retry_runtime._resume_portfolio_decision(decision_id)
        resumed = await registry.portfolio_executions.list_by_decision(decision_id)
        assert broker.external_submission_count == 2
        assert [record.status.value for record in resumed] == ["FILLED", "FILLED"]
    finally:
        stack["gateway"].submit = original_submit
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_crash_after_next_submission_before_component_state_update(
    tmp_path, monkeypatch
) -> None:
    from joker.persistence.cognitive_execution_provenance import (
        PortfolioComponentStatus,
    )

    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        decision_id = str(result["_target_portfolio_decision"]["decision_id"])
        positions = result["_target_authorized_positions"]
        first_client_id = stack["tracked_requests"][0].client_order_id
        broker.fill_order(broker.list_open_orders()[0].order_id)
        await stack["deps"].execution_runtime.poll_order_status(first_client_id)

        runtime, crash_registry = await _restart_portfolio_runtime(stack)
        repo = crash_registry.portfolio_executions
        original_transition = repo.transition
        crashed = False

        async def _crash_before_second_state_update(tuple_id, **kwargs):
            nonlocal crashed
            if (
                not crashed
                and tuple_id == str(positions[1]["position_tuple_id"])
                and kwargs.get("status") == PortfolioComponentStatus.FILLED
            ):
                crashed = True
                raise RuntimeError("simulated crash before second state update")
            return await original_transition(tuple_id, **kwargs)

        repo.transition = _crash_before_second_state_update  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="simulated crash"):
            await runtime._resume_portfolio_decision(decision_id)
        assert broker.external_submission_count == 2

        repo.transition = original_transition  # type: ignore[method-assign]
        retry_runtime, registry = await _restart_portfolio_runtime(stack)
        await retry_runtime._resume_portfolio_decision(decision_id)
        resumed = await registry.portfolio_executions.list_by_decision(decision_id)
        assert broker.external_submission_count == 2
        assert [record.status.value for record in resumed] == ["FILLED", "FILLED"]
        assert len({record.client_order_id for record in resumed}) == 2
    finally:
        await _teardown_stack(stack)


async def _prepare_pending_reoptimization(tmp_path, monkeypatch):
    from joker.persistence.cognitive_execution_provenance import (
        CognitiveExecutionProvenanceRegistry,
    )
    from joker.objectives import full_chain_optimizer as fco_mod

    broker = ControllablePaperBroker(["open", "filled"])
    stack = await _two_contract_stack(tmp_path, monkeypatch, broker=broker)
    result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
    decision_id = str(result["_target_portfolio_decision"]["decision_id"])
    broker.fill_order(broker.list_open_orders()[0].order_id)
    service = stack["objective_service"]
    original_get_state = service.get_state

    async def _materially_changed_capital():
        current = await original_get_state()
        return current.model_copy(update={"available_capital_usd": Decimal("1.00")})

    service.get_state = _materially_changed_capital  # type: ignore[method-assign]
    runtime, registry = await _restart_portfolio_runtime(stack)
    await runtime._resume_portfolio_decision(decision_id)
    components = await registry.portfolio_executions.list_by_decision(decision_id)
    pending = await registry.portfolio_reoptimizations.list_pending(
        session_id=components[0].session_id,
        broker_account_identity=components[0].broker_account_identity,
        trading_date=components[0].trading_date,
    )
    assert len(pending) == 1
    request = pending[0]
    assert request.open_positions
    assert request.already_filled_tuple_ids == (components[0].authorized_position_tuple_id,)
    monkeypatch.setattr(
        fco_mod, "optimize_full_chain", stack["real_optimize_full_chain"]
    )
    # Reopen the registry just as a new process would.
    registry = CognitiveExecutionProvenanceRegistry(stack["deps"].db_path)
    await registry.initialize()
    stack["deps"].provenance_registry = registry
    runtime._deps.provenance_registry = registry
    return stack, runtime, registry, request, result, original_get_state


@pytest.mark.asyncio
async def test_new_decision_without_terminal_node_does_not_complete_request(
    tmp_path, monkeypatch
) -> None:
    from joker.persistence.cognitive_execution_provenance import (
        PortfolioReoptimizationStatus,
    )

    stack, runtime, registry, request, _result, _original_get_state = (
        await _prepare_pending_reoptimization(tmp_path, monkeypatch)
    )
    try:
        class _IncompleteGraph:
            async def ainvoke(self, state, config=None):
                return {
                    **state,
                    "_target_portfolio_decision": {
                        "decision_id": str(uuid4()),
                        "action": "wait",
                        "authorized_positions": [],
                    },
                    "_target_authorized_positions": [],
                    "node_trace": [],
                    "errors": [],
                }

        runtime._decision_graph = _IncompleteGraph()
        await runtime._resume_pending_portfolio_reoptimizations()
        failed = await registry.portfolio_reoptimizations.get(request.request_id)
        assert failed is not None
        assert failed.status == PortfolioReoptimizationStatus.FAILED
        assert failed.failure_reason == "reoptimization_terminal_outcome_missing"
        assert failed.attempt_count == 1
        assert failed.last_attempt_run_id == runtime._run_id
    finally:
        await _teardown_stack(stack)


def _validation_runtime_and_request():
    from joker.runtime.cognitive_agent_runtime import CognitiveAgentRuntime

    runtime = CognitiveAgentRuntime.__new__(CognitiveAgentRuntime)
    runtime._deps = SimpleNamespace()
    runtime._session_id = "session-a"
    request = SimpleNamespace(
        request_id="request-a",
        session_id="session-a",
        broker_account_identity="paper-a",
        trading_date="2026-08-05",
        original_portfolio_decision_id="decision-old",
        already_filled_tuple_ids=("tuple-filled",),
        remaining_authorized_tuple_ids=("tuple-old-pending",),
        open_positions=({"contract_id": CONTRACT_ID, "quantity": 1},),
    )
    base = {
        "_portfolio_reoptimization_request_id": request.request_id,
        "_portfolio_execution_owner": {
            "session_id": request.session_id,
            "broker_account_identity": request.broker_account_identity,
            "trading_date": request.trading_date,
        },
        "node_trace": [{"node_name": "persist_cycle", "status": "completed"}],
        "errors": [],
    }
    return runtime, request, base


@pytest.mark.asyncio
async def test_new_decision_with_blocking_error_does_not_complete_request() -> None:
    runtime, request, base = _validation_runtime_and_request()
    valid, reason, _decision_id, _action = await runtime._validate_reoptimization_result(
        request,
        {
            **base,
            "errors": [{"error_code": "validation_failed"}],
            "_target_portfolio_decision": {
                "decision_id": "decision-new",
                "action": "wait",
                "authorized_positions": [],
            },
            "_target_authorized_positions": [],
        },
    )
    assert valid is False
    assert reason == "reoptimization_graph_has_blocking_error"


@pytest.mark.asyncio
async def test_wait_reoptimization_completes_only_with_empty_authority() -> None:
    runtime, request, base = _validation_runtime_and_request()
    stale = [{"position_tuple_id": "tuple-old-pending", "contract_id": SECOND_CONTRACT_ID}]
    valid, reason, _decision_id, _action = await runtime._validate_reoptimization_result(
        request,
        {
            **base,
            "_target_portfolio_decision": {
                "decision_id": "decision-new",
                "action": "wait",
                "authorized_positions": stale,
            },
            "_target_authorized_positions": stale,
            "execution_proposal": object(),
        },
    )
    assert valid is False
    assert reason == "wait_reoptimization_retains_authority"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tuple_id", "contract_id", "expected_reason"),
    [
        ("tuple-old-pending", SECOND_CONTRACT_ID, "replacement_reuses_old_tuple"),
        ("tuple-new", CONTRACT_ID, "replacement_selects_existing_contract"),
    ],
)
async def test_enter_reoptimization_rejects_stale_authority(
    tuple_id, contract_id, expected_reason
) -> None:
    runtime, request, base = _validation_runtime_and_request()
    positions = [
        {
            "position_tuple_id": tuple_id,
            "contract_id": contract_id,
            "decision_id": "decision-new",
        }
    ]
    valid, reason, _decision_id, _action = await runtime._validate_reoptimization_result(
        request,
        {
            **base,
            "_target_portfolio_decision": {
                "decision_id": "decision-new",
                "action": "enter",
                "authorized_positions": positions,
            },
            "_target_authorized_positions": positions,
            "_reoptimization_excluded_contract_ids": [CONTRACT_ID],
            "execution_proposal": object(),
        },
    )
    assert valid is False
    assert reason == expected_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_action", ["WAIT", "ENTER"])
async def test_valid_compiled_reoptimization_completes(
    tmp_path, monkeypatch, replacement_action
) -> None:
    from joker.persistence.cognitive_execution_provenance import (
        PortfolioComponentResolutionStatus,
        PortfolioReoptimizationStatus,
    )

    stack, runtime, registry, request, original_result, original_get_state = (
        await _prepare_pending_reoptimization(tmp_path, monkeypatch)
    )
    try:
        if replacement_action == "WAIT":
            stack["deps"].kill_switch = True
        else:
            stack["objective_service"].get_state = original_get_state
        cycle_id = f"portfolio-reoptimization-{request.request_id}"
        register_full_path_canned(
            stack["fake_model_provider"],
            UUID(request.latest_snapshot_id),
            cycle_id,
            session=stack["deps"].session_id,
        )
        captured_result = {}
        original_validator = runtime._validate_reoptimization_result

        async def _capture_validation(persisted_request, result_state):
            captured_result.update(result_state)
            return await original_validator(persisted_request, result_state)

        runtime._validate_reoptimization_result = _capture_validation
        runtime._decision_graph = build_cognitive_graph(stack["deps"])
        await runtime._resume_pending_portfolio_reoptimizations()

        completed = await registry.portfolio_reoptimizations.get(request.request_id)
        assert completed is not None
        assert completed.status == PortfolioReoptimizationStatus.COMPLETED, (
            completed.failure_reason,
            request.open_positions,
            await runtime._open_position_contract_ids(),
            (captured_result.get("_target_portfolio_decision") or {}).get(
                "authorized_positions"
            ),
            captured_result.get("_reoptimization_expected_objective_version"),
            (captured_result.get("_target_portfolio_decision") or {}).get(
                "objective_version"
            ),
            (captured_result.get("_target_portfolio_decision") or {}).get(
                "submission_objective_version"
            ),
        )
        assert completed.replacement_decision_id
        assert completed.replacement_decision_id != request.original_portfolio_decision_id
        assert completed.replacement_action == replacement_action
        preserved = await registry.portfolio_executions.list_by_decision(
            request.original_portfolio_decision_id
        )
        assert preserved[0].status.value == "FILLED"
        assert preserved[0].contract_id == original_result["_target_authorized_positions"][0][
            "contract_id"
        ]
        assert preserved[1].status.value == "REOPTIMIZATION_REQUIRED"
        assert preserved[1].resolution_status == PortfolioComponentResolutionStatus.SUPERSEDED
        assert preserved[1].superseded_by_reoptimization_request_id == request.request_id
        assert preserved[1].superseded_by_decision_id == completed.replacement_decision_id
        if replacement_action == "WAIT":
            assert stack["broker"].external_submission_count == 1
        else:
            assert stack["broker"].external_submission_count == 2
    finally:
        await _teardown_stack(stack)
