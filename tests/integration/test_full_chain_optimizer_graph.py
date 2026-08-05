from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

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
from tests.cognitive.task2_canned import CONTRACT_ID
from tests.integration.test_goal_driven_full_graph import (
    _prepare_stack,
    _teardown_stack,
)

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


def _ctx(*, snapshot_id=None) -> TargetAttainmentContext:
    return TargetAttainmentContext(
        objective_id=uuid4(),
        snapshot_id=snapshot_id or uuid4(),
        authorised_capital_usd=Decimal("200"),
        available_capital_usd=Decimal("200"),
        reserved_capital_usd=Decimal("0"),
        realised_pnl_usd=Decimal("0"),
        unrealised_pnl_usd=Decimal("0"),
        target_profit_usd=Decimal("20"),
        remaining_goal_gap_usd=Decimal("10"),
        time_remaining_seconds=300,
        objective_duration_seconds=3600,
        elapsed_seconds=3300,
        open_position_count=0,
        working_order_count=0,
        max_concurrent_positions=1,
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
        result = await stack["graph"].ainvoke(
            stack["state"], config=stack["config"]
        )
        trace_names = [trace.node_name for trace in result.get("node_trace") or ()]
        assert "entry_tactician" in trace_names
        assert not any(
            error.error_code == "missing_proposal"
            for error in result.get("errors") or ()
        )
        decision = result.get("_target_portfolio_decision") or {}
        assert decision.get("action") == "enter"
        assert result.get("_portfolio_review_context")
        assert (
            result.get("_portfolio_review_finalization") or {}
        ).get("action") == "preserve_exact_tuple"
        assert len(stack["submitted"]) == len(
            result.get("_target_authorized_positions") or ()
        ), [(error.error_code, error.message) for error in result.get("errors") or ()]
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
        result = await build_cognitive_graph(deps).ainvoke(
            stack["state"], config=stack["config"]
        )
        positions = result.get("_target_authorized_positions") or []
        proposal = result["execution_proposal"]
        assert [
            (str(leg.strategy_id), leg.contract_id, leg.quantity)
            for leg in proposal.legs
        ] == [
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
        strategies = list(kwargs.get("strategies") or [])
        strategy_id = (
            UUID(str(strategies[0].strategy_id)) if strategies else uuid4()
        )
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
                capital_allocation=(
                    Decimal(premium_a) * Decimal(qty_a) * Decimal("100")
                ).quantize(Decimal("0.01")),
                maximum_loss=(
                    Decimal(premium_a) * Decimal(qty_a) * Decimal("100")
                ).quantize(Decimal("0.01")),
                snapshot_id=snapshot_id,
                objective_version=kwargs["ctx"].objective_version,
                decision_id=decision_id,
                evaluated_objective_fingerprint=fingerprint,
            ),
            AuthorizedPositionTuple(
                position_tuple_id=uuid4(),
                strategy_id=strategy_id,
                contract_id=SECOND_CONTRACT_ID,
                quantity=qty_b,
                evaluation_premium=Decimal(premium_b),
                capital_allocation=(
                    Decimal(premium_b) * Decimal(qty_b) * Decimal("100")
                ).quantize(Decimal("0.01")),
                maximum_loss=(
                    Decimal(premium_b) * Decimal(qty_b) * Decimal("100")
                ).quantize(Decimal("0.01")),
                snapshot_id=snapshot_id,
                objective_version=kwargs["ctx"].objective_version,
                decision_id=decision_id,
                evaluated_objective_fingerprint=fingerprint,
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
        )
        return FullChainOptimizationResult(
            universe=result.universe,
            selection_specs=result.selection_specs,
            shared_scenario_grid=result.shared_scenario_grid,
            contract_outcomes=result.contract_outcomes,
            decision=decision,
        )

    monkeypatch.setattr(fco_mod, "optimize_full_chain", _wrapped)


async def _two_contract_stack(tmp_path, monkeypatch):
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
        objective_duration=timedelta(minutes=4),
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
    )
    _enable_full_chain(stack["deps"])
    _force_two_component_optimize(monkeypatch, qty_a=1, qty_b=3)
    provenance = CognitiveExecutionProvenanceRegistry(stack["deps"].db_path)
    await provenance.initialize()
    stack["deps"].provenance_registry = provenance
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


async def _seed_first_component_provenance(stack, result) -> str:
    """Persist component-0 as already submitted for restart simulation."""
    from joker.objectives.decision_fingerprint import ObjectiveDecisionFingerprint
    from joker.persistence.cognitive_execution_provenance import (
        ExecutionProvenanceRecord,
    )
    from joker.runtime.order_action_gateway import working_orders_from_projection

    positions = result.get("_target_authorized_positions") or []
    decision = result.get("_target_portfolio_decision") or {}
    decision_id = str(decision["decision_id"])
    first = positions[0]
    proposal = result["execution_proposal"]
    objective = await stack["objective_service"].get_state()
    projection = (
        await stack["deps"].projection_loader()
        if stack["deps"].projection_loader is not None
        else None
    )
    fingerprint = ObjectiveDecisionFingerprint.from_state(
        objective,
        working_order_count=len(working_orders_from_projection(projection)),
        broker_identity="PaperBroker",
        broker_eligible=True,
        reconciliation_eligible=True,
    )
    client_order_id = f"seeded-{first['position_tuple_id']}"
    await stack["deps"].provenance_registry.record(
        ExecutionProvenanceRecord(
            client_order_id=client_order_id,
            proposal_id=str(proposal.proposal_id),
            decision_id=str(proposal.decision_id),
            strategy_id=str(first["strategy_id"]),
            cycle_id=str(result.get("cycle_id") or "cycle-hist-ev"),
            snapshot_id=str(first["snapshot_id"]),
            contract_id=str(first["contract_id"]),
            session_id=stack["deps"].session_id,
            kind="entry",
            extra={
                "target_portfolio_decision_id": decision_id,
                "selected_portfolio_id": str(decision.get("selected_portfolio_id")),
                "authorized_position_tuple_id": str(first["position_tuple_id"]),
                "component_index": 0,
                "component_count": 2,
                "evaluated_objective_version": int(first["objective_version"]),
                "post_submission_objective_fingerprint": fingerprint.canonical_json,
                "post_submission_objective_version": int(objective.version),
            },
        )
    )
    return client_order_id


@pytest.mark.asyncio
async def test_compiled_graph_preserves_distinct_portfolio_quantities(
    tmp_path, monkeypatch
) -> None:
    stack = await _two_contract_stack(tmp_path, monkeypatch)
    try:
        result = await stack["graph"].ainvoke(
            stack["state"], config=stack["config"]
        )
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
                        "target_portfolio_decision_id": (
                            request.target_portfolio_decision_id
                        ),
                        "selected_portfolio_id": request.selected_portfolio_id,
                        "authorized_position_tuple_id": (
                            request.authorized_position_tuple_id
                        ),
                        "component_index": request.component_index,
                        "component_count": request.component_count,
                    },
                )
            )
            return OrderActionResult(
                submitted=True,
                client_order_id=request.client_order_id,
                broker_order=SimpleNamespace(order_id=request.client_order_id),
                working_orders={},
            )

        gateway.submit = _approve  # type: ignore[method-assign]
        stack["tracked_requests"].clear()
        result = await stack["graph"].ainvoke(
            stack["state"], config=stack["config"]
        )
        positions = result.get("_target_authorized_positions") or []
        decision = result.get("_target_portfolio_decision") or {}
        assert len(stack["tracked_requests"]) == len(positions) == 2
        decision_id = str(decision["decision_id"])
        for index, (request, position) in enumerate(
            zip(stack["tracked_requests"], positions, strict=True)
        ):
            assert request.target_portfolio_decision_id == decision_id
            assert request.authorized_position_tuple_id == str(
                position["position_tuple_id"]
            )
            assert request.component_index == index
            assert request.selected_portfolio_id == str(
                decision["selected_portfolio_id"]
            )
        recorded = await stack[
            "deps"
        ].provenance_registry.list_by_target_portfolio_decision_id(decision_id)
        assert len(recorded) == 2
        assert {
            str((row.extra or {}).get("authorized_position_tuple_id"))
            for row in recorded
        } == {str(p["position_tuple_id"]) for p in positions}
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_restart_after_first_component_does_not_duplicate_or_skip(
    tmp_path, monkeypatch
) -> None:
    stack = await _two_contract_stack(tmp_path, monkeypatch)
    try:
        from joker.runtime.order_action_gateway import OrderActionResult

        # First pass: build authoritative two-component proposal without submitting.
        async def _block_all(request):
            return OrderActionResult(
                submitted=False,
                client_order_id=request.client_order_id,
                blocked_reason="defer_submission_for_restart_test",
                working_orders={},
            )

        stack["gateway"].submit = _block_all  # type: ignore[method-assign]
        result = await stack["graph"].ainvoke(
            stack["state"], config=stack["config"]
        )
        positions = result.get("_target_authorized_positions") or []
        decision = result.get("_target_portfolio_decision") or {}
        assert len(positions) == 2
        decision_id = str(decision["decision_id"])
        seeded_id = await _seed_first_component_provenance(stack, result)

        resume_calls: list = []

        async def _approve_remaining(request):
            resume_calls.append(request)
            return OrderActionResult(
                submitted=True,
                client_order_id=request.client_order_id,
                broker_order=SimpleNamespace(order_id=request.client_order_id),
                working_orders={},
            )

        stack["gateway"].submit = _approve_remaining  # type: ignore[method-assign]
        submit_node = stack["graph"].nodes["submit_execution_command"]
        resume_state: CognitiveGraphState = {
            **result,
            "execution_command_id": None,
            "_execution_command_ids": None,
            "errors": [],
            "_block_new_entries": False,
        }
        resumed = await submit_node.ainvoke(resume_state)
        command_ids = list(resumed.get("_execution_command_ids") or [])
        assert command_ids[0] == seeded_id
        assert len(resume_calls) == 1
        assert resume_calls[0].component_index == 1
        assert resume_calls[0].authorized_position_tuple_id == str(
            positions[1]["position_tuple_id"]
        )
        assert resume_calls[0].target_portfolio_decision_id == decision_id
        assert len(command_ids) == 2
        assert command_ids[1] == resume_calls[0].client_order_id
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_remaining_components_revalidate_after_restart(
    tmp_path, monkeypatch
) -> None:
    stack = await _two_contract_stack(tmp_path, monkeypatch)
    try:
        from joker.runtime.order_action_gateway import OrderActionResult

        async def _block_all(request):
            return OrderActionResult(
                submitted=False,
                client_order_id=request.client_order_id,
                blocked_reason="defer_submission_for_restart_test",
                working_orders={},
            )

        stack["gateway"].submit = _block_all  # type: ignore[method-assign]
        result = await stack["graph"].ainvoke(
            stack["state"], config=stack["config"]
        )
        assert len(result.get("_target_authorized_positions") or []) == 2
        await _seed_first_component_provenance(stack, result)

        svc = stack["objective_service"]
        original_get = svc.get_state

        async def _tight_capital():
            current = await original_get()
            return current.model_copy(
                update={"available_capital_usd": Decimal("1.00")}
            )

        svc.get_state = _tight_capital  # type: ignore[method-assign]
        remaining_calls: list = []

        async def _should_not_submit(request):
            remaining_calls.append(request)
            return OrderActionResult(
                submitted=True,
                client_order_id=request.client_order_id,
                broker_order=SimpleNamespace(order_id=request.client_order_id),
                working_orders={},
            )

        stack["gateway"].submit = _should_not_submit  # type: ignore[method-assign]
        submit_node = stack["graph"].nodes["submit_execution_command"]
        resume_state: CognitiveGraphState = {
            **result,
            "execution_command_id": None,
            "_execution_command_ids": None,
            "errors": [],
            "_block_new_entries": False,
        }
        resumed = await submit_node.ainvoke(resume_state)
        assert resumed.get("_block_new_entries") is True
        assert remaining_calls == []
        assert any(
            getattr(error, "error_code", None)
            == "target_attainment_recalculation_required"
            for error in (resumed.get("errors") or [])
        )
    finally:
        await _teardown_stack(stack)
