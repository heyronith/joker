"""Paper vs live capture-mode: identical pre-broker graph commands."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.config.settings import AppSettings
from joker.models.fake_provider import FakeModelProvider
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
from joker.objectives.service import SessionObjectiveService
from joker.persistence.migrations import apply_task1_migrations
from joker.runtime.cognitive_session_factory import (
    prepare_cognitive_live_session,
    prepare_cognitive_paper_session,
)
from joker.runtime.execution_runtime import ExecutionCommand
from joker.runtime.order_action_gateway import OrderActionGateway, OrderActionKind, OrderActionRequest
from joker.schemas.domain import BrokerOrder, OptionContract, OrderIntent
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.broker._live_helpers import make_live_client
from tests.cognitive.task2_canned import CONTRACT_ID

ET = ZoneInfo("America/New_York")


def _entry_request(**kwargs) -> OrderActionRequest:
    defaults = {
        "action": OrderActionKind.ENTRY,
        "client_order_id": "z" * 32,
        "contract_id": CONTRACT_ID,
        "side": "buy",
        "quantity": 1,
        "order_type": "limit",
        "limit_price": 1.10,
        "snapshot_id": "snap-1",
        "strategy_id": uuid4(),
        "estimate_id": uuid4(),
        "proposal_id": uuid4(),
        "decision_id": uuid4(),
        "cycle_id": "cycle-1",
    }
    defaults.update(kwargs)
    return OrderActionRequest(**defaults)


def _command_fields(cmd: ExecutionCommand) -> dict:
    intent = cmd.intent
    contract = intent.contract
    return {
        "contract": (
            f"{contract.symbol}:{contract.expiration.isoformat()}:"
            f"{contract.strike}:{contract.option_type}"
        ),
        "side": intent.side,
        "quantity": intent.quantity,
        "limit_price": intent.limit_price,
        "position_intent": intent.position_intent,
        "broker_account_id": cmd.broker_account_id,
    }


@pytest.mark.asyncio
async def test_public_paper_live_graph_command_equivalence(tmp_path, monkeypatch) -> None:
    """Paper and capture-only live sessions compile identical execution commands."""
    from joker.runtime import order_action_gateway as gw_mod

    db = tmp_path / "equiv.db"
    apply_task1_migrations(db)
    apply_objective_migrations(db)
    obj_repo = ObjectiveRepository(db)
    objective_service = SessionObjectiveService(
        obj_repo, require_positive_expected_value=False
    )
    definition = await objective_service.create_objective(
        session_id="equiv-session",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=4),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await objective_service.confirm_objective(definition.objective_id)

    paper_app = AppSettings(
        db_path=str(db),
        live_trading_enabled=False,
        evolution={"enabled": True},
        objective={"enabled": True, "require_positive_expected_value": False},
        cognitive_graph={"enabled": True},
    )
    live_app = AppSettings(
        mode=SafetyMode.LIVE_GATED,
        live_trading_enabled=True,
        db_path=str(db),
        broker={"provider": "webull_live"},
        evolution={"enabled": True},
        objective={"enabled": True, "require_positive_expected_value": False},
        cognitive_graph={"enabled": True},
    )
    fake = FakeModelProvider(available=True)
    clock = FrozenExchangeClock(datetime(2026, 7, 1, 10, 0, tzinfo=ET), calendar=MarketCalendar())

    paper_session = await prepare_cognitive_paper_session(
        app_settings=paper_app,
        objective_service=objective_service,
        broker=PaperBroker(slippage_pct=0),
        db_path=db,
        session_id="equiv-session",
        fake_model_provider=fake,
        clock=clock,
        start_cognitive_agent=False,
        start_evolution_workers=False,
    )
    live_client, _, _ = make_live_client(tmp_path, capture_only=True)
    live_session = await prepare_cognitive_live_session(
        app_settings=live_app,
        objective_service=objective_service,
        broker=live_client,
        db_path=db,
        session_id="equiv-session-live",
        fake_model_provider=fake,
        clock=clock,
        start_cognitive_agent=False,
        start_evolution_workers=False,
    )
    try:
        async def _fake_load(deps, snapshot_id):
            return (
                SimpleNamespace(snapshot_id=snapshot_id, trading_date=date(2026, 7, 1)),
                SimpleNamespace(usable_for_execution=True),
                SimpleNamespace(surface_id=uuid4(), contracts=()),
                (),
            )

        def _fixed_compile(self, request, **kwargs):
            intent = OrderIntent(
                intent_id=request.client_order_id,
                candidate_id=str(request.proposal_id or uuid4()),
                contract=OptionContract(
                    symbol="SPY",
                    expiration=date(2026, 7, 1),
                    strike=500.0,
                    option_type="call",
                ),
                side=request.side,
                order_type=request.order_type,
                quantity=request.quantity,
                limit_price=request.limit_price,
                position_intent="BUY_TO_OPEN",
            )
            return ExecutionCommand(
                client_order_id=request.client_order_id,
                intent=intent,
                broker_account_id=request.broker_account_id,
            )

        monkeypatch.setattr(gw_mod, "load_snapshot_truth", _fake_load)
        monkeypatch.setattr(OrderActionGateway, "_validate_and_compile", _fixed_compile)
        monkeypatch.setattr(
            OrderActionGateway,
            "_maybe_live_preview",
            AsyncMock(return_value=None),
        )

        paper_cmds: list[ExecutionCommand] = []
        live_cmds: list[ExecutionCommand] = []

        async def _intercept_paper(cmd: ExecutionCommand) -> BrokerOrder:
            paper_cmds.append(cmd)
            raise RuntimeError("stop-before-broker")

        async def _intercept_live(cmd: ExecutionCommand) -> BrokerOrder:
            live_cmds.append(cmd)
            raise RuntimeError("stop-before-broker")

        paper_rt = paper_session.bridge.execution_runtime
        live_rt = live_session.bridge.execution_runtime
        paper_session.graph_deps.execution_runtime.submit_execution_command = _intercept_paper  # type: ignore[method-assign]
        live_session.graph_deps.execution_runtime.submit_execution_command = _intercept_live  # type: ignore[method-assign]
        paper_rt.submit_execution_command = _intercept_paper  # type: ignore[method-assign]
        live_rt.submit_execution_command = _intercept_live  # type: ignore[method-assign]

        from dataclasses import replace

        paper_gateway = OrderActionGateway(
            replace(paper_session.graph_deps, objective_service=None)
        )
        live_gateway = OrderActionGateway(
            replace(live_session.graph_deps, objective_service=None)
        )

        request = _entry_request(
            broker_account_id=paper_session.graph_deps.execution_runtime._broker_account_id,
        )

        with pytest.raises(RuntimeError, match="stop-before-broker"):
            await paper_gateway.submit(request)
        live_request = _entry_request(
            client_order_id="y" * 32,
            broker_account_id=live_session.graph_deps.execution_runtime._broker_account_id,
        )
        with pytest.raises(RuntimeError, match="stop-before-broker"):
            await live_gateway.submit(live_request)

        paper_fields = _command_fields(paper_cmds[0])
        live_fields = _command_fields(live_cmds[0])
        for key in ("contract", "side", "quantity", "limit_price", "position_intent"):
            assert paper_fields[key] == live_fields[key]
        assert paper_fields["broker_account_id"] != live_fields["broker_account_id"]
    finally:
        await paper_session.shutdown()
        await live_session.shutdown()


def test_live_capture_mode_preserves_payload_without_placement(tmp_path) -> None:
    """Lightweight capture payload check — no broker placement."""
    from tests.broker._live_helpers import make_intent

    client, api, _ = make_live_client(tmp_path, capture_only=True)
    intent = make_intent()
    order = client.submit_order(intent)
    assert order.status == "pending"
    assert api.placed == []
    assert len(client.captured_payloads) == 1
    payload = client.captured_payloads[0]
    assert payload["limit_price"] == "1.10"
    assert payload["quantity"] == "1"
    assert payload["position_intent"] == "BUY_TO_OPEN"
    assert payload["side"] == "BUY"
