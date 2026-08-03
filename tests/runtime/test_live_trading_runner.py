"""LiveTradingRunner gates and shared session factory."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.broker.webull_live import create_mock_live_trade_api
from joker.config.settings import AppSettings, EnvSettings
from joker.runtime.cognitive_session_factory import prepare_cognitive_paper_session
from joker.runtime.live_activation import create_live_activation
from joker.runtime.live_preflight import run_production_preflight
from joker.runtime.live_trading import LiveTradingError, LiveTradingRunner
from joker.broker.account_truth import hash_account_id


def _live_env(**kw) -> EnvSettings:
    base = dict(
        OPENAI_API_KEY="k",
        WEBULL_LIVE_TRADING_ENABLED=True,
        WEBULL_LIVE_APP_KEY="lk",
        WEBULL_LIVE_APP_SECRET="ls",
        WEBULL_LIVE_ACCESS_TOKEN="lt",
        WEBULL_LIVE_ACCOUNT_ID="LIVE_ACCT_1",
        WEBULL_LIVE_API_ENV="prod",
    )
    base.update(kw)
    return EnvSettings(**base)  # type: ignore[arg-type]


def _live_app(tmp_path) -> AppSettings:
    return AppSettings(
        mode=SafetyMode.LIVE_GATED,
        live_trading_enabled=True,
        db_path=str(tmp_path / "live.db"),
        broker={"provider": "webull_live"},
        evolution={"enabled": True},
        objective={
            "enabled": True,
            "require_positive_expected_value": True,
        },
        cognitive_graph={"enabled": True},
    )


def test_live_runner_rejects_paper_broker_mode(tmp_path) -> None:
    app = AppSettings(
        mode=SafetyMode.PAPER,
        live_trading_enabled=False,
        evolution={"enabled": True},
        objective={"enabled": True},
    )
    activation = create_live_activation(
        account_id_hash="x",
        objective_id=uuid4(),
        authorized_capital_usd=Decimal("1000"),
    )
    with pytest.raises(LiveTradingError, match="LIVE_GATED"):
        LiveTradingRunner(
            app_settings=app,
            env=_live_env(),
            objective_service=MagicMock(),
            activation=activation,
        )


@pytest.mark.asyncio
async def test_paper_runner_rejects_live_broker(tmp_path) -> None:
    from joker.broker.webull_live import WebullLiveClient

    app = AppSettings(
        mode=SafetyMode.PAPER,
        live_trading_enabled=False,
        db_path=str(tmp_path / "p.db"),
        evolution={"enabled": True},
        objective={"enabled": True},
    )
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    live = WebullLiveClient(
        _live_env(),
        app_settings=_live_app(tmp_path),
        trade_api=api,
        skip_account_list_check=True,
        capture_only=True,
    )
    obj = MagicMock()
    obj.repository = None
    with pytest.raises(ValueError, match="rejects webull_live"):
        await prepare_cognitive_paper_session(
            app_settings=app, objective_service=obj, broker=live
        )


@pytest.mark.asyncio
async def test_live_runner_requires_confirmed_objective(tmp_path) -> None:
    app = _live_app(tmp_path)
    env = _live_env()
    activation = create_live_activation(
        account_id_hash=hash_account_id("LIVE_ACCT_1"),
        objective_id=uuid4(),
        authorized_capital_usd=Decimal("5000"),
    )
    obj = MagicMock()
    obj.get_state = AsyncMock(
        return_value=MagicMock(status="pending_confirmation")
    )
    runner = LiveTradingRunner(
        app_settings=app,
        env=env,
        objective_service=obj,
        activation=activation,
        trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
        capture_only=True,
        db_path=tmp_path / "live.db",
    )
    with pytest.raises(LiveTradingError, match="confirmed objective"):
        await runner.start(start_cognitive_agent=False)


def test_production_preflight_performs_no_mutation(tmp_path) -> None:
    app = _live_app(tmp_path)
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    report = run_production_preflight(
        app_settings=app,
        env=_live_env(),
        trade_api=api,
        skip_network=False,
    )
    assert report.mutated is False
    assert any("mutation: none" in c for c in report.checks)
    assert report.ok is True
    assert api.placed == []


def test_live_health_exposes_unresolved_broker_state(tmp_path) -> None:
    from joker.broker.reconciliation import ReconciliationReport, ReconciliationFinding
    from datetime import datetime, timezone

    app = _live_app(tmp_path)
    activation = create_live_activation(
        account_id_hash=hash_account_id("LIVE_ACCT_1"),
        objective_id=uuid4(),
        authorized_capital_usd=Decimal("5000"),
    )
    obj = MagicMock()
    obj.get_state = AsyncMock(return_value=MagicMock(status="active"))
    runner = LiveTradingRunner(
        app_settings=app,
        env=_live_env(),
        objective_service=obj,
        activation=activation,
        trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
        capture_only=True,
        db_path=tmp_path / "live.db",
    )
    runner.last_reconciliation = ReconciliationReport(
        captured_at=datetime.now(timezone.utc),
        account_id_hash=activation.account_id_hash,
        findings=(
            ReconciliationFinding(
                kind="submission_unknown",
                severity="critical",
                client_order_id="x",
            ),
        ),
        degraded=True,
        entries_blocked=True,
        unknown_submissions=1,
    )
    runner._degraded_reasons = ["submission_unknown"]
    health = runner.health()
    assert health.unknown_submissions == 1
    assert health.reconciliation_clean is False
    assert health.entries_permitted is False
