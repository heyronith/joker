"""Production preflight: linked SPY 0DTE truth + real preview, never placement."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from joker.app.safety import SafetyMode
from joker.broker.webull_live import create_mock_live_trade_api
from joker.config.settings import AppSettings
from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.market.quality import (
    DataQualityCode,
    DataQualityFinding,
    DataQualityReport,
    DataQualitySeverity,
)
from joker.market.snapshots import MarketSnapshot, UnderlyingSnapshot
from joker.persistence.migrations import apply_task1_migrations
from joker.runtime.live_preflight import run_production_preflight
from joker.time.clock import SessionPhase
from tests.broker._live_helpers import live_env

ET = timezone.utc


def _live_app(tmp_path) -> AppSettings:
    return AppSettings(
        mode=SafetyMode.LIVE_GATED,
        live_trading_enabled=True,
        db_path=str(tmp_path / "live.db"),
        broker={"provider": "webull_live"},
    )


def _seed_market_truth(
    db_path,
    *,
    trading_day: date,
    underlying_symbol: str = "SPY",
    surface_id=None,
    link_surface: bool = True,
    persist_surface: bool = True,
    usable_for_execution: bool = True,
    findings: tuple[DataQualityFinding, ...] = (),
    contract_expiry: date | None = None,
    ask: str = "1.10",
    now: datetime | None = None,
) -> tuple[MarketSnapshot, OptionSurfaceSnapshot, DataQualityReport]:
    apply_task1_migrations(db_path)
    now = now or datetime.now(timezone.utc)
    surface_id = surface_id or uuid4()
    dq_id = uuid4()
    snap_id = uuid4()
    expiry = contract_expiry or trading_day
    contract = OptionContractSnapshot(
        contract_id=f"SPY{expiry.strftime('%y%m%d')}C00500000",
        symbol="SPY",
        expiry=expiry,
        strike=Decimal("500"),
        option_type="call",
        bid=Decimal("1.00"),
        ask=Decimal(ask),
        mid=Decimal("1.05"),
        quote_timestamp=now,
        quote_age_ms=0,
    )
    surface = OptionSurfaceSnapshot(
        surface_id=surface_id,
        exchange_time=now,
        trading_date=trading_day,
        underlying_symbol="SPY",
        underlying_price=Decimal("500"),
        contracts=(contract,),
        source="test",
    )
    quality = DataQualityReport(
        report_id=dq_id,
        snapshot_id=snap_id,
        severity=(
            DataQualitySeverity.OK
            if usable_for_execution and not findings
            else DataQualitySeverity.ERROR
        ),
        findings=findings,
        usable_for_reasoning=True,
        usable_for_execution=usable_for_execution,
    )
    snapshot = MarketSnapshot(
        snapshot_id=snap_id,
        exchange_time=now,
        trading_date=trading_day,
        underlying=UnderlyingSnapshot(
            symbol=underlying_symbol,
            exchange_time=now,
            last=Decimal("500"),
            bid=Decimal("499.9"),
            ask=Decimal("500.1"),
            source="test",
        ),
        option_surface_id=surface_id if link_surface else None,
        data_quality_id=dq_id,
    )
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO market_snapshots
            (snapshot_id, trading_date, exchange_time, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(snap_id),
            trading_day.isoformat(),
            now.isoformat(),
            snapshot.model_dump_json(),
            now.isoformat(),
        ),
    )
    if persist_surface:
        conn.execute(
            """
            INSERT INTO option_surfaces
                (surface_id, trading_date, exchange_time, underlying_symbol,
                 payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(surface_id),
                trading_day.isoformat(),
                now.isoformat(),
                surface.underlying_symbol,
                surface.model_dump_json(),
                now.isoformat(),
            ),
        )
    conn.execute(
        """
        INSERT INTO data_quality_reports
            (report_id, snapshot_id, session_id, severity,
             usable_for_reasoning, usable_for_execution, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(dq_id),
            str(snap_id),
            "preflight",
            quality.severity.value,
            1,
            1 if usable_for_execution else 0,
            quality.model_dump_json(),
            now.isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return snapshot, surface, quality


def _open_session_preflight(tmp_path, api, *, seed_kwargs=None):
    trading_day = date(2026, 7, 1)
    db = tmp_path / "live.db"
    _seed_market_truth(db, trading_day=trading_day, **(seed_kwargs or {}))
    app = _live_app(tmp_path)
    env = live_env(WEBULL_MARKET_DATA_ENABLED=True, WEBULL_APP_KEY="k")
    with patch("joker.runtime.live_preflight.SystemExchangeClock") as clock_cls:
        clock = MagicMock()
        clock.session_phase.return_value = SessionPhase.REGULAR
        clock.trading_date.return_value = trading_day
        clock_cls.return_value = clock
        report = run_production_preflight(
            app_settings=app,
            env=env,
            trade_api=api,
            check_market_data=True,
        )
    return report


def test_preflight_rejects_nested_non_spy_snapshot(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    report = _open_session_preflight(
        tmp_path, api, seed_kwargs={"underlying_symbol": "QQQ"}
    )
    assert report.live_snapshot_ok is False
    assert report.operational_ready is False
    assert any("underlying.symbol" in c and "QQQ" in c for c in report.checks)
    assert api.placed == []
    assert api.previewed == []


def test_preflight_rejects_unlinked_surface(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    report = _open_session_preflight(
        tmp_path,
        api,
        seed_kwargs={"link_surface": True, "persist_surface": False},
    )
    assert report.current_0dte_surface_ok is False
    assert report.operational_ready is False
    assert any("linked option_surface_id not found" in c for c in report.checks)
    assert api.previewed == []


def test_preflight_rejects_execution_unusable_quality_report(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    report = _open_session_preflight(
        tmp_path, api, seed_kwargs={"usable_for_execution": False}
    )
    assert report.current_0dte_surface_ok is False
    assert report.production_preview_ok is False
    assert any("usable_for_execution" in c for c in report.checks)
    assert api.previewed == []


def test_preflight_rejects_partial_surface(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    findings = (
        DataQualityFinding(
            code=DataQualityCode.PARTIAL_OPTION_SURFACE,
            severity=DataQualitySeverity.ERROR,
            message="partial surface",
            symbol="SPY",
        ),
    )
    report = _open_session_preflight(
        tmp_path,
        api,
        seed_kwargs={"usable_for_execution": True, "findings": findings},
    )
    assert report.current_0dte_surface_ok is False
    assert any("partial_option_surface" in c for c in report.checks)
    assert api.previewed == []


def test_preflight_calls_preview_order(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    report = _open_session_preflight(tmp_path, api)
    assert report.live_snapshot_ok is True
    assert report.current_0dte_surface_ok is True
    assert report.production_preview_ok is True
    assert len(api.previewed) == 1
    preview = api.previewed[0]
    assert preview["quantity"] == "1"
    assert preview.get("position_intent") == "BUY_TO_OPEN" or preview.get(
        "side"
    ) == "BUY"
    assert api.placed == []


def test_preflight_rejects_preview_failure(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.preview_reject = "INSUFFICIENT_BP"
    report = _open_session_preflight(tmp_path, api)
    assert report.production_preview_ok is False
    assert report.operational_ready is False
    assert any("production_preview: fail" in c for c in report.checks)
    assert len(api.previewed) == 1
    assert api.placed == []


def test_preflight_operational_ready_after_real_preview(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    report = _open_session_preflight(tmp_path, api)
    assert report.configuration_ok is True
    assert report.account_truth_ok is True
    assert report.database_ok is True
    assert report.market_session_open is True
    assert report.live_snapshot_ok is True
    assert report.current_0dte_surface_ok is True
    assert report.production_preview_ok is True
    assert report.operational_ready is True
    assert report.ok is True
    assert any("production_preview: ok accepted" in c for c in report.checks)
    assert len(api.previewed) == 1
    assert api.placed == []


def test_preflight_never_calls_place_order(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    original_place = api.place_order

    def _boom(*a, **k):
        raise AssertionError("place_order must never be called during preflight")

    api.place_order = _boom  # type: ignore[method-assign]
    report = _open_session_preflight(tmp_path, api)
    assert report.operational_ready is True
    assert len(api.previewed) == 1
    # Restore for cleanliness
    api.place_order = original_place  # type: ignore[method-assign]
