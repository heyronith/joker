"""Production read-only preflight — never places orders."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from joker.app.safety import SafetyMode
from joker.broker.account_truth import hash_account_id, mask_account_id
from joker.broker.webull_live import (
    HttpWebullLiveTradeApi,
    validate_live_broker_startup,
)
from joker.config.settings import AppSettings, EnvSettings
from joker.config.validation import redact_secrets
from joker.time.calendar import MarketCalendar
from joker.time.clock import SessionPhase, SystemExchangeClock


@dataclass(frozen=True)
class LivePreflightReport:
    """Explicit readiness statuses — closed market is never operational_ready."""

    ok: bool  # alias of operational_ready (fail-closed production gate)
    configuration_ok: bool
    account_truth_ok: bool
    database_ok: bool
    market_session_open: bool
    live_snapshot_ok: bool
    current_0dte_surface_ok: bool
    production_preview_ok: bool
    operational_ready: bool
    account_id_hash: str | None
    account_id_masked: str | None
    checks: tuple[str, ...]
    captured_at: datetime
    mutated: bool = False  # always False for this service


def run_production_preflight(
    *,
    app_settings: AppSettings,
    env: EnvSettings,
    trade_api: Any | None = None,
    skip_network: bool = False,
    check_market_data: bool = True,
) -> LivePreflightReport:
    """Authenticate and read account/market health without placement."""
    checks: list[str] = []
    captured = datetime.now(timezone.utc)

    configuration_ok = True
    account_truth_ok = False
    database_ok = False
    market_session_open = False
    live_snapshot_ok = False
    current_0dte_surface_ok = False
    production_preview_ok = False

    if app_settings.mode is not SafetyMode.LIVE_GATED:
        checks.append("configuration: fail — require LIVE_GATED")
        configuration_ok = False
    else:
        checks.append("configuration: ok mode=LIVE_GATED")
    if not app_settings.live_trading_enabled:
        checks.append("configuration: fail — live_trading_enabled=false")
        configuration_ok = False
    else:
        checks.append("configuration: ok live_trading_enabled")
    if not env.webull_live_trading_enabled:
        checks.append("configuration: fail — WEBULL_LIVE_TRADING_ENABLED=false")
        configuration_ok = False
    else:
        checks.append("configuration: ok WEBULL_LIVE_TRADING_ENABLED")

    creds = env.live_credentials_env()
    missing = creds.missing_fields()
    if missing:
        checks.append("configuration: fail — credentials missing: " + ", ".join(missing))
        configuration_ok = False
        return _report(
            checks=checks,
            captured=captured,
            configuration_ok=False,
            account_truth_ok=False,
            database_ok=False,
            market_session_open=False,
            live_snapshot_ok=False,
            current_0dte_surface_ok=False,
            production_preview_ok=False,
            account_id_hash=None,
            account_id_masked=None,
        )
    checks.append("configuration: ok credentials present (redacted)")
    if creds.api_env != "prod":
        checks.append(f"configuration: fail — api_env={creds.api_env!r} (require prod)")
        configuration_ok = False
        return _report(
            checks=checks,
            captured=captured,
            configuration_ok=False,
            account_truth_ok=False,
            database_ok=False,
            market_session_open=False,
            live_snapshot_ok=False,
            current_0dte_surface_ok=False,
            production_preview_ok=False,
            account_id_hash=hash_account_id(str(creds.account_id)),
            account_id_masked=mask_account_id(str(creds.account_id)),
        )
    checks.append("configuration: ok api_env=prod")

    # SQLite integrity against the configured production database (read-only).
    db_path = Path(app_settings.db_path)
    if not db_path.exists():
        checks.append(f"database: fail — configured database missing: {db_path}")
        database_ok = False
    else:
        try:
            uri = f"file:{db_path.resolve()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                fk = conn.execute("PRAGMA foreign_key_check").fetchall()
            finally:
                conn.close()
            if integrity != "ok" or fk:
                checks.append(
                    f"database: fail — integrity={integrity!r} fk_rows={len(fk)}"
                )
                database_ok = False
            else:
                checks.append(f"database: ok (read-only {db_path.name})")
                database_ok = True
        except Exception as exc:
            checks.append(f"database: fail — {type(exc).__name__}: {exc}")
            database_ok = False

    account_hash = hash_account_id(str(creds.account_id))
    masked = mask_account_id(str(creds.account_id))

    if skip_network:
        checks.append("account_truth: skipped (skip_network)")
        checks.append("mutation: none (read-only preflight)")
        return _report(
            checks=checks,
            captured=captured,
            configuration_ok=configuration_ok,
            account_truth_ok=False,
            database_ok=database_ok,
            market_session_open=False,
            live_snapshot_ok=False,
            current_0dte_surface_ok=False,
            production_preview_ok=False,
            account_id_hash=account_hash,
            account_id_masked=masked,
        )

    api = trade_api or HttpWebullLiveTradeApi(creds)
    try:
        validate_live_broker_startup(
            app_settings=app_settings, env=env, trade_api=api
        )
        checks.append("account_truth: ok identity exact match")
        account_truth_ok = True
    except Exception as exc:
        checks.append(
            "account_truth: fail — " + redact_secrets(str(exc), env=env)
        )
        account_truth_ok = False
        checks.append("mutation: none (read-only preflight)")
        return _report(
            checks=checks,
            captured=captured,
            configuration_ok=configuration_ok,
            account_truth_ok=False,
            database_ok=database_ok,
            market_session_open=False,
            live_snapshot_ok=False,
            current_0dte_surface_ok=False,
            production_preview_ok=False,
            account_id_hash=account_hash,
            account_id_masked=masked,
        )

    try:
        balance = api.get_balance(str(creds.account_id))
        checks.append(
            "account_truth: ok balance keys="
            + ",".join(sorted(str(k) for k in balance.keys())[:8])
        )
    except Exception as exc:
        checks.append(
            "account_truth: fail — balance " + redact_secrets(str(exc), env=env)
        )
        account_truth_ok = False

    try:
        positions = api.get_positions(str(creds.account_id))
        checks.append(f"account_truth: ok positions count={len(positions)}")
    except Exception as exc:
        checks.append(
            "account_truth: fail — positions " + redact_secrets(str(exc), env=env)
        )
        account_truth_ok = False

    try:
        orders = api.list_open_orders(str(creds.account_id))
        checks.append(f"account_truth: ok open_orders count={len(orders)}")
        production_preview_ok = True
        checks.append("production_preview: ok read-only open-order listing")
    except Exception as exc:
        checks.append(
            "account_truth: fail — open_orders " + redact_secrets(str(exc), env=env)
        )
        account_truth_ok = False
        production_preview_ok = False
        checks.append("production_preview: fail — open order listing unavailable")

    if check_market_data:
        market_session_open, live_snapshot_ok, current_0dte_surface_ok = (
            _evaluate_market_readiness(checks, env=env, app_settings=app_settings)
        )
    else:
        checks.append("market_session_open: skipped (check_market_data=false)")
        checks.append("live_snapshot: skipped")
        checks.append("current_0dte_surface: skipped")

    checks.append("mutation: none (read-only preflight)")
    return _report(
        checks=checks,
        captured=captured,
        configuration_ok=configuration_ok,
        account_truth_ok=account_truth_ok,
        database_ok=database_ok,
        market_session_open=market_session_open,
        live_snapshot_ok=live_snapshot_ok,
        current_0dte_surface_ok=current_0dte_surface_ok,
        production_preview_ok=production_preview_ok,
        account_id_hash=account_hash,
        account_id_masked=masked,
    )


def _report(
    *,
    checks: list[str],
    captured: datetime,
    configuration_ok: bool,
    account_truth_ok: bool,
    database_ok: bool,
    market_session_open: bool,
    live_snapshot_ok: bool,
    current_0dte_surface_ok: bool,
    production_preview_ok: bool,
    account_id_hash: str | None,
    account_id_masked: str | None,
) -> LivePreflightReport:
    operational_ready = (
        configuration_ok
        and account_truth_ok
        and database_ok
        and market_session_open
        and live_snapshot_ok
        and current_0dte_surface_ok
        and production_preview_ok
    )
    checks.append(
        "operational_ready: "
        + ("ok" if operational_ready else "fail — not ready for live market operation")
    )
    return LivePreflightReport(
        ok=operational_ready,
        configuration_ok=configuration_ok,
        account_truth_ok=account_truth_ok,
        database_ok=database_ok,
        market_session_open=market_session_open,
        live_snapshot_ok=live_snapshot_ok,
        current_0dte_surface_ok=current_0dte_surface_ok,
        production_preview_ok=production_preview_ok,
        operational_ready=operational_ready,
        account_id_hash=account_id_hash,
        account_id_masked=account_id_masked,
        checks=tuple(checks),
        captured_at=captured,
        mutated=False,
    )


def _evaluate_market_readiness(
    checks: list[str],
    *,
    env: EnvSettings,
    app_settings: AppSettings,
) -> tuple[bool, bool, bool]:
    """Return (market_session_open, live_snapshot_ok, current_0dte_surface_ok)."""
    creds_present = bool(env.webull_market_data_enabled and env.webull_app_key)
    if not creds_present:
        checks.append("market_session_open: fail — market-data credentials absent")
        checks.append("live_snapshot: fail — credentials absent")
        checks.append("current_0dte_surface: fail — credentials absent")
        return False, False, False
    checks.append("market-data credentials: present")

    clock = SystemExchangeClock(calendar=MarketCalendar())
    phase = clock.session_phase()
    trading_day = clock.trading_date()
    market_open = phase is SessionPhase.REGULAR
    if not market_open:
        checks.append(
            f"market_session_open: fail — session_phase={phase.value} "
            f"(exchange_date={trading_day.isoformat()})"
        )
        checks.append("live_snapshot: fail — market session not open")
        checks.append("current_0dte_surface: fail — market session not open")
        return False, False, False
    checks.append(
        f"market_session_open: ok regular (exchange_date={trading_day.isoformat()})"
    )

    db_path = Path(app_settings.db_path)
    if not db_path.exists():
        checks.append("live_snapshot: fail — no database")
        checks.append("current_0dte_surface: fail — no database")
        return True, False, False

    snap_ok = _verify_current_snapshot(
        checks, db_path=db_path, trading_day=trading_day, app_settings=app_settings
    )
    surface_ok = _verify_current_0dte_surface(
        checks, db_path=db_path, trading_day=trading_day, app_settings=app_settings
    )
    return True, snap_ok, surface_ok


def _verify_current_snapshot(
    checks: list[str],
    *,
    db_path: Path,
    trading_day: date,
    app_settings: AppSettings,
) -> bool:
    stale_limit = float(
        getattr(app_settings.data_quality, "underlying_stale_seconds", 5.0) or 5.0
    )
    # Allow a slightly wider window for preflight persistence lag.
    max_age = max(stale_limit, 60.0)
    try:
        uri = f"file:{db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "market_snapshots" not in tables:
                checks.append("live_snapshot: fail — market_snapshots table missing")
                return False
            row = conn.execute(
                """
                SELECT trading_date, exchange_time, payload_json
                FROM market_snapshots
                ORDER BY exchange_time DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        checks.append(f"live_snapshot: fail — {type(exc).__name__}: {exc}")
        return False

    if row is None:
        checks.append("live_snapshot: fail — no persisted snapshot")
        return False

    row_date, exchange_time_raw, payload_json = row
    if str(row_date) != trading_day.isoformat():
        checks.append(
            f"live_snapshot: fail — trading_date={row_date} "
            f"!= exchange_date={trading_day.isoformat()}"
        )
        return False
    try:
        exchange_time = datetime.fromisoformat(str(exchange_time_raw))
        if exchange_time.tzinfo is None:
            exchange_time = exchange_time.replace(tzinfo=timezone.utc)
    except Exception:
        checks.append("live_snapshot: fail — unparseable exchange_time")
        return False
    age = (datetime.now(timezone.utc) - exchange_time.astimezone(timezone.utc)).total_seconds()
    if age > max_age:
        checks.append(
            f"live_snapshot: fail — stale age_seconds={age:.1f} max={max_age}"
        )
        return False
    try:
        payload = json.loads(payload_json) if payload_json else {}
    except Exception:
        payload = {}
    symbol = str(payload.get("symbol") or payload.get("underlying_symbol") or "")
    if symbol and symbol.upper() != "SPY":
        checks.append(f"live_snapshot: fail — symbol={symbol!r} (require SPY)")
        return False
    checks.append(
        f"live_snapshot: ok trading_date={row_date} age_seconds={age:.1f}"
    )
    return True


def _verify_current_0dte_surface(
    checks: list[str],
    *,
    db_path: Path,
    trading_day: date,
    app_settings: AppSettings,
) -> bool:
    stale_limit = float(
        getattr(app_settings.data_quality, "option_stale_seconds", 10.0) or 10.0
    )
    max_age = max(stale_limit, 120.0)
    try:
        uri = f"file:{db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "option_surfaces" not in tables:
                checks.append(
                    "current_0dte_surface: fail — option_surfaces table missing"
                )
                return False
            row = conn.execute(
                """
                SELECT trading_date, exchange_time, underlying_symbol, payload_json
                FROM option_surfaces
                ORDER BY exchange_time DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            conn.close()
    except Exception as exc:
        checks.append(f"current_0dte_surface: fail — {type(exc).__name__}: {exc}")
        return False

    if row is None:
        checks.append("current_0dte_surface: fail — no persisted surface")
        return False

    row_date, exchange_time_raw, underlying, payload_json = row
    if str(row_date) != trading_day.isoformat():
        checks.append(
            f"current_0dte_surface: fail — trading_date={row_date} "
            f"!= exchange_date={trading_day.isoformat()}"
        )
        return False
    if str(underlying or "").upper() != "SPY":
        checks.append(
            f"current_0dte_surface: fail — underlying={underlying!r} (require SPY)"
        )
        return False
    try:
        exchange_time = datetime.fromisoformat(str(exchange_time_raw))
        if exchange_time.tzinfo is None:
            exchange_time = exchange_time.replace(tzinfo=timezone.utc)
    except Exception:
        checks.append("current_0dte_surface: fail — unparseable exchange_time")
        return False
    age = (datetime.now(timezone.utc) - exchange_time.astimezone(timezone.utc)).total_seconds()
    if age > max_age:
        checks.append(
            f"current_0dte_surface: fail — stale age_seconds={age:.1f} max={max_age}"
        )
        return False

    try:
        payload = json.loads(payload_json) if payload_json else {}
    except Exception:
        checks.append("current_0dte_surface: fail — unparseable payload")
        return False

    contracts = payload.get("contracts") or payload.get("quotes") or []
    if not contracts:
        checks.append("current_0dte_surface: fail — empty contract set")
        return False

    # Require at least one contract with today's expiration (0DTE).
    expiry_ok = False
    for c in contracts:
        if not isinstance(c, dict):
            continue
        exp = c.get("expiry") or c.get("expiration") or c.get("expiration_date")
        if exp is None:
            continue
        exp_s = str(exp)[:10]
        if exp_s == trading_day.isoformat():
            expiry_ok = True
            break
        # Nested contract object
        nested = c.get("contract") if isinstance(c.get("contract"), dict) else None
        if nested:
            exp2 = nested.get("expiry") or nested.get("expiration")
            if exp2 is not None and str(exp2)[:10] == trading_day.isoformat():
                expiry_ok = True
                break
    if not expiry_ok:
        # Also accept surfaces tagged is_0dte / trading_date match with contracts.
        if any(
            isinstance(c, dict) and (c.get("is_0dte") or (c.get("contract") or {}).get("is_0dte"))
            for c in contracts
        ):
            expiry_ok = True
    if not expiry_ok:
        checks.append(
            "current_0dte_surface: fail — no SPY 0DTE contracts for exchange date"
        )
        return False

    checks.append(
        f"current_0dte_surface: ok contracts={len(contracts)} "
        f"age_seconds={age:.1f}"
    )
    return True
