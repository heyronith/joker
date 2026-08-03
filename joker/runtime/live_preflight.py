"""Production read-only preflight — never places orders."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
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


@dataclass(frozen=True)
class LivePreflightReport:
    ok: bool
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

    if app_settings.mode is not SafetyMode.LIVE_GATED:
        checks.append("mode: fail — require LIVE_GATED")
    else:
        checks.append("mode: ok LIVE_GATED")
    if not app_settings.live_trading_enabled:
        checks.append("app.live_trading_enabled: fail")
    else:
        checks.append("app.live_trading_enabled: ok")
    if not env.webull_live_trading_enabled:
        checks.append("WEBULL_LIVE_TRADING_ENABLED: fail")
    else:
        checks.append("WEBULL_LIVE_TRADING_ENABLED: ok")

    creds = env.live_credentials_env()
    missing = creds.missing_fields()
    if missing:
        checks.append("credentials: fail — " + ", ".join(missing))
        return LivePreflightReport(
            ok=False,
            account_id_hash=None,
            account_id_masked=None,
            checks=tuple(checks),
            captured_at=captured,
        )
    checks.append("credentials: ok (redacted)")
    if creds.api_env != "prod":
        checks.append(f"api_env: fail — {creds.api_env!r} (require prod)")
        return LivePreflightReport(
            ok=False,
            account_id_hash=None,
            account_id_masked=mask_account_id(str(creds.account_id)),
            checks=tuple(checks),
            captured_at=captured,
        )
    checks.append("api_env: ok prod")

    # SQLite integrity against the configured production database (read-only).
    db_path = Path(app_settings.db_path)
    if not db_path.exists():
        checks.append(f"sqlite: fail — configured database missing: {db_path}")
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
                    f"sqlite: fail — integrity={integrity!r} fk_rows={len(fk)}"
                )
            else:
                checks.append(f"sqlite: ok (read-only {db_path.name})")
        except Exception as exc:
            checks.append(f"sqlite: fail — {type(exc).__name__}: {exc}")

    account_hash = hash_account_id(str(creds.account_id))
    masked = mask_account_id(str(creds.account_id))

    if skip_network:
        checks.append("network: skipped")
        ok = all("fail" not in c for c in checks)
        return LivePreflightReport(
            ok=ok,
            account_id_hash=account_hash,
            account_id_masked=masked,
            checks=tuple(checks),
            captured_at=captured,
        )

    api = trade_api or HttpWebullLiveTradeApi(creds)
    try:
        validate_live_broker_startup(
            app_settings=app_settings, env=env, trade_api=api
        )
        checks.append("account identity: ok exact match")
    except Exception as exc:
        checks.append(
            "account identity: fail — " + redact_secrets(str(exc), env=env)
        )
        return LivePreflightReport(
            ok=False,
            account_id_hash=account_hash,
            account_id_masked=masked,
            checks=tuple(checks),
            captured_at=captured,
        )

    try:
        balance = api.get_balance(str(creds.account_id))
        checks.append(
            "balance truth: ok keys="
            + ",".join(sorted(str(k) for k in balance.keys())[:8])
        )
    except Exception as exc:
        checks.append("balance truth: fail — " + redact_secrets(str(exc), env=env))

    try:
        positions = api.get_positions(str(creds.account_id))
        checks.append(f"positions truth: ok count={len(positions)}")
    except Exception as exc:
        checks.append("positions truth: fail — " + redact_secrets(str(exc), env=env))

    try:
        orders = api.list_open_orders(str(creds.account_id))
        checks.append(f"open orders truth: ok count={len(orders)}")
    except Exception as exc:
        checks.append("open orders: fail — " + redact_secrets(str(exc), env=env))

    if check_market_data:
        _append_market_data_checks(checks, env=env, app_settings=app_settings)

    # Explicit: no order placement performed.
    checks.append("mutation: none (read-only preflight)")
    ok = all("fail" not in c for c in checks)
    return LivePreflightReport(
        ok=ok,
        account_id_hash=account_hash,
        account_id_masked=masked,
        checks=tuple(checks),
        captured_at=captured,
        mutated=False,
    )


def _append_market_data_checks(
    checks: list[str],
    *,
    env: EnvSettings,
    app_settings: AppSettings,
) -> None:
    """Distinguish credentials / market closed / live snapshot / 0DTE surface."""
    creds_present = bool(env.webull_market_data_enabled and env.webull_app_key)
    if not creds_present:
        checks.append("market-data credentials: absent")
        checks.append("market-data live snapshot: not verified")
        checks.append("market-data 0DTE surface: not verified")
        return
    checks.append("market-data credentials: present")

    # Session clock — closed market is not a credentials failure.
    try:
        from joker.time.clock import ExchangeClock

        clock = ExchangeClock()
        if hasattr(clock, "is_market_open") and not clock.is_market_open():
            checks.append("market-data session: market closed")
            checks.append("market-data live snapshot: not verified (market closed)")
            checks.append("market-data 0DTE surface: not verified (market closed)")
            return
        checks.append("market-data session: open or clock unavailable")
    except Exception:
        checks.append("market-data session: clock unavailable")

    # Prefer persisted snapshot / surface from the configured production DB.
    db_path = Path(app_settings.db_path)
    if not db_path.exists():
        checks.append("market-data live snapshot: not verified (no database)")
        checks.append("market-data 0DTE surface: not verified (no database)")
        return
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
            snap_ok = False
            surface_ok = False
            if "market_snapshots" in tables or "snapshots" in tables:
                table = "market_snapshots" if "market_snapshots" in tables else "snapshots"
                row = conn.execute(
                    f"SELECT 1 FROM {table} ORDER BY 1 DESC LIMIT 1"
                ).fetchone()
                snap_ok = row is not None
            if "option_surfaces" in tables or "option_surface" in tables:
                table = (
                    "option_surfaces"
                    if "option_surfaces" in tables
                    else "option_surface"
                )
                row = conn.execute(
                    f"SELECT 1 FROM {table} ORDER BY 1 DESC LIMIT 1"
                ).fetchone()
                surface_ok = row is not None
        finally:
            conn.close()
        checks.append(
            "market-data live snapshot: "
            + ("verified (persisted)" if snap_ok else "not verified")
        )
        checks.append(
            "market-data 0DTE surface: "
            + ("verified (persisted)" if surface_ok else "not verified")
        )
    except Exception as exc:
        checks.append(f"market-data live snapshot: not verified ({type(exc).__name__})")
        checks.append("market-data 0DTE surface: not verified")
