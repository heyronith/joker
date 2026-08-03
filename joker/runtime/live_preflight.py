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
from joker.persistence.migrations import apply_task1_migrations


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

    # SQLite integrity on ephemeral migrate.
    try:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="joker-live-preflight-") as tmp:
            db = Path(tmp) / "preflight.db"
            apply_task1_migrations(db)
            conn = sqlite3.connect(db)
            try:
                integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                fk = conn.execute("PRAGMA foreign_key_check").fetchall()
            finally:
                conn.close()
        if integrity != "ok" or fk:
            checks.append("sqlite: fail")
        else:
            checks.append("sqlite: ok")
    except Exception as exc:
        checks.append(f"sqlite: fail — {type(exc).__name__}")

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
        if env.webull_market_data_enabled and env.webull_app_key:
            checks.append("market-data credentials: present")
        else:
            checks.append(
                "market-data / SPY surface: unavailable or not configured "
                "(may be closed session)"
            )

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
