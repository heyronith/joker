"""Production read-only preflight — never places orders."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from joker.app.safety import SafetyMode
from joker.broker.account_truth import hash_account_id, mask_account_id
from joker.broker.webull_live import (
    HttpWebullLiveTradeApi,
    validate_live_broker_startup,
)
from joker.broker.webull_trade_api import (
    build_option_limit_order_payload,
    new_client_order_id,
)
from joker.config.settings import AppSettings, EnvSettings
from joker.config.validation import redact_secrets
from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.market.quality import DataQualityCode, DataQualityReport
from joker.market.snapshots import MarketSnapshot
from joker.schemas.domain import OptionContract, OrderIntent
from joker.time.calendar import MarketCalendar
from joker.time.clock import SessionPhase, SystemExchangeClock

_BLOCKING_SURFACE_CODES = frozenset(
    {
        DataQualityCode.PARTIAL_OPTION_SURFACE,
        DataQualityCode.OPTION_SURFACE_UNAVAILABLE,
        DataQualityCode.INSUFFICIENT_OPTION_SURFACE,
        DataQualityCode.REPORT_UNAVAILABLE,
    }
)


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


@dataclass(frozen=True)
class _VerifiedMarketTruth:
    snapshot: MarketSnapshot
    surface: OptionSurfaceSnapshot
    quality: DataQualityReport
    preview_contract: OptionContractSnapshot


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
        checks.append("production_preview: fail — skipped (skip_network)")
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
    except Exception as exc:
        checks.append(
            "account_truth: fail — open_orders " + redact_secrets(str(exc), env=env)
        )
        account_truth_ok = False

    verified: _VerifiedMarketTruth | None = None
    if check_market_data:
        market_session_open, live_snapshot_ok, current_0dte_surface_ok, verified = (
            _evaluate_market_readiness(checks, env=env, app_settings=app_settings)
        )
    else:
        checks.append("market_session_open: skipped (check_market_data=false)")
        checks.append("live_snapshot: skipped")
        checks.append("current_0dte_surface: skipped")
        checks.append("production_preview: fail — market checks skipped")

    if (
        check_market_data
        and account_truth_ok
        and live_snapshot_ok
        and current_0dte_surface_ok
        and verified is not None
    ):
        production_preview_ok = _run_production_preview(
            checks,
            api=api,
            account_id=str(creds.account_id),
            env=env,
            verified=verified,
        )
    elif check_market_data:
        checks.append(
            "production_preview: fail — verified current SPY 0DTE market truth required"
        )
        production_preview_ok = False

    # Explicit: no order placement performed.
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
) -> tuple[bool, bool, bool, _VerifiedMarketTruth | None]:
    """Return session/snapshot/surface flags and linked verified market truth."""
    creds_present = bool(env.webull_market_data_enabled and env.webull_app_key)
    if not creds_present:
        checks.append("market_session_open: fail — market-data credentials absent")
        checks.append("live_snapshot: fail — credentials absent")
        checks.append("current_0dte_surface: fail — credentials absent")
        return False, False, False, None
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
        return False, False, False, None
    checks.append(
        f"market_session_open: ok regular (exchange_date={trading_day.isoformat()})"
    )

    db_path = Path(app_settings.db_path)
    if not db_path.exists():
        checks.append("live_snapshot: fail — no database")
        checks.append("current_0dte_surface: fail — no database")
        return True, False, False, None

    verified = _load_verified_market_truth(
        checks,
        db_path=db_path,
        trading_day=trading_day,
        app_settings=app_settings,
    )
    if verified is None:
        return True, False, False, None
    return True, True, True, verified


def _load_verified_market_truth(
    checks: list[str],
    *,
    db_path: Path,
    trading_day: date,
    app_settings: AppSettings,
) -> _VerifiedMarketTruth | None:
    """Latest snapshot → linked surface → DQ report → preview contract."""
    stale_underlying = float(
        getattr(app_settings.data_quality, "underlying_stale_seconds", 5.0) or 5.0
    )
    max_snap_age = max(stale_underlying, 60.0)
    stale_option = float(
        getattr(app_settings.data_quality, "option_stale_seconds", 10.0) or 10.0
    )
    max_surface_age = max(stale_option, 120.0)

    try:
        uri = f"file:{db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    except Exception as exc:
        checks.append(f"live_snapshot: fail — {type(exc).__name__}: {exc}")
        checks.append("current_0dte_surface: fail — database unavailable")
        return None

    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "market_snapshots" not in tables:
            checks.append("live_snapshot: fail — market_snapshots table missing")
            checks.append("current_0dte_surface: fail — snapshot missing")
            return None
        row = conn.execute(
            """
            SELECT payload_json
            FROM market_snapshots
            ORDER BY exchange_time DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            checks.append("live_snapshot: fail — no persisted snapshot")
            checks.append("current_0dte_surface: fail — snapshot missing")
            return None
        try:
            snapshot = MarketSnapshot.model_validate_json(row[0])
        except Exception as exc:
            checks.append(
                f"live_snapshot: fail — invalid MarketSnapshot ({type(exc).__name__})"
            )
            checks.append("current_0dte_surface: fail — snapshot invalid")
            return None

        if snapshot.trading_date != trading_day:
            checks.append(
                f"live_snapshot: fail — trading_date={snapshot.trading_date.isoformat()} "
                f"!= exchange_date={trading_day.isoformat()}"
            )
            checks.append("current_0dte_surface: fail — snapshot trading date mismatch")
            return None
        age = (
            datetime.now(timezone.utc)
            - snapshot.exchange_time.astimezone(timezone.utc)
        ).total_seconds()
        if age > max_snap_age:
            checks.append(
                f"live_snapshot: fail — stale age_seconds={age:.1f} max={max_snap_age}"
            )
            checks.append("current_0dte_surface: fail — snapshot stale")
            return None
        symbol = str(snapshot.underlying.symbol or "").upper()
        if symbol != "SPY":
            checks.append(
                f"live_snapshot: fail — underlying.symbol={symbol!r} (require SPY)"
            )
            checks.append("current_0dte_surface: fail — non-SPY snapshot")
            return None
        if snapshot.option_surface_id is None:
            checks.append("live_snapshot: fail — option_surface_id missing")
            checks.append("current_0dte_surface: fail — snapshot has no linked surface")
            return None
        checks.append(
            f"live_snapshot: ok trading_date={snapshot.trading_date.isoformat()} "
            f"age_seconds={age:.1f} snapshot_id={snapshot.snapshot_id}"
        )

        if "option_surfaces" not in tables:
            checks.append("current_0dte_surface: fail — option_surfaces table missing")
            return None
        surface_row = conn.execute(
            """
            SELECT payload_json
            FROM option_surfaces
            WHERE surface_id = ?
            """,
            (str(snapshot.option_surface_id),),
        ).fetchone()
        if surface_row is None:
            checks.append(
                "current_0dte_surface: fail — linked option_surface_id not found "
                f"({snapshot.option_surface_id})"
            )
            return None
        try:
            surface = OptionSurfaceSnapshot.model_validate_json(surface_row[0])
        except Exception as exc:
            checks.append(
                "current_0dte_surface: fail — invalid OptionSurfaceSnapshot "
                f"({type(exc).__name__})"
            )
            return None
        if surface.surface_id != snapshot.option_surface_id:
            checks.append(
                "current_0dte_surface: fail — surface_id mismatch vs snapshot link"
            )
            return None
        if surface.trading_date != trading_day:
            checks.append(
                "current_0dte_surface: fail — trading_date="
                f"{surface.trading_date.isoformat()} "
                f"!= exchange_date={trading_day.isoformat()}"
            )
            return None
        if str(surface.underlying_symbol or "").upper() != "SPY":
            checks.append(
                "current_0dte_surface: fail — underlying_symbol="
                f"{surface.underlying_symbol!r} (require SPY)"
            )
            return None
        surface_age = (
            datetime.now(timezone.utc)
            - surface.exchange_time.astimezone(timezone.utc)
        ).total_seconds()
        if surface_age > max_surface_age:
            checks.append(
                "current_0dte_surface: fail — stale "
                f"age_seconds={surface_age:.1f} max={max_surface_age}"
            )
            return None
        if not surface.contracts:
            checks.append("current_0dte_surface: fail — empty contract set")
            return None

        if "data_quality_reports" not in tables:
            checks.append(
                "current_0dte_surface: fail — data_quality_reports table missing"
            )
            return None
        dq_row = conn.execute(
            "SELECT payload FROM data_quality_reports WHERE report_id = ?",
            (str(snapshot.data_quality_id),),
        ).fetchone()
        if dq_row is None:
            checks.append(
                "current_0dte_surface: fail — linked data_quality_id not found "
                f"({snapshot.data_quality_id})"
            )
            return None
        try:
            quality = DataQualityReport.model_validate_json(dq_row[0])
        except Exception as exc:
            checks.append(
                "current_0dte_surface: fail — invalid DataQualityReport "
                f"({type(exc).__name__})"
            )
            return None
        if quality.report_id != snapshot.data_quality_id:
            checks.append(
                "current_0dte_surface: fail — data_quality report_id mismatch"
            )
            return None
        if not quality.usable_for_execution:
            checks.append(
                "current_0dte_surface: fail — data quality not usable_for_execution"
            )
            return None
        blocking = [
            f.code.value
            for f in quality.findings
            if f.code in _BLOCKING_SURFACE_CODES
        ]
        if blocking:
            checks.append(
                "current_0dte_surface: fail — blocking surface findings: "
                + ",".join(sorted(set(blocking)))
            )
            return None

        preview_contract = _choose_preview_contract(
            surface, trading_day=trading_day
        )
        if preview_contract is None:
            checks.append(
                "current_0dte_surface: fail — no SPY 0DTE contract with usable ask"
            )
            return None
        checks.append(
            "current_0dte_surface: ok "
            f"surface_id={surface.surface_id} "
            f"contracts={len(surface.contracts)} "
            f"preview_contract={preview_contract.contract_id}"
        )
        return _VerifiedMarketTruth(
            snapshot=snapshot,
            surface=surface,
            quality=quality,
            preview_contract=preview_contract,
        )
    finally:
        conn.close()


def _choose_preview_contract(
    surface: OptionSurfaceSnapshot,
    *,
    trading_day: date,
) -> OptionContractSnapshot | None:
    """Pick one verified 0DTE contract with a positive ask for the preview."""
    underlying = surface.underlying_price
    candidates = [
        c
        for c in surface.contracts
        if c.expiry == trading_day and c.ask is not None and Decimal(c.ask) > 0
    ]
    if not candidates:
        return None
    if underlying is None:
        return candidates[0]

    def _distance(c: OptionContractSnapshot) -> Decimal:
        return abs(Decimal(c.strike) - Decimal(underlying))

    return min(candidates, key=_distance)


def _run_production_preview(
    checks: list[str],
    *,
    api: Any,
    account_id: str,
    env: EnvSettings,
    verified: _VerifiedMarketTruth,
) -> bool:
    """Call Webull preview_order only — never place_order."""
    contract_snap = verified.preview_contract
    limit = float(contract_snap.ask) if contract_snap.ask is not None else None
    if limit is None or limit <= 0:
        checks.append("production_preview: fail — preview contract ask unavailable")
        return False
    intent = OrderIntent(
        intent_id=new_client_order_id(),
        candidate_id="production_preflight",
        contract=OptionContract(
            symbol="SPY",
            expiration=contract_snap.expiry,
            strike=float(contract_snap.strike),
            option_type=contract_snap.option_type,
            is_0dte=True,
        ),
        side="buy",
        order_type="limit",
        quantity=1,
        limit_price=limit,
        position_intent="BUY_TO_OPEN",
    )
    try:
        payload = build_option_limit_order_payload(
            intent, client_order_id=intent.intent_id, account_id=account_id
        )
    except Exception as exc:
        checks.append(
            "production_preview: fail — payload "
            + redact_secrets(str(exc), env=env)
        )
        return False

    try:
        raw = api.preview_order(account_id, [payload])
    except Exception as exc:
        checks.append(
            "production_preview: fail — "
            + redact_secrets(str(exc), env=env)
        )
        return False

    if not isinstance(raw, dict):
        checks.append("production_preview: fail — unexpected preview response shape")
        return False
    if not _preview_accepted(raw):
        code = raw.get("reject_code") or raw.get("code") or "rejected"
        msg = raw.get("reject_message") or raw.get("message") or ""
        checks.append(
            "production_preview: fail — preview not accepted "
            f"code={code!r} message={redact_secrets(str(msg), env=env)[:120]!r}"
        )
        return False

    checks.append(
        "production_preview: ok accepted "
        f"contract={contract_snap.contract_id} qty=1 BUY_TO_OPEN "
        f"limit={limit:.2f} (no placement)"
    )
    return True


def _preview_accepted(raw: dict[str, Any]) -> bool:
    accepted = bool(raw.get("accepted", True))
    reject_code = str(raw.get("reject_code") or raw.get("code") or "") or None
    if reject_code and str(reject_code).upper() not in {"0", "OK", "SUCCESS"}:
        accepted = False
    if raw.get("accepted") is False:
        accepted = False
    return accepted
