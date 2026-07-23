"""Webull SPY 0DTE options data diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone

from joker.config.settings import EnvSettings
from joker.config.validation import redact_secrets
from joker.data.webull_api import HttpWebullMarketApi, WebullApiError
from joker.data.webull_config import WebullMarketConfigError, validate_webull_market_env
from joker.data.webull_endpoints import endpoint_status_map, get_endpoint
from joker.data.webull_errors import OptionEndpointUnverified
from joker.data.webull_option_symbols import build_atm_candidate_symbols
from joker.data.webull_options_api import (
    HttpWebullOptionsMarketApi,
    WebullOptionsMarketApi,
)
from joker.data.webull_options_provider import WebullOptionsDataProvider
from joker.schemas.options_data import OptionDataDiagnosticReport


def _check_status(
    report: OptionDataDiagnosticReport,
    name: str,
    *,
    pass_label: str,
    fail_label: str,
    exc: Exception | None = None,
) -> None:
    ep = get_endpoint(name)
    base = "verified" if ep.verified else "unverified"
    if isinstance(exc, OptionEndpointUnverified):
        report.endpoint_status[name] = "unverified"
        report.checks.append(f"{name}: unverified — {exc}")
    elif exc is not None:
        kind = "fail"
        if isinstance(exc, WebullApiError):
            if exc.error_code == "ENDPOINT_MISMATCH":
                kind = "endpoint mismatch"
            elif exc.subscription_related:
                kind = "subscription failure"
            elif exc.rate_limited:
                kind = "rate limit"
        report.endpoint_status[name] = f"{base}/{kind}"
        report.checks.append(f"{name}: fail — {fail_label}")
    else:
        report.endpoint_status[name] = f"{base}/{pass_label}"
        report.checks.append(f"{name}: {pass_label}")


def run_options_diagnostics(
    env: EnvSettings,
    *,
    symbol: str = "SPY",
    stock_api: object | None = None,
    options_api: WebullOptionsMarketApi | None = None,
) -> OptionDataDiagnosticReport:
    now = datetime.now(timezone.utc)
    report = OptionDataDiagnosticReport(checked_at=now)
    report.endpoint_status = {
        k: v for k, v in endpoint_status_map().items()
        if k.startswith(("stock_", "option_"))
    }

    try:
        validate_webull_market_env(env)
        report.credentials_present = True
    except WebullMarketConfigError as exc:
        report.likely_issue = "missing credentials"
        report.checks.append(f"credentials: fail — {exc}")
        return report

    options_api = options_api or HttpWebullOptionsMarketApi(env)
    stock_api = stock_api or HttpWebullMarketApi(env)
    provider = WebullOptionsDataProvider(env=env, api=options_api)

    try:
        auth = options_api.authenticate()
        report.auth_pass = auth.success
        report.checks.append(f"auth: {'pass' if auth.success else 'fail'}")
        if not auth.success:
            report.likely_issue = "auth failure"
            return report
    except Exception as exc:
        report.checks.append(f"auth: fail — {redact_secrets(str(exc), env=env)}")
        report.likely_issue = "auth failure"
        return report

    underlying_price: float | None = None
    try:
        quote = stock_api.get_snapshot(symbol)
        underlying_price = quote.price
        report.spy_snapshot_pass = True
        _check_status(report, "stock_snapshot", pass_label="pass", fail_label="")
    except OptionEndpointUnverified as exc:
        report.checks.append(f"SPY snapshot: fail — endpoint unverified")
        report.likely_issue = "endpoint unverified"
    except WebullApiError as exc:
        safe = redact_secrets(str(exc), env=env)
        report.checks.append(f"SPY snapshot: fail — {safe}")
        _check_status(report, "stock_snapshot", pass_label="pass", fail_label=safe, exc=exc)
        if exc.subscription_related:
            report.likely_issue = "OpenAPI subscription required"
        elif exc.error_code == "ENDPOINT_MISMATCH":
            report.likely_issue = "endpoint mismatch"
    except Exception as exc:
        report.checks.append(f"SPY snapshot: fail — {redact_secrets(str(exc), env=env)}")

    expiration = provider.market_today()
    contracts: list = []
    discovery_via_api = False
    try:
        contracts = provider.discover_contracts(symbol, expiration)
        discovery_via_api = bool(contracts)
        report.contract_discovery_pass = discovery_via_api
        report.same_day_expiration_found = discovery_via_api
        _check_status(report, "option_chain", pass_label="pass", fail_label="no contracts")
        report.checks.append(
            f"contract discovery: {'pass' if contracts else 'fail'} — {len(contracts)} contracts"
        )
        if not contracts:
            report.likely_issue = report.likely_issue or "no 0DTE contracts found"
    except OptionEndpointUnverified as exc:
        report.contract_discovery_pass = False
        report.same_day_expiration_found = False
        _check_status(report, "option_chain", pass_label="pass", fail_label="", exc=exc)
        report.likely_issue = report.likely_issue or "endpoint unverified"
    except WebullApiError as exc:
        safe = redact_secrets(str(exc), env=env)
        report.checks.append(f"contract discovery: fail — {safe}")
        _check_status(report, "option_chain", pass_label="pass", fail_label=safe, exc=exc)
        if exc.subscription_related:
            report.likely_issue = "OpenAPI subscription required"
        elif exc.error_code == "ENDPOINT_MISMATCH":
            report.likely_issue = "endpoint mismatch"
    except Exception as exc:
        report.checks.append(f"contract discovery: fail — {redact_secrets(str(exc), env=env)}")

    call_snap = put_snap = None
    if underlying_price is not None:
        if not contracts:
            contracts = build_atm_candidate_symbols(symbol, expiration, underlying_price)
            report.same_day_expiration_found = True
            report.checks.append(
                f"OSI candidate symbols: pass — {len(contracts)} constructed (chain unverified)"
            )
        try:
            candidates = provider.select_atm_candidates(contracts, underlying_price, expiration)
            if candidates.atm_call and candidates.atm_call.contract_id:
                call_snap = provider.fetch_snapshot(candidates.atm_call)
                report.atm_call_snapshot_pass = call_snap is not None
            if candidates.atm_put and candidates.atm_put.contract_id:
                put_snap = provider.fetch_snapshot(candidates.atm_put)
                report.atm_put_snapshot_pass = put_snap is not None
            _check_status(
                report,
                "option_snapshot",
                pass_label="pass" if call_snap and put_snap else "partial",
                fail_label="snapshot failed",
            )
            report.checks.append(f"ATM call: {'pass' if call_snap else 'fail'}")
            report.checks.append(f"ATM put: {'pass' if put_snap else 'fail'}")
        except OptionEndpointUnverified as exc:
            _check_status(report, "option_snapshot", pass_label="pass", fail_label="", exc=exc)
            report.likely_issue = report.likely_issue or "endpoint unverified"
        except WebullApiError as exc:
            safe = redact_secrets(str(exc), env=env)
            report.checks.append(f"ATM snapshots: fail — {safe}")
            _check_status(report, "option_snapshot", pass_label="pass", fail_label=safe, exc=exc)
            if exc.subscription_related:
                report.likely_issue = "OpenAPI subscription required"
            elif exc.error_code == "ENDPOINT_MISMATCH":
                report.likely_issue = "endpoint mismatch"
            elif exc.rate_limited:
                report.likely_issue = "rate limit"

    snaps = [s for s in (call_snap, put_snap) if s is not None]
    if snaps:
        avail = snaps[0].field_availability
        report.bid_ask_available = all(s.bid is not None and s.ask is not None for s in snaps)
        report.volume_available = any(s.volume is not None for s in snaps)
        report.open_interest_available = any(s.open_interest is not None for s in snaps)
        report.iv_available = any(s.implied_volatility is not None for s in snaps)
        report.greeks_available = avail.delta or avail.gamma or avail.theta or avail.vega
        delayed = [s.delayed for s in snaps if s.delayed is not None]
        if delayed:
            report.delayed_status = "delayed" if any(delayed) else "real-time"

    bars_contract = contracts[0] if contracts else None
    try:
        if bars_contract:
            options_api.get_option_bars(bars_contract, "M1")
        report.historical_bars = "yes"
        _check_status(report, "option_bars", pass_label="pass", fail_label="")
    except OptionEndpointUnverified as exc:
        report.historical_bars = "unknown"
        _check_status(report, "option_bars", pass_label="pass", fail_label="", exc=exc)
    except WebullApiError as exc:
        report.historical_bars = "no"
        _check_status(report, "option_bars", pass_label="pass", fail_label=str(exc), exc=exc)
    except Exception:
        report.historical_bars = "no"

    try:
        if bars_contract:
            options_api.get_option_ticks(bars_contract)
        report.ticks = "yes"
        _check_status(report, "option_tick", pass_label="pass", fail_label="")
    except OptionEndpointUnverified as exc:
        report.ticks = "unknown"
        _check_status(report, "option_tick", pass_label="pass", fail_label="", exc=exc)
    except WebullApiError as exc:
        report.ticks = "no"
        _check_status(report, "option_tick", pass_label="pass", fail_label=str(exc), exc=exc)
    except Exception:
        report.ticks = "no"

    capability = provider.build_capability_report(
        call_snap,
        put_snap,
        contract_discovery_succeeded=report.contract_discovery_pass,
        auth_pass=report.auth_pass,
        same_day_expiration=report.same_day_expiration_found,
    )
    report.capability = capability
    provider.verified = capability.verified

    if capability.verified:
        report.likely_issue = None
    elif not report.likely_issue:
        report.likely_issue = "required option fields missing"

    return report
