"""Full Webull options verification report generation."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from joker.config.settings import EnvSettings
from joker.data.options_diagnostics import run_options_diagnostics
from joker.data.webull_capability import WebullOptionsCapability, save_capability
from joker.schemas.options_data import OptionDataDiagnosticReport


def _next_action(report: OptionDataDiagnosticReport) -> str:
    if not report.credentials_present:
        return "Configure WEBULL_APP_KEY and WEBULL_APP_SECRET in .env"
    if not report.auth_pass:
        return "Fix authentication — verify token and signed request headers"
    if report.endpoint_status.get("option_chain") == "unverified":
        return "Use OSI symbol construction until option chain endpoint is verified"
    if report.likely_issue and "subscription" in report.likely_issue.lower():
        return "Subscribe to Webull OpenAPI market-data (Advanced Quotes)"
    if not report.bid_ask_available:
        return "Option snapshots missing bid/ask — check subscription or symbol format"
    if report.capability and report.capability.verified:
        return "Options data usable for shadow candidate building (still no broker execution)"
    return "Run joker data capture-webull-contract and inspect field-shape artifacts"


def build_capability_from_report(
    report: OptionDataDiagnosticReport,
    *,
    symbol: str,
    expiration: date | None = None,
) -> WebullOptionsCapability:
    cap = report.capability
    verified = bool(cap and cap.verified)
    return WebullOptionsCapability(
        checked_at=report.checked_at,
        symbol=symbol,
        auth_pass=report.auth_pass,
        contract_discovery_verified=report.endpoint_status.get("option_chain") == "verified",
        contract_discovery_succeeded=report.contract_discovery_pass,
        snapshot_verified=report.endpoint_status.get("option_snapshot") == "verified",
        snapshot_succeeded=report.atm_call_snapshot_pass and report.atm_put_snapshot_pass,
        bid_ask_available=report.bid_ask_available,
        timestamp_available=bool(cap and cap.verified),
        same_day_expiration_found=report.same_day_expiration_found,
        volume_available=report.volume_available,
        open_interest_available=report.open_interest_available,
        iv_available=report.iv_available,
        greeks_available=report.greeks_available,
        delayed_status=report.delayed_status,
        usable_for_shadow=verified,
        usable_for_replay_capture=verified,
        likely_issue=report.likely_issue,
        expiration_tested=expiration,
        endpoint_status=dict(report.endpoint_status),
        missing_required_fields=_missing_required(report),
        optional_missing_fields=list(cap.unavailable_fields) if cap else [],
    )


def _missing_required(report: OptionDataDiagnosticReport) -> list[str]:
    missing: list[str] = []
    if not report.bid_ask_available:
        missing.append("bid/ask")
    if report.atm_call_snapshot_pass is False or report.atm_put_snapshot_pass is False:
        missing.append("ATM call/put snapshot")
    if not report.same_day_expiration_found:
        missing.append("same-day expiration contracts")
    return missing


def generate_verification_report(
    env: EnvSettings,
    *,
    symbol: str = "SPY",
    reports_dir: Path | None = None,
    stock_api: object | None = None,
    options_api: object | None = None,
) -> tuple[Path, OptionDataDiagnosticReport, WebullOptionsCapability]:
    diag = run_options_diagnostics(
        env,
        symbol=symbol,
        stock_api=stock_api,
        options_api=options_api,
    )
    expiration = date.today()
    capability = build_capability_from_report(diag, symbol=symbol, expiration=expiration)
    save_capability(capability)

    out_dir = reports_dir or Path("reports/webull")
    out_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).date().isoformat()
    path = out_dir / f"options_verification_{day}.md"

    lines = [
        f"# Webull Options Verification — {day}",
        "",
        f"Symbol: **{symbol}**",
        f"Checked at: {diag.checked_at.isoformat()}",
        "",
        "## Credentials & Auth",
        f"- Credentials present: {'yes' if diag.credentials_present else 'no'}",
        f"- Auth passed: {'yes' if diag.auth_pass else 'no'}",
        "",
        "## Endpoint Status",
    ]
    for name, status in sorted(diag.endpoint_status.items()):
        lines.append(f"- {name}: **{status}**")
    lines.extend(
        [
            "",
            "## Market Data Results",
            f"- SPY stock snapshot: {'pass' if diag.spy_snapshot_pass else 'fail'}",
            f"- Same-day expiration found: {'yes' if diag.same_day_expiration_found else 'no'}",
            f"- ATM call snapshot: {'pass' if diag.atm_call_snapshot_pass else 'fail'}",
            f"- ATM put snapshot: {'pass' if diag.atm_put_snapshot_pass else 'fail'}",
            "",
            "## Required Fields",
            f"- Bid/ask: {'yes' if diag.bid_ask_available else 'no'}",
            f"- Quote timestamp: {'yes' if capability.timestamp_available else 'no'}",
            f"- Contract identity: {'yes' if diag.same_day_expiration_found else 'no'}",
            "",
            "## Optional Fields",
            f"- Volume: {'yes' if diag.volume_available else 'no'}",
            f"- Open interest: {'yes' if diag.open_interest_available else 'no'}",
            f"- IV: {'yes' if diag.iv_available else 'no'}",
            f"- Greeks: {'yes' if diag.greeks_available else 'no'}",
            "",
            "## Usability",
            f"- Usable for joker shadow mode: **{'yes' if capability.usable_for_shadow else 'no'}**",
            f"- Usable for paper simulation: **{'yes' if capability.usable_for_replay_capture else 'no'}**",
            f"- External options provider needed: **{'yes' if not capability.usable_for_shadow else 'no'}**",
            "",
            "## Issues",
            f"- Likely issue: {diag.likely_issue or 'none'}",
            f"- Missing required: {', '.join(capability.missing_required_fields) or 'none'}",
            "",
            "## Next Action",
            _next_action(diag),
            "",
            "> No secrets are included in this report.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, diag, capability
