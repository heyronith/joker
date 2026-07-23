"""Webull market-data subscription and permission diagnostics."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from datetime import datetime, timezone

from joker.config.settings import EnvSettings
from joker.config.validation import redact_secrets
from joker.data.webull_api import WebullApiError, WebullMarketApi, WebullQuote
from joker.data.webull_config import (
    WebullMarketConfigError,
    ensure_live_trading_disabled,
    validate_webull_market_env,
)


@dataclass
class DiagnosticCheck:
    name: str
    status: str  # pass | fail | skip
    detail: str = ""


@dataclass
class WebullDiagnosticReport:
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    credentials_present: bool = False
    sdk_available: bool = False
    checks: list[DiagnosticCheck] = field(default_factory=list)
    likely_issue: str | None = None
    quote_delayed: bool | None = None

    def to_lines(self) -> list[str]:
        lines = [
            f"Webull market-data diagnostics ({self.checked_at.isoformat()})",
            f"- credentials: {'present' if self.credentials_present else 'missing'}",
            f"- sdk/import: {'available' if self.sdk_available else 'not installed (using httpx REST)'}",
        ]
        for check in self.checks:
            lines.append(f"- {check.name}: {check.status}" + (f" — {check.detail}" if check.detail else ""))
        if self.quote_delayed is not None:
            lines.append(f"- quote latency: {'delayed' if self.quote_delayed else 'real-time'}")
        if self.likely_issue:
            lines.append(f"- likely issue: {self.likely_issue}")
        return lines


def _sdk_available() -> bool:
    return importlib.util.find_spec("webull") is not None


def run_webull_diagnostics(
    env: EnvSettings,
    api: WebullMarketApi | None = None,
) -> WebullDiagnosticReport:
    report = WebullDiagnosticReport(
        sdk_available=_sdk_available(),
    )
    try:
        ensure_live_trading_disabled(env)
        validate_webull_market_env(env)
        report.credentials_present = True
        report.checks.append(DiagnosticCheck("credentials", "pass"))
    except WebullMarketConfigError as exc:
        report.checks.append(
            DiagnosticCheck("credentials", "fail", redact_secrets(str(exc), env=env))
        )
        report.likely_issue = "Missing or invalid Webull market-data credentials"
        return report

    if api is None:
        from joker.data.webull_api import HttpWebullMarketApi

        api = HttpWebullMarketApi(env)

    try:
        auth = api.authenticate()
        if auth.success:
            report.checks.append(DiagnosticCheck("auth", "pass", auth.message))
        else:
            report.checks.append(DiagnosticCheck("auth", "fail", auth.message))
            report.likely_issue = "Authentication failed — verify app key/secret and region"
            return report
    except Exception as exc:
        safe = redact_secrets(str(exc), env=env)
        report.checks.append(DiagnosticCheck("auth", "fail", safe))
        report.likely_issue = "Authentication failed"
        return report

    try:
        quote = api.get_snapshot("SPY")
        report.checks.append(DiagnosticCheck("snapshot", "pass", f"SPY ${quote.price:.2f}"))
        report.quote_delayed = quote.delayed
    except WebullApiError as exc:
        safe = redact_secrets(str(exc), env=env)
        report.checks.append(DiagnosticCheck("snapshot", "fail", safe))
        if exc.subscription_related:
            report.likely_issue = "OpenAPI market-data subscription required"
        elif exc.rate_limited:
            report.likely_issue = "Rate limit exceeded — retry later"
        else:
            report.likely_issue = "Snapshot access failed"
    except Exception as exc:
        safe = redact_secrets(str(exc), env=env)
        report.checks.append(DiagnosticCheck("snapshot", "fail", safe))
        report.likely_issue = "Snapshot access failed"

    try:
        candles = api.get_candles("SPY", "1m")
        if candles:
            report.checks.append(DiagnosticCheck("candles", "pass", f"{len(candles)} bars"))
        else:
            report.checks.append(DiagnosticCheck("candles", "pass", "empty response"))
    except WebullApiError as exc:
        safe = redact_secrets(str(exc), env=env)
        report.checks.append(DiagnosticCheck("candles", "fail", safe))
        if exc.subscription_related and not report.likely_issue:
            report.likely_issue = "OpenAPI market-data subscription required"
    except Exception as exc:
        safe = redact_secrets(str(exc), env=env)
        report.checks.append(DiagnosticCheck("candles", "fail", safe))

    try:
        count = 0
        for _ in api.stream_quotes("SPY", duration_seconds=0.01, poll_interval_seconds=0.01):
            count += 1
            if count >= 1:
                break
        report.checks.append(DiagnosticCheck("streaming", "pass", f"{count} quote(s)"))
    except WebullApiError as exc:
        safe = redact_secrets(str(exc), env=env)
        report.checks.append(DiagnosticCheck("streaming", "fail", safe))
        if exc.subscription_related and not report.likely_issue:
            report.likely_issue = "OpenAPI market-data subscription required"
    except Exception as exc:
        safe = redact_secrets(str(exc), env=env)
        report.checks.append(DiagnosticCheck("streaming", "fail", safe))

    if not report.likely_issue:
        failed = [c for c in report.checks if c.status == "fail"]
        if failed:
            report.likely_issue = failed[0].detail or failed[0].name

    return report
