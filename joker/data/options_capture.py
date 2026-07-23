"""Capture normalized Webull options snapshots to JSONL."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from joker.config.settings import EnvSettings
from joker.config.validation import redact_secrets
from joker.data.webull_api import HttpWebullMarketApi, WebullApiError
from joker.data.webull_errors import OptionEndpointUnverified
from joker.data.webull_options_provider import WebullOptionsDataProvider, create_webull_options_provider
from joker.compliance.opra_sanitizer import capture_field_summary, snapshot_to_safe_metadata


def resolve_expiration(expiration: str, provider: WebullOptionsDataProvider) -> date:
    if expiration.lower() in ("today", "0dte", "same-day"):
        return provider.market_today()
    return date.fromisoformat(expiration)


def capture_options_snapshot(
    env: EnvSettings,
    *,
    symbol: str = "SPY",
    expiration: str = "today",
    captures_dir: Path | None = None,
    options_provider: WebullOptionsDataProvider | None = None,
    stock_api: object | None = None,
) -> tuple[Path, dict]:
    """Fetch ATM call/put snapshots and write safe JSONL capture."""
    provider = options_provider or create_webull_options_provider(env)
    stock = stock_api or HttpWebullMarketApi(env)
    exp = resolve_expiration(expiration, provider)

    try:
        spy_quote = stock.get_snapshot(symbol)
        underlying_price = spy_quote.price
    except WebullApiError as exc:
        raise WebullApiError(redact_secrets(str(exc), env=env)) from exc

    try:
        contracts = provider.discover_contracts(symbol, exp)
    except OptionEndpointUnverified:
        contracts = provider.discover_osi_candidates(underlying_price, exp)
    except Exception:
        contracts = []
    if not contracts:
        raise WebullApiError(f"No option contracts found for {symbol} exp {exp}")

    candidates = provider.select_atm_candidates(contracts, underlying_price, exp)
    snapshots: list[OptionSnapshot] = []
    for contract in (candidates.atm_call, candidates.atm_put):
        if contract and contract.contract_id:
            snapshots.append(provider.fetch_snapshot(contract))

    out_dir = captures_dir or Path("data/captures/options")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{symbol.lower()}_0dte_{exp.isoformat()}_{ts}.jsonl"

    summary = {
        "symbol": symbol,
        "expiration": exp.isoformat(),
        "source": "webull_opra",
        "is_real_webull_data": True,
        "data_classification": "NON_PRICE_DECISION_METADATA",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_count": len(snapshots),
        "verified": provider.verified,
        "underlying_price_available": underlying_price is not None,
    }

    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_type": "capture.meta", **summary}) + "\n")
        for snap in snapshots:
            safe = snapshot_to_safe_metadata(snap, underlying_price=underlying_price)
            safe["event_type"] = "option.snapshot.safe"
            handle.write(json.dumps(safe) + "\n")

    return path, summary
