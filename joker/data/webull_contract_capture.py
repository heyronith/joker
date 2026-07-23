"""Capture redacted Webull response shapes for endpoint contract verification."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from joker.config.settings import EnvSettings
from joker.config.validation import redact_secrets
from joker.data.webull_api import HttpWebullMarketApi, WebullApiError
from joker.data.webull_endpoints import WEBULL_ENDPOINTS, get_endpoint
from joker.data.webull_errors import OptionEndpointUnverified
from joker.data.webull_http import WebullHttpClient
from joker.data.webull_option_symbols import build_atm_candidate_symbols
from joker.data.webull_options_api import HttpWebullOptionsMarketApi
from joker.compliance.opra_sanitizer import capture_field_summary
from joker.data.webull_response_capture import summarize_response, write_contract_capture


def _endpoint_status(name: str) -> str:
    ep = get_endpoint(name)
    return "verified" if ep.verified else "unverified"


def capture_webull_contract(
    env: EnvSettings,
    *,
    symbol: str = "SPY",
    include_options: bool = True,
    output_dir: Path | None = None,
    stock_api: object | None = None,
    options_api: object | None = None,
    http_client: WebullHttpClient | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Authenticate and capture redacted response-shape summaries."""
    summaries: list[dict[str, Any]] = []
    http = http_client or WebullHttpClient(env)
    stock = stock_api or HttpWebullMarketApi(env, http_client=http)
    options = options_api or HttpWebullOptionsMarketApi(env, http_client=http)

    auth = stock.authenticate() if hasattr(stock, "authenticate") else http.authenticate_legacy()
    summaries.append(
        {
            "endpoint": "auth",
            "status_code": 200 if auth.success else 401,
            "classification": None if auth.success else "auth failure",
            "presence": {},
            "verified": False,
        }
    )
    if not auth.success:
        path = write_contract_capture(summaries, output_dir=output_dir, env=env)
        return path, summaries

    # Stock snapshot
    try:
        quote = stock.get_snapshot(symbol)
        summaries.append(
            summarize_response(
                endpoint_name="stock_snapshot",
                status_code=200,
                payload={
                    "symbol": quote.symbol,
                    "price": quote.price,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "timestamp": quote.timestamp.isoformat(),
                    "delayed": quote.delayed,
                },
            )
        )
    except WebullApiError as exc:
        summaries.append(
            summarize_response(
                endpoint_name="stock_snapshot",
                status_code=exc.status_code or 500,
                payload={},
                error=redact_secrets(str(exc), env=env),
                classification=getattr(exc, "error_code", None),
            )
        )

    # Stock candles (may be unverified)
    ep_bars = get_endpoint("stock_bars")
    if ep_bars.verified:
        try:
            candles = stock.get_candles(symbol, "M1")
            sample = [
                {
                    "timestamp": c.timestamp.isoformat(),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles[:1]
            ]
            summaries.append(
                summarize_response(
                    endpoint_name="stock_bars",
                    status_code=200,
                    payload=sample,
                )
            )
        except (WebullApiError, OptionEndpointUnverified) as exc:
            summaries.append(
                summarize_response(
                    endpoint_name="stock_bars",
                    status_code=getattr(exc, "status_code", None) or 400,
                    payload={},
                    error=redact_secrets(str(exc), env=env),
                    classification="endpoint unverified"
                    if isinstance(exc, OptionEndpointUnverified)
                    else getattr(exc, "error_code", None),
                )
            )
    else:
        summaries.append(
            summarize_response(
                endpoint_name="stock_bars",
                status_code=0,
                payload={},
                error="endpoint unverified",
                classification="endpoint unverified",
            )
        )

    if include_options:
        summaries.append(
            {
                "endpoint": "option_chain",
                "status_code": 0,
                "endpoint_status": _endpoint_status("option_chain"),
                "classification": "endpoint unverified",
                "error": "No official chain endpoint — use OSI symbol construction",
            }
        )

        exp = date.today()
        underlying_price: float | None = None
        try:
            underlying_price = stock.get_snapshot(symbol).price
        except WebullApiError:
            pass

        if underlying_price is not None:
            candidates = build_atm_candidate_symbols(symbol, exp, underlying_price)
            atm_call = next(c for c in candidates if c.option_type == "call" and c.strike == round(underlying_price))
            atm_put = next(c for c in candidates if c.option_type == "put" and c.strike == round(underlying_price))
            for label, contract in (("option_snapshot_call", atm_call), ("option_snapshot_put", atm_put)):
                try:
                    snap = options.get_option_snapshot(contract)
                    summaries.append(
                        summarize_response(
                            endpoint_name=label,
                            status_code=200,
                            payload=capture_field_summary(snap.model_dump(mode="json")),
                        )
                    )
                except OptionEndpointUnverified as exc:
                    summaries.append(
                        summarize_response(
                            endpoint_name=label,
                            status_code=0,
                            payload={},
                            error=str(exc),
                            classification="endpoint unverified",
                        )
                    )
                except WebullApiError as exc:
                    summaries.append(
                        summarize_response(
                            endpoint_name=label,
                            status_code=exc.status_code or 500,
                            payload={},
                            error=redact_secrets(str(exc), env=env),
                            classification=exc.error_code,
                        )
                    )

    for name, ep in WEBULL_ENDPOINTS.items():
        if name.startswith("option_") or name.startswith("stock_"):
            summaries.append(
                {
                    "endpoint_registry": name,
                    "method": ep.method,
                    "path": ep.path,
                    "verified": ep.verified,
                    "rate_limit_per_minute": ep.rate_limit_per_minute,
                    "required_params": list(ep.required_params),
                }
            )

    path = write_contract_capture(summaries, output_dir=output_dir, env=env)
    return path, summaries
