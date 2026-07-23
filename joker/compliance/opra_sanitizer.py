"""OPRA-safe sanitization for persistence, reports, and OpenAI inputs."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from joker.compliance.data_classification import (
    DataClassification,
    SOURCE_WEBULL_OPRA_SAFE,
    classify_market_event,
    is_opra_source,
)

logger = logging.getLogger(__name__)

OPRA_VALUE_FIELDS = frozenset(
    {
        "bid",
        "ask",
        "mid",
        "last",
        "spread_pct",
        "volume",
        "open_interest",
        "implied_volatility",
        "imp_vol",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "quote_timestamp",
        "entry_limit_price",
        "entry_price",
        "exit_price",
        "simulated_entry",
        "simulated_exit",
        "simulated_pnl",
        "stop_price",
        "take_profit_price",
        "latest_option_mid",
        "call_bid",
        "call_ask",
        "call_mid",
        "call_spread_pct",
        "put_bid",
        "put_ask",
        "put_mid",
        "put_spread_pct",
        "option_quote_timestamp",
    }
)

OPRA_IDENTIFIER_FIELDS = frozenset(
    {
        "contract_id",
        "instrument_id",
        "osi_symbol",
        "call_id",
        "put_id",
        "selected_call_contract",
        "selected_put_contract",
    }
)

OPTION_EVENT_TYPES = frozenset(
    {
        "option.snapshot",
        "option_quote",
        "options.snapshots",
        "exit.shadow",
        "option_quote_safe",
        "option.snapshot.safe",
    }
)

_REDACT_PATTERNS = [
    (re.compile(r'"bid"\s*:\s*[-+]?\d+(?:\.\d+)?', re.I), '"bid": "[REDACTED_OPRA]"'),
    (re.compile(r'"ask"\s*:\s*[-+]?\d+(?:\.\d+)?', re.I), '"ask": "[REDACTED_OPRA]"'),
    (re.compile(r'"mid"\s*:\s*[-+]?\d+(?:\.\d+)?', re.I), '"mid": "[REDACTED_OPRA]"'),
    (re.compile(r'"delta"\s*:\s*[-+]?\d+(?:\.\d+)?', re.I), '"delta": "[REDACTED_OPRA]"'),
]


class RawOpraViolationError(ValueError):
    """Raised when raw OPRA data would be persisted or sent to OpenAI."""


def _is_synthetic_obj(obj: dict[str, Any]) -> bool:
    if obj.get("is_synthetic") is True:
        return True
    classification = classify_market_event(obj)
    if classification == DataClassification.SYNTHETIC_DATA:
        return True
    source = str(obj.get("source", "")).lower()
    if any(
        token in source
        for token in (
            "synthetic",
            "mock_option",
            "mock_stock",
            "synthetic_option",
            "synthetic_stock",
            "synthetic_replay",
        )
    ):
        return True
    if source in ("replay", "mock") and classification != DataClassification.RAW_OPRA:
        return True
    return False


def _in_opra_context(obj: dict[str, Any], parent_opra: bool) -> bool:
    if parent_opra:
        return True
    if _is_synthetic_obj(obj):
        return False
    classification = classify_market_event(obj)
    if classification in (
        DataClassification.RAW_OPRA,
        DataClassification.DERIVED_OPRA_PRICE,
    ):
        return True
    if classification == DataClassification.STOCK_MARKET_DATA:
        return False
    if classification == DataClassification.NON_PRICE_DECISION_METADATA:
        return False
    source = str(obj.get("source", ""))
    if is_opra_source(source):
        return True
    event_type = str(obj.get("event_type", ""))
    if event_type in OPTION_EVENT_TYPES:
        return True
    if (
        obj.get("option_type")
        and obj.get("strike") is not None
        and any(obj.get(k) is not None for k in ("bid", "ask", "mid", "spread_pct"))
    ):
        return is_opra_source(source) or event_type in OPTION_EVENT_TYPES
    if "contract" in obj and isinstance(obj["contract"], dict):
        contract = obj["contract"]
        if contract.get("option_type") and any(k in obj for k in ("bid", "ask", "mid")):
            return is_opra_source(source) or is_opra_source(str(contract.get("source", "")))
    if "quote" in obj and isinstance(obj["quote"], dict) and obj["quote"].get("bid") is not None:
        return parent_opra
    return False


def _contains_raw_opra_values(obj: Any, *, in_opra_context: bool = False) -> bool:
    if isinstance(obj, dict):
        if _is_synthetic_obj(obj):
            return False
        opra_ctx = _in_opra_context(obj, in_opra_context)
        if opra_ctx:
            for key, value in obj.items():
                if key in OPRA_VALUE_FIELDS and value is not None:
                    return True
        for value in obj.values():
            if _contains_raw_opra_values(value, in_opra_context=opra_ctx):
                return True
    elif isinstance(obj, list):
        return any(_contains_raw_opra_values(item, in_opra_context=in_opra_context) for item in obj)
    return False


def assert_no_raw_opra(obj: Any) -> None:
    if _contains_raw_opra_values(obj):
        raise RawOpraViolationError("Raw OPRA market data detected in payload")


def redact_opra_values(text: str) -> str:
    out = text
    for pattern, repl in _REDACT_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def _pass_fail(value: bool | None) -> str:
    return "PASS" if value else "FAIL"


def _contract_role(option_type: str | None) -> str | None:
    if option_type == "call":
        return "ATM_CALL"
    if option_type == "put":
        return "ATM_PUT"
    return None


def _moneyness_bucket(strike: float | None, underlying: float | None, option_type: str | None) -> str:
    if strike is None or underlying is None or not option_type:
        return "UNKNOWN"
    diff = abs(strike - underlying)
    if diff < 1.0:
        return "ATM"
    if option_type == "call":
        return "ITM" if strike < underlying else "OTM"
    return "ITM" if strike > underlying else "OTM"


def contract_safe_metadata(
    contract: dict[str, Any] | None,
    *,
    underlying_price: float | None = None,
    selected: bool = True,
) -> dict[str, Any]:
    if not contract:
        return {"contract_selected": False}
    option_type = contract.get("option_type")
    strike = contract.get("strike")
    role = _contract_role(str(option_type) if option_type else None)
    direction = f"long_{option_type}" if option_type in ("call", "put") else None
    exp = contract.get("expiration")
    expiration_type = "0DTE"
    if isinstance(exp, date):
        expiration_type = "0DTE" if exp == date.today() else "DATED"
    return {
        "contract_role": role,
        "selected_direction": direction,
        "expiration_type": expiration_type,
        "moneyness_bucket": _moneyness_bucket(
            float(strike) if strike is not None else None,
            underlying_price,
            str(option_type) if option_type else None,
        ),
        "contract_selected": selected,
    }


def snapshot_to_safe_metadata(snapshot: Any, *, underlying_price: float | None = None) -> dict[str, Any]:
    """Convert OptionSnapshot-like object to non-price decision metadata."""
    if hasattr(snapshot, "model_dump"):
        data = snapshot.model_dump(mode="json")
    elif isinstance(snapshot, dict):
        data = snapshot
    else:
        return {"contract_quality": "UNKNOWN"}

    avail = data.get("field_availability") or {}
    if hasattr(snapshot, "field_availability"):
        avail = snapshot.field_availability.model_dump()

    spread_ok = bool(data.get("bid")) and bool(data.get("ask"))
    if data.get("spread_pct") is not None and isinstance(data.get("spread_pct"), (int, float)):
        spread_ok = spread_ok and float(data["spread_pct"]) <= 15.0

    contract = data.get("contract") if isinstance(data.get("contract"), dict) else {}
    safe_contract = contract_safe_metadata(
        contract,
        underlying_price=underlying_price or data.get("underlying_price"),
        selected=True,
    )

    return {
        "data_classification": DataClassification.NON_PRICE_DECISION_METADATA.value,
        "source": SOURCE_WEBULL_OPRA_SAFE,
        **safe_contract,
        "spread_check": _pass_fail(spread_ok),
        "freshness_check": _pass_fail(bool(data.get("quote_timestamp"))),
        "bid_ask_available": bool(data.get("bid") is not None and data.get("ask") is not None),
        "greeks_available": any(avail.get(k) for k in ("delta", "gamma", "theta", "vega")),
        "iv_available": bool(avail.get("implied_volatility")),
        "open_interest_available": bool(avail.get("open_interest")),
        "volume_available": bool(avail.get("volume")),
        "contract_quality": _pass_fail(spread_ok and bool(data.get("quote_timestamp"))),
    }


def shadow_safe_metadata(
    *,
    setup_id: str | None = None,
    direction: str | None = None,
    spread_check: str = "PASS",
    freshness_check: str = "PASS",
    max_premium_check: str = "PASS",
    candidate_created: bool = False,
    would_trade_created: bool = False,
    risk_reason_codes: list[str] | None = None,
    shadow_result_label: str | None = None,
    exit_reason: str | None = None,
    risk_multiple_bucket: str | None = None,
    contract_role: str | None = None,
) -> dict[str, Any]:
    return {
        "data_classification": DataClassification.NON_PRICE_DECISION_METADATA.value,
        "setup_id": setup_id,
        "selected_direction": direction,
        "contract_role": contract_role,
        "expiration_type": "0DTE",
        "spread_check": spread_check,
        "freshness_check": freshness_check,
        "max_premium_check": max_premium_check,
        "candidate_created": candidate_created,
        "would_trade_created": would_trade_created,
        "risk_reason_code": risk_reason_codes or [],
        "shadow_result_label": shadow_result_label,
        "exit_reason": exit_reason,
        "risk_multiple_bucket": risk_multiple_bucket,
        "contract_selected": candidate_created,
    }


def exit_decision_safe_metadata(reason: str, *, shadow_result_label: str | None = None) -> dict[str, Any]:
    return shadow_safe_metadata(
        exit_reason=reason,
        shadow_result_label=shadow_result_label,
        would_trade_created=True,
    )


def _sanitize_node(obj: Any, *, in_opra_context: bool = False) -> Any:
    if isinstance(obj, dict):
        if _is_synthetic_obj(obj):
            return obj
        if classify_market_event(obj) == DataClassification.STOCK_MARKET_DATA:
            return obj
        opra_ctx = _in_opra_context(obj, in_opra_context)
        if opra_ctx and "contract" in obj and any(k in obj for k in ("bid", "ask", "mid")):
            return snapshot_to_safe_metadata(obj)
        if opra_ctx and obj.get("event_type") == "option_quote":
            return {
                "event_type": "option_quote_safe",
                "data_classification": DataClassification.NON_PRICE_DECISION_METADATA.value,
                "source": SOURCE_WEBULL_OPRA_SAFE,
                "contract_role": _contract_role(obj.get("option_type")),
                "selected_direction": f"long_{obj['option_type']}"
                if obj.get("option_type") in ("call", "put")
                else None,
                "expiration_type": "0DTE",
                "spread_check": _pass_fail(obj.get("spread_pct") is not None),
                "freshness_check": _pass_fail(bool(obj.get("quote_timestamp"))),
                "bid_ask_available": bool(obj.get("bid") and obj.get("ask")),
                "contract_selected": True,
            }
        cleaned: dict[str, Any] = {}
        for key, value in obj.items():
            if opra_ctx and key in OPRA_VALUE_FIELDS:
                continue
            if opra_ctx and key in OPRA_IDENTIFIER_FIELDS:
                continue
            cleaned[key] = _sanitize_node(value, in_opra_context=opra_ctx)
        if opra_ctx and not _is_synthetic_obj(obj):
            cleaned.setdefault(
                "data_classification",
                DataClassification.NON_PRICE_DECISION_METADATA.value,
            )
            cleaned["opra_sanitized"] = True
        return cleaned
    if isinstance(obj, list):
        return [_sanitize_node(item, in_opra_context=in_opra_context) for item in obj]
    return obj


def sanitize_for_persistence(obj: Any) -> Any:
    had_raw = _contains_raw_opra_values(obj)
    sanitized = _sanitize_node(obj)
    if had_raw:
        logger.warning("OPRA persistence guard: raw OPRA values stripped before write")
    assert_no_raw_opra(sanitized)
    return sanitized


def sanitize_for_openai(obj: Any) -> Any:
    return sanitize_for_persistence(obj)


def sanitize_for_report(obj: Any) -> Any:
    return sanitize_for_persistence(obj)


def capture_field_summary(payload: Any) -> dict[str, Any]:
    """Shape-only summary for contract capture — no raw OPRA values."""
    if isinstance(payload, dict):
        if any(k in payload for k in ("bid", "ask", "contract")):
            return snapshot_to_safe_metadata(payload)
        return {
            "top_level_keys": list(payload.keys())[:30],
            "field_types": {k: type(v).__name__ for k, v in list(payload.items())[:30]},
            "presence": {
                "bid": "bid" in payload,
                "ask": "ask" in payload,
                "timestamp": any(k in payload for k in ("quote_timestamp", "timestamp", "quote_time")),
                "volume": "volume" in payload,
                "open_interest": "open_interest" in payload,
                "iv": "imp_vol" in payload or "implied_volatility" in payload,
                "greeks": any(k in payload for k in ("delta", "gamma", "theta", "vega")),
            },
        }
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return capture_field_summary(payload[0])
    return {"top_level_type": type(payload).__name__}
