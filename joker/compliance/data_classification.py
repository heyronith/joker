"""Data classification and policy for OPRA / market-data governance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# Explicit source taxonomy (Phase 21A.1)
SOURCE_WEBULL_STOCK = "webull_stock"
SOURCE_WEBULL_OPRA = "webull_opra"
SOURCE_WEBULL_OPRA_SAFE = "webull_opra_safe"
SOURCE_SYNTHETIC_OPTION = "synthetic_option"
SOURCE_SYNTHETIC_STOCK = "synthetic_stock"
SOURCE_MOCK_OPTION = "mock_option"
SOURCE_MOCK_STOCK = "mock_stock"

_OPRA_SOURCES = frozenset(
    {
        SOURCE_WEBULL_OPRA,
        "webull_option",
        "opra",
        "us_option",
    }
)

_STOCK_SOURCES = frozenset(
    {
        SOURCE_WEBULL_STOCK,
        SOURCE_MOCK_STOCK,
        SOURCE_SYNTHETIC_STOCK,
    }
)

_SYNTHETIC_SOURCES = frozenset(
    {
        SOURCE_SYNTHETIC_OPTION,
        SOURCE_SYNTHETIC_STOCK,
        SOURCE_MOCK_OPTION,
        SOURCE_MOCK_STOCK,
        "synthetic_replay",
        "replay",
        "mock",
    }
)

_STOCK_EVENT_TYPES = frozenset({"spy_quote", "spy_candle", "market.quote", "market.candle"})
_OPTION_EVENT_TYPES = frozenset(
    {
        "option_quote",
        "option.snapshot",
        "option.snapshot.safe",
        "options.snapshots",
        "option_quote_safe",
    }
)


class DataClassification(str, Enum):
    RAW_OPRA = "RAW_OPRA"
    DERIVED_OPRA_PRICE = "DERIVED_OPRA_PRICE"
    NON_PRICE_DECISION_METADATA = "NON_PRICE_DECISION_METADATA"
    SYSTEM_METADATA = "SYSTEM_METADATA"
    SYNTHETIC_DATA = "SYNTHETIC_DATA"
    STOCK_MARKET_DATA = "STOCK_MARKET_DATA"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DataPolicy:
    persist_allowed: bool
    openai_allowed: bool
    display_allowed: bool
    memory_allowed: bool


_POLICIES: dict[DataClassification, DataPolicy] = {
    DataClassification.RAW_OPRA: DataPolicy(
        persist_allowed=False,
        openai_allowed=False,
        display_allowed=True,
        memory_allowed=True,
    ),
    DataClassification.DERIVED_OPRA_PRICE: DataPolicy(
        persist_allowed=False,
        openai_allowed=False,
        display_allowed=True,
        memory_allowed=True,
    ),
    DataClassification.NON_PRICE_DECISION_METADATA: DataPolicy(
        persist_allowed=True,
        openai_allowed=True,
        display_allowed=True,
        memory_allowed=True,
    ),
    DataClassification.SYSTEM_METADATA: DataPolicy(
        persist_allowed=True,
        openai_allowed=True,
        display_allowed=True,
        memory_allowed=True,
    ),
    DataClassification.SYNTHETIC_DATA: DataPolicy(
        persist_allowed=True,
        openai_allowed=True,
        display_allowed=True,
        memory_allowed=True,
    ),
    DataClassification.STOCK_MARKET_DATA: DataPolicy(
        persist_allowed=True,
        openai_allowed=True,
        display_allowed=True,
        memory_allowed=True,
    ),
    DataClassification.UNKNOWN: DataPolicy(
        persist_allowed=False,
        openai_allowed=False,
        display_allowed=False,
        memory_allowed=False,
    ),
}


def policy_for(classification: DataClassification) -> DataPolicy:
    return _POLICIES.get(classification, _POLICIES[DataClassification.UNKNOWN])


def _normalize_source(source: str | None) -> str:
    if not source:
        return ""
    return source.lower().replace("-", "_")


def is_opra_source(source: str | None) -> bool:
    """True only for explicit OPRA/option sources — not generic webull stock."""
    normalized = _normalize_source(source)
    if not normalized:
        return False
    if normalized in _OPRA_SOURCES:
        return True
    if normalized == "webull":
        return False
    if "synthetic" in normalized and "option" in normalized:
        return False
    if normalized.endswith("_option") and "synthetic" not in normalized and "mock" not in normalized:
        return True
    return False


def is_stock_source(source: str | None) -> bool:
    normalized = _normalize_source(source)
    if normalized in _STOCK_SOURCES:
        return True
    if normalized == "webull":
        return True
    return normalized.endswith("_stock")


def _has_option_markers(obj: dict[str, Any]) -> bool:
    if obj.get("option_type") and obj.get("strike") is not None:
        return True
    contract = obj.get("contract")
    if isinstance(contract, dict) and contract.get("option_type"):
        return True
    if obj.get("contract_id") and obj.get("option_type"):
        return True
    if any(k in obj for k in ("spread_pct", "greeks_available", "open_interest")):
        if obj.get("event_type") in _OPTION_EVENT_TYPES or obj.get("option_type"):
            return True
    return False


def _has_derived_option_prices(obj: dict[str, Any]) -> bool:
    return any(
        obj.get(k) is not None
        for k in ("bid", "ask", "mid", "spread_pct", "delta", "gamma", "theta", "vega")
    ) and _has_option_markers(obj)


def classify_market_event(obj: Any) -> DataClassification:
    """Classify a market event or payload using explicit labels and structure."""
    if not isinstance(obj, dict):
        return DataClassification.UNKNOWN

    explicit = obj.get("data_classification")
    if explicit:
        try:
            return DataClassification(str(explicit))
        except ValueError:
            return DataClassification.UNKNOWN

    if obj.get("is_synthetic") is True:
        return DataClassification.SYNTHETIC_DATA

    source = _normalize_source(str(obj.get("source", "")))
    event_type = str(obj.get("event_type", ""))

    if source in _SYNTHETIC_SOURCES or source.startswith("synthetic"):
        return DataClassification.SYNTHETIC_DATA
    if source in (SOURCE_MOCK_OPTION, SOURCE_MOCK_STOCK) or source.startswith("mock_"):
        return DataClassification.SYNTHETIC_DATA

    if source == SOURCE_WEBULL_OPRA_SAFE:
        return DataClassification.NON_PRICE_DECISION_METADATA

    if any(
        obj.get(k) in ("PASS", "FAIL")
        for k in ("spread_check", "freshness_check", "contract_quality", "max_premium_check")
    ):
        return DataClassification.NON_PRICE_DECISION_METADATA

    if event_type in _STOCK_EVENT_TYPES or (source in _STOCK_SOURCES and not _has_option_markers(obj)):
        return DataClassification.STOCK_MARKET_DATA

    if is_opra_source(source) or event_type in _OPTION_EVENT_TYPES:
        if _has_derived_option_prices(obj) or "contract" in obj:
            if obj.get("bid") is not None or obj.get("ask") is not None or obj.get("mid") is not None:
                return DataClassification.RAW_OPRA
            return DataClassification.DERIVED_OPRA_PRICE
        if is_opra_source(source):
            return DataClassification.RAW_OPRA

    if source == "webull" and _has_option_markers(obj) and _has_derived_option_prices(obj):
        return DataClassification.RAW_OPRA

    if source == "webull" and event_type in _STOCK_EVENT_TYPES:
        return DataClassification.STOCK_MARKET_DATA

    if event_type in _STOCK_EVENT_TYPES:
        return DataClassification.STOCK_MARKET_DATA

    return DataClassification.UNKNOWN


def classify_option_source(source: str | None, *, synthetic: bool = False) -> DataClassification:
    if synthetic:
        return DataClassification.SYNTHETIC_DATA
    normalized = _normalize_source(source)
    if normalized in _SYNTHETIC_SOURCES or "synthetic" in normalized:
        return DataClassification.SYNTHETIC_DATA
    if is_opra_source(source):
        return DataClassification.RAW_OPRA
    return DataClassification.UNKNOWN
