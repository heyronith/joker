"""Normalize and validate Webull option snapshot responses."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from joker.compliance.data_classification import DataClassification, SOURCE_SYNTHETIC_OPTION
from joker.schemas.options_data import (
    OptionContractMetadata,
    OptionDataQualityWarning,
    OptionFieldAvailability,
    OptionSnapshot,
)
from joker.schemas.replay import OptionQuoteEvent


class OptionQuoteValidationError(Exception):
    """Option quote failed validation for tradable use."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _parse_ts(raw: Any) -> datetime:
    if isinstance(raw, (int, float)):
        ts = float(raw)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(raw, str):
        text = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    raise OptionQuoteValidationError("MISSING_TIMESTAMP", "Unknown timestamp format")


def normalize_webull_option_snapshot(
    contract: OptionContractMetadata,
    data: dict[str, Any],
) -> OptionSnapshot:
    """Normalize raw Webull option quote into OptionSnapshot with field tracking."""
    avail = OptionFieldAvailability()
    bid = data.get("bid")
    ask = data.get("ask")
    last = data.get("last") or data.get("latestPrice")
    volume = data.get("volume")
    oi = data.get("openInterest") or data.get("open_interest")
    iv = data.get("impliedVolatility") or data.get("imp_vol") or data.get("iv")
    delta = data.get("delta")
    gamma = data.get("gamma")
    theta = data.get("theta")
    vega = data.get("vega")
    ts_raw = (
        data.get("quote_time")
        or data.get("timestamp")
        or data.get("quoteTime")
        or data.get("tradeTime")
    )
    delayed = data.get("delayed", data.get("isDelayed"))

    if bid is not None:
        avail.bid = True
        bid = float(bid)
    if ask is not None:
        avail.ask = True
        ask = float(ask)
    if last is not None:
        avail.last = True
        last = float(last)
    if volume is not None:
        avail.volume = True
        volume = int(volume)
    if oi is not None:
        avail.open_interest = True
        oi = int(oi)
    if iv is not None:
        avail.implied_volatility = True
        iv = float(iv)
    if delta is not None:
        avail.delta = True
        delta = float(delta)
    if gamma is not None:
        avail.gamma = True
        gamma = float(gamma)
    if theta is not None:
        avail.theta = True
        theta = float(theta)
    if vega is not None:
        avail.vega = True
        vega = float(vega)

    quote_ts: datetime | None = None
    if ts_raw is not None:
        avail.quote_timestamp = True
        quote_ts = _parse_ts(ts_raw)
    if delayed is not None:
        avail.delayed = True

    mid: float | None = None
    spread_pct: float | None = None
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
        avail.mid = True
        if mid > 0:
            spread_pct = ((ask - bid) / mid) * 100.0
            avail.spread_pct = True

    if contract.contract_id:
        avail.contract_id = True
    if contract.instrument_id or data.get("instrument_id"):
        avail.instrument_id = True

    return OptionSnapshot(
        contract=contract,
        bid=bid,
        ask=ask,
        mid=mid,
        last=last,
        spread_pct=spread_pct,
        volume=volume,
        open_interest=oi,
        implied_volatility=iv,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        quote_timestamp=quote_ts,
        delayed=bool(delayed) if delayed is not None else None,
        source="webull_opra",
        field_availability=avail,
        data_classification=DataClassification.RAW_OPRA.value,
        persist_allowed=False,
        openai_allowed=False,
        is_synthetic=False,
    )


def validate_tradable_snapshot(
    snapshot: OptionSnapshot,
    *,
    reference_time: datetime | None = None,
    max_spread_pct: float = 15.0,
    quote_max_age_seconds: int = 30,
    allow_delayed_quotes: bool = True,
    feed_max_silence_seconds: int = 60,
    delayed_quote_max_age_seconds: int = 900,
    received_at: datetime | None = None,
) -> list[OptionDataQualityWarning]:
    """Validate snapshot for tradable quote use. Raises on hard failures."""
    from joker.data.freshness import FreshnessConfig, evaluate_quote_freshness

    warnings: list[OptionDataQualityWarning] = []

    if snapshot.bid is None or snapshot.bid <= 0:
        raise OptionQuoteValidationError("MISSING_BID", "Bid required for tradable quote")
    if snapshot.ask is None or snapshot.ask <= 0:
        raise OptionQuoteValidationError("MISSING_ASK", "Ask required for tradable quote")
    if snapshot.quote_timestamp is None:
        raise OptionQuoteValidationError("MISSING_TIMESTAMP", "Quote timestamp required")
    if snapshot.mid is None or snapshot.mid <= 0:
        raise OptionQuoteValidationError("INVALID_MID", "Mid must be positive")

    if snapshot.delayed:
        warnings.append(
            OptionDataQualityWarning(code="DELAYED", message="Quote marked as delayed")
        )

    if snapshot.spread_pct is not None and snapshot.spread_pct > max_spread_pct:
        raise OptionQuoteValidationError(
            "WIDE_SPREAD",
            f"Spread {snapshot.spread_pct:.1f}% exceeds max {max_spread_pct}%",
        )

    if reference_time is not None:
        verdict = evaluate_quote_freshness(
            quote_timestamp=snapshot.quote_timestamp,
            reference_time=reference_time,
            delayed=bool(snapshot.delayed),
            received_at=received_at,
            config=FreshnessConfig(
                quote_max_age_seconds=quote_max_age_seconds,
                feed_max_silence_seconds=feed_max_silence_seconds,
                delayed_quote_max_age_seconds=delayed_quote_max_age_seconds,
                allow_delayed_quotes=allow_delayed_quotes,
            ),
        )
        if not verdict.ok:
            raise OptionQuoteValidationError(
                verdict.reason or "STALE_QUOTE",
                f"Quote freshness failed: {verdict.reason} "
                f"(exchange_age={verdict.exchange_age_seconds})",
            )

    return warnings


def snapshot_to_quote_event(snapshot: OptionSnapshot) -> OptionQuoteEvent:
    """Convert validated OptionSnapshot to replay-compatible OptionQuoteEvent."""
    if snapshot.bid is None or snapshot.ask is None or snapshot.mid is None:
        raise OptionQuoteValidationError("MISSING_BID_ASK", "Bid/ask required for event")
    if snapshot.quote_timestamp is None:
        raise OptionQuoteValidationError("MISSING_TIMESTAMP", "Timestamp required")
    contract = snapshot.contract
    if not contract.contract_id:
        raise OptionQuoteValidationError("MISSING_CONTRACT_ID", "Contract ID required")

    classification = (
        DataClassification.SYNTHETIC_DATA.value
        if snapshot.is_synthetic or snapshot.source in (
            "replay",
            "mock",
            "synthetic",
            "synthetic_option",
            "mock_option",
            "synthetic_replay",
        )
        else DataClassification.RAW_OPRA.value
    )
    is_synthetic = classification == DataClassification.SYNTHETIC_DATA.value

    source_label = SOURCE_SYNTHETIC_OPTION if is_synthetic else snapshot.source

    received_at = datetime.now(timezone.utc)
    return OptionQuoteEvent(
        timestamp=snapshot.quote_timestamp,
        symbol=contract.underlying_symbol,
        source=source_label,
        contract_id=contract.contract_id,
        expiration=contract.expiration,
        strike=contract.strike,
        option_type=contract.option_type,
        bid=snapshot.bid,
        ask=snapshot.ask,
        mid=snapshot.mid,
        spread_pct=snapshot.spread_pct or 0.0,
        volume=snapshot.volume,
        open_interest=snapshot.open_interest,
        quote_timestamp=snapshot.quote_timestamp,
        data_classification=classification,
        persist_allowed=is_synthetic,
        openai_allowed=is_synthetic,
        is_synthetic=is_synthetic,
        delayed=bool(snapshot.delayed) if snapshot.delayed is not None else False,
        received_at=received_at,
    )
