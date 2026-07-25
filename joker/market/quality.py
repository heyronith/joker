"""Typed data-quality evaluation for market observations and snapshots."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from joker.market.bars import MarketBar
from joker.market.observations import (
    OptionQuoteObservation,
    QuoteObservation,
    TradeObservation,
    UnderlyingObservation,
)
from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.market.snapshots import DataQualitySnapshot, UnderlyingSnapshot


class DataQualitySeverity(StrEnum):
    """Highest severity present in a quality report."""

    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataQualityCode(StrEnum):
    """Stable finding codes — no strategy or trade-direction semantics."""

    STALE_UNDERLYING = "stale_underlying"
    STALE_OPTION = "stale_option"
    CROSSED_MARKET = "crossed_market"
    LOCKED_MARKET = "locked_market"
    MISSING_BID = "missing_bid"
    MISSING_ASK = "missing_ask"
    NEGATIVE_PRICE = "negative_price"
    EXTREME_SPREAD = "extreme_spread"
    MISSING_INTERVAL = "missing_interval"
    INCOMPLETE_BAR = "incomplete_bar"
    CLOCK_SKEW = "clock_skew"
    DUPLICATE_OBSERVATION = "duplicate_observation"
    CUMULATIVE_VOLUME_REGRESSION = "cumulative_volume_regression"
    INSUFFICIENT_OPTION_SURFACE = "insufficient_option_surface"


class DataQualityFinding(BaseModel):
    """Single data-quality finding."""

    model_config = ConfigDict(frozen=True)

    code: DataQualityCode
    severity: DataQualitySeverity
    message: str
    symbol: str | None = None
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class DataQualityReport(BaseModel):
    """Aggregated data-quality assessment for a market snapshot."""

    model_config = ConfigDict(frozen=True)

    report_id: UUID = Field(default_factory=uuid4)
    snapshot_id: UUID | None = None
    severity: DataQualitySeverity
    findings: tuple[DataQualityFinding, ...] = ()
    usable_for_reasoning: bool = True
    usable_for_execution: bool = True

    def to_snapshot(self) -> DataQualitySnapshot:
        """Compact snapshot form for MarketSnapshot.data_quality_id linkage."""
        return DataQualitySnapshot(
            data_quality_id=self.report_id,
            severity=self.severity.value,
            finding_codes=tuple(f.code.value for f in self.findings),
            usable_for_reasoning=self.usable_for_reasoning,
            usable_for_execution=self.usable_for_execution,
        )


class DataQualityConfig(BaseModel):
    """Thresholds for quality evaluation (no strategy preferences)."""

    model_config = ConfigDict(frozen=True)

    underlying_max_age_seconds: float = 30.0
    option_max_age_seconds: float = 60.0
    max_relative_spread: float = 0.25
    max_clock_skew_seconds: float = 5.0
    min_option_contracts: int = 1
    expected_1m_intervals: int | None = None


_SEVERITY_RANK = {
    DataQualitySeverity.OK: 0,
    DataQualitySeverity.INFO: 1,
    DataQualitySeverity.WARNING: 2,
    DataQualitySeverity.ERROR: 3,
    DataQualitySeverity.CRITICAL: 4,
}


def _max_severity(findings: list[DataQualityFinding]) -> DataQualitySeverity:
    if not findings:
        return DataQualitySeverity.OK
    return max(findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity


def _age_seconds(ts: datetime, now: datetime) -> float:
    if ts.tzinfo is None or now.tzinfo is None:
        raise ValueError("Naive datetimes are not allowed in quality evaluation")
    return (now - ts.astimezone(now.tzinfo)).total_seconds()


def evaluate_data_quality(
    *,
    now: datetime,
    underlying: UnderlyingSnapshot | UnderlyingObservation | QuoteObservation | None = None,
    option_quotes: list[OptionQuoteObservation] | tuple[OptionQuoteObservation, ...] | None = None,
    option_surface: OptionSurfaceSnapshot | None = None,
    bars_1m: list[MarketBar] | tuple[MarketBar, ...] | None = None,
    bars_5m: list[MarketBar] | tuple[MarketBar, ...] | None = None,
    trades: list[TradeObservation] | tuple[TradeObservation, ...] | None = None,
    observation_ids: list[UUID] | tuple[UUID, ...] | None = None,
    prior_cumulative_volume: int | None = None,
    current_cumulative_volume: int | None = None,
    snapshot_id: UUID | None = None,
    config: DataQualityConfig | None = None,
) -> DataQualityReport:
    """
    Evaluate market-data integrity checks.

    Does not encode trade direction or strategy preferences.
    """
    if now.tzinfo is None:
        raise ValueError("evaluate_data_quality requires timezone-aware now")

    cfg = config or DataQualityConfig()
    findings: list[DataQualityFinding] = []

    _check_underlying(underlying, now, cfg, findings)
    _check_option_quotes(option_quotes, now, cfg, findings)
    _check_option_surface(option_surface, now, cfg, findings)
    _check_bars(bars_1m, bars_5m, cfg, findings)
    _check_clock_skew(underlying, option_quotes, option_surface, now, cfg, findings)
    _check_duplicates(observation_ids, findings)
    _check_cum_vol(prior_cumulative_volume, current_cumulative_volume, findings)
    _check_trades(trades, findings)

    severity = _max_severity(findings)
    usable_for_reasoning = severity not in {
        DataQualitySeverity.CRITICAL,
    }
    usable_for_execution = severity not in {
        DataQualitySeverity.ERROR,
        DataQualitySeverity.CRITICAL,
    } and not any(
        f.code
        in {
            DataQualityCode.STALE_UNDERLYING,
            DataQualityCode.CROSSED_MARKET,
            DataQualityCode.NEGATIVE_PRICE,
            DataQualityCode.INSUFFICIENT_OPTION_SURFACE,
        }
        for f in findings
    )

    return DataQualityReport(
        snapshot_id=snapshot_id,
        severity=severity,
        findings=tuple(findings),
        usable_for_reasoning=usable_for_reasoning,
        usable_for_execution=usable_for_execution,
    )


def _check_underlying(
    underlying: UnderlyingSnapshot | UnderlyingObservation | QuoteObservation | None,
    now: datetime,
    cfg: DataQualityConfig,
    findings: list[DataQualityFinding],
) -> None:
    if underlying is None:
        return

    symbol = underlying.symbol
    if isinstance(underlying, UnderlyingSnapshot):
        ts = underlying.exchange_time
        bid, ask, last = underlying.bid, underlying.ask, underlying.last
    else:
        ts = underlying.source_timestamp
        bid, ask = underlying.bid, underlying.ask
        last = underlying.last

    age = _age_seconds(ts, now)
    if age > cfg.underlying_max_age_seconds:
        findings.append(
            DataQualityFinding(
                code=DataQualityCode.STALE_UNDERLYING,
                severity=DataQualitySeverity.ERROR,
                message=f"Underlying quote age {age:.1f}s exceeds {cfg.underlying_max_age_seconds}s",
                symbol=symbol,
                details={"age_seconds": age},
            )
        )

    _check_book(symbol, bid, ask, last, findings)


def _check_book(
    symbol: str,
    bid: Decimal | None,
    ask: Decimal | None,
    last: Decimal | None,
    findings: list[DataQualityFinding],
    *,
    max_relative_spread: float = 0.25,
) -> None:
    if bid is None:
        findings.append(
            DataQualityFinding(
                code=DataQualityCode.MISSING_BID,
                severity=DataQualitySeverity.WARNING,
                message="Missing bid",
                symbol=symbol,
            )
        )
    if ask is None:
        findings.append(
            DataQualityFinding(
                code=DataQualityCode.MISSING_ASK,
                severity=DataQualitySeverity.WARNING,
                message="Missing ask",
                symbol=symbol,
            )
        )

    for label, price in (("bid", bid), ("ask", ask), ("last", last)):
        if price is not None and price < 0:
            findings.append(
                DataQualityFinding(
                    code=DataQualityCode.NEGATIVE_PRICE,
                    severity=DataQualitySeverity.CRITICAL,
                    message=f"Negative {label} price: {price}",
                    symbol=symbol,
                    details={"field": label, "price": float(price)},
                )
            )

    if bid is not None and ask is not None:
        if ask < bid:
            findings.append(
                DataQualityFinding(
                    code=DataQualityCode.CROSSED_MARKET,
                    severity=DataQualitySeverity.ERROR,
                    message=f"Crossed market bid={bid} ask={ask}",
                    symbol=symbol,
                )
            )
        elif ask == bid:
            findings.append(
                DataQualityFinding(
                    code=DataQualityCode.LOCKED_MARKET,
                    severity=DataQualitySeverity.INFO,
                    message=f"Locked market bid=ask={bid}",
                    symbol=symbol,
                )
            )
        else:
            mid = (bid + ask) / Decimal("2")
            if mid > 0:
                rel = float((ask - bid) / mid)
                if rel > max_relative_spread:
                    findings.append(
                        DataQualityFinding(
                            code=DataQualityCode.EXTREME_SPREAD,
                            severity=DataQualitySeverity.WARNING,
                            message=f"Relative spread {rel:.4f} exceeds {max_relative_spread}",
                            symbol=symbol,
                            details={"relative_spread": rel},
                        )
                    )


def _check_option_quotes(
    option_quotes: list[OptionQuoteObservation] | tuple[OptionQuoteObservation, ...] | None,
    now: datetime,
    cfg: DataQualityConfig,
    findings: list[DataQualityFinding],
) -> None:
    if not option_quotes:
        return
    for oq in option_quotes:
        age = _age_seconds(oq.source_timestamp, now)
        if age > cfg.option_max_age_seconds:
            findings.append(
                DataQualityFinding(
                    code=DataQualityCode.STALE_OPTION,
                    severity=DataQualitySeverity.WARNING,
                    message=f"Option quote age {age:.1f}s exceeds {cfg.option_max_age_seconds}s",
                    symbol=oq.contract_symbol,
                    details={"age_seconds": age},
                )
            )
        _check_book(
            oq.contract_symbol,
            oq.bid,
            oq.ask,
            oq.last,
            findings,
            max_relative_spread=cfg.max_relative_spread,
        )


def _check_option_surface(
    surface: OptionSurfaceSnapshot | None,
    now: datetime,
    cfg: DataQualityConfig,
    findings: list[DataQualityFinding],
) -> None:
    if surface is None:
        return
    if len(surface.contracts) < cfg.min_option_contracts:
        findings.append(
            DataQualityFinding(
                code=DataQualityCode.INSUFFICIENT_OPTION_SURFACE,
                severity=DataQualitySeverity.ERROR,
                message=(
                    f"Option surface has {len(surface.contracts)} contracts; "
                    f"minimum is {cfg.min_option_contracts}"
                ),
                symbol=surface.underlying_symbol,
            )
        )
        return

    for c in surface.contracts:
        age_s = c.quote_age_ms / 1000.0
        if age_s > cfg.option_max_age_seconds:
            findings.append(
                DataQualityFinding(
                    code=DataQualityCode.STALE_OPTION,
                    severity=DataQualitySeverity.WARNING,
                    message=f"Surface contract {c.contract_id} age {age_s:.1f}s",
                    symbol=c.contract_id,
                    details={"age_seconds": age_s},
                )
            )
        _check_contract_book(c, cfg, findings)


def _check_contract_book(
    c: OptionContractSnapshot,
    cfg: DataQualityConfig,
    findings: list[DataQualityFinding],
) -> None:
    _check_book(
        c.contract_id,
        c.bid,
        c.ask,
        c.last,
        findings,
        max_relative_spread=cfg.max_relative_spread,
    )


def _check_bars(
    bars_1m: list[MarketBar] | tuple[MarketBar, ...] | None,
    bars_5m: list[MarketBar] | tuple[MarketBar, ...] | None,
    cfg: DataQualityConfig,
    findings: list[DataQualityFinding],
) -> None:
    for bars in (bars_1m, bars_5m):
        if not bars:
            continue
        ordered = sorted(bars, key=lambda b: b.start)
        for bar in ordered:
            if bar.incomplete:
                findings.append(
                    DataQualityFinding(
                        code=DataQualityCode.INCOMPLETE_BAR,
                        severity=DataQualitySeverity.WARNING,
                        message=f"Incomplete {bar.timeframe.value} bar starting {bar.start.isoformat()}",
                        symbol=bar.symbol,
                    )
                )
        # Missing interval detection for contiguous 1m sequences when expected count set.
        if bars is bars_1m and cfg.expected_1m_intervals is not None:
            if len(ordered) < cfg.expected_1m_intervals:
                findings.append(
                    DataQualityFinding(
                        code=DataQualityCode.MISSING_INTERVAL,
                        severity=DataQualitySeverity.WARNING,
                        message=(
                            f"Expected {cfg.expected_1m_intervals} 1m bars, "
                            f"got {len(ordered)}"
                        ),
                        symbol=ordered[0].symbol if ordered else None,
                    )
                )
        if bars is bars_1m and len(ordered) >= 2:
            for prev, cur in zip(ordered, ordered[1:]):
                if prev.end < cur.start:
                    findings.append(
                        DataQualityFinding(
                            code=DataQualityCode.MISSING_INTERVAL,
                            severity=DataQualitySeverity.WARNING,
                            message=(
                                f"Gap between {prev.end.isoformat()} and {cur.start.isoformat()}"
                            ),
                            symbol=cur.symbol,
                        )
                    )


def _check_clock_skew(
    underlying: UnderlyingSnapshot | UnderlyingObservation | QuoteObservation | None,
    option_quotes: list[OptionQuoteObservation] | tuple[OptionQuoteObservation, ...] | None,
    surface: OptionSurfaceSnapshot | None,
    now: datetime,
    cfg: DataQualityConfig,
    findings: list[DataQualityFinding],
) -> None:
    timestamps: list[tuple[str, datetime]] = []
    if isinstance(underlying, UnderlyingSnapshot):
        timestamps.append((underlying.symbol, underlying.exchange_time))
    elif underlying is not None:
        timestamps.append((underlying.symbol, underlying.source_timestamp))
        timestamps.append((underlying.symbol, underlying.received_timestamp))

    if option_quotes:
        for oq in option_quotes:
            timestamps.append((oq.contract_symbol, oq.source_timestamp))
            timestamps.append((oq.contract_symbol, oq.received_timestamp))

    if surface is not None:
        timestamps.append((surface.underlying_symbol, surface.exchange_time))

    for symbol, ts in timestamps:
        skew = abs(_age_seconds(ts, now))
        # Future timestamps (negative age) or large forward skew.
        forward = (ts.astimezone(now.tzinfo) - now).total_seconds()
        if forward > cfg.max_clock_skew_seconds:
            findings.append(
                DataQualityFinding(
                    code=DataQualityCode.CLOCK_SKEW,
                    severity=DataQualitySeverity.ERROR,
                    message=f"Clock skew: timestamp {forward:.1f}s ahead of now",
                    symbol=symbol,
                    details={"skew_seconds": forward},
                )
            )
        elif skew > 3600 and forward <= 0:
            # Extreme past skew vs exchange clock — informational only when stale checks exist.
            pass


def _check_duplicates(
    observation_ids: list[UUID] | tuple[UUID, ...] | None,
    findings: list[DataQualityFinding],
) -> None:
    if not observation_ids:
        return
    seen: set[UUID] = set()
    for oid in observation_ids:
        if oid in seen:
            findings.append(
                DataQualityFinding(
                    code=DataQualityCode.DUPLICATE_OBSERVATION,
                    severity=DataQualitySeverity.WARNING,
                    message=f"Duplicate observation_id {oid}",
                    details={"observation_id": str(oid)},
                )
            )
        else:
            seen.add(oid)


def _check_cum_vol(
    prior: int | None,
    current: int | None,
    findings: list[DataQualityFinding],
) -> None:
    if prior is None or current is None:
        return
    if current < prior:
        findings.append(
            DataQualityFinding(
                code=DataQualityCode.CUMULATIVE_VOLUME_REGRESSION,
                severity=DataQualitySeverity.ERROR,
                message=f"Cumulative volume fell from {prior} to {current}",
                details={"prior": prior, "current": current},
            )
        )


def _check_trades(
    trades: list[TradeObservation] | tuple[TradeObservation, ...] | None,
    findings: list[DataQualityFinding],
) -> None:
    if not trades:
        return
    for trade in trades:
        if trade.price < 0:
            findings.append(
                DataQualityFinding(
                    code=DataQualityCode.NEGATIVE_PRICE,
                    severity=DataQualitySeverity.CRITICAL,
                    message=f"Negative trade price {trade.price}",
                    symbol=trade.symbol,
                )
            )
