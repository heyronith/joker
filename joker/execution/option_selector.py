"""SPY 0DTE option contract selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from joker.data.freshness import FreshnessConfig, evaluate_quote_freshness
from joker.data.options_normalizer import snapshot_to_quote_event
from joker.schemas.domain import OptionContract, OptionQuote
from joker.schemas.options_data import OptionSnapshot
from joker.schemas.replay import OptionQuoteEvent, SelectedOptionContract


class OptionSelectionError(Exception):
    pass


@dataclass(frozen=True)
class OptionSelectorConfig:
    max_spread_pct: float = 15.0
    max_premium_usd: float = 200.0
    quote_max_age_seconds: int = 30
    allowed_symbol: str = "SPY"
    allow_delayed_quotes: bool = True
    feed_max_silence_seconds: int = 60
    delayed_quote_max_age_seconds: int = 900
    # When True (agent_led), soft spread/premium caps become advisory — still select.
    soft_liquidity_advisory: bool = False


class OptionSelector:
    """Select near-ATM 0DTE long call/put from live or replay quotes."""

    REJECT_STALE = "STALE_QUOTE"
    REJECT_WIDE_SPREAD = "WIDE_SPREAD"
    REJECT_MISSING = "MISSING_QUOTE"
    REJECT_PREMIUM = "MAX_PREMIUM"
    REJECT_NOT_0DTE = "NOT_0DTE"
    REJECT_DELAYED = "DELAYED_NOT_ALLOWED"
    REJECT_FEED_SILENT = "FEED_SILENT"

    def __init__(self, config: OptionSelectorConfig | None = None) -> None:
        self.config = config or OptionSelectorConfig()
        self.last_advisories: list[str] = []

    def _freshness_config(self) -> FreshnessConfig:
        return FreshnessConfig(
            quote_max_age_seconds=self.config.quote_max_age_seconds,
            feed_max_silence_seconds=self.config.feed_max_silence_seconds,
            delayed_quote_max_age_seconds=self.config.delayed_quote_max_age_seconds,
            allow_delayed_quotes=self.config.allow_delayed_quotes,
        )

    def _validate_quote(self, event: OptionQuoteEvent, reference_time: datetime) -> list[str]:
        advisories: list[str] = []
        if event.symbol != self.config.allowed_symbol:
            raise OptionSelectionError(f"Wrong symbol: {event.symbol}")
        if event.bid <= 0 or event.ask <= 0:
            raise OptionSelectionError(self.REJECT_MISSING)
        verdict = evaluate_quote_freshness(
            quote_timestamp=event.quote_timestamp,
            reference_time=reference_time,
            delayed=event.delayed,
            received_at=event.received_at,
            config=self._freshness_config(),
        )
        if not verdict.ok:
            if verdict.reason == "DELAYED_NOT_ALLOWED":
                raise OptionSelectionError(self.REJECT_DELAYED)
            if verdict.reason == "FEED_SILENT":
                raise OptionSelectionError(self.REJECT_FEED_SILENT)
            raise OptionSelectionError(self.REJECT_STALE)
        if event.spread_pct > self.config.max_spread_pct:
            if self.config.soft_liquidity_advisory:
                advisories.append(f"{self.REJECT_WIDE_SPREAD}:{event.spread_pct:.1f}")
            else:
                raise OptionSelectionError(self.REJECT_WIDE_SPREAD)
        premium = event.mid * 100
        if premium > self.config.max_premium_usd:
            if self.config.soft_liquidity_advisory:
                advisories.append(f"{self.REJECT_PREMIUM}:{premium:.1f}")
            else:
                raise OptionSelectionError(self.REJECT_PREMIUM)
        return advisories

    def select_from_events(
        self,
        quotes: list[OptionQuoteEvent],
        direction: str,
        underlying_price: float,
        reference_time: datetime,
    ) -> SelectedOptionContract:
        self.last_advisories = []
        option_type = "call" if direction == "long_call" else "put"
        candidates = [q for q in quotes if q.option_type == option_type]
        if not candidates:
            raise OptionSelectionError(self.REJECT_MISSING)

        candidates.sort(key=lambda q: abs(q.strike - underlying_price))
        last_error: OptionSelectionError | None = None
        for event in candidates:
            try:
                advisories = self._validate_quote(event, reference_time)
                self.last_advisories = advisories
                contract = OptionContract(
                    symbol=event.symbol,
                    expiration=event.expiration,
                    strike=event.strike,
                    option_type=event.option_type,
                    is_0dte=True,
                )
                quote = OptionQuote(
                    contract=contract,
                    bid=event.bid,
                    ask=event.ask,
                    last=event.mid,
                    timestamp=event.quote_timestamp,
                    source=event.source,
                    data_classification=event.data_classification,
                    persist_allowed=event.persist_allowed,
                    openai_allowed=event.openai_allowed,
                    is_synthetic=event.is_synthetic,
                    delayed=event.delayed,
                    received_at=event.received_at,
                )
                reason = f"near-ATM {option_type} strike {event.strike}"
                if advisories:
                    reason = f"{reason} advisory={','.join(advisories)}"
                return SelectedOptionContract(
                    contract_id=event.contract_id,
                    contract=contract,
                    quote=quote,
                    selection_reason=reason,
                    underlying_price=underlying_price,
                )
            except OptionSelectionError as exc:
                last_error = exc
                continue
        raise last_error or OptionSelectionError(self.REJECT_MISSING)

    def select_from_snapshots(
        self,
        snapshots: list[OptionSnapshot],
        direction: str,
        underlying_price: float,
        reference_time: datetime,
    ) -> SelectedOptionContract:
        """Select from Webull OptionSnapshot objects."""
        events = [snapshot_to_quote_event(s) for s in snapshots]
        return self.select_from_events(events, direction, underlying_price, reference_time)

    def select_from_surface(
        self,
        surface,
        direction: str,
        underlying_price: float,
        reference_time: datetime,
    ) -> SelectedOptionContract:
        """Select nearest ATM from a full OptionSurfaceSnapshot (compatibility path)."""
        from joker.schemas.replay import OptionQuoteEvent

        option_type = "call" if direction == "long_call" else "put"
        events: list[OptionQuoteEvent] = []
        for c in surface.contracts:
            if c.option_type != option_type:
                continue
            if c.bid is None or c.ask is None:
                continue
            mid = float(c.mid) if c.mid is not None else (float(c.bid) + float(c.ask)) / 2.0
            spread_pct = 0.0
            if mid > 0:
                spread_pct = ((float(c.ask) - float(c.bid)) / mid) * 100.0
            events.append(
                OptionQuoteEvent(
                    timestamp=c.quote_timestamp,
                    source="option_surface",
                    contract_id=c.contract_id,
                    expiration=c.expiry,
                    strike=float(c.strike),
                    option_type=c.option_type,
                    bid=float(c.bid),
                    ask=float(c.ask),
                    mid=mid,
                    spread_pct=spread_pct,
                    quote_timestamp=c.quote_timestamp,
                )
            )
        return self.select_from_events(events, direction, underlying_price, reference_time)
