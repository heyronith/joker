"""Webull SPY 0DTE options data provider — discovery, caching, normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from joker.config.settings import EnvSettings
from joker.data.options_normalizer import (
    OptionQuoteValidationError,
    snapshot_to_quote_event,
    validate_tradable_snapshot,
)
from joker.data.options_rate_limit import RateLimitExceeded, RateLimiter, TTLCache
from joker.data.webull_errors import OptionEndpointUnverified, WebullApiError
from joker.data.webull_config import validate_webull_market_env
from joker.data.webull_option_symbols import build_atm_candidate_symbols
from joker.data.webull_options_api import (
    HttpWebullOptionsMarketApi,
    MockWebullOptionsMarketApi,
    WebullOptionsMarketApi,
)
from joker.schemas.options_data import (
    OptionContractMetadata,
    OptionDataCapabilityReport,
    OptionDataQualityWarning,
    OptionSnapshot,
)
from joker.schemas.replay import OptionQuoteEvent

MARKET_TZ = ZoneInfo("America/New_York")
ALLOWED_SYMBOL = "SPY"


@dataclass
class AtmCandidateSet:
    expiration: date
    underlying_price: float
    atm_call: OptionContractMetadata | None = None
    atm_put: OptionContractMetadata | None = None
    otm_call: OptionContractMetadata | None = None
    otm_put: OptionContractMetadata | None = None
    itm_call: OptionContractMetadata | None = None
    itm_put: OptionContractMetadata | None = None


@dataclass
class WebullOptionsDataProvider:
    """Discover and fetch SPY 0DTE option quotes — no order submission."""

    env: EnvSettings
    api: WebullOptionsMarketApi | None = None
    contract_cache_ttl_seconds: float = 300.0
    snapshot_cache_ttl_seconds: float = 5.0
    max_spread_pct: float = 15.0
    quote_max_age_seconds: int = 30
    allow_delayed_quotes: bool = True
    feed_max_silence_seconds: int = 60
    delayed_quote_max_age_seconds: int = 900
    _verified: bool = False
    _contract_discovery_succeeded: bool = False
    _contract_cache: TTLCache[list[OptionContractMetadata]] = field(
        default_factory=lambda: TTLCache(300.0)
    )
    _snapshot_cache: TTLCache[OptionSnapshot] = field(
        default_factory=lambda: TTLCache(5.0)
    )
    _rate_limiter: RateLimiter = field(default_factory=lambda: RateLimiter(60, 60.0))
    _last_warnings: list[OptionDataQualityWarning] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.api is None:
            self.api = HttpWebullOptionsMarketApi(self.env)
        self._contract_cache = TTLCache(self.contract_cache_ttl_seconds)
        self._snapshot_cache = TTLCache(self.snapshot_cache_ttl_seconds)

    @property
    def verified(self) -> bool:
        return self._verified

    @verified.setter
    def verified(self, value: bool) -> None:
        self._verified = value

    @property
    def contract_discovery_succeeded(self) -> bool:
        return self._contract_discovery_succeeded

    @property
    def last_warnings(self) -> list[OptionDataQualityWarning]:
        return list(self._last_warnings)

    def market_today(self) -> date:
        return datetime.now(MARKET_TZ).date()

    def authenticate(self) -> bool:
        result = self.api.authenticate()
        return result.success

    def discover_contracts(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> list[OptionContractMetadata]:
        if symbol.upper() != ALLOWED_SYMBOL:
            raise WebullApiError(f"Only {ALLOWED_SYMBOL} supported; got {symbol}")
        exp = expiration or self.market_today()
        cache_key = f"{symbol}:{exp.isoformat()}"
        cached = self._contract_cache.get(cache_key)
        if cached is not None:
            return cached
        self._rate_limiter.acquire()
        contracts = self.api.find_option_contracts(symbol, exp)
        self._contract_discovery_succeeded = bool(contracts)
        self._contract_cache.set(cache_key, contracts)
        return contracts

    def discover_osi_candidates(
        self,
        underlying_price: float,
        expiration: date | None = None,
        symbol: str = ALLOWED_SYMBOL,
    ) -> list[OptionContractMetadata]:
        """Construct OSI symbols when chain API is unverified."""
        exp = expiration or self.market_today()
        return build_atm_candidate_symbols(symbol, exp, underlying_price)

    @staticmethod
    def select_atm_candidates(
        contracts: list[OptionContractMetadata],
        underlying_price: float,
        expiration: date,
    ) -> AtmCandidateSet:
        result = AtmCandidateSet(expiration=expiration, underlying_price=underlying_price)
        if not contracts:
            return result

        calls = sorted(
            [c for c in contracts if c.option_type == "call"],
            key=lambda c: abs(c.strike - underlying_price),
        )
        puts = sorted(
            [c for c in contracts if c.option_type == "put"],
            key=lambda c: abs(c.strike - underlying_price),
        )
        if calls:
            result.atm_call = calls[0]
            if len(calls) > 1:
                otm = next((c for c in calls if c.strike > underlying_price), None)
                itm = next((c for c in calls if c.strike < underlying_price), None)
                result.otm_call = otm
                result.itm_call = itm
        if puts:
            result.atm_put = puts[0]
            if len(puts) > 1:
                otm = next((c for c in puts if c.strike < underlying_price), None)
                itm = next((c for c in puts if c.strike > underlying_price), None)
                result.otm_put = otm
                result.itm_put = itm
        return result

    def fetch_snapshot(self, contract: OptionContractMetadata) -> OptionSnapshot:
        key = contract.contract_id or ""
        if not key:
            raise WebullApiError("Missing contract_id")
        cached = self._snapshot_cache.get(key)
        if cached is not None:
            return cached
        try:
            self._rate_limiter.acquire()
            snapshot = self.api.get_option_snapshot(contract)
        except RateLimitExceeded as exc:
            raise WebullApiError(str(exc), rate_limited=True) from exc
        self._snapshot_cache.set(key, snapshot)
        return snapshot

    def fetch_atm_snapshots(
        self,
        underlying_price: float,
        expiration: date | None = None,
        *,
        allow_osi_fallback: bool = True,
    ) -> tuple[OptionSnapshot | None, OptionSnapshot | None]:
        exp = expiration or self.market_today()
        contracts: list[OptionContractMetadata]
        try:
            contracts = self.discover_contracts(ALLOWED_SYMBOL, exp)
        except OptionEndpointUnverified:
            if not allow_osi_fallback:
                raise
            contracts = self.discover_osi_candidates(underlying_price, exp)
        if not contracts:
            return None, None
        candidates = self.select_atm_candidates(contracts, underlying_price, exp)
        call_snap = (
            self.fetch_snapshot(candidates.atm_call)
            if candidates.atm_call and candidates.atm_call.contract_id
            else None
        )
        put_snap = (
            self.fetch_snapshot(candidates.atm_put)
            if candidates.atm_put and candidates.atm_put.contract_id
            else None
        )
        return call_snap, put_snap

    def to_quote_events(
        self,
        snapshots: list[OptionSnapshot],
        *,
        require_tradable: bool = True,
        reference_time: datetime | None = None,
        allow_delayed_quotes: bool = True,
        feed_max_silence_seconds: int = 60,
        delayed_quote_max_age_seconds: int = 900,
    ) -> list[OptionQuoteEvent]:
        ref = reference_time or datetime.now(timezone.utc)
        received_at = datetime.now(timezone.utc)
        events: list[OptionQuoteEvent] = []
        self._last_warnings = []
        for snap in snapshots:
            if require_tradable:
                warnings = validate_tradable_snapshot(
                    snap,
                    reference_time=ref,
                    max_spread_pct=self.max_spread_pct,
                    quote_max_age_seconds=self.quote_max_age_seconds,
                    allow_delayed_quotes=allow_delayed_quotes,
                    feed_max_silence_seconds=feed_max_silence_seconds,
                    delayed_quote_max_age_seconds=delayed_quote_max_age_seconds,
                    received_at=received_at,
                )
                self._last_warnings.extend(warnings)
            event = snapshot_to_quote_event(snap)
            events.append(event.model_copy(update={"received_at": received_at}))
        return events

    def build_capability_report(
        self,
        call_snap: OptionSnapshot | None,
        put_snap: OptionSnapshot | None,
        *,
        contract_discovery_succeeded: bool = False,
        auth_pass: bool = False,
        same_day_expiration: bool = False,
    ) -> OptionDataCapabilityReport:
        snaps = [s for s in (call_snap, put_snap) if s is not None]
        if not snaps:
            return OptionDataCapabilityReport(
                contract_discovery=contract_discovery_succeeded,
                verified=False,
            )
        bid_ask = all(s.bid is not None and s.ask is not None for s in snaps)
        timestamps = all(s.quote_timestamp is not None for s in snaps)
        verified = (
            auth_pass
            and same_day_expiration
            and report_atm_pass(call_snap, put_snap)
            and bid_ask
            and timestamps
        )
        all_unavail: set[str] = set()
        for s in snaps:
            all_unavail.update(s.field_availability.unavailable_fields())
        return OptionDataCapabilityReport(
            contract_discovery=contract_discovery_succeeded,
            same_day_expiration=same_day_expiration,
            snapshot_bid_ask=bid_ask,
            volume=any(s.volume is not None for s in snaps),
            open_interest=any(s.open_interest is not None for s in snaps),
            implied_volatility=any(s.implied_volatility is not None for s in snaps),
            greeks=any(s.delta is not None or s.gamma is not None for s in snaps),
            historical_bars="unknown",
            ticks="unknown",
            delayed_data=any(s.delayed for s in snaps if s.delayed is not None),
            verified=verified,
            unavailable_fields=sorted(all_unavail),
        )

    def is_available(self) -> bool:
        from joker.data.webull_capability import capability_usable_for_shadow

        return self._verified or capability_usable_for_shadow()


def report_atm_pass(
    call_snap: OptionSnapshot | None,
    put_snap: OptionSnapshot | None,
) -> bool:
    return call_snap is not None and put_snap is not None


def create_webull_options_provider(
    env: EnvSettings,
    *,
    api: WebullOptionsMarketApi | None = None,
    app_settings: object | None = None,
) -> WebullOptionsDataProvider:
    validate_webull_market_env(env)
    provider = WebullOptionsDataProvider(env=env, api=api)
    if app_settings is not None:
        risk = getattr(app_settings, "risk", None)
        if risk is not None:
            provider.max_spread_pct = getattr(risk, "max_spread_pct", 15.0)
            provider.quote_max_age_seconds = getattr(risk, "quote_max_age_seconds", 30)
            provider.allow_delayed_quotes = getattr(risk, "allow_delayed_quotes", True)
            provider.feed_max_silence_seconds = getattr(risk, "feed_max_silence_seconds", 60)
            provider.delayed_quote_max_age_seconds = getattr(
                risk, "delayed_quote_max_age_seconds", 900
            )
    return provider
