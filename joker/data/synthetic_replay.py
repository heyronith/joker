"""Generate synthetic SPY 0DTE replay fixtures for tests.

All data is explicitly synthetic — not real market data or performance claims.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from joker.schemas.domain import Candle
from joker.schemas.replay import OptionQuoteEvent, SpyCandleEvent, SpyQuoteEvent

from joker.compliance.data_classification import (
    DataClassification,
    SOURCE_SYNTHETIC_OPTION,
    SOURCE_SYNTHETIC_STOCK,
)
TRADING_DAY = date(2026, 7, 1)
BASE_TS = datetime(2026, 7, 1, 14, 0, 0, tzinfo=timezone.utc)


def _ts(minutes: int) -> datetime:
    return BASE_TS + timedelta(minutes=minutes)


def _event_dict(model) -> dict:
    return json.loads(model.model_dump_json())


def build_synthetic_replay_events() -> list[dict]:
    """Build a full synthetic intraday sequence."""
    events: list[dict] = []
    spy = 550.0
    vwap = 550.0
    candles: list[float] = []

    def add_spy_quote(minute: int, price: float) -> None:
        events.append(
            _event_dict(
                SpyQuoteEvent(
                    timestamp=_ts(minute),
                    symbol="SPY",
                    source=SOURCE_SYNTHETIC_STOCK,
                    price=price,
                    bid=round(price - 0.01, 2),
                    ask=round(price + 0.01, 2),
                )
            )
        )

    def add_candle(minute: int, o: float, h: float, lo: float, c: float, vol: float = 10000) -> None:
        candles.append(c)
        events.append(
            _event_dict(
                SpyCandleEvent(
                    timestamp=_ts(minute),
                    symbol="SPY",
                    source=SOURCE_SYNTHETIC_STOCK,
                    candle=Candle(
                        symbol="SPY",
                        timestamp=_ts(minute),
                        open=o,
                        high=h,
                        low=lo,
                        close=c,
                        volume=vol,
                    ),
                )
            )
        )

    def add_option(
        minute: int,
        strike: float,
        option_type: str,
        bid: float,
        ask: float,
        *,
        quote_age_minutes: int = 0,
        contract_suffix: str = "",
    ) -> None:
        mid = round((bid + ask) / 2, 4)
        spread_pct = round(((ask - bid) / mid) * 100, 2) if mid > 0 else 100.0
        cid = f"SPY_{TRADING_DAY.isoformat()}_{strike}_{option_type}{contract_suffix}"
        events.append(
            _event_dict(
                OptionQuoteEvent(
                    timestamp=_ts(minute),
                    symbol="SPY",
                    source=SOURCE_SYNTHETIC_OPTION,
                    contract_id=cid,
                    expiration=TRADING_DAY,
                    strike=strike,
                    option_type=option_type,  # type: ignore[arg-type]
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    spread_pct=spread_pct,
                    volume=500,
                    open_interest=1000,
                    quote_timestamp=_ts(minute - quote_age_minutes),
                    is_synthetic=True,
                    data_classification=DataClassification.SYNTHETIC_DATA.value,
                )
            )
        )

    # Phase 1: trend up (minutes 0-10)
    for i, p in enumerate([550.0, 550.3, 550.8, 551.2, 551.5, 552.0, 552.3, 552.8, 553.0, 553.2, 553.5]):
        add_spy_quote(i, p)
        if i % 2 == 0:
            add_candle(i, p - 0.2, p + 0.3, p - 0.3, p)
        add_option(i, 553.0, "call", 1.0, 1.1)

    # Phase 2: chop — declining call premium for stop-loss exit testing
    chop_prices = [553.0, 552.5, 551.0, 550.5, 550.2, 550.0, 550.3, 550.1, 550.0, 550.2]
    call_mids = [1.05, 0.9, 0.7, 0.6, 0.55, 0.5, 0.48, 0.45, 0.42, 0.4]
    for j, p in enumerate(chop_prices):
        m = 11 + j
        add_spy_quote(m, p)
        add_candle(m, p, p + 0.1, p - 0.1, p, vol=8000)
        mid = call_mids[j]
        add_option(m, 553.0, "call", round(mid - 0.05, 2), round(mid + 0.05, 2))
        add_option(m, 550.0, "put", 0.85, 0.95)

    # Phase 3: breakdown (minutes 21-30)
    for j, p in enumerate([549.5, 548.8, 548.0, 547.2, 546.5, 546.0, 545.5, 545.0, 544.5, 544.0]):
        m = 21 + j
        add_spy_quote(m, p)
        add_candle(m, p + 0.2, p + 0.3, p - 0.2, p)
        add_option(m, 545.0, "put", 1.2, 1.35)

    # Wide spread case (minute 31)
    add_spy_quote(31, 544.0)
    add_option(31, 544.0, "put", 0.5, 1.5, contract_suffix="_wide")

    # Stale quote case (minute 32)
    add_spy_quote(32, 544.0)
    add_option(32, 544.0, "put", 1.0, 1.1, quote_age_minutes=120, contract_suffix="_stale")

    # Recovery with tight spread for exit testing (minutes 33-40)
    for j, p in enumerate([544.5, 545.0, 545.5, 546.0, 546.5, 547.0, 547.5, 548.0]):
        m = 33 + j
        add_spy_quote(m, p)
        add_option(m, 545.0, "put", max(0.3, 1.5 - j * 0.15), max(0.35, 1.6 - j * 0.15))

    return events


def write_synthetic_replay(path: Path | None = None) -> Path:
    path = path or Path("data/replays/spy_0dte_synthetic_day.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    events = build_synthetic_replay_events()
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# SYNTHETIC REPLAY DATA — not real market data or performance claims\n")
        for event in events:
            handle.write(json.dumps(event, default=str) + "\n")
    return path


if __name__ == "__main__":
    out = write_synthetic_replay()
    print(f"Wrote {out}")
