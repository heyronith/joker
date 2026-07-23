"""Replay and market-event schemas."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from joker.compliance.data_classification import (
    DataClassification,
    SOURCE_WEBULL_STOCK,
)
from joker.schemas.domain import Candle, MarketSnapshot, OptionContract, OptionQuote, SCHEMA_VERSION, VersionedModel


class ReplaySpeedMode(str, Enum):
    STEP = "step"
    REALTIME = "realtime"
    ACCELERATED = "accelerated"
    DETERMINISTIC = "deterministic"


class MarketEventBase(VersionedModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime
    symbol: str = "SPY"
    source: str = "replay"
    event_type: str


class SpyQuoteEvent(MarketEventBase):
    event_type: Literal["spy_quote"] = "spy_quote"
    price: float
    bid: float | None = None
    ask: float | None = None
    data_classification: str = DataClassification.STOCK_MARKET_DATA.value


class SpyCandleEvent(MarketEventBase):
    event_type: Literal["spy_candle"] = "spy_candle"
    candle: Candle
    data_classification: str = DataClassification.STOCK_MARKET_DATA.value


class OptionQuoteEvent(MarketEventBase):
    event_type: Literal["option_quote"] = "option_quote"
    contract_id: str
    expiration: date
    strike: float
    option_type: Literal["call", "put"]
    bid: float
    ask: float
    mid: float
    spread_pct: float
    volume: int | None = None
    open_interest: int | None = None
    quote_timestamp: datetime
    data_classification: str = DataClassification.SYNTHETIC_DATA.value
    persist_allowed: bool = True
    openai_allowed: bool = True
    is_synthetic: bool = True
    delayed: bool = False
    received_at: datetime | None = None


class OptionChainSnapshot(MarketEventBase):
    event_type: Literal["option_chain"] = "option_chain"
    expiration: date
    contract_ids: list[str] = Field(default_factory=list)
    underlying_price: float


MarketEvent = Annotated[
    Union[SpyQuoteEvent, SpyCandleEvent, OptionQuoteEvent, OptionChainSnapshot],
    Field(discriminator="event_type"),
]


class ReplayMetadata(VersionedModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    trading_day: date
    symbol: str = "SPY"
    is_synthetic: bool = True
    description: str = ""
    event_count: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    source_file: str | None = None


class ReplayClock(BaseModel):
    """Tracks replay time independently of wall clock."""

    current_time: datetime
    trading_day: date
    speed_multiplier: float = 1.0
    mode: ReplaySpeedMode = ReplaySpeedMode.DETERMINISTIC

    def advance_to(self, ts: datetime) -> None:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=self.current_time.tzinfo)
        if ts < self.current_time:
            raise ValueError("Replay clock cannot go backwards")
        self.current_time = ts


class ReplaySession(VersionedModel):
    metadata: ReplayMetadata
    events: list[MarketEvent] = Field(default_factory=list)


class SelectedOptionContract(VersionedModel):
    contract_id: str
    contract: OptionContract
    quote: OptionQuote
    selection_reason: str
    underlying_price: float


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIME_STOP = "time_stop"
    INVALIDATION = "invalidation"
    END_OF_DAY = "end_of_day"


class ExitDecision(VersionedModel):
    position_id: str
    reason: ExitReason
    exit_price: float
    message: str = ""


class ReplaySummary(VersionedModel):
    run_id: str
    session_name: str
    is_synthetic: bool
    mock_agents: bool = True
    events_processed: int
    signals_detected: int
    trades_entered: int
    trades_exited: int
    final_pnl_usd: float
    risk_rejections: int
    option_selector_rejections: int = 0
    playbook_validation_approved: bool = False
    council_blocked: bool = False
    failures: list[str] = Field(default_factory=list)
