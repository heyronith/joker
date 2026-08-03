"""Domain schemas for joker trading objects."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1"


class VersionedModel(BaseModel):
    schema_version: str = SCHEMA_VERSION


class Candle(VersionedModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OptionContract(VersionedModel):
    symbol: str = "SPY"
    expiration: date
    strike: float
    option_type: Literal["call", "put"]
    is_0dte: bool = True

    @model_validator(mode="after")
    def validate_0dte(self) -> "OptionContract":
        if not self.is_0dte:
            raise ValueError("Only 0DTE options are supported in V1")
        return self


class OptionQuote(VersionedModel):
    contract: OptionContract
    bid: float
    ask: float
    last: float | None = None
    timestamp: datetime
    source: str = "synthetic"
    data_classification: str = "SYNTHETIC_DATA"
    persist_allowed: bool = True
    openai_allowed: bool = True
    is_synthetic: bool = True
    delayed: bool = False
    received_at: datetime | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        if mid <= 0:
            return 100.0
        return ((self.ask - self.bid) / mid) * 100.0


class MarketSnapshot(VersionedModel):
    symbol: str
    timestamp: datetime
    price: float
    bid: float | None = None
    ask: float | None = None
    candles: list[Candle] = Field(default_factory=list)


class MarketRegime(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHOP = "chop"
    UNKNOWN = "unknown"


class TechnicalFeatures(VersionedModel):
    symbol: str
    as_of: datetime
    vwap: float | None = None
    previous_high: float | None = None
    previous_low: float | None = None
    premarket_high: float | None = None
    premarket_low: float | None = None
    intraday_high: float | None = None
    intraday_low: float | None = None
    momentum_5m: float | None = None
    distance_from_vwap_pct: float | None = None
    trend_label: str = "unknown"
    volume_confirmed: bool | None = None
    is_stale: bool = False
    # Richer session context for DecisionAgent
    candle_count: int = 0
    opening_range_high: float | None = None
    opening_range_low: float | None = None
    distance_from_or_high_pct: float | None = None
    distance_from_or_low_pct: float | None = None
    distance_from_prev_high_pct: float | None = None
    distance_from_prev_low_pct: float | None = None
    vwap_upper_band: float | None = None
    vwap_lower_band: float | None = None
    momentum_15m: float | None = None
    range_15m_pct: float | None = None
    extension_label: str = "unknown"  # near_vwap | extended_up | extended_down | unknown
    minutes_from_open: float | None = None
    minutes_to_close: float | None = None
    day_part: str = "unknown"

class AgentOpinionMetadata(VersionedModel):
    """Structured agent extras — OpenAI strict JSON schema (no free-form dict)."""

    model_config = ConfigDict(extra="forbid")

    notes: str = Field(default="", description="Optional non-actionable notes from the agent")


class AgentOpinion(VersionedModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    regime: MarketRegime | None = None
    metadata: AgentOpinionMetadata = Field(default_factory=AgentOpinionMetadata)

    @field_validator("metadata", mode="before")
    @classmethod
    def _coerce_metadata(cls, value: object) -> object:
        if value is None or value == {}:
            return AgentOpinionMetadata()
        if isinstance(value, dict):
            return AgentOpinionMetadata(
                notes=str(value.get("notes", "") or ""),
            )
        return value


class CommunicatorResponse(VersionedModel):
    """Structured CommunicatorAgent reply — no broker actions."""

    answer: str
    data_available: bool = True
    refused_advice: bool = False
    refused_injection: bool = False


class AgentCouncilDecision(VersionedModel):
    run_id: str
    timestamp: datetime
    opinions: list[AgentOpinion]
    synthesis_summary: str
    playbook_id: str | None = None


class PlaybookSetup(VersionedModel):
    """Setup with structured entry rules the runtime can execute (not prose-only)."""

    model_config = ConfigDict(extra="forbid")

    setup_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    direction: Literal["long_call", "long_put"]
    enabled: bool = True
    entry_conditions: list[str] = Field(default_factory=list)
    stop_rule: str
    take_profit_rule: str
    # Structured rules — MarketEventHandler evaluates these
    require_trend: Literal["trend_up", "trend_down", "chop", "any"] = "any"
    vwap_side: Literal["above", "below", "either"] = "either"
    min_vwap_distance_pct: float = 0.0
    min_momentum_pct: float = 0.0
    max_momentum_pct: float | None = None
    stop_pct: float = Field(default=0.5, ge=0.05, le=0.95)
    take_profit_pct: float = Field(default=1.0, ge=0.1, le=5.0)
    time_stop_minutes: int = Field(default=30, ge=1, le=390)


class Playbook(VersionedModel):
    model_config = ConfigDict(extra="forbid")

    playbook_id: str = Field(default_factory=lambda: str(uuid4()))
    trading_day: date
    title: str
    summary: str
    setups: list[PlaybookSetup]
    approved: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PlaybookPatch(VersionedModel):
    model_config = ConfigDict(extra="forbid")

    patch_id: str = Field(default_factory=lambda: str(uuid4()))
    playbook_id: str
    author_agent: str
    reason: str
    disable_setup_ids: list[str] = Field(default_factory=list)
    enable_setup_ids: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TradeProposal(VersionedModel):
    """Agent-proposed entry intent. Under agent_led, soft risk caps are advisory only."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str = ""
    setup_id: str
    direction: Literal["long_call", "long_put"]
    propose_entry: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    stop_pct: float = Field(default=0.5, ge=0.05, le=0.95)
    take_profit_pct: float = Field(default=1.0, ge=0.1, le=5.0)


class IntradayCouncilResult(VersionedModel):
    """Structured intraday agent output: optional patch + optional trade proposal."""

    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    patch: PlaybookPatch | None = None
    proposal: TradeProposal | None = None


class SessionLesson(VersionedModel):
    """Postmarket lessons persisted for next-day memory (no raw OPRA)."""

    model_config = ConfigDict(extra="forbid")

    trading_day: date
    summary: str
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    next_day_hints: list[str] = Field(default_factory=list)
    final_pnl_usd: float = 0.0
    trades_entered: int = 0
    risk_rejections: int = 0


class DayMemoryBundle(VersionedModel):
    """Compact prior-session context injected into agent prompts."""

    model_config = ConfigDict(extra="forbid")

    as_of: date
    prior_lessons: list[SessionLesson] = Field(default_factory=list)
    recent_pnl_usd: float = 0.0
    recent_trade_count: int = 0
    recent_risk_reject_codes: list[str] = Field(default_factory=list)
    recent_playbook_titles: list[str] = Field(default_factory=list)
    memory_available: bool = False


class TradeCandidate(VersionedModel):
    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    setup_id: str
    contract: OptionContract
    quote: OptionQuote
    direction: Literal["long_call", "long_put"]
    entry_limit_price: float
    stop_price: float
    take_profit_price: float
    quantity: int = Field(default=1, ge=1, le=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RiskConfig(VersionedModel):
    max_daily_loss_usd: float
    max_trades_per_day: int
    max_open_positions: int
    max_premium_usd: float
    max_spread_pct: float
    quote_max_age_seconds: int
    allowed_symbol: str = "SPY"
    kill_switch: bool = False
    allow_delayed_quotes: bool = True
    feed_max_silence_seconds: int = 60
    delayed_quote_max_age_seconds: int = 900
    policy: Literal["strict", "agent_led"] = "strict"
    # Hard capital ceiling for the session (premium notional); 0 disables check
    authorized_capital_usd: float = 0.0
    reserved_capital_usd: float = 0.0


class RiskDecision(VersionedModel):
    candidate_id: str
    approved: bool
    reason_codes: list[str] = Field(default_factory=list)
    message: str = ""
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    policy: str = "strict"


class IntradayDecision(VersionedModel):
    """Realtime agent decision — primary authority under execution_mode=agent_led.

    Two-step entry: propose → confirm (abandon/hold clears). Direct 'enter' is
    accepted only as a confirm synonym when a pending proposal exists.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["hold", "propose", "confirm", "abandon", "enter"] = "hold"
    direction: Literal["long_call", "long_put"] | None = None
    setup_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    stop_pct: float = Field(default=0.5, ge=0.05, le=0.95)
    take_profit_pct: float = Field(default=1.0, ge=0.1, le=5.0)
    patch: PlaybookPatch | None = None
    summary: str = ""
    # Optional confirm conditions the agent wants checked (informational)
    confirm_note: str = ""
    # Capital allocation hints (enforced by CapitalBudget — never trusted raw)
    capital_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    target_contracts: int | None = Field(default=None, ge=1, le=100)
    allocation_style: Literal["auto", "aggressive", "split", "conservative"] = "auto"
    # Structured edge estimates (required on propose/confirm; enforced deterministically)
    win_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_r: float | None = None
    expected_value_usd: float | None = None


class OrderIntent(VersionedModel):
    intent_id: str = Field(default_factory=lambda: str(uuid4()))
    candidate_id: str
    contract: OptionContract
    side: Literal["buy", "sell"]
    order_type: Literal["limit", "market"] = "limit"
    quantity: int = 1
    limit_price: float | None = None
    # Explicit options position intent for broker submission (never inferred from side alone).
    position_intent: (
        Literal["BUY_TO_OPEN", "BUY_TO_CLOSE", "SELL_TO_OPEN", "SELL_TO_CLOSE"] | None
    ) = None


class BrokerOrder(VersionedModel):
    order_id: str
    intent_id: str
    status: Literal["pending", "open", "filled", "cancelled", "rejected"]
    contract: OptionContract
    side: Literal["buy", "sell"]
    quantity: int
    limit_price: float | None = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)


class Fill(VersionedModel):
    fill_id: str = Field(default_factory=lambda: str(uuid4()))
    order_id: str
    price: float
    quantity: int
    filled_at: datetime = Field(default_factory=datetime.utcnow)
    slippage_pct: float = 0.0


class Position(VersionedModel):
    position_id: str = Field(default_factory=lambda: str(uuid4()))
    contract: OptionContract
    quantity: int
    avg_entry_price: float
    is_open: bool = True
    opened_at: datetime = Field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None
    realized_pnl_usd: float | None = None


class DailyState(VersionedModel):
    trading_day: date
    run_id: str
    mode: str
    daily_pnl_usd: float = 0.0
    trades_count: int = 0
    open_positions: int = 0
    kill_switch: bool = False
    playbook_approved: bool = False
