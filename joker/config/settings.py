"""Application settings and configuration schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from joker.app.safety import SafetyMode


class RiskSettings(BaseModel):
    max_daily_loss_usd: float = 100.0
    max_trades_per_day: int = 1
    max_open_positions: int = 1
    max_premium_usd: float = 100.0
    max_spread_pct: float = 15.0
    quote_max_age_seconds: int = 30
    allowed_symbol: str = "SPY"
    kill_switch: bool = False
    # Webull OpenAPI often serves delayed quotes; allow with feed-silence checks.
    allow_delayed_quotes: bool = True
    feed_max_silence_seconds: int = 60
    delayed_quote_max_age_seconds: int = 900
    # strict = all soft caps enforce; agent_led = hard floors only (paper/sandbox).
    policy: str = "strict"  # strict | agent_led


class PaperSettings(BaseModel):
    initial_balance_usd: float = 25000.0
    slippage_pct: float = 2.0
    default_spread_pct: float = 5.0


class CapitalSettings(BaseModel):
    """Daily authorized trading capital and profit goal (paper/sandbox risk budget)."""

    # Defaults used when CLI confirmation is skipped via flags
    authorized_usd: float = 500.0
    target_profit_pct: float = 20.0
    max_concurrent_positions: int = 1
    max_contracts_per_trade: int = 20
    min_contracts_per_trade: int = 1
    # Require interactive confirm on `joker paper run` unless --yes / flags set
    require_session_confirm: bool = True
    # Stop seeking new entries after daily profit goal is met (fail-safe greed brake)
    pause_entries_when_goal_met: bool = True
    # EV / aggression policy
    aggression_mode: str = "goal_adaptive"  # fixed | goal_adaptive
    max_kelly_fraction: float = 0.35
    min_win_probability: float = 0.45
    behind_goal_boost: float = 0.15
    ahead_goal_dampen: float = 0.15


class AgentSettings(BaseModel):
    mock_agents: bool = True
    council_timeout_seconds: int = 120
    max_retries: int = 2
    # Agent runtime: cognitive_graph | legacy | null
    runtime: str = "legacy"
    # Agentic paper loop
    # rules_hybrid = structured rules can auto-enter; agent_led = AI is sole entry authority
    execution_mode: str = "rules_hybrid"  # rules_hybrid | agent_led
    intraday_enabled: bool = True
    intraday_interval_seconds: float = 300.0
    decision_interval_seconds: float = 45.0
    max_intraday_calls_per_session: int = 12
    max_decision_calls_per_session: int = 40
    max_proposals_per_session: int = 3
    min_proposal_confidence: float = 0.55
    # Two-step propose → confirm
    require_propose_before_enter: bool = True
    confirm_ttl_seconds: float = 120.0
    max_confirm_spy_drift_pct: float = 0.20
    max_confirm_option_mid_worsen_pct: float = 15.0
    # Latency: skip LLM when no cheap edge; fast-confirm pending proposals
    use_edge_prefilter: bool = True
    fast_confirm_min_seconds: float = 8.0
    memory_lookback_days: int = 5
    postmarket_learner_enabled: bool = True


class LoggingSettings(BaseModel):
    level: str = "INFO"
    redact_env_keys: list[str] = Field(
        default_factory=lambda: [
            "OPENAI_API_KEY",
            "WEBULL_APP_KEY",
            "WEBULL_APP_SECRET",
            "WEBULL_TRADE_PIN",
            "WEBULL_ACCESS_TOKEN",
            "WEBULL_PAPER_ACCOUNT_ID",
            "WEBULL_TRADE_APP_KEY",
            "WEBULL_TRADE_APP_SECRET",
            "WEBULL_TRADE_ACCESS_TOKEN",
        ]
    )


class BrokerSettings(BaseModel):
    """Broker selection. Default local PaperBroker; webull_paper uses Webull paper account."""

    provider: str = "paper"  # paper | webull_paper
    require_explicit_confirmation: bool = True
    webull_paper_trading_enabled: bool = False


class DataSettings(BaseModel):
    default_provider: str = "webull"
    quote_poll_interval_seconds: float = 1.0
    # Task 1 observation/bar settings (aliases under market_data in YAML)
    observation_poll_seconds: float = 1.0
    bar_timeframes: list[str] = Field(default_factory=lambda: ["1m", "5m"])
    late_observation_tolerance_seconds: float = 2.0


class ExchangeSettings(BaseModel):
    """Canonical exchange clock/calendar configuration."""

    calendar: str = "XNYS"
    timezone: str = "America/New_York"


class MarketDataSettings(BaseModel):
    observation_poll_seconds: float = 1.0
    bar_timeframes: list[str] = Field(default_factory=lambda: ["1m", "5m"])
    late_observation_tolerance_seconds: float = 2.0


class DataQualitySettings(BaseModel):
    underlying_stale_seconds: float = 5.0
    option_stale_seconds: float = 10.0
    maximum_relative_spread: float = 0.25


class RuntimeSettings(BaseModel):
    event_handler_timeout_seconds: float = 10.0
    reconciliation_interval_seconds: float = 5.0


class CognitivePositionSettings(BaseModel):
    enabled: bool = True
    minimum_reassessment_interval_seconds: float = 5.0
    prioritise_over_new_entries: bool = True


class CognitiveContextSettings(BaseModel):
    max_1m_bars: int = 60
    max_5m_bars: int = 36
    maximum_option_rows_per_request: int = 80
    maximum_context_characters: int = 60_000


class CognitiveGraphSettings(BaseModel):
    enabled: bool = True
    max_parallel_agents: int = 5
    max_cycle_seconds: int = 90
    max_debate_rounds: int = 2
    max_strategy_candidates: int = 3
    max_hypotheses_per_cycle: int = 5
    max_strategy_switches: int = 1
    max_agent_data_requests: int = 1
    market_snapshot_coalescing: bool = True
    legacy_fallback_enabled: bool = False
    position: CognitivePositionSettings = Field(default_factory=CognitivePositionSettings)
    context: CognitiveContextSettings = Field(default_factory=CognitiveContextSettings)


class PersistenceSettings(BaseModel):
    database_url: str = "sqlite:///data/joker.db"
    checkpoint_database_url: str = "sqlite:///data/joker_checkpoints.db"


def _default_models_config() -> Any:
    from joker.models.schemas import ModelsConfig

    return ModelsConfig()


class AppSettings(BaseModel):
    """Merged application settings from YAML config and environment."""

    mode: SafetyMode = SafetyMode.PAPER
    live_trading_enabled: bool = False
    data_dir: Path = Path("data")
    db_path: Path = Path("data/joker.db")
    event_log_dir: Path = Path("data/logs/jsonl")
    reports_dir: Path = Path("data/reports")
    risk: RiskSettings = Field(default_factory=RiskSettings)
    paper: PaperSettings = Field(default_factory=PaperSettings)
    capital: CapitalSettings = Field(default_factory=CapitalSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)
    data_quality: DataQualitySettings = Field(default_factory=DataQualitySettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    persistence: PersistenceSettings = Field(default_factory=PersistenceSettings)
    cognitive_graph: CognitiveGraphSettings = Field(default_factory=CognitiveGraphSettings)
    models: Any = Field(default_factory=_default_models_config)

    @field_validator("mode", mode="before")
    @classmethod
    def _parse_mode(cls, value: Any) -> SafetyMode:
        if isinstance(value, SafetyMode):
            return value
        return SafetyMode.from_string(str(value))

    @field_validator("data_dir", "db_path", "event_log_dir", "reports_dir", mode="before")
    @classmethod
    def _parse_path(cls, value: Any) -> Path:
        return Path(value)

    def model_post_init(self, __context: Any) -> None:
        if self.live_trading_enabled and self.mode is not SafetyMode.LIVE_GATED:
            raise ValueError(
                "live_trading_enabled requires mode LIVE_GATED. "
                "Refusing to enable live trading in paper/shadow mode."
            )


class EnvSettings(BaseSettings):
    """Environment-backed secrets and overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5.4-mini", alias="OPENAI_MODEL")
    joker_config: str = Field(default="config/paper.yaml", alias="JOKER_CONFIG")
    joker_data_dir: str | None = Field(default=None, alias="JOKER_DATA_DIR")
    joker_log_level: str = Field(default="INFO", alias="JOKER_LOG_LEVEL")
    webull_app_key: str | None = Field(default=None, alias="WEBULL_APP_KEY")
    webull_app_secret: str | None = Field(default=None, alias="WEBULL_APP_SECRET")
    webull_device_id: str | None = Field(default=None, alias="WEBULL_DEVICE_ID")
    webull_trade_pin: str | None = Field(default=None, alias="WEBULL_TRADE_PIN")
    webull_region: str = Field(default="US", alias="WEBULL_REGION")
    webull_api_env: str = Field(default="uat", alias="WEBULL_API_ENV")
    webull_access_token: str | None = Field(default=None, alias="WEBULL_ACCESS_TOKEN")
    webull_market_data_enabled: bool = Field(default=False, alias="WEBULL_MARKET_DATA_ENABLED")
    webull_live_trading_enabled: bool = Field(default=False, alias="WEBULL_LIVE_TRADING_ENABLED")
    # Paper-account broker orders only (not real-money live).
    webull_paper_trading_enabled: bool = Field(
        default=False, alias="WEBULL_PAPER_TRADING_ENABLED"
    )
    webull_paper_account_id: str | None = Field(
        default=None, alias="WEBULL_PAPER_ACCOUNT_ID"
    )
    # Optional separate OpenAPI paper/sandbox trade credentials.
    # When set, market data can stay on prod while orders use sandbox/papertrade keys.
    webull_trade_app_key: str | None = Field(default=None, alias="WEBULL_TRADE_APP_KEY")
    webull_trade_app_secret: str | None = Field(
        default=None, alias="WEBULL_TRADE_APP_SECRET"
    )
    webull_trade_app_id: str | None = Field(default=None, alias="WEBULL_TRADE_APP_ID")
    webull_trade_api_env: str | None = Field(default=None, alias="WEBULL_TRADE_API_ENV")
    webull_trade_access_token: str | None = Field(
        default=None, alias="WEBULL_TRADE_ACCESS_TOKEN"
    )

    @field_validator("webull_live_trading_enabled")
    @classmethod
    def _reject_live_trading_flag(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "WEBULL_LIVE_TRADING_ENABLED must remain false — "
                "real-money broker execution is not enabled in this release. "
                "Use WEBULL_PAPER_TRADING_ENABLED for paper-account orders."
            )
        return value

    def trade_credentials_env(self) -> "EnvSettings":
        """
        Env view used by the trade HTTP client.

        Prefers WEBULL_TRADE_* when present so prod market-data keys can stay unchanged.
        """
        updates: dict[str, object] = {}
        if self.webull_trade_app_key:
            updates["webull_app_key"] = self.webull_trade_app_key
        if self.webull_trade_app_secret:
            updates["webull_app_secret"] = self.webull_trade_app_secret
        if self.webull_trade_api_env:
            updates["webull_api_env"] = self.webull_trade_api_env.strip().lower()
        elif self.webull_paper_trading_enabled and not self.webull_trade_app_key:
            # Same keys for trade: still prefer sandbox host if user set API_ENV=sandbox.
            pass
        if self.webull_trade_access_token:
            updates["webull_access_token"] = self.webull_trade_access_token
        if not updates:
            return self
        return self.model_copy(update=updates)
