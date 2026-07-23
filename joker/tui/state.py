"""Dashboard view state for TUI panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from joker.app.safety import SafetyMode


@dataclass
class DashboardState:
    mode: SafetyMode = SafetyMode.PAPER
    live_trading_enabled: bool = False
    trading_day: date = field(default_factory=date.today)
    run_id: str | None = None

    daily_pnl_usd: float = 0.0
    trades_count: int = 0
    open_positions: int = 0
    kill_switch: bool = False
    max_daily_loss_usd: float = 100.0

    market_symbol: str = "SPY"
    market_price: float | None = None
    market_bid: float | None = None
    market_ask: float | None = None
    market_regime: str = "unknown"
    vwap_distance_pct: float | None = None

    # Real data status (Phase 18)
    data_provider: str = "mock"
    data_mode: str = "offline"
    last_data_update: str | None = None
    feed_health: str = "OK"
    permission_warning: str | None = None
    market_data_only: bool = True
    broker_execution_enabled: bool = False
    options_data_available: bool = False
    options_provider_name: str = "unavailable"
    odte_chain_available: bool = False
    selected_call_contract: str | None = None
    selected_put_contract: str | None = None
    call_bid: float | None = None
    call_ask: float | None = None
    call_mid: float | None = None
    call_spread_pct: float | None = None
    put_bid: float | None = None
    put_ask: float | None = None
    put_mid: float | None = None
    put_spread_pct: float | None = None
    option_quote_timestamp: str | None = None
    options_unavailable_fields: list[str] = field(default_factory=list)

    council_status: str = "idle"
    council_summary: str = "No council session active."

    playbook_title: str = "No active playbook"
    playbook_status: str = "unapproved"
    playbook_summary: str = ""

    trade_state: str = "IDLE"
    active_order_id: str | None = None
    position_summary: str = "Flat"

    # Replay mode (Phase 16)
    replay_mode: bool = False
    replay_timestamp: str | None = None
    replay_is_synthetic: bool = False
    latest_option_mid: float | None = None
    last_risk_decision: str | None = None
    last_replay_event: str | None = None

    event_lines: list[str] = field(default_factory=list)
    communicator_input: str = ""
    communicator_history: list[str] = field(default_factory=list)

    selected_panel: int = 0
    panel_count: int = 9
    opra_display_notice: str = "OPRA values displayed locally only; not stored."

    def display_state_ephemeral(self) -> dict[str, Any]:
        """In-memory/TUI-only state including raw OPRA option values."""
        return {
            "call_bid": self.call_bid,
            "call_ask": self.call_ask,
            "call_mid": self.call_mid,
            "call_spread_pct": self.call_spread_pct,
            "put_bid": self.put_bid,
            "put_ask": self.put_ask,
            "put_mid": self.put_mid,
            "put_spread_pct": self.put_spread_pct,
            "option_quote_timestamp": self.option_quote_timestamp,
            "latest_option_mid": self.latest_option_mid,
            "selected_call_contract": self.selected_call_contract,
            "selected_put_contract": self.selected_put_contract,
            "opra_notice": self.opra_display_notice,
        }

    def persisted_state_safe(self) -> dict[str, Any]:
        """Safe subset for logs/reports — no raw OPRA values."""
        from joker.compliance.opra_sanitizer import sanitize_for_persistence

        return sanitize_for_persistence(
            {
                "data_provider": self.data_provider,
                "options_data_available": self.options_data_available,
                "options_provider_name": self.options_provider_name,
                "odte_chain_available": self.odte_chain_available,
                "call_contract_selected": bool(self.selected_call_contract),
                "put_contract_selected": bool(self.selected_put_contract),
                "options_unavailable_fields": self.options_unavailable_fields,
                "bid_ask_available": bool(self.call_bid and self.call_ask),
                "spread_check": "PASS" if self.call_spread_pct is not None else "FAIL",
                "freshness_check": "PASS" if self.option_quote_timestamp else "FAIL",
            }
        )

    def mode_label(self) -> str:
        if self.replay_mode:
            label = "REPLAY — synthetic data" if self.replay_is_synthetic else "REPLAY"
            return label
        if self.mode is SafetyMode.SHADOW:
            return "SHADOW — no orders submitted"
        if self.mode is SafetyMode.LIVE_GATED and self.live_trading_enabled:
            return "LIVE_GATED — broker enabled"
        if self.mode is SafetyMode.LIVE_GATED:
            return "LIVE_GATED — broker disabled"
        return "PAPER — simulated execution"
