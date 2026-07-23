"""Strengthened TUI tests."""

from __future__ import annotations

from joker.app.safety import SafetyMode
from joker.tui.app import JokerApp, PANEL_RENDERERS
from joker.tui.panels import render_daily_risk, render_trade
from joker.tui.state import DashboardState


def test_tui_app_initializes_paper_mode() -> None:
    app = JokerApp()
    assert app.dashboard_state.mode is SafetyMode.PAPER
    assert app.dashboard_state.live_trading_enabled is False


def test_panels_render_substantive_content() -> None:
    state = DashboardState(
        mode=SafetyMode.SHADOW,
        run_id="run-1",
        market_price=551.25,
        daily_pnl_usd=-120.50,
        trades_count=2,
        trade_state="RISK_CHECK",
        kill_switch=True,
    )
    mode_text = PANEL_RENDERERS[0](state)
    assert "SHADOW" in mode_text
    risk_text = render_daily_risk(state)
    assert "-120.50" in risk_text
    assert "Kill switch: ON" in risk_text
    trade_text = render_trade(state)
    assert "RISK_CHECK" in trade_text


def test_panel_navigation_wraps() -> None:
    state = DashboardState(selected_panel=8)
    state.selected_panel = (state.selected_panel + 1) % state.panel_count
    assert state.selected_panel == 0


def test_safe_quit_records_shutdown_event() -> None:
    app = JokerApp()
    app._run_active = True
    app.action_quit()
    assert app._run_active is False
    assert any("shutdown" in line for line in app.dashboard_state.event_lines)
