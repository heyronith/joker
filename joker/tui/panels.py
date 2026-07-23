"""TUI panel widgets."""

from __future__ import annotations

from textual.widgets import Static

from joker.tui.state import DashboardState


class PanelBox(Static):
    """Base panel with title."""

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.panel_title = title


def render_system_mode(state: DashboardState) -> str:
    replay_line = ""
    if state.replay_mode:
        replay_line = f"\nReplay time: {state.replay_timestamp or '—'}"
        if state.replay_is_synthetic:
            replay_line += "\n[Synthetic replay — not real market data]"
    return (
        f"[bold]System Mode[/bold]\n"
        f"Mode: {state.mode.value}\n"
        f"{state.mode_label()}\n"
        f"Run ID: {state.run_id or '—'}\n"
        f"Day: {state.trading_day.isoformat()}"
        f"{replay_line}"
    )


def render_daily_risk(state: DashboardState) -> str:
    return (
        f"[bold]Daily Risk State[/bold]\n"
        f"P&L: ${state.daily_pnl_usd:,.2f}\n"
        f"Trades: {state.trades_count}\n"
        f"Open positions: {state.open_positions}\n"
        f"Max loss: ${state.max_daily_loss_usd:,.2f}\n"
        f"Kill switch: {'ON' if state.kill_switch else 'off'}"
    )


def render_market(state: DashboardState) -> str:
    price = f"${state.market_price:,.2f}" if state.market_price else "—"
    bid = f"${state.market_bid:,.2f}" if state.market_bid else "—"
    ask = f"${state.market_ask:,.2f}" if state.market_ask else "—"
    vwap = (
        f"{state.vwap_distance_pct:+.2f}%"
        if state.vwap_distance_pct is not None
        else "—"
    )
    opt = "unavailable" if not state.options_data_available else "—"
    return (
        f"[bold]Market State[/bold]\n"
        f"Symbol: {state.market_symbol}\n"
        f"Price: {price}\n"
        f"Bid/Ask: {bid} / {ask}\n"
        f"Option data: {opt}\n"
        f"Regime: {state.market_regime}\n"
        f"VWAP dist: {vwap}"
    )


def render_real_data(state: DashboardState) -> str:
    price = f"${state.market_price:,.2f}" if state.market_price else "—"
    bid = f"${state.market_bid:,.2f}" if state.market_bid else "—"
    ask = f"${state.market_ask:,.2f}" if state.market_ask else "—"
    warning = state.permission_warning or "—"
    return (
        f"[bold]Real Data Status[/bold]\n"
        f"Provider: {state.data_provider}\n"
        f"Data mode: {state.data_mode}\n"
        f"SPY price: {price}\n"
        f"Bid/Ask: {bid} / {ask}\n"
        f"Last update: {state.last_data_update or '—'}\n"
        f"Feed health: {state.feed_health}\n"
        f"Subscription warning: {warning}\n"
        f"Market data only: {'yes' if state.market_data_only else 'no'}\n"
        f"Broker execution: {'enabled' if state.broker_execution_enabled else 'disabled'}"
    )


def render_options_data(state: DashboardState) -> str:
    call = state.selected_call_contract or "—"
    put = state.selected_put_contract or "—"
    unavail = ", ".join(state.options_unavailable_fields) or "—"
    return (
        f"[bold]Options Data[/bold]\n"
        f"Provider: {state.options_provider_name}\n"
        f"0DTE chain: {'available' if state.odte_chain_available else 'unavailable'}\n"
        f"Call: {call}\n"
        f"  bid/ask/mid/spread: "
        f"{state.call_bid or '—'}/{state.call_ask or '—'}/"
        f"{state.call_mid or '—'}/{state.call_spread_pct or '—'}\n"
        f"Put: {put}\n"
        f"  bid/ask/mid/spread: "
        f"{state.put_bid or '—'}/{state.put_ask or '—'}/"
        f"{state.put_mid or '—'}/{state.put_spread_pct or '—'}\n"
        f"Quote time: {state.option_quote_timestamp or '—'}\n"
        f"Unavailable fields: {unavail}\n"
        f"Subscription warning: {state.permission_warning or '—'}\n"
        f"[dim]{state.opra_display_notice}[/dim]"
    )


def render_council(state: DashboardState) -> str:
    return (
        f"[bold]Agent Council[/bold]\n"
        f"Status: {state.council_status}\n"
        f"{state.council_summary}"
    )


def render_playbook(state: DashboardState) -> str:
    return (
        f"[bold]Active Playbook[/bold]\n"
        f"{state.playbook_title}\n"
        f"Status: {state.playbook_status}\n"
        f"{state.playbook_summary or '—'}"
    )


def render_trade(state: DashboardState) -> str:
    risk = state.last_risk_decision or "—"
    return (
        f"[bold]Trade State[/bold]\n"
        f"State: {state.trade_state}\n"
        f"Order: {state.active_order_id or '—'}\n"
        f"Risk: {risk}\n"
        f"{state.position_summary}"
    )


def render_events(state: DashboardState) -> str:
    lines = state.event_lines[-8:] or ["No events yet."]
    body = "\n".join(lines)
    return f"[bold]Event Log[/bold]\n{body}"
