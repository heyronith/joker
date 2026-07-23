"""Textual TUI application."""

from __future__ import annotations

from datetime import date

from textual.app import App, ComposeResult
from textual.containers import Container, Grid, Vertical
from textual.widgets import Footer, Header, Input, Static

from joker.app.safety import SafetyMode
from joker.tui.panels import (
    render_council,
    render_daily_risk,
    render_events,
    render_market,
    render_playbook,
    render_real_data,
    render_options_data,
    render_system_mode,
    render_trade,
)
from joker.tui.state import DashboardState

PANEL_RENDERERS = [
    render_system_mode,
    render_daily_risk,
    render_market,
    render_real_data,
    render_options_data,
    render_council,
    render_playbook,
    render_trade,
    render_events,
]


class JokerApp(App[None]):
    """Main joker interactive terminal application."""

    TITLE = "joker"
    SUB_TITLE = "SPY 0DTE Research"

    CSS = """
    Screen {
        layout: vertical;
    }
    #dashboard-grid {
        height: 1fr;
        grid-size: 2 4;
        grid-gutter: 1;
        padding: 1;
    }
    .panel {
        border: solid $accent;
        padding: 1;
        height: auto;
        min-height: 5;
    }
    .panel-focused {
        border: solid $warning;
    }
    #communicator-box {
        height: auto;
        max-height: 8;
        border: solid $primary;
        padding: 1;
    }
    #communicator-input {
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("up", "select_prev", "Prev panel"),
        ("down", "select_next", "Next panel"),
        ("left", "select_prev", "Prev panel"),
        ("right", "select_next", "Next panel"),
        ("enter", "focus_communicator", "Communicator"),
        ("1", "set_mode('PAPER')", "Paper"),
        ("2", "set_mode('SHADOW')", "Shadow"),
        ("3", "set_mode('LIVE_GATED')", "Live gated"),
    ]

    def __init__(self, state: DashboardState | None = None) -> None:
        super().__init__()
        self.dashboard_state = state or DashboardState()
        self._panel_widgets: list[Static] = []
        self._run_active = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Grid(id="dashboard-grid"):
            for idx in range(len(PANEL_RENDERERS)):
                widget = Static("", id=f"panel-{idx}", classes="panel")
                self._panel_widgets.append(widget)
                yield widget
        with Vertical(id="communicator-box"):
            yield Static("[bold]Communicator Agent[/bold]", id="communicator-label")
            yield Input(placeholder="Ask the system anything…", id="communicator-input")
            yield Static("", id="communicator-history")
        yield Footer()

    def on_mount(self) -> None:
        self._run_active = True
        self.refresh_panels()

    def refresh_panels(self) -> None:
        for idx, renderer in enumerate(PANEL_RENDERERS):
            content = renderer(self.dashboard_state)
            widget = self._panel_widgets[idx]
            classes = "panel panel-focused" if idx == self.dashboard_state.selected_panel else "panel"
            widget.update(content)
            widget.set_classes(classes)
        history = self.dashboard_state.communicator_history[-3:]
        history_text = "\n".join(history) if history else "No messages yet."
        self.query_one("#communicator-history", Static).update(history_text)

    def action_select_prev(self) -> None:
        self.dashboard_state.selected_panel = (
            self.dashboard_state.selected_panel - 1
        ) % self.dashboard_state.panel_count
        self.refresh_panels()

    def action_select_next(self) -> None:
        self.dashboard_state.selected_panel = (
            self.dashboard_state.selected_panel + 1
        ) % self.dashboard_state.panel_count
        self.refresh_panels()

    def action_focus_communicator(self) -> None:
        self.query_one("#communicator-input", Input).focus()

    def action_set_mode(self, mode: str) -> None:
        self.dashboard_state.mode = SafetyMode.from_string(mode)
        if self.dashboard_state.mode is not SafetyMode.LIVE_GATED:
            self.dashboard_state.live_trading_enabled = False
        self.dashboard_state.event_lines.append(f"Mode set to {mode}")
        self.refresh_panels()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "communicator-input":
            return
        message = event.value.strip()
        if not message:
            return
        from joker.agents.communicator import CommunicatorAgent

        agent = CommunicatorAgent()
        state = {
            "mode": self.dashboard_state.mode.value,
            "trade_state": self.dashboard_state.trade_state,
            "council_status": self.dashboard_state.council_status,
            "playbook_title": self.dashboard_state.playbook_title,
            "playbook_status": self.dashboard_state.playbook_status,
            "market_price": self.dashboard_state.market_price,
            "kill_switch": self.dashboard_state.kill_switch,
        }
        reply = agent.answer(message, state)
        self.dashboard_state.communicator_history.append(f"You: {message}\nAgent: {reply}")
        self.dashboard_state.event_lines.append(f"user.message: {message[:80]}")
        event.input.value = ""
        self.refresh_panels()

    def action_quit(self) -> None:
        if self._run_active:
            self.dashboard_state.event_lines.append("system: safe shutdown initiated")
            self._run_active = False
        self.exit()
