"""Interactive session capital / goal confirmation for paper runs."""

from __future__ import annotations

from dataclasses import dataclass

import typer
from rich.console import Console
from rich.table import Table

from joker.config.settings import CapitalSettings
from joker.risk.capital import CapitalBudget, CapitalPlan


@dataclass(frozen=True)
class SessionCapitalInput:
    authorized_usd: float
    target_profit_pct: float
    max_concurrent_positions: int
    max_contracts_per_trade: int
    pause_entries_when_goal_met: bool = True


def plan_from_settings(settings: CapitalSettings) -> CapitalPlan:
    return CapitalPlan(
        authorized_usd=float(settings.authorized_usd),
        target_profit_pct=float(settings.target_profit_pct),
        max_concurrent_positions=int(settings.max_concurrent_positions),
        max_contracts_per_trade=int(settings.max_contracts_per_trade),
        min_contracts_per_trade=int(settings.min_contracts_per_trade),
        aggression_mode=str(settings.aggression_mode),
        max_kelly_fraction=float(settings.max_kelly_fraction),
        min_win_probability=float(settings.min_win_probability),
        behind_goal_boost=float(settings.behind_goal_boost),
        ahead_goal_dampen=float(settings.ahead_goal_dampen),
    )


def confirm_session_capital(
    settings: CapitalSettings,
    *,
    console: Console | None = None,
    authorized_usd: float | None = None,
    target_profit_pct: float | None = None,
    max_concurrent_positions: int | None = None,
    yes: bool = False,
) -> CapitalBudget:
    """
    Confirm daily authorized capital and profit goal before the session arms.

    Pass --yes / CLI values to skip prompts (automation / tests).
    """
    out = console or Console()
    auth = float(authorized_usd if authorized_usd is not None else settings.authorized_usd)
    target = float(
        target_profit_pct if target_profit_pct is not None else settings.target_profit_pct
    )
    concurrent = int(
        max_concurrent_positions
        if max_concurrent_positions is not None
        else settings.max_concurrent_positions
    )

    if not yes and settings.require_session_confirm:
        out.print("\n[bold]Session capital & goal confirmation[/bold]")
        out.print(
            "[dim]Authorized capital = max premium you are willing to risk today "
            "(paper/sandbox). Not live money.[/dim]"
        )
        auth = typer.prompt("Authorized capital to risk today (USD)", default=auth, type=float)
        target = typer.prompt(
            "Target profit % on authorized capital",
            default=target,
            type=float,
        )
        concurrent = typer.prompt(
            "Max concurrent positions (1 recommended for v1 exits)",
            default=concurrent,
            type=int,
        )
        if concurrent != 1:
            out.print(
                "[yellow]Note: exit manager currently tracks one open trade; "
                "keep concurrent=1 unless you accept limited multi-position support.[/yellow]"
            )
            concurrent = 1

    if auth <= 0:
        raise typer.BadParameter("authorized capital must be > 0")
    if target < 0:
        raise typer.BadParameter("target profit % must be >= 0")

    plan = CapitalPlan(
        authorized_usd=auth,
        target_profit_pct=target,
        max_concurrent_positions=max(1, concurrent),
        max_contracts_per_trade=int(settings.max_contracts_per_trade),
        min_contracts_per_trade=int(settings.min_contracts_per_trade),
        aggression_mode=str(settings.aggression_mode),
        max_kelly_fraction=float(settings.max_kelly_fraction),
        min_win_probability=float(settings.min_win_probability),
        behind_goal_boost=float(settings.behind_goal_boost),
        ahead_goal_dampen=float(settings.ahead_goal_dampen),
    )
    budget = CapitalBudget(plan=plan)

    table = Table(title="Confirmed session goals")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Authorized capital", f"${plan.authorized_usd:,.2f}")
    table.add_row("Target profit %", f"{plan.target_profit_pct:.1f}%")
    table.add_row("Target profit $", f"${plan.target_profit_usd:,.2f}")
    table.add_row("Max concurrent positions", str(plan.max_concurrent_positions))
    table.add_row("Max contracts / trade", str(plan.max_contracts_per_trade))
    out.print(table)

    if not yes and settings.require_session_confirm:
        ok = typer.confirm("Proceed with this capital & goal?", default=True)
        if not ok:
            raise typer.Abort()

    return budget
