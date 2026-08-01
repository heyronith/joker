"""Interactive session capital / goal / objective confirmation for paper runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from joker.config.settings import AppSettings, CapitalSettings
from joker.objectives.deadline import DeadlineParseError, resolve_deadline, time_remaining_seconds
from joker.objectives.events import BoundedOperatorEventProjection
from joker.objectives.feasibility import GoalFeasibilityEngine
from joker.objectives.repository import ObjectiveRepository
from joker.objectives.scoring import ObjectiveStrategyScorer
from joker.objectives.service import SessionObjectiveService
from joker.objectives.sizing import DeterministicObjectiveSizer
from joker.risk.capital import CapitalBudget, CapitalPlan


@dataclass
class SessionObjectiveBundle:
    """Confirmed objective + legacy CapitalBudget mirror for MarketEventHandler."""

    capital_budget: CapitalBudget
    objective_service: SessionObjectiveService | None = None
    objective_id: str | None = None
    deadline_exchange_time: datetime | None = None


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
    """Legacy capital confirmation (objective-disabled profiles)."""
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


async def confirm_session_objective(
    app_settings: AppSettings,
    *,
    session_id: str,
    db_path: Path,
    console: Console | None = None,
    authorized_usd: float | None = None,
    target_profit_pct: float | None = None,
    target_deadline: str | None = None,
    max_concurrent_positions: int | None = None,
    acknowledge_total_loss: bool = False,
    yes: bool = False,
    exchange_tz: str | None = None,
) -> SessionObjectiveBundle:
    """Confirm and persist a durable Task-1 session objective before Task 2 starts."""
    out = console or Console()
    obj_settings = app_settings.objective
    capital = app_settings.capital
    tz = exchange_tz or str(app_settings.exchange.timezone)

    if yes:
        missing: list[str] = []
        if authorized_usd is None:
            missing.append("--authorized-capital")
        if target_profit_pct is None:
            missing.append("--target-profit-pct")
        if obj_settings.require_deadline and not target_deadline:
            missing.append("--target-deadline")
        if max_concurrent_positions is None:
            missing.append("--max-concurrent-positions")
        if obj_settings.require_total_loss_acknowledgement and not acknowledge_total_loss:
            missing.append("--acknowledge-total-loss")
        if missing:
            raise typer.BadParameter(
                "--yes requires explicit values for: " + ", ".join(missing)
            )

    auth = authorized_usd
    target = target_profit_pct
    concurrent = max_concurrent_positions
    deadline_raw = target_deadline
    ack = acknowledge_total_loss

    if not yes:
        out.print("\n[bold]Session objective confirmation[/bold]")
        out.print(
            "[dim]Authorised capital = max premium at risk in paper/sandbox. "
            "Never inferred from broker buying power.[/dim]"
        )
        auth = float(
            typer.prompt(
                "Authorised capital (USD)",
                default=float(auth if auth is not None else capital.authorized_usd),
                type=float,
            )
        )
        target = float(
            typer.prompt(
                "Target profit %",
                default=float(target if target is not None else capital.target_profit_pct),
                type=float,
            )
        )
        deadline_raw = str(
            typer.prompt(
                "Target deadline (e.g. 15:30 ET or ISO timestamp)",
                default=deadline_raw or "15:30 ET",
            )
        )
        concurrent = int(
            typer.prompt(
                "Max concurrent positions",
                default=int(
                    concurrent
                    if concurrent is not None
                    else obj_settings.default_max_concurrent_positions
                ),
                type=int,
            )
        )
        if obj_settings.require_total_loss_acknowledgement:
            ack = typer.confirm(
                f"I acknowledge that the full authorised capital "
                f"(${auth:,.2f}) may be lost in this paper account",
                default=False,
            )

    if auth is None or float(auth) <= 0:
        raise typer.BadParameter("authorised capital must be > 0")
    if target is None or float(target) < 0:
        raise typer.BadParameter("target profit % must be >= 0")
    if concurrent is None or int(concurrent) < 1:
        raise typer.BadParameter("max concurrent positions must be >= 1")
    if obj_settings.require_deadline and not deadline_raw:
        raise typer.BadParameter("target deadline is required")
    if obj_settings.require_total_loss_acknowledgement and not ack:
        raise typer.BadParameter("total-loss acknowledgement is required")

    try:
        deadline = resolve_deadline(str(deadline_raw), exchange_tz=tz)
    except DeadlineParseError as exc:
        raise typer.BadParameter(str(exc)) from exc

    profit = Decimal(str(auth)) * Decimal(str(target)) / Decimal("100")
    ending = Decimal(str(auth)) + profit
    remaining = time_remaining_seconds(deadline, exchange_tz=tz)

    table = Table(title="Resolved session objective")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Authorised capital", f"${float(auth):,.2f}")
    table.add_row("Target profit %", f"{float(target):.1f}%")
    table.add_row("Target profit $", f"${float(profit):,.2f}")
    table.add_row("Target ending equity", f"${float(ending):,.2f}")
    table.add_row("Deadline (exchange)", deadline.isoformat())
    table.add_row("Time remaining", f"{remaining}s")
    table.add_row("Max concurrent positions", str(int(concurrent)))
    table.add_row("Total-loss acknowledged", "yes" if ack else "no")
    out.print(table)

    if not yes:
        ok = typer.confirm("Confirm and arm this objective?", default=True)
        if not ok:
            raise typer.Abort()

    repo = ObjectiveRepository(db_path)
    events = BoundedOperatorEventProjection(
        capacity=int(getattr(obj_settings, "operator_event_capacity", 256))
    )
    service = SessionObjectiveService(
        repo,
        exchange_tz=tz,
        operator_events=events,
        pause_entries_when_goal_met=bool(obj_settings.pause_entries_when_goal_met),
        stop_new_entries_at_deadline=bool(obj_settings.stop_new_entries_at_deadline),
        require_positive_expected_value=bool(obj_settings.require_positive_expected_value),
        minimum_win_probability=float(obj_settings.minimum_win_probability),
    )
    definition = await service.create_objective(
        session_id=session_id,
        authorised_capital_usd=auth,
        target_profit_pct=target,
        deadline_exchange_time=deadline,
        max_concurrent_positions=int(concurrent),
        accepted_total_loss_risk=bool(ack),
        pause_entries_when_goal_met=bool(obj_settings.pause_entries_when_goal_met),
    )
    state = await service.confirm_objective(definition.objective_id)
    out.print(
        f"[green]Objective confirmed[/green] id={definition.objective_id} "
        f"status={state.status} version={state.version}"
    )

    plan = CapitalPlan(
        authorized_usd=float(auth),
        target_profit_pct=float(target),
        max_concurrent_positions=int(concurrent),
        max_contracts_per_trade=int(
            getattr(obj_settings, "maximum_authorised_contracts", capital.max_contracts_per_trade)
        ),
        min_contracts_per_trade=int(capital.min_contracts_per_trade),
        aggression_mode=str(capital.aggression_mode),
        max_kelly_fraction=float(capital.max_kelly_fraction),
        min_win_probability=float(obj_settings.minimum_win_probability),
        behind_goal_boost=float(capital.behind_goal_boost),
        ahead_goal_dampen=float(capital.ahead_goal_dampen),
    )
    return SessionObjectiveBundle(
        capital_budget=CapitalBudget(plan=plan),
        objective_service=service,
        objective_id=str(definition.objective_id),
        deadline_exchange_time=deadline,
    )


def build_objective_engines(
    app_settings: AppSettings,
    *,
    episode_repository: Any | None = None,
    evaluation_repository: Any | None = None,
    dataset_repository: Any | None = None,
    objective_repository: ObjectiveRepository | None = None,
):
    """Construct objective engines with explicit Task-3 repository injection.

    Do not guess evolution DB paths from settings. Callers that already own
    initialized Task-3 repositories must pass those same instances.

    Returns:
        ObjectiveEngineBundle
    """
    from joker.objectives.engine_bundle import (
        HistoricalSourceDiagnostic,
        ObjectiveEngineBundle,
    )
    from joker.objectives.historical_outcomes import (
        HistoricalOutcomeService,
        build_historical_outcome_service_from_evolution_repos,
    )

    obj = app_settings.objective
    capital = app_settings.capital
    hist_settings = obj.historical_outcomes

    diagnostic_reason: str | None = None
    if episode_repository is not None and evaluation_repository is not None:
        historical_service = build_historical_outcome_service_from_evolution_repos(
            episode_repo=episode_repository,
            evaluation_repo=evaluation_repository,
            dataset_repo=dataset_repository,
            settings=hist_settings,
            repository=objective_repository,
        )
    else:
        diagnostic_reason = (
            "task3_repositories_unavailable:"
            f"episodes={episode_repository is not None},"
            f"evaluations={evaluation_repository is not None}"
        )
        historical_service = HistoricalOutcomeService(
            settings=hist_settings,
            repository=objective_repository,
            source_diagnostic_reason=diagnostic_reason,
        )

    if objective_repository is not None:
        historical_service.attach_objective_repository(objective_repository)

    diagnostic = HistoricalSourceDiagnostic(
        episode_loader_configured=historical_service.uses_repository_loaders,
        evaluation_loader_configured=historical_service.uses_repository_loaders,
        dataset_loader_configured=historical_service.uses_dataset_loader,
        objective_repository_attached=historical_service.objective_repository_attached,
        reason=diagnostic_reason or historical_service.source_diagnostic_reason,
        cold_start=not historical_service.uses_repository_loaders,
    )

    return ObjectiveEngineBundle(
        feasibility_engine=GoalFeasibilityEngine(
            minimum_samples_for_numeric_probability=int(
                obj.feasibility.minimum_samples_for_numeric_probability
            ),
        ),
        objective_strategy_scorer=ObjectiveStrategyScorer(
            require_positive_expected_value=bool(obj.require_positive_expected_value),
            minimum_win_probability=float(obj.minimum_win_probability),
            allow_ordinal_when_probability_unavailable=bool(
                obj.feasibility.allow_ordinal_scoring_when_probability_unavailable
            ),
        ),
        capital_sizer=DeterministicObjectiveSizer(
            max_capital_fraction=float(obj.sizing.max_capital_fraction),
            max_probe_fraction=float(obj.sizing.max_probe_fraction),
            prohibit_loss_multiplier=bool(obj.sizing.prohibit_loss_multiplier),
            minimum_win_probability=float(obj.minimum_win_probability),
            require_positive_expected_value=bool(obj.require_positive_expected_value),
            maximum_authorised_contracts=int(obj.maximum_authorised_contracts),
            min_contracts=int(capital.min_contracts_per_trade),
            aggression_mode=str(capital.aggression_mode),
            max_kelly_fraction=float(capital.max_kelly_fraction),
            behind_goal_boost=float(capital.behind_goal_boost),
            ahead_goal_dampen=float(capital.ahead_goal_dampen),
        ),
        historical_outcome_service=historical_service,
        historical_outcome_settings=hist_settings,
        source_diagnostic=diagnostic,
    )
