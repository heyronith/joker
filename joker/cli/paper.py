"""Live paper trading CLI — real Webull data + auto paper orders."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from joker.broker.factory import webull_paper_env_ready
from joker.runtime.live_cli import format_live_event

paper_app = typer.Typer(
    help="Live paper trading: real Webull data, agentic decisions, auto paper orders"
)
console = Console()


def _broker_status_row(env) -> tuple[str, str]:
    if webull_paper_env_ready(env):
        return (
            "Broker",
            "[green]Webull paper account (auto orders)[/green]",
        )
    return (
        "Broker",
        "[yellow]local PaperBroker (simulated fills)[/yellow]",
    )


@paper_app.command("preflight")
def paper_preflight(
    symbol: str = typer.Option("SPY", "--symbol"),
    config: Optional[str] = typer.Option(
        "config/paper.yaml", "--config", "-c", envvar="JOKER_CONFIG"
    ),
    skip_model_check: bool = typer.Option(False, "--skip-model-check"),
    require_webull_paper: bool = typer.Option(
        False,
        "--require-webull-paper",
        help="Fail unless WEBULL_PAPER_TRADING_ENABLED + account id are set",
    ),
) -> None:
    """Verify Webull data, OPRA, safety gates, and paper broker readiness."""
    from joker.compliance.opra_scanner import discover_db_paths, scan_local_opra
    from joker.config.validation import validate_startup
    from joker.data.webull_capability import capability_usable_for_shadow, load_capability

    if symbol.upper() != "SPY":
        console.print("[red]Only SPY is supported.[/red]")
        raise typer.Exit(code=1)

    result = validate_startup(config_path=config, skip_model_check=skip_model_check)
    app_settings = result.app_settings
    env = result.env_settings

    table = Table(title="Live paper preflight")
    table.add_column("Check")
    table.add_column("Status")

    table.add_row("Config valid", "[green]pass[/green]")
    table.add_row("Mode", app_settings.mode.value)
    table.add_row(
        "live_trading_enabled",
        "[green]false[/green]"
        if not app_settings.live_trading_enabled
        else "[red]true (blocked)[/red]",
    )
    table.add_row(
        "Webull market data",
        "[green]enabled[/green]" if env.webull_market_data_enabled else "[red]disabled[/red]",
    )
    table.add_row(
        "Webull live money env",
        "[green]false[/green]"
        if not env.webull_live_trading_enabled
        else "[red]true (blocked)[/red]",
    )
    table.add_row(*_broker_status_row(env))
    if env.webull_paper_trading_enabled:
        table.add_row(
            "WEBULL_PAPER_ACCOUNT_ID",
            "[green]set[/green]" if env.webull_paper_account_id else "[red]missing[/red]",
        )
    table.add_row("Default provider", app_settings.data.default_provider)
    table.add_row(
        "Agents",
        "mock" if app_settings.agents.mock_agents else "OpenAI council + intraday",
    )
    table.add_row(
        "Allow delayed quotes",
        "yes" if app_settings.risk.allow_delayed_quotes else "no",
    )

    cap = load_capability()
    cap_ok = capability_usable_for_shadow()
    table.add_row(
        "Options capability (OPRA)",
        "[green]usable[/green]" if cap_ok else "[red]not verified[/red]",
    )
    if cap:
        table.add_row("Capability checked at", cap.checked_at.isoformat())

    db_paths = discover_db_paths(Path("."), app_settings.db_path)
    scan = scan_local_opra(root=Path("."), db_paths=db_paths)
    opra_violations = len(scan.violations)
    table.add_row(
        "OPRA compliance scan",
        f"[green]{opra_violations} violations[/green]"
        if opra_violations == 0
        else f"[red]{opra_violations} possible violation(s)[/red]",
    )

    console.print(table)

    failed = False
    if app_settings.live_trading_enabled:
        console.print("[red]Disable live_trading_enabled for paper sessions.[/red]")
        failed = True
    if app_settings.mode.value != "PAPER":
        console.print("[red]Use config/paper.yaml (mode PAPER).[/red]")
        failed = True
    if not env.webull_market_data_enabled:
        console.print("[red]Enable WEBULL_MARKET_DATA_ENABLED in .env[/red]")
        failed = True
    if app_settings.data.default_provider != "webull":
        console.print("[red]data.default_provider must be webull for live paper.[/red]")
        failed = True
    if opra_violations > 0:
        console.print("[yellow]Run: joker compliance quarantine-opra-artifacts[/yellow]")
        failed = True
    if not cap_ok:
        console.print("[yellow]Run: joker data verify-webull-options --symbol SPY[/yellow]")
        failed = True
    if require_webull_paper and not webull_paper_env_ready(env):
        console.print(
            "[red]Webull paper broker required. Set WEBULL_PAPER_TRADING_ENABLED=true "
            "and WEBULL_PAPER_ACCOUNT_ID (see `joker broker accounts`).[/red]"
        )
        failed = True

    if failed:
        raise typer.Exit(code=1)

    if webull_paper_env_ready(env):
        console.print(
            "[green]Paper preflight passed[/green] — real Webull data + "
            "Webull paper-account auto orders. Real money stays off."
        )
    else:
        console.print(
            "[green]Paper preflight passed[/green] — real Webull data + "
            "local PaperBroker fills. To auto-place on Webull paper account:\n"
            "  1) joker broker accounts\n"
            "  2) set WEBULL_PAPER_TRADING_ENABLED=true and WEBULL_PAPER_ACCOUNT_ID\n"
            "  3) joker broker preflight && joker paper preflight --require-webull-paper"
        )


@paper_app.command("run")
def paper_run(
    symbol: str = typer.Option("SPY", "--symbol"),
    duration_minutes: Optional[float] = typer.Option(
        None,
        "--duration-minutes",
        help=(
            "Wall-clock runtime minutes. When omitted, equals the objective duration "
            "(default 60). Must be >= objective duration when both are set."
        ),
    ),
    objective_duration_minutes: Optional[float] = typer.Option(
        None,
        "--objective-duration-minutes",
        help=(
            "Objective deadline = current exchange time + N minutes "
            f"(default {60.0} when neither this nor --target-deadline is set). "
            "Mutually exclusive with --target-deadline."
        ),
    ),
    config: Optional[str] = typer.Option(
        "config/paper.yaml", "--config", "-c", envvar="JOKER_CONFIG"
    ),
    use_openai: bool = typer.Option(
        True,
        "--use-openai/--mock-agents",
        help="OpenAI premarket/intraday agents (default) or local mock agents",
    ),
    skip_model_check: bool = typer.Option(False, "--skip-model-check"),
    skip_preflight: bool = typer.Option(
        False, "--skip-preflight", help="Skip preflight (not recommended)"
    ),
    require_webull_paper: bool = typer.Option(
        False,
        "--require-webull-paper",
        help="Refuse to start unless Webull paper-account auto orders are enabled",
    ),
    authorized_capital: Optional[float] = typer.Option(
        None,
        "--authorized-capital",
        help="Max premium USD to risk today (required with --yes when objective.enabled)",
    ),
    target_profit_pct: Optional[float] = typer.Option(
        None,
        "--target-profit-pct",
        help="Profit goal as %% of authorised capital",
    ),
    target_deadline: Optional[str] = typer.Option(
        None,
        "--target-deadline",
        help=(
            'Absolute deadline as "15:30 ET" or timezone-aware ISO timestamp. '
            "Mutually exclusive with --objective-duration-minutes."
        ),
    ),
    max_concurrent_positions: Optional[int] = typer.Option(
        None,
        "--max-concurrent-positions",
        help="Maximum concurrent positions for this session",
    ),
    acknowledge_total_loss: bool = typer.Option(
        False,
        "--acknowledge-total-loss",
        help="Acknowledge that the full authorised capital may be lost (paper)",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip prompts only when all required objective values + ack are provided",
    ),
    heartbeat_seconds: float = typer.Option(
        5.0,
        "--heartbeat-seconds",
        help="How often to print SPY heartbeat lines (decision events print immediately)",
    ),
    graph_view: Optional[str] = typer.Option(
        None,
        "--graph-view",
        help="Cognitive graph evidence view: compact, verbose, or json",
    ),
    objective_session: str = typer.Option(
        "new",
        "--objective-session",
        help="Objective lifecycle: 'new' starts a new objective; 'resume' recovers the active one",
    ),
) -> None:
    """Run the full live paper loop: monitor → decide → risk → auto order → log."""
    import asyncio
    import subprocess

    from joker.broker.account_truth import hash_account_id
    from joker.cli.paper_goal_timing import (
        PaperGoalTimingError,
        format_timing_banner,
        resolve_paper_goal_timing,
    )
    from joker.cli.session_confirm import (
        confirm_session_capital,
        confirm_session_objective,
        has_unresolved_portfolio_work,
        recover_session_objective_bundle,
        validate_objective_session_action,
        validate_resume_mutation_flags,
    )
    from joker.config.validation import validate_startup
    from joker.objectives.deadline import time_remaining_seconds
    from joker.runtime.live_paper_runner import LivePaperRunConfig, LivePaperRunner
    from joker.runtime.paper_goal_result import (
        PaperGoalResult,
        append_jsonl,
        build_manifest,
        classify_paper_goal,
        evidence_dir,
        sqlite_checks,
        write_json,
    )
    from joker.time.calendar import MarketCalendar
    from joker.time.clock import SystemExchangeClock

    if symbol.upper() != "SPY":
        console.print("[red]Only SPY is supported.[/red]")
        raise typer.Exit(code=1)
    objective_session = objective_session.strip().lower()
    if objective_session not in {"new", "resume"}:
        console.print("[red]--objective-session must be 'new' or 'resume'[/red]")
        raise typer.Exit(code=1)

    result = validate_startup(config_path=config, skip_model_check=skip_model_check)
    from joker.cli.graph_view import (
        GRAPH_EVENT_TYPES,
        GraphView,
        render_graph_event,
    )

    configured_graph_view = getattr(
        getattr(result.app_settings, "full_chain_optimizer", None),
        "cli_graph_view",
        "compact",
    )
    try:
        resolved_graph_view = GraphView(graph_view or configured_graph_view)
    except ValueError:
        console.print("[red]--graph-view must be compact, verbose, or json[/red]")
        raise typer.Exit(code=1) from None
    if not skip_preflight:
        from joker.data.webull_capability import capability_usable_for_shadow

        if not result.env_settings.webull_market_data_enabled:
            console.print("[red]WEBULL_MARKET_DATA_ENABLED required[/red]")
            raise typer.Exit(code=1)
        if not capability_usable_for_shadow():
            console.print(
                "[red]Options capability not verified. "
                "Run: joker data verify-webull-options --symbol SPY[/red]"
            )
            raise typer.Exit(code=1)
        if result.app_settings.live_trading_enabled:
            console.print("[red]live_trading_enabled must be false[/red]")
            raise typer.Exit(code=1)

    if result.env_settings.webull_live_trading_enabled:
        console.print(
            "[red]WEBULL_LIVE_TRADING_ENABLED must remain false for paper goal tests.[/red]"
        )
        raise typer.Exit(code=1)

    if require_webull_paper and not webull_paper_env_ready(result.env_settings):
        console.print(
            "[red]--require-webull-paper set but Webull paper env is not ready. "
            "Run `joker broker accounts` then set WEBULL_PAPER_TRADING_ENABLED and "
            "WEBULL_PAPER_ACCOUNT_ID.[/red]"
        )
        raise typer.Exit(code=1)

    # Fail closed for profiles that pin broker.provider=webull_paper — never fall
    # through to local PaperBroker even when --require-webull-paper is omitted.
    yaml_provider = (result.app_settings.broker.provider or "paper").strip().lower()
    if yaml_provider in {"webull_paper", "webull"} and not webull_paper_env_ready(
        result.env_settings
    ):
        console.print(
            "[red]Config broker.provider=webull_paper requires "
            "WEBULL_PAPER_TRADING_ENABLED=true, WEBULL_PAPER_ACCOUNT_ID, and "
            "WEBULL_LIVE_TRADING_ENABLED=false. Refusing PaperBroker fallback.[/red]"
        )
        raise typer.Exit(code=1)

    if require_webull_paper and not use_openai:
        console.print(
            "[red]Paper goal test refuses --mock-agents when --require-webull-paper "
            "is set. Use configured non-fake model providers.[/red]"
        )
        raise typer.Exit(code=1)

    if result.app_settings.mode.value != "PAPER":
        console.print("[yellow]Forcing PAPER mode for this session.[/yellow]")
        result.app_settings = result.app_settings.model_copy(
            update={"mode": "PAPER", "live_trading_enabled": False}
        )

    # Resolve stable broker/session authority before any new-objective timing.
    # A resume is defined exclusively by its durable objective deadline.
    exchange_tz = str(result.app_settings.exchange.timezone)
    calendar = MarketCalendar()
    exchange_clock = SystemExchangeClock(calendar=calendar)
    exchange_now = exchange_clock.now()

    objective_enabled = bool(getattr(result.app_settings.objective, "enabled", False))
    if not objective_enabled and (
        objective_session == "resume"
        or
        objective_duration_minutes is not None
        or target_deadline is not None
        or require_webull_paper
    ):
        console.print(
            "[yellow]Enabling session objective for paper goal-test workflow.[/yellow]"
        )
        result.app_settings = result.app_settings.model_copy(
            update={
                "objective": result.app_settings.objective.model_copy(
                    update={"enabled": True}
                )
            }
        )
        objective_enabled = True

    broker_ready = webull_paper_env_ready(result.env_settings)
    if require_webull_paper and not broker_ready:
        console.print("[red]Webull paper broker required; refusing local PaperBroker.[/red]")
        raise typer.Exit(code=1)
    broker_label = (
        "Webull paper account (auto orders)"
        if broker_ready
        else "local PaperBroker (simulated fills)"
    )
    paper_account_hash = None
    if result.env_settings.webull_paper_account_id:
        paper_account_hash = hash_account_id(
            str(result.env_settings.webull_paper_account_id).strip()
        )

    from joker.runtime.cognitive_session import (
        paper_account_identity,
        stable_cognitive_session_id,
    )

    broker_kind = "webull_paper" if broker_ready else "local_paper"
    account_identity = paper_account_identity(
        broker_kind=broker_kind,
        env=result.env_settings,
    )
    session_id = stable_cognitive_session_id(
        trading_date=exchange_now.date(),
        account_identity=account_identity,
    )
    task1_db = Path(result.app_settings.db_path).resolve().parent / "joker_task1.db"
    provenance_db = task1_db.with_name(task1_db.stem + "_cognitive_provenance.db")

    objective_service = None
    objective_id = None
    bundle = None
    existing_definition = None
    existing_state = None
    if objective_enabled:
        from joker.objectives.repository import ObjectiveRepository

        objective_repo = ObjectiveRepository(task1_db)
        existing_definition = objective_repo.latest_definition_for_session(session_id)
        if existing_definition is not None:
            bundle = asyncio.run(
                recover_session_objective_bundle(
                    result.app_settings,
                    session_id=session_id,
                    db_path=task1_db,
                    exchange_tz=exchange_tz,
                    now=exchange_now,
                )
            )
            existing_state = objective_repo.latest_state_for_session(session_id)
        try:
            validate_objective_session_action(
                objective_session,
                has_definition=existing_definition is not None,
                latest_status=(
                    str(existing_state.status) if existing_state is not None else None
                ),
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        if objective_session == "resume":
            try:
                validate_resume_mutation_flags(
                    objective_duration_minutes=objective_duration_minutes,
                    target_deadline=target_deadline,
                    authorized_capital=authorized_capital,
                    target_profit_pct=target_profit_pct,
                    max_concurrent_positions=max_concurrent_positions,
                )
            except ValueError as exc:
                console.print(
                    f"[red]{exc}[/red]"
                )
                raise typer.Exit(code=1) from exc
            if bundle is None or bundle.deadline_exchange_time is None:
                console.print("[red]Active objective recovery failed closed.[/red]")
                raise typer.Exit(code=1)
            console.print(
                f"[green]Resuming durable objective session[/green] {session_id}"
            )
        else:
            if asyncio.run(
                has_unresolved_portfolio_work(
                    provenance_db_path=provenance_db,
                    session_id=session_id,
                    broker_account_identity=account_identity,
                    trading_date=exchange_now.date().isoformat(),
                )
            ):
                console.print(
                    "[red]Durable portfolio execution or reoptimization work is still "
                    "owned by this account/date session. Resume and reconcile it before "
                    "starting a new objective.[/red]"
                )
                raise typer.Exit(code=1)

    resolved_objective_duration = objective_duration_minutes
    resolved_target_deadline = target_deadline
    if objective_session == "resume" and objective_enabled:
        assert bundle is not None and bundle.deadline_exchange_time is not None
        resolved_objective_duration = None
        resolved_target_deadline = bundle.deadline_exchange_time.isoformat()
    elif not yes and objective_duration_minutes is None and target_deadline is None:
        console.print("\n[bold]Deadline mode[/bold]")
        mode = typer.prompt(
            "Deadline mode (1=Relative duration, 2=Absolute exchange deadline)",
            default="1",
        )
        if str(mode).strip() in {"2", "absolute", "Absolute", "A", "a"}:
            resolved_target_deadline = str(
                typer.prompt(
                    "Target deadline (e.g. 11:30 ET or ISO timestamp)",
                    default="15:30 ET",
                )
            )
        else:
            resolved_objective_duration = float(
                typer.prompt(
                    "Objective duration in minutes",
                    default=60.0,
                    type=float,
                )
            )
    try:
        timing = resolve_paper_goal_timing(
            objective_duration_minutes=resolved_objective_duration,
            target_deadline=resolved_target_deadline,
            duration_minutes=duration_minutes,
            exchange_tz=exchange_tz,
            calendar=calendar,
            now=exchange_now,
            require_regular_session=True,
        )
    except PaperGoalTimingError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if objective_enabled and objective_session == "new":
        bundle = asyncio.run(
            confirm_session_objective(
                result.app_settings,
                session_id=session_id,
                db_path=task1_db,
                console=console,
                authorized_usd=authorized_capital,
                target_profit_pct=target_profit_pct,
                target_deadline=None,
                deadline_exchange_time=timing.objective_deadline,
                confirmed_at_exchange_time=timing.exchange_now,
                max_concurrent_positions=max_concurrent_positions,
                acknowledge_total_loss=acknowledge_total_loss,
                yes=yes,
                exchange_tz=exchange_tz,
            )
        )

    banner = format_timing_banner(timing)
    console.print(
        f"[bold]Exchange now[/bold] {banner['exchange_now']}  "
        f"[bold]objective deadline[/bold] {banner['objective_deadline']}  "
        f"[bold]market session remaining[/bold] "
        f"{banner['remaining_market_session_seconds']}s"
    )

    if objective_enabled:
        assert bundle is not None
        capital_budget = bundle.capital_budget
        objective_service = bundle.objective_service
        objective_id = bundle.objective_id
    else:
        if objective_session == "resume":
            console.print("[red]Objective resume could not activate durable objective mode.[/red]")
            raise typer.Exit(code=1)
        capital_budget = confirm_session_capital(
            result.app_settings.capital,
            console=console,
            authorized_usd=authorized_capital,
            target_profit_pct=target_profit_pct,
            max_concurrent_positions=max_concurrent_positions,
            yes=yes,
        )

    # Final confirmed objective banner (required for goal-test).
    target_profit_usd = float(capital_budget.plan.target_profit_usd)
    obj_table = Table(title="Confirmed paper goal session")
    obj_table.add_column("Field")
    obj_table.add_column("Value")
    obj_table.add_row("objective_id", str(objective_id or "—"))
    obj_table.add_row("session_id", session_id)
    obj_table.add_row("authorized capital", f"${capital_budget.authorized_usd:,.2f}")
    obj_table.add_row(
        "target profit percent", f"{capital_budget.plan.target_profit_pct:.2f}%"
    )
    obj_table.add_row("target profit USD", f"${target_profit_usd:,.2f}")
    obj_table.add_row("exchange start time", timing.exchange_now.isoformat())
    obj_table.add_row("exchange deadline", timing.objective_deadline.isoformat())
    obj_table.add_row(
        "objective duration", f"{timing.objective_duration_minutes:.2f} minutes"
    )
    obj_table.add_row(
        "runtime duration", f"{timing.runtime_duration_minutes:.2f} minutes"
    )
    obj_table.add_row(
        "maximum concurrent positions",
        str(capital_budget.plan.max_concurrent_positions),
    )
    obj_table.add_row("paper broker account hash", paper_account_hash or "—")
    console.print(obj_table)

    # Evidence package root (created up-front; filled after run).
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    exchange_date = SystemExchangeClock(calendar=MarketCalendar()).trading_date()
    artifacts = evidence_dir(
        code_sha=code_sha, exchange_date=exchange_date, session_id=session_id
    )
    write_json(
        artifacts / "objective.json",
        {
            "objective_id": objective_id,
            "session_id": session_id,
            "policy": str(getattr(result.app_settings.objective, "policy", None)),
            "shadow_baseline_enabled": bool(
                getattr(result.app_settings.objective, "shadow_baseline_enabled", False)
            ),
            "deadline_mode": (
                "absolute"
                if timing.objective_source == "absolute_deadline"
                else "relative"
            ),
            "authorized_capital_usd": capital_budget.authorized_usd,
            "target_profit_pct": capital_budget.plan.target_profit_pct,
            "target_profit_usd": target_profit_usd,
            **banner,
            "paper_account_hash": paper_account_hash,
        },
    )
    # Ensure target-attainment evidence streams exist even if no cycles emit.
    for name in (
        "target-attainment-context.jsonl",
        "contract-candidate-evaluations.jsonl",
        "candidate-quantity-evaluations.jsonl",
        "target-probability-estimates.jsonl",
        "no-trade-evaluations.jsonl",
        "baseline-shadow-decisions.jsonl",
        "meta-target-review.jsonl",
        "urgency-transitions.jsonl",
    ):
        (artifacts / name).touch(exist_ok=True)
    write_json(
        artifacts / "deadline-state.json",
        {
            "objective_deadline": timing.objective_deadline.isoformat(),
            "exchange_now_at_start": timing.exchange_now.isoformat(),
            "session_close": timing.session_close.isoformat(),
            "shutdown_grace_seconds": timing.shutdown_grace_seconds,
        },
    )
    write_json(
        artifacts / "environment-summary.json",
        {
            "code_sha": code_sha,
            "branch": branch,
            "mode": "PAPER",
            "live_trading_enabled": False,
            "webull_live_trading_enabled": False,
            "webull_paper_trading_enabled": bool(
                result.env_settings.webull_paper_trading_enabled
            ),
            "webull_trade_api_env": result.env_settings.webull_trade_api_env,
            "broker_provider": result.app_settings.broker.provider,
            "require_webull_paper": require_webull_paper,
            "paper_account_hash": paper_account_hash,
            "mock_agents": not use_openai,
            "objective_enabled": objective_enabled,
        },
    )

    runner = LivePaperRunner(result.app_settings, result.env_settings)
    last_heartbeat = 0.0
    latest_graph_action = "—"
    latest_no_trade_reason = "—"
    progress_peaks = {
        "max_unrealized": 0.0,
        "min_realized": 0.0,
        "reserved_peak": 0.0,
    }
    starting_realized = 0.0
    graph_cycles = 0
    no_trade_decisions = 0
    entry_proposals = 0
    entry_approvals = 0

    def on_event(event_type: str, payload: dict) -> None:
        nonlocal latest_graph_action, latest_no_trade_reason
        nonlocal graph_cycles, no_trade_decisions, entry_proposals, entry_approvals
        if event_type in GRAPH_EVENT_TYPES:
            optimizer_settings = result.app_settings.full_chain_optimizer
            line = render_graph_event(
                event_type,
                payload,
                view=resolved_graph_view,
                top_contract_rows=optimizer_settings.cli_top_contract_rows,
                top_portfolio_rows=optimizer_settings.cli_top_portfolio_rows,
            )
        else:
            line = format_live_event(event_type, payload)
        latest_graph_action = f"{event_type}"
        if event_type in {
            "agent.decision",
            "agent.propose",
            "agent.prefilter_skip",
            "risk.decision",
            "cognitive.no_trade",
            "graph.no_trade",
        }:
            reason = (
                payload.get("reason")
                or payload.get("blocked_reason")
                or payload.get("summary")
                or payload.get("action")
            )
            if reason and payload.get("approved") is False:
                latest_no_trade_reason = str(reason)[:120]
                no_trade_decisions += 1
            if payload.get("action") in {"propose", "enter"} or event_type.endswith(
                "propose"
            ):
                entry_proposals += 1
            if payload.get("approved") is True or payload.get("action") in {
                "confirm",
                "enter",
            }:
                entry_approvals += 1
        if "cycle" in event_type or event_type in {
            "cognitive.cycle",
            "graph.cycle_complete",
        }:
            graph_cycles += 1

        # Evidence streams (redacted on write).
        if "historical" in event_type or "ev" in event_type:
            append_jsonl(
                artifacts / "historical-ev-decisions.jsonl",
                {"event": event_type, "payload": payload},
            )
        if "capital" in event_type or event_type == "capital.sized":
            append_jsonl(
                artifacts / "capital-sizing.jsonl",
                {"event": event_type, "payload": payload},
            )
        if event_type.startswith("order.") or "fill" in event_type:
            append_jsonl(
                artifacts / "order-lifecycle.jsonl",
                {"event": event_type, "payload": payload},
            )
        if "position" in event_type or "exit" in event_type:
            append_jsonl(
                artifacts / "position-lifecycle.jsonl",
                {"event": event_type, "payload": payload},
            )
        if event_type.startswith("agent.") or event_type.startswith("cognitive.") or event_type.startswith("graph.") or event_type.startswith("risk."):
            append_jsonl(
                artifacts / "graph-decisions.jsonl",
                {"event": event_type, "payload": payload},
            )
        ta = payload.get("target_attainment_decision") or payload.get(
            "_target_attainment_decision"
        )
        if ta and isinstance(ta, dict):
            append_jsonl(
                artifacts / "target-attainment-context.jsonl",
                {
                    "event": event_type,
                    "snapshot_id": payload.get("snapshot_id"),
                    "decision": {
                        k: ta.get(k)
                        for k in (
                            "decision_id",
                            "action",
                            "feasibility",
                            "selected_strategy_id",
                            "selected_quantity",
                            "selected_capital_usd",
                            "probability_delta",
                            "reason_codes",
                        )
                    },
                },
            )
            for qev in ta.get("quantity_evaluations") or []:
                append_jsonl(
                    artifacts / "candidate-quantity-evaluations.jsonl",
                    {"event": event_type, "evaluation": qev},
                )
                if isinstance(qev, dict) and qev.get("p_goal"):
                    append_jsonl(
                        artifacts / "target-probability-estimates.jsonl",
                        {
                            "kind": "candidate_quantity",
                            "strategy_id": qev.get("strategy_id"),
                            "quantity": qev.get("quantity"),
                            "p_goal": qev.get("p_goal"),
                        },
                    )
            if ta.get("no_trade"):
                append_jsonl(
                    artifacts / "no-trade-evaluations.jsonl",
                    {"event": event_type, "evaluation": ta.get("no_trade")},
                )
            if ta.get("baseline_shadow"):
                append_jsonl(
                    artifacts / "baseline-shadow-decisions.jsonl",
                    {
                        "event": event_type,
                        "shadow": ta.get("baseline_shadow"),
                        "authoritative_action": ta.get("action"),
                        "executes_shadow": False,
                    },
                )
            for qev in ta.get("quantity_evaluations") or []:
                if isinstance(qev, dict) and qev.get("contract_id"):
                    append_jsonl(
                        artifacts / "contract-candidate-evaluations.jsonl",
                        {
                            "event": event_type,
                            "strategy_id": qev.get("strategy_id"),
                            "contract_id": qev.get("contract_id"),
                            "quantity": qev.get("quantity"),
                            "evaluation_premium_usd": qev.get("evaluation_premium_usd"),
                            "selected": qev.get("selected"),
                        },
                    )
        review = payload.get("_meta_target_review") or payload.get("meta_target_review")
        if review:
            append_jsonl(
                artifacts / "meta-target-review.jsonl",
                {"event": event_type, "review": review},
            )

        important = {
            "order.accepted",
            "order.partial_fill",
            "order.final_fill",
            "order.submitted",
            "signal.detected",
            "agent.execute",
            "agent.propose",
            "agent.confirm_executed",
            "agent.outcome",
            "capital.sized",
            "risk.decision",
            "objective.achieved",
            "objective.missed",
        }
        if event_type in GRAPH_EVENT_TYPES:
            console.print(line, markup=False)
        elif event_type.startswith("order.") or event_type in important:
            console.print(f"[bold cyan]» {line}[/bold cyan]")
        elif event_type in ("agent.decision",) and payload.get("action") in (
            "propose",
            "confirm",
            "enter",
        ):
            console.print(f"[bold green]» {line}[/bold green]")
        elif event_type in ("capital.sized", "option.advisory", "agent.prefilter_skip"):
            console.print(f"[cyan]» {line}[/cyan]")
        elif event_type == "risk.decision" and payload.get("approved"):
            console.print(f"[bold green]» {line}[/bold green]")
        elif event_type == "risk.decision":
            console.print(f"[yellow]» {line}[/yellow]")
        elif event_type.endswith("failure") or event_type.endswith("failed") or event_type.endswith("error"):
            console.print(f"[red]» {line}[/red]")
        else:
            console.print(f"[dim]» {line}[/dim]")

    def on_state(state: dict) -> None:
        nonlocal last_heartbeat
        now_m = time.monotonic()
        if now_m - last_heartbeat < max(0.5, heartbeat_seconds):
            return
        last_heartbeat = now_m

        realized = float(state.get("paper_pnl") or state.get("realized_pnl") or 0.0)
        unrealized = float(state.get("unrealized_pnl") or 0.0)
        reserved = float(state.get("capital_reserved") or 0.0)
        available = float(
            state.get("capital_available")
            if state.get("capital_available") is not None
            else capital_budget.available_usd
        )
        progress_peaks["max_unrealized"] = max(
            progress_peaks["max_unrealized"], unrealized
        )
        progress_peaks["min_realized"] = min(progress_peaks["min_realized"], realized)
        progress_peaks["reserved_peak"] = max(
            progress_peaks["reserved_peak"], reserved
        )
        goal_gap = target_profit_usd - realized
        rem = time_remaining_seconds(
            timing.objective_deadline, exchange_tz=exchange_tz
        )
        exchange_now = SystemExchangeClock(calendar=MarketCalendar()).now()
        append_jsonl(
            artifacts / "objective-progress.jsonl",
            {
                "exchange_time": exchange_now.isoformat(),
                "time_remaining_seconds": rem,
                "spy": state.get("market_price"),
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "goal_gap_usd": goal_gap,
                "authorized": capital_budget.authorized_usd,
                "available": available,
                "reserved": reserved,
                "open": state.get("open_trade"),
                "pending": state.get("pending_order"),
                "latest_graph_action": latest_graph_action,
                "latest_no_trade_reason": latest_no_trade_reason,
            },
        )
        console.print(
            f"  ET {exchange_now.strftime('%H:%M:%S')}  rem={rem}s  "
            f"SPY ${state.get('market_price', '—')}  "
            f"md={state.get('feed_health')}  "
            f"opt={'ok' if state.get('options_available') else 'n/a'}  "
            f"tgt=${target_profit_usd:,.2f}  "
            f"realized=${realized:,.2f}  unreal=${unrealized:,.2f}  "
            f"gap=${goal_gap:,.2f}  "
            f"auth=${capital_budget.authorized_usd:,.0f}  "
            f"avail=${available:,.2f}  res=${reserved:,.2f}  "
            f"open={state.get('open_trade')}  working={state.get('pending_order')}  "
            f"action={latest_graph_action}  "
            f"no_trade={latest_no_trade_reason}"
        )

    exec_mode = (
        result.app_settings.agents.execution_mode or "rules_hybrid"
    ).strip().lower()
    risk_policy = (result.app_settings.risk.policy or "strict").strip().lower()
    console.print(
        f"[bold]Starting live paper loop[/bold] — "
        f"runtime={timing.runtime_duration_minutes:.1f}m, "
        f"objective={timing.objective_duration_minutes:.1f}m, "
        f"agents={'openai' if use_openai else 'mock'}, broker={broker_label}"
    )
    console.print(
        f"[dim]execution_mode={exec_mode} risk.policy={risk_policy}[/dim]"
    )
    if exec_mode == "agent_led":
        console.print(
            "[dim]AI is primary entry authority (~45s). Flow: propose → confirm. "
            "Soft caps advisory. Hard floors still block. "
            "Session memory feeds recent decisions/outcomes back to the agent.[/dim]"
        )
    console.print(
        "[dim]Decision / risk / order events stream live. "
        "Heartbeat lines are throttled. Ctrl+C to stop.[/dim]"
    )

    run_result = runner.run(
        LivePaperRunConfig(
            symbol=symbol,
            duration_seconds=timing.runtime_seconds,
            mock_agents=not use_openai,
            require_options=True,
            capital_budget=capital_budget,
            objective_service=objective_service,
            cognitive_session_id_override=session_id if objective_enabled else None,
            objective_deadline_exchange=timing.objective_deadline,
            shutdown_grace_seconds=timing.shutdown_grace_seconds,
        ),
        on_state=on_state,
        on_event=on_event,
    )

    for err in run_result.errors:
        console.print(f"[yellow]{err}[/yellow]")
    for fail in run_result.failures:
        console.print(f"[red]failure:[/red] {fail}")

    ending_realized = float(
        getattr(run_result.summary, "final_pnl_usd", None)
        if run_result.summary is not None
        else run_result.paper_pnl_usd
    )
    open_remaining = int(run_result.open_positions_remaining or 0)
    working_remaining = int(run_result.working_orders_remaining or 0)
    recon_clean: bool | None = run_result.reconciliation_clean
    deadline_reached = (
        time_remaining_seconds(timing.objective_deadline, exchange_tz=exchange_tz) == 0
        or bool(run_result.objective_deadline_reached)
    )
    classification, reason = classify_paper_goal(
        ending_realized_pnl_usd=ending_realized,
        target_profit_usd=target_profit_usd,
        open_positions_remaining=open_remaining,
        working_orders_remaining=working_remaining,
        reconciliation_clean=recon_clean if recon_clean is not None else True,
        deadline_reached=bool(
            getattr(run_result, "objective_deadline_reached", False) or deadline_reached
        ),
        system_operational=not bool(run_result.failures),
        session_failed_errors=list(run_result.failures),
    )
    goal_result = PaperGoalResult(
        classification=classification,
        objective_id=objective_id,
        session_id=session_id,
        authorized_capital_usd=float(capital_budget.authorized_usd),
        target_profit_pct=float(capital_budget.plan.target_profit_pct),
        target_profit_usd=target_profit_usd,
        objective_duration_minutes=timing.objective_duration_minutes,
        starting_realized_pnl_usd=starting_realized,
        ending_realized_pnl_usd=ending_realized,
        max_unrealized_gain_usd=progress_peaks["max_unrealized"],
        max_drawdown_usd=abs(min(0.0, progress_peaks["min_realized"])),
        capital_reserved_peak_usd=progress_peaks["reserved_peak"],
        graph_cycles=graph_cycles or int(
            getattr(run_result.summary, "events_processed", 0) or 0
        ),
        entry_proposals=entry_proposals,
        entry_approvals=entry_approvals,
        trades_entered=int(
            getattr(run_result.summary, "trades_entered", 0) or 0
        ),
        trades_exited=int(getattr(run_result.summary, "trades_exited", 0) or 0),
        no_trade_decisions=no_trade_decisions,
        goal_achieved=classification == "PAPER_OBJECTIVE_ACHIEVED",
        open_positions_remaining=open_remaining,
        working_orders_remaining=working_remaining,
        reconciliation_clean=recon_clean if recon_clean is not None else True,
        reason=reason,
    )
    write_json(artifacts / "final-result.json", goal_result.to_dict())
    write_json(
        artifacts / "episode-summary.json",
        {
            "session_id": session_id,
            "summary": (
                run_result.summary.model_dump()
                if run_result.summary is not None and hasattr(run_result.summary, "model_dump")
                else None
            ),
            "events_processed": run_result.events_processed,
            "feed_health": run_result.feed_health,
            "broker_kind": run_result.broker_kind,
            "errors": run_result.errors,
            "failures": run_result.failures,
        },
    )
    write_json(
        artifacts / "reconciliation.json",
        {
            "reconciliation_clean": goal_result.reconciliation_clean,
            "open_positions_remaining": open_remaining,
            "working_orders_remaining": working_remaining,
            "broker_kind": run_result.broker_kind,
        },
    )
    write_json(
        artifacts / "sqlite-checks.json",
        sqlite_checks(
            [
                Path(result.app_settings.db_path),
                task1_db,
            ]
        ),
    )
    # Placeholder market/surface summaries if not populated by runner hooks.
    if not (artifacts / "market-data-summary.json").exists():
        write_json(
            artifacts / "market-data-summary.json",
            {
                "feed_health": run_result.feed_health,
                "options_available": run_result.options_available,
            },
        )
    if not (artifacts / "option-surface-summary.json").exists():
        write_json(
            artifacts / "option-surface-summary.json",
            {"options_available": run_result.options_available},
        )

    model_providers: list[str] = []
    models = getattr(result.app_settings, "models", None)
    if models is not None:
        if getattr(getattr(models, "ollama", None), "enabled", False):
            model_providers.append("ollama")
        if getattr(getattr(models, "openai", None), "enabled", False):
            model_providers.append("openai")
    write_json(
        artifacts / "manifest.json",
        build_manifest(
            code_sha=code_sha,
            branch=branch,
            timing=banner,
            objective_id=objective_id,
            session_id=session_id,
            paper_account_hash=paper_account_hash,
            model_providers=model_providers,
            artifact_dir=artifacts,
        ),
    )

    # Final metrics table
    metrics = Table(title="Paper goal-test result")
    metrics.add_column("Metric")
    metrics.add_column("Value")
    rows = [
        ("Authorized capital", f"${goal_result.authorized_capital_usd:,.2f}"),
        ("Target profit percentage", f"{goal_result.target_profit_pct:.2f}%"),
        ("Target profit USD", f"${goal_result.target_profit_usd:,.2f}"),
        ("Objective duration", f"{goal_result.objective_duration_minutes:.0f} minutes"),
        ("Starting realized P&L baseline", f"${goal_result.starting_realized_pnl_usd:,.2f}"),
        ("Ending realized P&L", f"${goal_result.ending_realized_pnl_usd:,.2f}"),
        ("Maximum unrealized gain", f"${goal_result.max_unrealized_gain_usd or 0:,.2f}"),
        ("Maximum drawdown", f"${goal_result.max_drawdown_usd or 0:,.2f}"),
        ("Capital reserved peak", f"${goal_result.capital_reserved_peak_usd or 0:,.2f}"),
        ("Number of graph cycles", str(goal_result.graph_cycles)),
        ("Entry proposals", str(goal_result.entry_proposals)),
        ("Entry approvals", str(goal_result.entry_approvals)),
        ("Trades entered", str(goal_result.trades_entered)),
        ("Trades exited", str(goal_result.trades_exited)),
        ("Wins", str(goal_result.wins)),
        ("Losses", str(goal_result.losses)),
        ("No-trade decisions", str(goal_result.no_trade_decisions)),
        ("Goal achieved", "yes" if goal_result.goal_achieved else "no"),
        ("Time goal achieved", goal_result.time_goal_achieved or "—"),
        ("Open positions remaining", str(goal_result.open_positions_remaining)),
        ("Working orders remaining", str(goal_result.working_orders_remaining)),
        (
            "Reconciliation clean",
            str(goal_result.reconciliation_clean),
        ),
        ("Final classification", goal_result.classification),
    ]
    for k, v in rows:
        metrics.add_row(k, v)
    console.print(metrics)
    console.print(f"Evidence: {artifacts}")

    if run_result.summary:
        s = run_result.summary
        status = "complete" if not (run_result.failures or run_result.errors) else "finished with errors"
        color = "green" if not (run_result.failures or run_result.errors) else "yellow"
        console.print(
            f"[{color}]Paper run {status}[/{color}] — events={s.events_processed} "
            f"signals={s.signals_detected} entered={s.trades_entered} "
            f"exited={s.trades_exited} pnl=${s.final_pnl_usd:.2f} "
            f"health={run_result.feed_health} broker={run_result.broker_kind}"
        )
    else:
        console.print("[red]Paper run ended without summary.[/red]")
        raise typer.Exit(code=1)

    if run_result.report_path:
        console.print(f"Report: {run_result.report_path}")

    if run_result.broker_kind == "webull_paper":
        console.print(
            "[yellow]PAPER ACCOUNT — orders were sent to Webull paper/sandbox. "
            "No real-money LIVE orders.[/yellow]"
        )
    else:
        console.print(
            "[yellow]LOCAL PAPER — fills were simulated by PaperBroker. "
            "Enable WEBULL_PAPER_TRADING_ENABLED for Webull paper-account auto orders.[/yellow]"
        )

    if classification == "PAPER_SESSION_FAILED" or run_result.failures:
        raise typer.Exit(code=1)


@paper_app.command("execution-smoke")
def paper_execution_smoke(
    config: Optional[str] = typer.Option(
        "config/paper-all-tasks.yaml", "--config", "-c", envvar="JOKER_CONFIG"
    ),
    require_sandbox: bool = typer.Option(
        False,
        "--require-sandbox",
        help="Required. Refuse unless trade API env is exactly sandbox.",
    ),
    confirm_place: bool = typer.Option(
        False,
        "--confirm-place",
        help="Required. Explicitly confirm placing one sandbox order.",
    ),
    skip_model_check: bool = typer.Option(False, "--skip-model-check"),
) -> None:
    """Place+cancel exactly one SPY 0DTE limit via Task 1 ExecutionRuntime (sandbox only)."""
    from joker.config.validation import validate_startup
    from joker.runtime.execution_smoke import ExecutionSmokeError, ExecutionSmokeRunner

    if not require_sandbox or not confirm_place:
        console.print(
            "[red]execution-smoke requires both --require-sandbox and --confirm-place.[/red]"
        )
        raise typer.Exit(code=1)

    result = validate_startup(config_path=config, skip_model_check=skip_model_check)
    if result.app_settings.mode.value != "PAPER":
        console.print("[red]mode must be PAPER[/red]")
        raise typer.Exit(code=1)
    if result.app_settings.live_trading_enabled:
        console.print("[red]live_trading_enabled must be false[/red]")
        raise typer.Exit(code=1)

    console.print(
        "[yellow]Sandbox execution-smoke: one $0.01 buy via ExecutionRuntime, "
        "then immediate cancel. Real money remains prohibited.[/yellow]"
    )
    runner = ExecutionSmokeRunner(
        result.app_settings,
        result.env_settings,
        require_sandbox=require_sandbox,
        confirm_place=confirm_place,
    )
    smoke = runner.run()
    summary = smoke.redacted_summary()
    console.print(summary)
    if not smoke.passed:
        for err in smoke.errors:
            console.print(f"[red]FAIL[/red] {err}")
        raise typer.Exit(code=1)
    console.print(
        "[green]PASS[/green] sandbox place+cancel via ExecutionRuntime; "
        f"orders {smoke.initial_open_orders}→{smoke.final_open_orders}, "
        f"positions {smoke.initial_positions}→{smoke.final_positions}"
    )
