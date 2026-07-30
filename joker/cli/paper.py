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
    duration_minutes: float = typer.Option(30.0, "--duration-minutes"),
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
        help='Deadline as "15:30 ET" or timezone-aware ISO timestamp',
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
) -> None:
    """Run the full live paper loop: monitor → decide → risk → auto order → log."""
    from joker.config.validation import validate_startup
    from joker.runtime.live_paper_runner import LivePaperRunConfig, LivePaperRunner

    if symbol.upper() != "SPY":
        console.print("[red]Only SPY is supported.[/red]")
        raise typer.Exit(code=1)

    result = validate_startup(config_path=config, skip_model_check=skip_model_check)
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

    if result.app_settings.mode.value != "PAPER":
        console.print("[yellow]Forcing PAPER mode for this session.[/yellow]")
        result.app_settings = result.app_settings.model_copy(
            update={"mode": "PAPER", "live_trading_enabled": False}
        )

    broker_ready = webull_paper_env_ready(result.env_settings)
    broker_label = (
        "Webull paper account (auto orders)"
        if broker_ready
        else "local PaperBroker (simulated fills)"
    )

    from joker.cli.session_confirm import confirm_session_capital, confirm_session_objective
    from joker.storage.models import new_run_id
    import asyncio

    objective_enabled = bool(getattr(result.app_settings.objective, "enabled", False))
    session_id = f"paper-{new_run_id()}"
    task1_db = Path(result.app_settings.db_path).resolve().parent / "joker_task1.db"

    objective_service = None
    if objective_enabled:
        bundle = asyncio.run(
            confirm_session_objective(
                result.app_settings,
                session_id=session_id,
                db_path=task1_db,
                console=console,
                authorized_usd=authorized_capital,
                target_profit_pct=target_profit_pct,
                target_deadline=target_deadline,
                max_concurrent_positions=max_concurrent_positions,
                acknowledge_total_loss=acknowledge_total_loss,
                yes=yes,
            )
        )
        capital_budget = bundle.capital_budget
        objective_service = bundle.objective_service
    else:
        capital_budget = confirm_session_capital(
            result.app_settings.capital,
            console=console,
            authorized_usd=authorized_capital,
            target_profit_pct=target_profit_pct,
            max_concurrent_positions=max_concurrent_positions,
            yes=yes,
        )

    runner = LivePaperRunner(result.app_settings, result.env_settings)
    last_heartbeat = 0.0

    def on_event(event_type: str, payload: dict) -> None:
        line = format_live_event(event_type, payload)
        if event_type.startswith("order.") or event_type in (
            "signal.detected",
            "agent.execute",
        ):
            console.print(f"[bold cyan]» {line}[/bold cyan]")
        elif event_type in ("agent.decision",) and payload.get("action") in (
            "propose",
            "confirm",
            "enter",
        ):
            console.print(f"[bold green]» {line}[/bold green]")
        elif event_type in ("agent.propose", "agent.confirm_executed", "agent.outcome"):
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
        now = time.monotonic()
        if now - last_heartbeat < max(0.5, heartbeat_seconds):
            return
        last_heartbeat = now
        mode = state.get("execution_mode") or ""
        decisions = state.get("decision_calls")
        decision_bit = f"  ai={decisions}" if decisions is not None else ""
        console.print(
            f"  SPY ${state.get('market_price', '—')}  "
            f"health={state.get('feed_health')}  "
            f"state={state.get('engine_state')}  "
            f"mode={mode}{decision_bit}  "
            f"cap=${state.get('capital_available', '—')} "
            f"goal={state.get('capital_goal_pct', '—')}%  "
            f"signals={state.get('signals')}  "
            f"orders={state.get('trades_entered')}/{state.get('trades_exited')}  "
            f"open={state.get('open_trade')} pending={state.get('pending_order')}  "
            f"pnl=${state.get('paper_pnl', 0):.2f}  "
            f"broker={state.get('broker')}"
        )

    exec_mode = (
        result.app_settings.agents.execution_mode or "rules_hybrid"
    ).strip().lower()
    risk_policy = (result.app_settings.risk.policy or "strict").strip().lower()
    console.print(
        f"[bold]Starting live paper loop[/bold] — duration={duration_minutes}m, "
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
            duration_seconds=duration_minutes * 60.0,
            mock_agents=not use_openai,
            require_options=True,
            capital_budget=capital_budget,
            objective_service=objective_service,
            cognitive_session_id_override=session_id if objective_enabled else None,
        ),
        on_state=on_state,
        on_event=on_event,
    )

    for err in run_result.errors:
        console.print(f"[yellow]{err}[/yellow]")
    for fail in run_result.failures:
        console.print(f"[red]failure:[/red] {fail}")

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

    if run_result.failures or run_result.errors:
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
