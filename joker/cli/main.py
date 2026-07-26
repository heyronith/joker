"""Developer CLI commands (debugging, testing, automation)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from joker.config.validation import ConfigValidationError, validate_startup

app = typer.Typer(
    name="joker",
    help="joker — local AI-assisted SPY 0DTE options trading research",
    no_args_is_help=False,
    invoke_without_command=True,
)
config_app = typer.Typer(help="Configuration commands")
app.add_typer(config_app, name="config")

from joker.cli.evolve import evolve_app

app.add_typer(evolve_app, name="evolve")

console = Console()


def _launch_tui() -> None:
    from joker.tui.app import JokerApp

    JokerApp().run()


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config profile",
        envvar="JOKER_CONFIG",
    ),
) -> None:
    """Launch interactive TUI when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return
    _launch_tui()


@config_app.command("validate")
def config_validate(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config profile",
        envvar="JOKER_CONFIG",
    ),
    skip_model_check: bool = typer.Option(
        False,
        "--skip-model-check",
        help="Skip OpenAI model availability check (for CI/dev)",
    ),
) -> None:
    """Validate configuration, environment, and model availability."""
    try:
        result = validate_startup(
            config_path=config,
            skip_model_check=skip_model_check,
        )
    except ConfigValidationError as exc:
        console.print(f"[red]Configuration invalid:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="joker configuration")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("Mode", result.app_settings.mode.value)
    table.add_row(
        "Live trading enabled",
        str(result.app_settings.live_trading_enabled),
    )
    table.add_row("Mock agents", str(result.app_settings.agents.mock_agents))
    table.add_row("OpenAI model", result.env_settings.openai_model)
    table.add_row("Data dir", str(result.app_settings.data_dir))
    table.add_row("DB path", str(result.app_settings.db_path))
    console.print(table)
    console.print("[green]Configuration valid.[/green]")
    if not skip_model_check:
        console.print(
            f"[green]OPENAI_MODEL '{result.env_settings.openai_model}' is available.[/green]"
        )


@config_app.command("show")
def config_show(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config profile",
        envvar="JOKER_CONFIG",
    ),
) -> None:
    """Show merged configuration (secrets redacted)."""
    from joker.config.loader import load_app_settings
    from joker.config.validation import redact_secrets

    app_settings, env_settings = load_app_settings(config_path=config)
    console.print(app_settings.model_dump_json(indent=2))
    console.print(
        redact_secrets(
            f"OpenAI model: {env_settings.openai_model}",
            env=env_settings,
        )
    )


@app.command("premarket")
def premarket_run(
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
    trading_day: Optional[str] = typer.Option(None, help="YYYY-MM-DD"),
    skip_model_check: bool = typer.Option(False, "--skip-model-check"),
    mock_agents: bool = typer.Option(
        False,
        "--mock-agents",
        help="Use deterministic local agents (offline)",
    ),
    use_openai: bool = typer.Option(
        False,
        "--use-openai",
        help="Use OpenAI-backed agents (requires valid OPENAI_API_KEY)",
    ),
) -> None:
    """Run premarket council workflow (developer command)."""
    from datetime import date

    from joker.config.loader import load_app_settings
    from joker.config.validation import validate_startup
    from joker.data.mock_provider import mock_spy_snapshot
    from joker.features.engine import FeatureEngine
    from joker.logging.event_log import EventLogWriter
    from joker.runtime.premarket import PremarketWorkflow
    from joker.runtime.run_manager import RunManager
    from joker.storage.database import ensure_database

    if mock_agents and use_openai:
        console.print("[red]Choose either --mock-agents or --use-openai, not both.[/red]")
        raise typer.Exit(code=1)

    result = validate_startup(
        config_path=config,
        skip_model_check=skip_model_check or mock_agents,
    )
    app_settings = result.app_settings
    if mock_agents:
        app_settings = app_settings.model_copy(
            update={"agents": app_settings.agents.model_copy(update={"mock_agents": True})}
        )
    elif use_openai:
        app_settings = app_settings.model_copy(
            update={"agents": app_settings.agents.model_copy(update={"mock_agents": False})}
        )
        if skip_model_check:
            console.print(
                "[yellow]Warning: running OpenAI agents without model validation.[/yellow]"
            )

    day = date.fromisoformat(trading_day) if trading_day else date.today()
    db = ensure_database(app_settings.db_path)
    event_log = EventLogWriter(
        app_settings.event_log_dir,
        redact_keys=app_settings.logging.redact_env_keys,
    )
    run_manager = RunManager(db, event_log, app_settings)
    run_id = run_manager.start_run(trading_day=day)
    features = FeatureEngine(max_age_seconds=999999).compute(mock_spy_snapshot())
    workflow = PremarketWorkflow(db, event_log, app_settings)
    mode_label = "mock" if app_settings.agents.mock_agents else "openai"
    console.print(f"[blue]Running premarket council ({mode_label} agents)...[/blue]")
    playbook = workflow.run(run_id, day, features, env_settings=result.env_settings)
    console.print(f"[green]Premarket playbook created:[/green] {playbook.title}")
    console.print(f"Report: {app_settings.reports_dir}/premarket/{day.isoformat()}.md")
    run_manager.end_run(run_id)


replay_app = typer.Typer(help="Replay market data sessions")
app.add_typer(replay_app, name="replay")


@replay_app.command("list")
def replay_list(
    replays_dir: Optional[str] = typer.Option(None, help="Directory containing replay JSONL files"),
) -> None:
    """List available replay files."""
    from joker.config.loader import load_app_settings

    app_settings, _ = load_app_settings()
    directory = Path(replays_dir) if replays_dir else Path("data/replays")
    if not directory.exists():
        console.print(f"[yellow]No replay directory at {directory}[/yellow]")
        raise typer.Exit(code=0)
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        console.print("[yellow]No replay files found.[/yellow]")
        raise typer.Exit(code=0)
    table = Table(title="Replay files")
    table.add_column("File")
    table.add_column("Synthetic")
    for f in files:
        synthetic = "yes" if "synthetic" in f.name.lower() else "unknown"
        table.add_row(str(f.name), synthetic)
    console.print(table)


@replay_app.command("inspect")
def replay_inspect(file: str) -> None:
    """Inspect a replay JSONL file."""
    from joker.data.replay_loader import inspect_replay

    path = Path(file)
    if not path.exists():
        path = Path("data/replays") / file
    try:
        info = inspect_replay(path)
    except Exception as exc:
        console.print(f"[red]Failed to inspect replay:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    table = Table(title=f"Replay: {info['name']}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in info.items():
        if key != "event_types":
            table.add_row(key, str(value))
    console.print(table)
    if info.get("event_types"):
        console.print(f"Event types: {info['event_types']}")
    if info.get("is_synthetic"):
        console.print("[yellow]Synthetic replay — not real market data.[/yellow]")


@replay_app.command("run")
def replay_run(
    file: str,
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
    mock_agents: bool = typer.Option(True, "--mock-agents", help="Use mock agents (default)"),
    use_openai: bool = typer.Option(False, "--use-openai", help="Use OpenAI agents"),
    speed: float = typer.Option(0.0, "--speed", help="Accelerated speed multiplier (0=deterministic)"),
    deterministic: bool = typer.Option(True, "--deterministic/--realtime", help="Deterministic mode"),
    skip_model_check: bool = typer.Option(True, "--skip-model-check"),
) -> None:
    """Run a full replay session."""
    from joker.config.validation import validate_startup
    from joker.runtime.replay_runner import ReplayRunConfig, ReplayRunner

    if mock_agents and use_openai:
        console.print("[red]Choose either --mock-agents or --use-openai.[/red]")
        raise typer.Exit(code=1)

    result = validate_startup(config_path=config, skip_model_check=skip_model_check)
    path = Path(file)
    if not path.exists():
        path = Path("data/replays") / file
    if not path.exists():
        console.print(f"[red]Replay file not found:[/red] {file}")
        raise typer.Exit(code=1)

    runner = ReplayRunner(result.app_settings, result.env_settings)
    run_result = runner.run(
        ReplayRunConfig(
            replay_path=path,
            deterministic=deterministic or speed <= 0,
            speed=speed if speed > 0 else 1.0,
            mock_agents=mock_agents and not use_openai,
        )
    )
    summary = run_result.summary
    if run_result.failures:
        console.print(f"[yellow]Replay completed with {len(run_result.failures)} warning(s)/failure(s)[/yellow]")
        for f in run_result.failures:
            console.print(f"  - {f}")
    label = "SYNTHETIC" if summary.is_synthetic else "REPLAY"
    console.print(f"[green]{label} replay completed[/green]")
    console.print(f"Events: {summary.events_processed}")
    console.print(f"Signals: {summary.signals_detected}")
    console.print(f"Trades entered: {summary.trades_entered}")
    console.print(f"Trades exited: {summary.trades_exited}")
    console.print(f"P&L: ${summary.final_pnl_usd:,.2f}")
    console.print(f"Risk rejections: {summary.risk_rejections}")
    if run_result.report_path:
        console.print(f"Report: {run_result.report_path}")


from joker.cli.data import data_app

app.add_typer(data_app, name="data")

from joker.cli.compliance import compliance_app

app.add_typer(compliance_app, name="compliance")

from joker.cli.shadow import shadow_app

app.add_typer(shadow_app, name="shadow")

from joker.cli.paper import paper_app

app.add_typer(paper_app, name="paper")

from joker.cli.broker import broker_app

app.add_typer(broker_app, name="broker")


@app.command("watch")
def watch_spy(
    symbol: str = typer.Argument("SPY", help="Symbol (SPY only)"),
    provider: str = typer.Option("webull", "--provider", "-p"),
    shadow: bool = typer.Option(True, "--shadow/--paper", help="Shadow mode (no orders)"),
    duration_seconds: Optional[float] = typer.Option(
        None, "--duration-seconds", help="Stream duration (default: single snapshot)"
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
    skip_model_check: bool = typer.Option(True, "--skip-model-check"),
) -> None:
    """Watch live SPY market data — features and logging only, no broker execution."""
    from joker.config.validation import validate_startup
    from joker.runtime.watch_runner import WatchRunConfig, WatchRunner

    if symbol.upper() != "SPY":
        console.print("[red]Only SPY is supported.[/red]")
        raise typer.Exit(code=1)

    result = validate_startup(config_path=config, skip_model_check=skip_model_check)
    runner = WatchRunner(result.app_settings, result.env_settings)

    def on_state(state: dict) -> None:
        console.print(
            f"  SPY ${state.get('market_price', '—')}  "
            f"health={state.get('feed_health')}  "
            f"provider={state.get('provider')}"
        )

    run_result = runner.run(
        WatchRunConfig(
            symbol=symbol,
            provider=provider,
            shadow=shadow,
            duration_seconds=duration_seconds,
        ),
        on_state=on_state if duration_seconds else None,
    )
    if run_result.errors:
        for err in run_result.errors:
            console.print(f"[yellow]{err}[/yellow]")
    console.print(
        f"[green]Watch complete[/green] — events={run_result.events_processed} "
        f"features={run_result.features_updated} health={run_result.feed_health}"
    )
    console.print("[yellow]Market-data only — no broker orders submitted.[/yellow]")
    if not run_result.options_available:
        console.print("[yellow]Options data unavailable in this phase.[/yellow]")
