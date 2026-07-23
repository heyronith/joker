"""Shadow mode CLI — preflight checks and real-data shadow runs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

shadow_app = typer.Typer(help="Real-data shadow mode (no broker orders)")
console = Console()


@shadow_app.command("preflight")
def shadow_preflight(
    symbol: str = typer.Option("SPY", "--symbol"),
    provider: str = typer.Option("webull", "--provider", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
    skip_model_check: bool = typer.Option(True, "--skip-model-check"),
) -> None:
    """Verify Webull stock/options capability and OPRA compliance before shadow."""
    from joker.compliance.opra_scanner import discover_db_paths, scan_local_opra
    from joker.config.loader import load_app_settings
    from joker.config.validation import validate_startup
    from joker.data.webull_capability import capability_usable_for_shadow, load_capability

    if symbol.upper() != "SPY":
        console.print("[red]Only SPY is supported.[/red]")
        raise typer.Exit(code=1)

    result = validate_startup(config_path=config, skip_model_check=skip_model_check)
    app_settings = result.app_settings
    env = result.env_settings

    table = Table(title="Shadow preflight")
    table.add_column("Check")
    table.add_column("Status")

    table.add_row("Config valid", "[green]pass[/green]")
    table.add_row("Mode", app_settings.mode.value)
    table.add_row(
        "Webull market data enabled",
        "yes" if env.webull_market_data_enabled else "[red]no[/red]",
    )

    cap = load_capability()
    cap_ok = capability_usable_for_shadow()
    table.add_row(
        "Options capability (OPRA)",
        "[green]usable[/green]" if cap_ok else "[yellow]not verified[/yellow]",
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
    table.add_row("DB paths scanned", str(len(scan.db_paths_scanned)))

    console.print(table)

    if not env.webull_market_data_enabled:
        console.print("[red]Enable WEBULL_MARKET_DATA_ENABLED in .env[/red]")
        raise typer.Exit(code=1)
    if opra_violations > 0:
        console.print("[yellow]Run: joker compliance quarantine-opra-artifacts[/yellow]")
        raise typer.Exit(code=1)
    if not cap_ok:
        console.print("[yellow]Run: joker data verify-webull-options --symbol SPY[/yellow]")
        raise typer.Exit(code=1)

    console.print("[green]Shadow preflight passed — OPRA governance OK, options capability ready.[/green]")


@shadow_app.command("run")
def shadow_run(
    symbol: str = typer.Option("SPY", "--symbol"),
    provider: str = typer.Option("webull", "--provider", "-p"),
    duration_minutes: float = typer.Option(30.0, "--duration-minutes"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
    skip_model_check: bool = typer.Option(True, "--skip-model-check"),
) -> None:
    """Run a real-data shadow session — OPRA in memory only, safe metadata persisted."""
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
            f"options={'yes' if state.get('options_available') else 'no'}  "
            f"health={state.get('feed_health')}"
        )

    run_result = runner.run(
        WatchRunConfig(
            symbol=symbol,
            provider=provider,
            shadow=True,
            use_options=True,
            duration_seconds=duration_minutes * 60.0,
        ),
        on_state=on_state,
    )
    if run_result.errors:
        for err in run_result.errors:
            console.print(f"[yellow]{err}[/yellow]")
    console.print(
        f"[green]Shadow run complete[/green] — events={run_result.events_processed} "
        f"features={run_result.features_updated} health={run_result.feed_health}"
    )
    console.print("[yellow]SHADOW — no broker orders. OPRA values not persisted.[/yellow]")
