"""Minimal live-trading operational commands (final UI selector not built here)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

live_app = typer.Typer(help="Real-money LIVE_GATED operational commands (minimal)")
console = Console()


@live_app.command("preflight")
def live_preflight(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", envvar="JOKER_CONFIG"
    ),
    skip_network: bool = typer.Option(
        False, "--skip-network", help="Validate env/SQLite only"
    ),
) -> None:
    """Production read-only preflight — never places orders."""
    from joker.app.safety import SafetyMode
    from joker.config.loader import load_app_settings
    from joker.config.settings import EnvSettings
    from joker.runtime.live_preflight import run_production_preflight

    app, _ = load_app_settings(config)
    # Allow inspecting live env even if YAML still says PAPER for dry checks.
    if app.mode is not SafetyMode.LIVE_GATED:
        console.print(
            "[yellow]Warning:[/yellow] config mode is not LIVE_GATED; "
            "preflight will report mode failure."
        )
    env = EnvSettings()  # type: ignore[call-arg]
    report = run_production_preflight(
        app_settings=app,
        env=env,
        skip_network=skip_network,
    )
    for line in report.checks:
        color = "red" if "fail" in line else "green"
        console.print(f"[{color}]{line}[/{color}]")
    if report.account_id_masked:
        console.print(f"account={report.account_id_masked} hash={report.account_id_hash}")
    console.print(f"mutated={report.mutated}")
    if not report.ok:
        raise typer.Exit(code=1)
    console.print("[green]Live preflight OK (read-only)[/green]")


@live_app.command("help-status")
def live_help_status() -> None:
    """Remind operators that live mode requires process-local activation."""
    console.print(
        "Real-money trading requires:\n"
        "  mode=LIVE_GATED\n"
        "  live_trading_enabled=true\n"
        "  WEBULL_LIVE_* credentials\n"
        "  process-local LiveActivation\n"
        "  startup reconciliation\n"
        "Final paper/live CLI selector is not built in this task."
    )
