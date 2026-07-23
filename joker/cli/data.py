"""CLI commands for market data providers (data-only)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
data_app = typer.Typer(help="Market data commands (no broker execution)")
spy_app = typer.Typer(help="SPY stock data (market-data only)")
data_app.add_typer(spy_app, name="spy")


@data_app.command("providers")
def data_providers() -> None:
    """List available market data providers."""
    from joker.data.provider_factory import list_providers

    table = Table(title="Market data providers")
    table.add_column("Provider")
    table.add_column("Description")
    for name, desc in list_providers():
        table.add_row(name, desc)
    console.print(table)
    console.print("[yellow]Market-data only — no options, orders, or account access.[/yellow]")


@data_app.command("validate")
def data_validate(
    provider: str = typer.Option("webull", "--provider", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Validate provider configuration (Webull credentials when provider=webull)."""
    from joker.config.loader import load_app_settings
    from joker.config.validation import safe_error_message
    from joker.data.provider_factory import ProviderKind
    from joker.data.webull_config import WebullMarketConfigError, validate_webull_market_env

    try:
        app_settings, env_settings = load_app_settings(config_path=config)
    except Exception as exc:
        console.print(f"[red]Config load failed:[/red] {safe_error_message(exc)}")
        raise typer.Exit(code=1) from exc

    kind = ProviderKind.from_string(provider)
    if kind is ProviderKind.WEBULL:
        try:
            validate_webull_market_env(env_settings)
        except WebullMarketConfigError as exc:
            console.print(f"[red]Webull config invalid:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print("[green]Webull market-data configuration valid.[/green]")
        console.print(f"Region: {env_settings.webull_region}")
        console.print(f"API env: {env_settings.webull_api_env}")
        console.print(f"Market data enabled flag: {env_settings.webull_market_data_enabled}")
    elif kind is ProviderKind.MOCK:
        console.print("[green]Mock provider requires no credentials.[/green]")
    elif kind is ProviderKind.REPLAY:
        console.print("[green]Replay provider requires a JSONL file path at run time.[/green]")


@data_app.command("diagnose-webull")
def data_diagnose_webull(
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Run Webull subscription and permission diagnostics."""
    from joker.config.loader import load_app_settings
    from joker.config.validation import safe_error_message
    from joker.data.webull_diagnostics import run_webull_diagnostics

    try:
        _, env_settings = load_app_settings(config_path=config)
        report = run_webull_diagnostics(env_settings)
    except Exception as exc:
        console.print(f"[red]Diagnostics failed:[/red] {safe_error_message(exc)}")
        raise typer.Exit(code=1) from exc

    for line in report.to_lines():
        if ": fail" in line:
            console.print(f"[red]{line}[/red]")
        elif ": pass" in line:
            console.print(f"[green]{line}[/green]")
        else:
            console.print(line)
    if report.likely_issue:
        console.print(f"[yellow]Likely issue:[/yellow] {report.likely_issue}")
        raise typer.Exit(code=1)


def _load_webull_provider(config: Optional[str], api: object | None = None):
    from joker.config.loader import load_app_settings
    from joker.data.provider_factory import create_market_provider

    app_settings, env_settings = load_app_settings(config_path=config)
    return create_market_provider(
        "webull",
        app_settings=app_settings,
        env_settings=env_settings,
        webull_api=api,
    )


@spy_app.command("snapshot")
def spy_snapshot(
    provider: str = typer.Option("webull", "--provider", "-p"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Fetch a single SPY stock snapshot."""
    from joker.data.webull_config import WebullMarketConfigError, safe_webull_error
    from joker.data.webull_market_provider import WebullMarketDataProvider

    if provider != "webull":
        console.print("[red]SPY snapshot requires --provider webull[/red]")
        raise typer.Exit(code=1)
    try:
        market_provider = _load_webull_provider(config)
    except WebullMarketConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    assert isinstance(market_provider, WebullMarketDataProvider)
    try:
        event = market_provider.fetch_snapshot_event()
        console.print(
            f"SPY ${event.price:,.2f}  bid={event.bid} ask={event.ask}  "
            f"source={event.source}  ts={event.timestamp.isoformat()}"
        )
    except Exception as exc:
        console.print(f"[red]Data error:[/red] {safe_webull_error(exc)}")
        raise typer.Exit(code=1) from exc


@spy_app.command("candles")
def spy_candles(
    provider: str = typer.Option("webull", "--provider", "-p"),
    timeframe: str = typer.Option("1m", "--timeframe"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Fetch SPY historical candles."""
    from joker.data.webull_config import WebullMarketConfigError, safe_webull_error
    from joker.data.webull_market_provider import WebullMarketDataProvider

    if provider != "webull":
        console.print("[red]SPY candles requires --provider webull[/red]")
        raise typer.Exit(code=1)
    try:
        market_provider = _load_webull_provider(config)
    except WebullMarketConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    assert isinstance(market_provider, WebullMarketDataProvider)
    try:
        events = market_provider.fetch_candle_events(timeframe)
        console.print(f"Fetched {len(events)} candle(s) ({timeframe})")
        for ev in events[-5:]:
            c = ev.candle
            console.print(
                f"  {c.timestamp.isoformat()} O={c.open} H={c.high} "
                f"L={c.low} C={c.close} V={c.volume}"
            )
    except Exception as exc:
        console.print(f"[red]Data error:[/red] {safe_webull_error(exc)}")
        raise typer.Exit(code=1) from exc


@spy_app.command("stream")
def spy_stream(
    provider: str = typer.Option("webull", "--provider", "-p"),
    duration_seconds: float = typer.Option(60.0, "--duration-seconds"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Stream SPY stock quotes for a duration."""
    from joker.data.webull_config import WebullMarketConfigError, safe_webull_error
    from joker.data.webull_market_provider import WebullMarketDataProvider

    if provider != "webull":
        console.print("[red]SPY stream requires --provider webull[/red]")
        raise typer.Exit(code=1)
    try:
        market_provider = _load_webull_provider(config)
    except WebullMarketConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    assert isinstance(market_provider, WebullMarketDataProvider)
    try:
        market_provider.prepare_stream(duration_seconds=duration_seconds)
        count = 0
        for event in market_provider.stream_events():
            if hasattr(event, "price"):
                console.print(
                    f"  quote ${event.price:,.2f} @ {event.timestamp.isoformat()}"
                )
            count += 1
        console.print(f"[green]Stream complete — {count} event(s)[/green]")
    except Exception as exc:
        console.print(f"[red]Data error:[/red] {safe_webull_error(exc)}")
        raise typer.Exit(code=1) from exc


options_app = typer.Typer(help="SPY 0DTE options data (verification only)")
data_app.add_typer(options_app, name="options")


@data_app.command("diagnose-options")
def data_diagnose_options(
    provider: str = typer.Option("webull", "--provider", "-p"),
    symbol: str = typer.Option("SPY", "--symbol", "-s"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Run SPY 0DTE options data diagnostics."""
    from joker.config.loader import load_app_settings
    from joker.config.validation import safe_error_message
    from joker.data.options_diagnostics import run_options_diagnostics

    if provider != "webull":
        console.print("[red]Options diagnostics currently require --provider webull[/red]")
        raise typer.Exit(code=1)
    try:
        _, env_settings = load_app_settings(config_path=config)
        report = run_options_diagnostics(env_settings, symbol=symbol)
    except Exception as exc:
        console.print(f"[red]Diagnostics failed:[/red] {safe_error_message(exc)}")
        raise typer.Exit(code=1) from exc

    for line in report.to_lines():
        if "fail" in line.lower() and ": pass" not in line:
            console.print(f"[red]{line}[/red]")
        elif ": pass" in line or ": yes" in line:
            console.print(f"[green]{line}[/green]")
        else:
            console.print(line)
    if report.likely_issue:
        console.print(f"[yellow]Likely issue:[/yellow] {report.likely_issue}")
        console.print(f"[yellow]Next action:[/yellow] {_options_next_action(report)}")
        raise typer.Exit(code=1)
    console.print("[green]Options data verification passed required fields.[/green]")


def _options_next_action(report) -> str:
    from joker.data.webull_verification import _next_action

    return _next_action(report)


@data_app.command("webull-auth")
def data_webull_auth(
    wait_seconds: float = typer.Option(
        0.0,
        "--wait",
        help="Seconds to poll token status after create (use after SMS verify in app)",
    ),
    trade: bool = typer.Option(
        False,
        "--trade",
        help="Auth using WEBULL_TRADE_* paper/sandbox keys (saves WEBULL_TRADE_ACCESS_TOKEN)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Ignore any existing token and create a new one",
    ),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Create Webull access token from app key+secret (no pre-made token required)."""
    from joker.config.loader import load_app_settings
    from joker.data.webull_auth_flow import run_webull_auth_flow
    from joker.data.webull_config import safe_webull_error

    try:
        _, env_settings = load_app_settings(config_path=config)
        if trade and not (
            env_settings.webull_trade_app_key and env_settings.webull_trade_app_secret
        ):
            console.print(
                "[red]--trade requires WEBULL_TRADE_APP_KEY and WEBULL_TRADE_APP_SECRET[/red]"
            )
            raise typer.Exit(code=1)
        result = run_webull_auth_flow(
            env_settings,
            wait_seconds=wait_seconds,
            for_trade=trade,
            force_recreate=force,
        )
    except Exception as exc:
        console.print(f"[red]Auth failed:[/red] {safe_webull_error(exc)}")
        raise typer.Exit(code=1) from exc

    console.print(f"API env: {result.api_env}")
    console.print(f"Token status: {result.status}")
    console.print(f"Token present: {result.token_present}")
    if result.token_saved_path:
        console.print(f"Token saved to: {result.token_saved_path}")
        dest = "WEBULL_TRADE_ACCESS_TOKEN" if trade else "WEBULL_ACCESS_TOKEN"
        console.print(f"[dim]Copy the token value into {dest} in .env[/dim]")
    if result.success:
        if trade:
            console.print(
                "[green]Trade auth ready — run joker broker accounts / preflight[/green]"
            )
        else:
            console.print(
                "[green]Webull auth ready — run joker data verify-webull-options[/green]"
            )
        return
    console.print(f"[yellow]{result.message}[/yellow]")
    raise typer.Exit(code=1)


@data_app.command("capture-webull-contract")
def data_capture_webull_contract(
    symbol: str = typer.Option("SPY", "--symbol", "-s"),
    include_options: bool = typer.Option(True, "--include-options/--no-include-options"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Capture redacted Webull response shapes for endpoint contract verification."""
    from joker.config.loader import load_app_settings
    from joker.data.webull_config import safe_webull_error
    from joker.data.webull_contract_capture import capture_webull_contract

    try:
        _, env_settings = load_app_settings(config_path=config)
        path, summaries = capture_webull_contract(
            env_settings,
            symbol=symbol,
            include_options=include_options,
        )
    except Exception as exc:
        console.print(f"[red]Capture failed:[/red] {safe_webull_error(exc)}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Captured {len(summaries)} shape summary(ies)[/green]")
    console.print(f"Output: {path}")


@data_app.command("verify-webull-options")
def data_verify_webull_options(
    symbol: str = typer.Option("SPY", "--symbol", "-s"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Run full Webull options verification and write Markdown report + capability cache."""
    from joker.config.loader import load_app_settings
    from joker.data.webull_config import safe_webull_error
    from joker.data.webull_verification import generate_verification_report

    try:
        _, env_settings = load_app_settings(config_path=config)
        path, report, capability = generate_verification_report(env_settings, symbol=symbol)
    except Exception as exc:
        console.print(f"[red]Verification failed:[/red] {safe_webull_error(exc)}")
        raise typer.Exit(code=1) from exc

    console.print(f"Report: {path}")
    console.print(f"Capability usable for shadow: {capability.usable_for_shadow}")
    if report.likely_issue:
        console.print(f"[yellow]Likely issue:[/yellow] {report.likely_issue}")
        raise typer.Exit(code=1)
    console.print("[green]Webull options verification passed.[/green]")


@options_app.command("snapshot")
def options_snapshot(
    provider: str = typer.Option("webull", "--provider", "-p"),
    symbol: str = typer.Option("SPY", "--symbol", "-s"),
    expiration: str = typer.Option("today", "--expiration", help="today or YYYY-MM-DD"),
    config: Optional[str] = typer.Option(None, "--config", "-c", envvar="JOKER_CONFIG"),
) -> None:
    """Capture ATM call/put option snapshots to JSONL."""
    from joker.config.loader import load_app_settings
    from joker.data.options_capture import capture_options_snapshot
    from joker.data.webull_config import safe_webull_error

    if provider != "webull":
        console.print("[red]Options snapshot requires --provider webull[/red]")
        raise typer.Exit(code=1)
    try:
        _, env_settings = load_app_settings(config_path=config)
        path, summary = capture_options_snapshot(
            env_settings,
            symbol=symbol,
            expiration=expiration,
        )
    except Exception as exc:
        console.print(f"[red]Capture failed:[/red] {safe_webull_error(exc)}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Captured {summary['snapshot_count']} snapshot(s)[/green]")
    console.print(f"Underlying: ${summary['underlying_price']:,.2f}")
    console.print(f"Expiration: {summary['expiration']}")
    console.print(f"Real Webull data: {summary['is_real_webull_data']}")
    console.print(f"Output: {path}")
