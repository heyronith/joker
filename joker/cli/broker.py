"""Broker CLI — Webull paper-account preflight and smoke (no live money)."""

from __future__ import annotations

from datetime import date
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

broker_app = typer.Typer(help="Webull paper-account broker commands (no live money)")
console = Console()


def _load_env():
    from joker.config.settings import EnvSettings

    return EnvSettings()  # type: ignore[call-arg]


@broker_app.command("preflight")
def broker_preflight(
    skip_network: bool = typer.Option(
        False,
        "--skip-network",
        help="Validate env only; do not call Webull",
    ),
) -> None:
    """Validate paper-trading env and optionally list accounts / balance."""
    from joker.broker.webull_trade_api import (
        HttpWebullTradeApi,
        WebullTradeConfigError,
        extract_cash_balance,
        validate_webull_paper_trade_env,
    )

    env = _load_env()
    try:
        validate_webull_paper_trade_env(env)
    except WebullTradeConfigError as exc:
        console.print(f"[red]Paper trading preflight failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    trade_env = env.trade_credentials_env()
    console.print("[green]Env OK[/green] — paper trading enabled, live money disabled")
    console.print(f"  account_id={env.webull_paper_account_id}")
    console.print(f"  market_api_env={env.webull_api_env}")
    console.print(f"  trade_api_env={trade_env.webull_api_env}")
    if env.webull_trade_app_key:
        console.print("  trade_credentials=WEBULL_TRADE_* (separate from market data)")
    else:
        console.print("  trade_credentials=shared WEBULL_APP_*")
    if env.webull_trade_app_id:
        console.print(f"  trade_app_id={env.webull_trade_app_id}")

    if skip_network:
        return

    try:
        api = HttpWebullTradeApi(env)
        accounts = api.list_accounts()
        balance = api.get_balance(str(env.webull_paper_account_id))
    except Exception as exc:
        console.print(f"[red]Network preflight failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Webull trade accounts")
    table.add_column("account_id")
    table.add_column("type")
    table.add_column("label")
    matched = False
    for row in accounts:
        aid = str(row.get("account_id", ""))
        if aid == env.webull_paper_account_id:
            matched = True
        table.add_row(
            aid,
            str(row.get("account_type", "")),
            str(row.get("account_label") or row.get("account_class") or ""),
        )
    console.print(table)
    if not matched:
        console.print(
            "[yellow]Warning: WEBULL_PAPER_ACCOUNT_ID not found in account list. "
            "Confirm you selected a paper/sandbox account.[/yellow]"
        )
    from joker.broker.webull_trade_api import account_looks_like_live_brokerage

    for row in accounts:
        if str(row.get("account_id", "")) != str(env.webull_paper_account_id):
            continue
        if account_looks_like_live_brokerage(row, api_env=trade_env.webull_api_env):
            console.print(
                "[red]Safety warning:[/red] configured account is labeled "
                f"[bold]{row.get('account_label') or row.get('account_class')}[/bold] "
                "on prod — that is a live brokerage account type, not app Paper Trading.\n"
                "Do [bold]not[/bold] run auto-orders against it unless you intentionally "
                "accept live-account risk (still blocked from LIVE money flag, but this "
                "account id is not simulated paper). Prefer Webull sandbox for API tests."
            )
    cash = extract_cash_balance(balance)
    console.print(f"[green]Balance query OK[/green] — cash≈${cash:,.2f}")
    # Show raw keys so mis-mapped balances are obvious.
    if isinstance(balance, dict):
        interesting = {
            k: balance.get(k)
            for k in (
                "total_cash_balance",
                "total_net_liquidation_value",
                "total_market_value",
                "total_day_profit_loss",
            )
            if k in balance
        }
        if interesting:
            console.print(f"[dim]Balance fields: {interesting}[/dim]")
    console.print(
        "[dim]WEBULL_LIVE_TRADING_ENABLED must stay false. "
        "Use WEBULL_TRADE_* + WEBULL_TRADE_API_ENV=sandbox for papertrade keys.[/dim]"
    )


@broker_app.command("accounts")
def broker_accounts() -> None:
    """List Webull accounts (requires paper trading env)."""
    from joker.broker.webull_trade_api import HttpWebullTradeApi, WebullTradeConfigError

    env = _load_env()
    try:
        api = HttpWebullTradeApi(env, require_account_id=False)
        accounts = api.list_accounts()
    except WebullTradeConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[red]Failed to list accounts:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Webull accounts — copy paper account_id into .env")
    table.add_column("account_id")
    table.add_column("type")
    table.add_column("label / class")
    for row in accounts:
        table.add_row(
            str(row.get("account_id", "")),
            str(row.get("account_type", "")),
            f"{row.get('account_label', '')} / {row.get('account_class', '')}",
        )
    console.print(table)
    console.print(
        "[dim]Set WEBULL_PAPER_ACCOUNT_ID to a sandbox/paper API account only. "
        "Do not use a funded live cash account.[/dim]"
    )
    for row in accounts:
        from joker.broker.webull_trade_api import account_looks_like_live_brokerage

        trade_env = env.trade_credentials_env()
        if account_looks_like_live_brokerage(row, api_env=trade_env.webull_api_env):
            console.print(
                "[yellow]Warning:[/yellow] "
                f"{row.get('account_id')} looks like a [bold]live brokerage[/bold] "
                f"account ({row.get('account_label') or row.get('account_class')}), "
                "not Webull app Paper Trading.\n"
                "  • App Paper Trading is a separate product and usually is [bold]not[/bold] "
                "this OpenAPI account list.\n"
                "  • For risk-free API order tests, use Webull [bold]sandbox[/bold] "
                "(api.sandbox.webull.com / WEBULL_TRADE_API_ENV=sandbox with papertrade keys), "
                "not prod Individual Cash.\n"
                "  • $0 cash on Individual Cash usually means an unfunded/empty live "
                "account — orders here can still be real-money once funded."
            )
            break


@broker_app.command("smoke-place")
def broker_smoke_place(
    strike: float = typer.Option(..., help="SPY option strike"),
    option_type: str = typer.Option("call", help="call or put"),
    limit_price: float = typer.Option(
        0.01,
        help="Limit price (use a low value so the order is unlikely to fill)",
    ),
    quantity: int = typer.Option(1, help="Contracts"),
    expiration: Optional[str] = typer.Option(
        None, help="YYYY-MM-DD (default: today, 0DTE)"
    ),
    confirm_place: bool = typer.Option(
        False,
        "--confirm-place",
        help="Actually submit to Webull paper account (otherwise dry-run only)",
    ),
    cancel_after: bool = typer.Option(
        True,
        "--cancel-after/--leave-open",
        help="Cancel immediately after place (recommended)",
    ),
) -> None:
    """Dry-run or place+cancel a SPY 0DTE limit on the paper account."""
    from joker.broker.webull import WebullClient
    from joker.broker.webull_trade_api import (
        build_option_limit_order_payload,
        validate_webull_paper_trade_env,
        WebullTradeConfigError,
    )
    from joker.schemas.domain import OptionContract, OrderIntent

    env = _load_env()
    try:
        validate_webull_paper_trade_env(env)
    except WebullTradeConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    opt = option_type.strip().lower()
    if opt not in {"call", "put"}:
        console.print("[red]option_type must be call or put[/red]")
        raise typer.Exit(code=1)

    exp = date.fromisoformat(expiration) if expiration else date.today()
    contract = OptionContract(
        symbol="SPY",
        expiration=exp,
        strike=strike,
        option_type=opt,  # type: ignore[arg-type]
        is_0dte=True,
    )
    intent = OrderIntent(
        candidate_id="smoke",
        contract=contract,
        side="buy",
        order_type="limit",
        quantity=quantity,
        limit_price=limit_price,
    )
    payload = build_option_limit_order_payload(intent)
    console.print("[blue]Order payload (redacted account):[/blue]")
    console.print(payload)

    if not confirm_place:
        console.print(
            "[yellow]Dry-run only.[/yellow] Re-run with --confirm-place to submit "
            "to the Webull paper account, then ideally cancel."
        )
        return

    console.print(
        f"[yellow]Submitting to paper account {env.webull_paper_account_id}...[/yellow]"
    )
    client = WebullClient(env)
    try:
        order = client.submit_order(intent)
    except Exception as exc:
        console.print(f"[red]Place failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Placed[/green] client_order_id={order.order_id} status={order.status}"
    )
    if cancel_after:
        try:
            cancelled = client.cancel_order(order.order_id)
            console.print(f"[green]Cancelled[/green] status={cancelled.status}")
        except Exception as exc:
            console.print(
                f"[yellow]Place succeeded but cancel failed:[/yellow] {exc}. "
                f"Cancel manually: client_order_id={order.order_id}"
            )
            raise typer.Exit(code=2) from exc
