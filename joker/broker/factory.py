"""Broker factory — paper local simulator vs Webull paper account."""

from __future__ import annotations

from dataclasses import dataclass

from joker.broker.interface import BrokerClient, PaperBroker
from joker.broker.webull import WebullClient
from joker.config.settings import AppSettings, EnvSettings


class BrokerFactoryError(Exception):
    pass


@dataclass(frozen=True)
class BrokerSelection:
    """Resolved broker for a session (never real-money live)."""

    client: BrokerClient
    kind: str  # local_paper | webull_paper
    auto_orders: bool
    label: str


def webull_paper_env_ready(env: EnvSettings) -> bool:
    return bool(
        env.webull_paper_trading_enabled
        and env.webull_paper_account_id
        and str(env.webull_paper_account_id).strip()
        and not env.webull_live_trading_enabled
    )


def create_broker(
    app_settings: AppSettings,
    env: EnvSettings,
    *,
    trade_api: object | None = None,
) -> BrokerClient:
    """
    Create a broker client from config.

    - provider=paper (default): local PaperBroker — no Webull orders
    - provider=webull_paper: Webull paper-account API (requires env flags)
    """
    provider = (app_settings.broker.provider or "paper").strip().lower()
    if provider == "paper":
        return PaperBroker(
            initial_balance=app_settings.paper.initial_balance_usd,
            slippage_pct=app_settings.paper.slippage_pct,
            default_spread_pct=app_settings.paper.default_spread_pct,
        )
    if provider in {"webull_paper", "webull"}:
        # "webull" alias only allowed when paper trading is enabled — never live.
        if not env.webull_paper_trading_enabled and not app_settings.broker.webull_paper_trading_enabled:
            raise BrokerFactoryError(
                "broker.provider=webull_paper requires WEBULL_PAPER_TRADING_ENABLED=true "
                "(or broker.webull_paper_trading_enabled in YAML). "
                "Real-money live remains disabled."
            )
        # Prefer env flag; YAML flag can enable when env also set.
        if not env.webull_paper_trading_enabled:
            raise BrokerFactoryError(
                "WEBULL_PAPER_TRADING_ENABLED must be true for Webull paper broker"
            )
        if env.webull_live_trading_enabled or app_settings.live_trading_enabled:
            raise BrokerFactoryError(
                "Refusing Webull broker: live money trading flags must remain false"
            )
        return WebullClient(env, trade_api=trade_api)  # type: ignore[arg-type]
    raise BrokerFactoryError(
        f"Unknown broker.provider={provider!r}. Use 'paper' or 'webull_paper'."
    )


def resolve_live_paper_broker(
    app_settings: AppSettings,
    env: EnvSettings,
    *,
    trade_api: object | None = None,
    broker: BrokerClient | None = None,
) -> BrokerSelection:
    """
    Resolve broker for live paper sessions.

    If WEBULL_PAPER_TRADING_ENABLED + account id are set, prefer Webull paper-account
    auto-orders (even when YAML still says provider=paper). Real money stays off.
    """
    if broker is not None:
        kind = "webull_paper" if isinstance(broker, WebullClient) else "local_paper"
        return BrokerSelection(
            client=broker,
            kind=kind,
            auto_orders=True,
            label="Webull paper account" if kind == "webull_paper" else "local PaperBroker",
        )

    yaml_provider = (app_settings.broker.provider or "paper").strip().lower()
    use_webull_paper = yaml_provider in {"webull_paper", "webull"} or webull_paper_env_ready(
        env
    )

    if use_webull_paper:
        if env.webull_live_trading_enabled or app_settings.live_trading_enabled:
            raise BrokerFactoryError(
                "Refusing Webull paper broker: live money flags must remain false"
            )
        if not webull_paper_env_ready(env):
            raise BrokerFactoryError(
                "Webull paper broker requires WEBULL_PAPER_TRADING_ENABLED=true, "
                "WEBULL_PAPER_ACCOUNT_ID, and WEBULL_LIVE_TRADING_ENABLED=false"
            )
        client = WebullClient(env, trade_api=trade_api)  # type: ignore[arg-type]
        return BrokerSelection(
            client=client,
            kind="webull_paper",
            auto_orders=True,
            label="Webull paper account (auto orders)",
        )

    client = PaperBroker(
        initial_balance=app_settings.paper.initial_balance_usd,
        slippage_pct=app_settings.paper.slippage_pct,
        default_spread_pct=app_settings.paper.default_spread_pct,
    )
    return BrokerSelection(
        client=client,
        kind="local_paper",
        auto_orders=True,
        label="local PaperBroker (simulated fills)",
    )
