"""Broker factory — local paper, Webull paper, Webull live."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from joker.app.safety import SafetyMode
from joker.broker.interface import BrokerClient, PaperBroker
from joker.broker.webull import WebullClient
from joker.config.settings import AppSettings, EnvSettings
from joker.persistence.broker_submission_journal import SyncBrokerSubmissionJournal


class BrokerFactoryError(Exception):
    pass


@dataclass(frozen=True)
class BrokerSelection:
    """Resolved broker for a session."""

    client: BrokerClient
    kind: str  # local_paper | webull_paper | webull_live
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
    journal_db_path: str | Path | None = None,
    activation: object | None = None,
    capture_only: bool = False,
) -> BrokerClient:
    """
    Create a broker client from config.

    - provider=paper (default): local PaperBroker — no Webull orders
    - provider=webull_paper: Webull paper-account API (requires env flags)
    - provider=webull_live: production WebullLiveClient (fail closed, no fallback)
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
                "Use provider=webull_live for production."
            )
        if not env.webull_paper_trading_enabled:
            raise BrokerFactoryError(
                "WEBULL_PAPER_TRADING_ENABLED must be true for Webull paper broker"
            )
        if env.webull_live_trading_enabled or app_settings.live_trading_enabled:
            raise BrokerFactoryError(
                "Refusing Webull paper broker: live money trading flags must remain false"
            )
        return WebullClient(env, trade_api=trade_api)  # type: ignore[arg-type]
    if provider == "webull_live":
        return create_live_broker(
            app_settings,
            env,
            trade_api=trade_api,
            journal_db_path=journal_db_path,
            activation=activation,
            capture_only=capture_only,
        )
    raise BrokerFactoryError(
        f"Unknown broker.provider={provider!r}. "
        "Use 'paper', 'webull_paper', or 'webull_live'."
    )


def create_live_broker(
    app_settings: AppSettings,
    env: EnvSettings,
    *,
    trade_api: object | None = None,
    activation: object | None = None,
    journal_db_path: str | Path | None = None,
    capture_only: bool = False,
    skip_account_list_check: bool = False,
    session_id: str | None = None,
    objective_id: str | None = None,
) -> BrokerClient:
    """Construct WebullLiveClient. Never falls back to paper or local PaperBroker.

    Placement-capable construction requires ``activation`` and ``journal_db_path``.
    ``capture_only=True`` may omit both for non-submitting tests.
    """
    from joker.broker.webull_live import WebullLiveClient
    from joker.persistence.session_pnl_baseline import SessionPnlBaselineStore

    if (app_settings.broker.provider or "").strip().lower() not in {
        "webull_live",
        "",
    } and app_settings.mode is not SafetyMode.LIVE_GATED:
        # Explicit factory call may omit provider when constructing for live runner.
        pass
    if app_settings.mode is not SafetyMode.LIVE_GATED:
        raise BrokerFactoryError(
            "create_live_broker requires mode LIVE_GATED — refusing fallback"
        )
    if not app_settings.live_trading_enabled:
        raise BrokerFactoryError(
            "create_live_broker requires live_trading_enabled=true — refusing fallback"
        )
    if not env.webull_live_trading_enabled:
        raise BrokerFactoryError(
            "create_live_broker requires WEBULL_LIVE_TRADING_ENABLED=true — refusing fallback"
        )
    if not capture_only:
        if activation is None:
            raise BrokerFactoryError(
                "create_live_broker requires LiveActivation for placement"
            )
        if journal_db_path is None:
            raise BrokerFactoryError(
                "create_live_broker requires journal_db_path for placement"
            )
    journal = None
    baseline_store = None
    if journal_db_path is not None:
        journal = SyncBrokerSubmissionJournal(journal_db_path)
        baseline_store = SessionPnlBaselineStore(journal_db_path)
    try:
        return WebullLiveClient(
            env,
            app_settings=app_settings,
            activation=activation,  # type: ignore[arg-type]
            trade_api=trade_api,  # type: ignore[arg-type]
            journal=journal,
            capture_only=capture_only,
            skip_account_list_check=skip_account_list_check,
            session_id=session_id,
            baseline_store=baseline_store,
            objective_id=objective_id,
        )
    except Exception as exc:
        raise BrokerFactoryError(
            f"webull_live initialization failed: {exc}. "
            "Refusing PaperBroker / webull_paper fallback."
        ) from exc


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
        from joker.broker.webull_live import WebullLiveClient

        if isinstance(broker, WebullLiveClient):
            raise BrokerFactoryError(
                "resolve_live_paper_broker refuses webull_live broker"
            )
        kind = "webull_paper" if isinstance(broker, WebullClient) else "local_paper"
        return BrokerSelection(
            client=broker,
            kind=kind,
            auto_orders=True,
            label="Webull paper account" if kind == "webull_paper" else "local PaperBroker",
        )

    yaml_provider = (app_settings.broker.provider or "paper").strip().lower()
    if yaml_provider == "webull_live":
        raise BrokerFactoryError(
            "resolve_live_paper_broker refuses provider=webull_live"
        )
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
