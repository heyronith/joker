"""Fail-closed all-task profile broker resolution tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from joker.app.safety import SafetyMode
from joker.broker.factory import (
    BrokerFactoryError,
    create_broker,
    resolve_live_paper_broker,
)
from joker.broker.interface import PaperBroker
from joker.broker.webull import WebullClient
from joker.config.loader import load_app_settings
from joker.config.settings import AppSettings, EnvSettings


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "paper-all-tasks.yaml"


def _env(**overrides: object) -> EnvSettings:
    base = {
        "OPENAI_API_KEY": "test-key",
        "WEBULL_PAPER_TRADING_ENABLED": "true",
        "WEBULL_PAPER_ACCOUNT_ID": "PAPER_ACCT_TEST",
        "WEBULL_LIVE_TRADING_ENABLED": "false",
        "WEBULL_TRADE_API_ENV": "sandbox",
        "WEBULL_TRADE_APP_KEY": "trade-key",
        "WEBULL_TRADE_APP_SECRET": "trade-secret",
        "WEBULL_TRADE_ACCESS_TOKEN": "trade-token",
    }
    base.update({k: str(v) if not isinstance(v, str) else v for k, v in overrides.items()})
    # Remove keys set to None sentinel
    cleaned = {k: v for k, v in base.items() if v is not None}
    return EnvSettings.model_validate(cleaned)


def test_paper_all_tasks_profile_pins_webull_paper() -> None:
    app, _env = load_app_settings(PROFILE)
    assert app.mode is SafetyMode.PAPER
    assert app.live_trading_enabled is False
    assert app.broker.provider == "webull_paper"
    assert app.broker.webull_paper_trading_enabled is True


def test_paper_all_tasks_resolves_webull_paper_when_env_valid() -> None:
    app, _ = load_app_settings(PROFILE)
    env = _env()
    spy_api = MagicMock()
    selection = resolve_live_paper_broker(app, env, trade_api=spy_api)
    assert selection.kind == "webull_paper"
    assert isinstance(selection.client, WebullClient)
    assert not isinstance(selection.client, PaperBroker)


def test_missing_paper_trading_enabled_fails() -> None:
    app, _ = load_app_settings(PROFILE)
    env = _env(WEBULL_PAPER_TRADING_ENABLED="false")
    with pytest.raises(BrokerFactoryError, match="WEBULL_PAPER"):
        resolve_live_paper_broker(app, env, trade_api=MagicMock())


def test_missing_paper_account_id_fails(monkeypatch) -> None:
    app, _ = load_app_settings(PROFILE)
    monkeypatch.delenv("WEBULL_PAPER_ACCOUNT_ID", raising=False)
    data = {
        "OPENAI_API_KEY": "test-key",
        "WEBULL_PAPER_TRADING_ENABLED": "true",
        "WEBULL_PAPER_ACCOUNT_ID": "",
        "WEBULL_LIVE_TRADING_ENABLED": "false",
        "WEBULL_TRADE_API_ENV": "sandbox",
        "WEBULL_TRADE_APP_KEY": "k",
        "WEBULL_TRADE_APP_SECRET": "s",
        "WEBULL_TRADE_ACCESS_TOKEN": "t",
    }
    env = EnvSettings.model_validate(data)
    env = env.model_copy(update={"webull_paper_account_id": None})
    assert not env.webull_paper_account_id
    with pytest.raises(BrokerFactoryError, match="WEBULL_PAPER_ACCOUNT_ID"):
        resolve_live_paper_broker(app, env, trade_api=MagicMock())


def test_live_trading_enabled_fails_at_settings_load() -> None:
    with pytest.raises(Exception, match="WEBULL_LIVE_TRADING_ENABLED"):
        EnvSettings.model_validate(
            {
                "OPENAI_API_KEY": "test-key",
                "WEBULL_LIVE_TRADING_ENABLED": "true",
            }
        )


def test_never_resolves_to_paper_broker() -> None:
    app, _ = load_app_settings(PROFILE)
    env = _env()
    selection = resolve_live_paper_broker(app, env, trade_api=MagicMock())
    assert not isinstance(selection.client, PaperBroker)
    with pytest.raises(BrokerFactoryError):
        # Even create_broker must not silently return PaperBroker for this provider.
        bad = env.model_copy(update={"webull_paper_trading_enabled": False})
        create_broker(app, bad)


def test_paper_run_cli_fails_closed_without_require_flag(monkeypatch) -> None:
    """broker.provider=webull_paper fails closed even without --require-webull-paper."""
    from typer.testing import CliRunner

    from joker.cli.paper import paper_app

    app, _ = load_app_settings(PROFILE)
    assert app.broker.provider == "webull_paper"

    class _Result:
        app_settings = app
        env_settings = _env(WEBULL_PAPER_TRADING_ENABLED="false")

    monkeypatch.setattr(
        "joker.config.validation.validate_startup",
        lambda **kwargs: _Result(),
    )
    runner = CliRunner()
    result = runner.invoke(
        paper_app,
        [
            "run",
            "--config",
            str(PROFILE),
            "--skip-preflight",
            "--skip-model-check",
            "--yes",
            "--authorized-capital",
            "50",
            "--target-profit-pct",
            "5",
            "--duration-minutes",
            "0.01",
        ],
    )
    assert result.exit_code != 0
    assert "webull_paper" in (result.output or "").lower() or "PaperBroker" in (
        result.output or ""
    )
