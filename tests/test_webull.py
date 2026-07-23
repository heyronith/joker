"""Phase 12 Webull adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from joker.app.safety import SafetyMode
from joker.broker.webull import MockWebullClient, WebullConfigError, validate_webull_env
from joker.config.loader import load_app_settings
from joker.config.settings import EnvSettings


def test_missing_webull_env_fails_when_validated() -> None:
    env = EnvSettings()  # type: ignore[call-arg]
    with pytest.raises(WebullConfigError, match="WEBULL"):
        validate_webull_env(env)


def test_paper_mode_no_webull_required(project_root: Path) -> None:
    app, _ = load_app_settings(project_root=project_root)
    assert app.mode is SafetyMode.PAPER


def test_mock_webull_no_network() -> None:
    client = MockWebullClient()
    assert client.list_open_orders() == []


def test_live_gated_requires_explicit_flag(project_root: Path) -> None:
    app, _ = load_app_settings(
        config_path="config/live.example.yaml",
        project_root=project_root,
    )
    assert app.live_trading_enabled is False
