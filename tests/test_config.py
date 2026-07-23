"""Phase 0 configuration tests."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from joker.app.safety import SafetyMode
from joker.config.loader import load_app_settings, load_yaml_config, merge_configs
from joker.config.settings import AppSettings, EnvSettings
from joker.config.validation import (
    ConfigValidationError,
    redact_secrets,
    safe_error_message,
    validate_mode,
    validate_openai_env,
    validate_startup,
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-unit-tests-only")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("JOKER_CONFIG", "config/paper.yaml")


def test_config_loads_from_yaml_and_env(project_root: Path, env_vars: None) -> None:
    app_settings, env_settings = load_app_settings(project_root=project_root)
    assert app_settings.mode is SafetyMode.PAPER
    assert app_settings.risk.allowed_symbol == "SPY"
    assert app_settings.risk.max_daily_loss_usd == 500.0
    assert app_settings.risk.max_premium_usd == 500.0
    assert app_settings.risk.max_trades_per_day == 5
    assert app_settings.risk.policy == "agent_led"
    assert app_settings.agents.execution_mode == "agent_led"
    assert env_settings.openai_api_key == "sk-test-key-for-unit-tests-only"
    assert env_settings.openai_model == "gpt-5.4-mini"


def test_yaml_merge_overrides_defaults(project_root: Path) -> None:
    default = load_yaml_config(project_root / "config" / "default.yaml")
    shadow = load_yaml_config(project_root / "config" / "shadow.yaml")
    merged = merge_configs(default, shadow)
    assert merged["mode"] == "SHADOW"


def test_missing_required_env_var_produces_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        EnvSettings(_env_file=None)  # type: ignore[call-arg]
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_invalid_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="Invalid safety mode"):
        SafetyMode.from_string("YOLO")

    with pytest.raises(ValidationError):
        AppSettings.model_validate({"mode": "INVALID"})


def test_live_mode_disabled_by_default(project_root: Path, env_vars: None) -> None:
    app_settings, _ = load_app_settings(project_root=project_root)
    assert app_settings.live_trading_enabled is False
    assert app_settings.mode.allows_broker_submit(app_settings.live_trading_enabled) is False


def test_live_trading_requires_live_gated_mode() -> None:
    with pytest.raises(ValidationError, match="live_trading_enabled requires mode LIVE_GATED"):
        AppSettings.model_validate(
            {"mode": "PAPER", "live_trading_enabled": True}
        )


def test_validate_mode_rejects_live_flag_in_paper_mode() -> None:
    app = AppSettings.model_validate({"mode": "PAPER", "live_trading_enabled": False})
    validate_mode(app)

    with pytest.raises(ValidationError, match="live_trading_enabled requires mode LIVE_GATED"):
        AppSettings.model_validate({"mode": "PAPER", "live_trading_enabled": True})


def test_secrets_never_printed_in_errors(env_vars: None) -> None:
    secret = "sk-test-key-for-unit-tests-only"
    env = EnvSettings()  # type: ignore[call-arg]
    message = safe_error_message(Exception(f"failed with key {secret}"), env=env)
    assert secret not in message
    assert "[REDACTED]" in message


def test_redact_secrets_patterns() -> None:
    text = "error: api_key=supersecretvalue and sk-abcdefghijklmnop"
    redacted = redact_secrets(text)
    assert "supersecretvalue" not in redacted
    assert "sk-abcdefghijklmnop" not in redacted


def test_validate_openai_placeholder_rejected(env_vars: None) -> None:
    env = EnvSettings()  # type: ignore[call-arg]
    env = env.model_copy(update={"openai_api_key": "sk-your-key-here"})
    with pytest.raises(ConfigValidationError, match="OPENAI_API_KEY"):
        validate_openai_env(env)


def test_validate_startup_with_mock_model_check(
    project_root: Path,
    env_vars: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(project_root)

    def fake_client_factory() -> MagicMock:
        client = MagicMock()
        client.models.list.return_value = MagicMock(
            data=[MagicMock(id="gpt-5.4-mini")]
        )
        return client

    result = validate_startup(
        skip_model_check=False,
        client_factory=fake_client_factory,
    )
    assert result.app_settings.mode is SafetyMode.PAPER


def test_validate_startup_model_unavailable(
    project_root: Path,
    env_vars: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(project_root)

    def fake_client_factory() -> MagicMock:
        client = MagicMock()
        client.models.list.return_value = MagicMock(data=[MagicMock(id="other-model")])
        return client

    with pytest.raises(ConfigValidationError, match="not available"):
        validate_startup(client_factory=fake_client_factory)
