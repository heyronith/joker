"""Startup validation for configuration, secrets, and model availability."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

from joker.app.safety import SafetyMode
from joker.config.loader import load_app_settings
from joker.config.settings import AppSettings, EnvSettings

SECRET_PATTERNS = (
    re.compile(r"sk-[a-zA-Z0-9_-]{10,}", re.IGNORECASE),
    re.compile(r"(api[_-]?key|secret|token|password|pin)\s*[:=]\s*\S+", re.IGNORECASE),
)


@dataclass(frozen=True)
class ValidationResult:
    app_settings: AppSettings
    env_settings: EnvSettings


class ConfigValidationError(Exception):
    """Raised when startup validation fails. Message is safe for display."""


def redact_secrets(text: str, env: EnvSettings | None = None) -> str:
    """Redact secret-looking substrings from log/error messages."""
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)

    if env is not None:
        for value in (
            env.openai_api_key,
            env.webull_app_key,
            env.webull_app_secret,
            env.webull_trade_pin,
            env.webull_access_token,
            env.webull_paper_account_id,
            env.webull_trade_app_key,
            env.webull_trade_app_secret,
            env.webull_trade_access_token,
            env.webull_live_app_key,
            env.webull_live_app_secret,
            env.webull_live_access_token,
            env.webull_live_account_id,
        ):
            if value and len(value) > 4:
                redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def safe_error_message(exc: Exception, env: EnvSettings | None = None) -> str:
    return redact_secrets(str(exc), env=env)


def validate_mode(app: AppSettings) -> None:
    if app.mode is SafetyMode.LIVE_GATED:
        if not app.live_trading_enabled:
            # LIVE_GATED without opt-in behaves like shadow — no broker submit.
            return
    elif app.live_trading_enabled:
        raise ConfigValidationError(
            "live_trading_enabled is true but mode is not LIVE_GATED. "
            "Live trading is disabled by default."
        )


def validate_openai_env(env: EnvSettings) -> None:
    if not env.openai_api_key or env.openai_api_key.startswith("sk-your-key"):
        raise ConfigValidationError(
            "OPENAI_API_KEY is missing or still set to the placeholder value. "
            "Copy .env.example to .env and set a valid key."
        )
    if not env.openai_model.strip():
        raise ConfigValidationError("OPENAI_MODEL must not be empty.")


def validate_webull_env(env: EnvSettings, app: AppSettings) -> None:
    if not app.mode.allows_broker_submit(app.live_trading_enabled):
        return
    missing = []
    if not env.webull_app_key:
        missing.append("WEBULL_APP_KEY")
    if not env.webull_app_secret:
        missing.append("WEBULL_APP_SECRET")
    if not env.webull_device_id:
        missing.append("WEBULL_DEVICE_ID")
    if not env.webull_trade_pin:
        missing.append("WEBULL_TRADE_PIN")
    if missing:
        raise ConfigValidationError(
            "Live-gated broker mode requires Webull credentials: "
            + ", ".join(missing)
        )


def validate_model_available(
    env: EnvSettings,
    client_factory: Callable[[], OpenAI] | None = None,
) -> None:
    """Verify configured OpenAI model exists; fail closed with clear error."""
    client = client_factory() if client_factory else OpenAI(api_key=env.openai_api_key)
    try:
        models = client.models.list()
        available = {m.id for m in models.data}
    except Exception as exc:
        raise ConfigValidationError(
            f"Unable to validate OPENAI_MODEL availability: {safe_error_message(exc, env)}"
        ) from exc

    if env.openai_model not in available:
        sample = sorted(available)[:8]
        raise ConfigValidationError(
            f"OPENAI_MODEL '{env.openai_model}' is not available for this API key. "
            f"Available models include: {', '.join(sample)}..."
        )


def validate_startup(
    config_path: str | None = None,
    project_root: str | None = None,
    skip_model_check: bool = False,
    client_factory: Callable[[], OpenAI] | None = None,
) -> ValidationResult:
    """Run all startup validations. Raises ConfigValidationError on failure."""
    from pathlib import Path

    root = Path(project_root) if project_root else Path.cwd()
    try:
        app_settings, env_settings = load_app_settings(
            config_path=config_path,
            project_root=root,
        )
    except Exception as exc:
        raise ConfigValidationError(safe_error_message(exc)) from exc

    validate_openai_env(env_settings)
    validate_mode(app_settings)
    validate_webull_env(env_settings, app_settings)

    if not skip_model_check:
        validate_model_available(env_settings, client_factory=client_factory)

    return ValidationResult(app_settings=app_settings, env_settings=env_settings)
