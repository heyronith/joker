"""Webull market-data configuration validation (data-only, no trading)."""

from __future__ import annotations

from enum import Enum

from joker.config.settings import EnvSettings
from joker.config.validation import redact_secrets


class WebullMarketConfigError(Exception):
    """Safe configuration error — message must not contain secrets."""


class WebullApiEnv(str, Enum):
    UAT = "uat"
    SANDBOX = "sandbox"
    PROD = "prod"

    @classmethod
    def from_string(cls, value: str) -> WebullApiEnv:
        normalized = value.strip().lower()
        # Treat older "uat" and official "sandbox" as test envs.
        aliases = {"test": "sandbox", "paper": "sandbox"}
        normalized = aliases.get(normalized, normalized)
        for member in cls:
            if member.value == normalized:
                return member
        raise WebullMarketConfigError(
            f"WEBULL_API_ENV must be 'sandbox', 'uat', or 'prod', got: {value!r}"
        )


def ensure_live_trading_disabled(env: EnvSettings) -> None:
    """Paper-trade / legacy gate — market-data validation no longer requires this."""
    if env.webull_live_trading_enabled:
        raise WebullMarketConfigError(
            "WEBULL_LIVE_TRADING_ENABLED is true. "
            "Paper-account order helpers refuse live mode; use WebullLiveClient."
        )


def validate_webull_market_env(env: EnvSettings) -> None:
    """Require market-data credentials when using Webull provider.

    Market data may run alongside live trading; live credentials are separate.
    """
    missing: list[str] = []
    if not env.webull_app_key:
        missing.append("WEBULL_APP_KEY")
    if not env.webull_app_secret:
        missing.append("WEBULL_APP_SECRET")
    if missing:
        raise WebullMarketConfigError(
            "Webull market-data credentials required: " + ", ".join(missing)
        )
    WebullApiEnv.from_string(env.webull_api_env)


def safe_webull_error(exc: Exception, env: EnvSettings | None = None) -> str:
    return redact_secrets(str(exc), env=env)
