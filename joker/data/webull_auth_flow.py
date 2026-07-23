"""Webull access-token creation and verification flow (key+secret only)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from joker.config.settings import EnvSettings
from joker.data.webull_http import WebullHttpClient


@dataclass
class WebullAuthFlowResult:
    success: bool
    status: str
    token_present: bool
    message: str
    api_env: str
    token_saved_path: Path | None = None


def _parse_token_payload(payload: object) -> tuple[str | None, str]:
    if not isinstance(payload, dict):
        return None, ""
    token = payload.get("token") or payload.get("accessToken") or payload.get("access_token")
    status = str(payload.get("status", "")).upper()
    return (str(token) if token else None), status


def run_webull_auth_flow(
    env: EnvSettings,
    *,
    wait_seconds: float = 0,
    poll_interval_seconds: float = 5.0,
    for_trade: bool = False,
    force_recreate: bool = False,
) -> WebullAuthFlowResult:
    """Create token from app key/secret; optionally poll until NORMAL."""
    auth_env = env.trade_credentials_env() if for_trade else env
    # When requesting a new trade token, ignore an existing market-data token.
    if for_trade:
        trade_tok = (env.webull_trade_access_token or "").strip()
        market_tok = (env.webull_access_token or "").strip()
        # Common mistake: copy prod WEBULL_ACCESS_TOKEN into WEBULL_TRADE_ACCESS_TOKEN.
        if force_recreate or not trade_tok or (market_tok and trade_tok == market_tok):
            auth_env = auth_env.model_copy(update={"webull_access_token": None})
        else:
            auth_env = auth_env.model_copy(update={"webull_access_token": trade_tok})
    elif force_recreate:
        auth_env = auth_env.model_copy(update={"webull_access_token": None})

    client = WebullHttpClient(auth_env)
    api_env = (auth_env.webull_api_env or "uat").lower()
    token_env_name = "WEBULL_TRADE_ACCESS_TOKEN" if for_trade else "WEBULL_ACCESS_TOKEN"

    if auth_env.webull_access_token:
        client.set_access_token(auth_env.webull_access_token)
        try:
            payload = client.check_token(auth_env.webull_access_token)
            _, status = _parse_token_payload(payload)
            if status == "NORMAL":
                return WebullAuthFlowResult(
                    success=True,
                    status=status,
                    token_present=True,
                    message=f"Using configured {token_env_name} (NORMAL)",
                    api_env=api_env,
                )
            return WebullAuthFlowResult(
                success=False,
                status=status or "UNKNOWN",
                token_present=True,
                message=(
                    f"Configured token status {status or 'UNKNOWN'} — "
                    "clear it (or pass --force) to recreate"
                ),
                api_env=api_env,
            )
        except Exception as exc:
            return WebullAuthFlowResult(
                success=False,
                status="CHECK_FAILED",
                token_present=True,
                message=str(exc),
                api_env=api_env,
            )

    auth = client.create_access_token()
    if not auth.token_present:
        return WebullAuthFlowResult(
            success=False,
            status="CREATE_FAILED",
            token_present=False,
            message=auth.message,
            api_env=api_env,
        )

    token = client.access_token
    if not token:
        return WebullAuthFlowResult(
            success=False,
            status="CREATE_FAILED",
            token_present=False,
            message="Token create returned no token",
            api_env=api_env,
        )

    try:
        payload = client.check_token(token)
        _, status = _parse_token_payload(payload)
    except Exception:
        status = "NORMAL" if auth.success else "PENDING"

    deadline = time.monotonic() + max(wait_seconds, 0)
    while status == "PENDING" and time.monotonic() < deadline:
        time.sleep(poll_interval_seconds)
        try:
            payload = client.check_token(token)
            _, status = _parse_token_payload(payload)
        except Exception:
            break

    if status == "NORMAL":
        saved = _save_token_hint(token, api_env, for_trade=for_trade)
        return WebullAuthFlowResult(
            success=True,
            status=status,
            token_present=True,
            message=f"Token active — add {token_env_name} to .env to reuse across runs",
            api_env=api_env,
            token_saved_path=saved,
        )

    if status == "PENDING":
        wait_cmd = (
            "joker data webull-auth --trade --wait 120"
            if for_trade
            else "joker data webull-auth --wait 120"
        )
        return WebullAuthFlowResult(
            success=False,
            status=status,
            token_present=True,
            message=(
                "Token created — verify in Webull app: Menu → Messages → OpenAPI Notifications, "
                f"enter SMS code, then run: {wait_cmd}"
            ),
            api_env=api_env,
        )

    return WebullAuthFlowResult(
        success=False,
        status=status or "UNKNOWN",
        token_present=True,
        message=auth.message or f"Token status {status}",
        api_env=api_env,
    )


def _save_token_hint(token: str, api_env: str, *, for_trade: bool = False) -> Path:
    """Write token to gitignored local file (user can copy to .env)."""
    if for_trade:
        path = Path("data/capabilities/webull_trade_access_token.txt")
        env_name = "WEBULL_TRADE_ACCESS_TOKEN"
    else:
        path = Path("data/capabilities/webull_access_token.txt")
        env_name = "WEBULL_ACCESS_TOKEN"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Webull access token ({api_env}) — copy value to {env_name} in .env\n"
        f"{token}\n",
        encoding="utf-8",
    )
    return path
