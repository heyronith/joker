"""Webull OpenAPI v2 signed request helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlparse

from joker.config.settings import EnvSettings


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def host_from_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    return parsed.netloc or base_url.removeprefix("https://").removeprefix("http://").split("/")[0]


def build_signature(
    *,
    app_secret: str,
    path: str,
    query_params: dict[str, Any] | None,
    signing_headers: dict[str, str],
    body_string: str | None = None,
) -> str:
    """Build HMAC-SHA1 signature per Webull OpenAPI v2 docs."""
    all_params: dict[str, str] = {}
    for key, value in (query_params or {}).items():
        all_params[str(key)] = str(value)
    all_params.update(signing_headers)

    str1 = "&".join(f"{key}={all_params[key]}" for key in sorted(all_params.keys()))
    if body_string:
        body_md5 = hashlib.md5(body_string.encode("utf-8")).hexdigest().upper()
        str3 = f"{path}&{str1}&{body_md5}"
    else:
        str3 = f"{path}&{str1}"
    encoded_string = quote(str3, safe="")
    signing_key = f"{app_secret}&"
    digest = hmac.new(
        signing_key.encode("utf-8"),
        encoded_string.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def build_signed_headers(
    env: EnvSettings,
    *,
    method: str,
    path: str,
    params: dict[str, Any] | None,
    access_token: str | None = "",
    host: str,
    body: dict[str, Any] | None = None,
) -> tuple[dict[str, str], str | None]:
    if not env.webull_app_key or not env.webull_app_secret:
        raise ValueError("Missing Webull app credentials for signed request")
    ts = utc_timestamp()
    nonce = str(uuid.uuid4())
    body_string: str | None = None
    if body is not None:
        body_string = json.dumps(body, separators=(",", ":"))
    signing_headers = {
        "x-app-key": env.webull_app_key,
        "x-timestamp": ts,
        "x-signature-algorithm": "HMAC-SHA1",
        "x-signature-version": "1.0",
        "x-signature-nonce": nonce,
        "host": host,
    }
    signature = build_signature(
        app_secret=env.webull_app_secret,
        path=path,
        query_params=params if method.upper() == "GET" else None,
        signing_headers=signing_headers,
        body_string=body_string if method.upper() != "GET" else None,
    )
    headers: dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-app-key": env.webull_app_key,
        "x-timestamp": ts,
        "x-signature-version": "1.0",
        "x-signature-algorithm": "HMAC-SHA1",
        "x-signature-nonce": nonce,
        "x-version": "v2",
        "x-signature": signature,
    }
    if access_token is not None:
        headers["x-access-token"] = access_token
    return headers, body_string


def build_auth_post_headers(
    env: EnvSettings,
    *,
    host: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[dict[str, str], str | None]:
    """Signed POST headers for auth endpoints (create/check token)."""
    headers, body_string = build_signed_headers(
        env,
        method="POST",
        path=path,
        params=None,
        access_token=None,
        host=host,
        body=body,
    )
    if env.webull_app_secret:
        headers["x-app-secret"] = env.webull_app_secret
    headers.pop("x-access-token", None)
    return headers, body_string


def build_token_create_headers(env: EnvSettings, *, host: str) -> dict[str, str]:
    """Headers for POST /openapi/auth/token/create."""
    headers, _ = build_auth_post_headers(env, host=host, path="/openapi/auth/token/create")
    return headers
