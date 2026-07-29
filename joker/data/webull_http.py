"""Shared HTTP transport for Webull market-data clients."""

from __future__ import annotations

from typing import Any

import httpx

from joker.config.settings import EnvSettings
from joker.config.validation import redact_secrets
from joker.data.webull_auth import build_signed_headers, build_token_create_headers, host_from_base_url
from joker.data.webull_endpoints import WEBULL_BASE_URLS, WebullEndpoint, get_endpoint
from joker.data.webull_errors import OptionEndpointUnverified, WebullApiError, WebullAuthResult


class WebullHttpClient:
    """Signed HTTP client for Webull market-data endpoints."""

    MARKET_DATA_CALLS_ENABLED = True

    def __init__(
        self,
        env: EnvSettings,
        *,
        client: httpx.Client | None = None,
        base_url: str | None = None,
    ) -> None:
        self._env = env
        api_env = (env.webull_api_env or "uat").lower()
        self._base_url = base_url or WEBULL_BASE_URLS.get(api_env, WEBULL_BASE_URLS["uat"])
        self._host = host_from_base_url(self._base_url)
        self._owns_client = client is None
        self._http = client or httpx.Client(timeout=30.0)
        self._access_token: str | None = env.webull_access_token or None

    def close(self) -> None:
        """Close the owned httpx client."""
        if self._owns_client:
            self._http.close()
            self._owns_client = False

    @property
    def access_token(self) -> str | None:
        return self._access_token

    def set_access_token(self, token: str) -> None:
        self._access_token = token

    def _safe_error(self, exc: Exception) -> str:
        return redact_secrets(str(exc), env=self._env)

    def classify_http_error(
        self,
        status: int,
        body: str,
        *,
        endpoint: WebullEndpoint,
    ) -> WebullApiError:
        lowered = body.lower()
        subscription = status == 403 or "subscription" in lowered or "permission" in lowered
        rate_limited = status == 429 or "rate limit" in lowered
        endpoint_mismatch = status == 404 or "not found" in lowered
        auth_failed = status == 401 or "unauthorized" in lowered
        malformed = status in (400, 417) or "invalid" in lowered
        issue = "request failed"
        if endpoint_mismatch:
            issue = "endpoint mismatch"
        elif subscription:
            issue = "subscription/permission"
        elif auth_failed:
            issue = "auth failure"
        elif rate_limited:
            issue = "rate limit"
        elif malformed:
            issue = "malformed response"
        return WebullApiError(
            f"Webull {endpoint.name} HTTP {status} ({issue})",
            status_code=status,
            subscription_related=subscription,
            rate_limited=rate_limited,
            error_code=issue.replace(" ", "_").upper(),
        )

    def request_json(
        self,
        endpoint_name: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        if not self.MARKET_DATA_CALLS_ENABLED:
            raise WebullApiError("Webull market-data calls are disabled")
        endpoint = get_endpoint(endpoint_name)
        if endpoint.method in ("MQTT",):
            raise WebullApiError(f"Endpoint {endpoint_name} is not HTTP")
        if endpoint.method == "GET" and not endpoint.verified:
            raise OptionEndpointUnverified(
                f"Endpoint '{endpoint_name}' not verified: {endpoint.path}"
            )
        token = self._access_token or ""
        headers, body_string = build_signed_headers(
            self._env,
            method=endpoint.method,
            path=endpoint.path,
            params=params if endpoint.method == "GET" else None,
            access_token=token,
            host=self._host,
            body=json_body if endpoint.method != "GET" else None,
        )
        try:
            if endpoint.method.upper() == "GET":
                response = self._http.get(
                    f"{self._base_url}{endpoint.path}",
                    params=params,
                    headers=headers,
                )
            else:
                response = self._http.request(
                    endpoint.method,
                    f"{self._base_url}{endpoint.path}",
                    content=body_string.encode("utf-8") if body_string else b"",
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise WebullApiError(f"Webull HTTP error: {self._safe_error(exc)}") from exc
        if response.status_code >= 400:
            raise self.classify_http_error(
                response.status_code,
                response.text[:500],
                endpoint=endpoint,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise WebullApiError("Malformed JSON in Webull response") from exc

    def authenticate_legacy(self) -> WebullAuthResult:
        """Create access token via signed POST /openapi/auth/token/create."""
        if not self._env.webull_app_key or not self._env.webull_app_secret:
            return WebullAuthResult(success=False, message="Missing app credentials")
        if self._access_token:
            return WebullAuthResult(success=True, token_present=True, message="Using existing token")
        endpoint = get_endpoint("auth_token_create")
        headers = build_token_create_headers(self._env, host=self._host)
        try:
            response = self._http.post(
                f"{self._base_url}{endpoint.path}",
                headers=headers,
            )
        except httpx.HTTPError as exc:
            return WebullAuthResult(success=False, message=self._safe_error(exc))
        if response.status_code >= 400:
            err = self.classify_http_error(
                response.status_code,
                response.text[:500],
                endpoint=endpoint,
            )
            return WebullAuthResult(success=False, message=str(err))
        try:
            payload = response.json()
        except ValueError:
            return WebullAuthResult(success=False, message="Malformed auth JSON")
        if not isinstance(payload, dict):
            return WebullAuthResult(success=False, message="Unexpected auth response")
        token = payload.get("token") or payload.get("accessToken") or payload.get("access_token")
        status = str(payload.get("status", "")).upper()
        if not token:
            return WebullAuthResult(success=False, message="Auth response missing token")
        self._access_token = str(token)
        if status == "PENDING":
            return WebullAuthResult(
                success=False,
                token_present=True,
                message=(
                    "Token created but status PENDING — enter SMS code in Webull app, "
                    "then re-run verification"
                ),
            )
        if status and status not in ("NORMAL", ""):
            return WebullAuthResult(
                success=False,
                token_present=True,
                message=f"Token status {status} — recreate or verify in Webull app",
            )
        return WebullAuthResult(success=True, token_present=True, message="Authenticated")

    def create_access_token(self) -> WebullAuthResult:
        """Alias for token creation — used by auth flow CLI."""
        return self.authenticate_legacy()

    def check_token(self, token: str) -> dict[str, Any]:
        """Check token status via POST /openapi/auth/token/check."""
        endpoint = get_endpoint("auth_token_check")
        from joker.data.webull_auth import build_auth_post_headers

        body = {"token": token}
        headers, body_string = build_auth_post_headers(
            self._env,
            host=self._host,
            path=endpoint.path,
            body=body,
        )
        try:
            response = self._http.post(
                f"{self._base_url}{endpoint.path}",
                content=body_string.encode("utf-8") if body_string else b"{}",
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise WebullApiError(f"Webull HTTP error: {self._safe_error(exc)}") from exc
        if response.status_code >= 400:
            raise self.classify_http_error(
                response.status_code,
                response.text[:500],
                endpoint=endpoint,
            )
        payload = response.json()
        if isinstance(payload, dict):
            status = str(payload.get("status", "")).upper()
            if status == "NORMAL":
                self._access_token = token
        return payload if isinstance(payload, dict) else {}
