"""Central registry of Webull OpenAPI market-data endpoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebullEndpoint:
    name: str
    method: str
    path: str
    verified: bool
    rate_limit_per_minute: int | None = None
    required_params: tuple[str, ...] = ()
    required_headers: tuple[str, ...] = (
        "x-app-key",
        "x-app-secret",
        "x-timestamp",
        "x-signature-version",
        "x-signature-algorithm",
        "x-signature-nonce",
        "x-access-token",
        "x-version",
        "x-signature",
    )
    docs_url: str | None = None
    notes: str = ""


SIGNED_HEADERS = WebullEndpoint("", "", "", True).required_headers

WEBULL_ENDPOINTS: dict[str, WebullEndpoint] = {
    "auth_token_create": WebullEndpoint(
        name="auth_token_create",
        method="POST",
        path="/openapi/auth/token/create",
        verified=True,
        rate_limit_per_minute=20,
        docs_url="https://developer.webull.com/apis/docs/reference/create-token/",
        notes="Signed POST; returns token + status (PENDING until Webull app SMS verify).",
    ),
    "auth_token_check": WebullEndpoint(
        name="auth_token_check",
        method="POST",
        path="/openapi/auth/token/check",
        verified=True,
        rate_limit_per_minute=20,
        docs_url="https://developer.webull.com/apis/docs/reference/check-token/",
        notes="Signed POST with JSON body {token}; returns PENDING/NORMAL/INVALID/EXPIRED.",
    ),
    "oauth_token": WebullEndpoint(
        name="oauth_token",
        method="POST",
        path="/openapi/oauth/token",
        verified=False,
        required_params=("appKey", "appSecret", "region"),
        required_headers=("Content-Type",),
        docs_url="https://developer.webull.com/apis/docs/authentication/overview",
        notes="Legacy endpoint — not available on US OpenAPI v2 UAT/prod.",
    ),
    "stock_snapshot": WebullEndpoint(
        name="stock_snapshot",
        method="GET",
        path="/openapi/market-data/stock/snapshot",
        verified=True,
        rate_limit_per_minute=60,
        required_params=("symbols", "category"),
        docs_url="https://developer.webull.com/apis/docs/reference/stock-snapshot",
        notes="category=US_STOCK. Response: array of snapshot objects.",
    ),
    "stock_bars": WebullEndpoint(
        name="stock_bars",
        method="GET",
        path="/openapi/market-data/stock/bars",
        verified=True,
        rate_limit_per_minute=60,
        # Live-verified 2026-07-24: param is `symbol` (singular), timespan `M1`.
        # `symbols` (plural) returns HTTP 400 Parameters not valid.
        required_params=("symbol", "category", "timespan"),
        docs_url="https://developer.webull.com/apis/docs/reference/stock-historical-bars",
        notes="category=US_STOCK; timespan=M1 (joker maps 1m→M1). Response: OHLCV array newest-first.",
    ),
    "stock_streaming": WebullEndpoint(
        name="stock_streaming",
        method="MQTT",
        path="data-api.webull.com",
        verified=True,
        docs_url="https://developer.webull.com/apis/market-data-api/getting-started",
        notes="Real-time via MQTT subscribe; HTTP polling uses stock_snapshot.",
    ),
    "option_chain": WebullEndpoint(
        name="option_chain",
        method="GET",
        path="/openapi/market/option/chain/{symbol}",
        verified=False,
        required_params=("expiration", "region"),
        docs_url=None,
        notes="No official chain endpoint in Webull OpenAPI docs (2026-03). Use OSI symbol construction.",
    ),
    "option_snapshot": WebullEndpoint(
        name="option_snapshot",
        method="GET",
        path="/openapi/market-data/option/snapshot",
        verified=True,
        rate_limit_per_minute=60,
        required_params=("symbols", "category"),
        docs_url="https://developer.webull.com/apis/docs/reference/option-snapshot",
        notes="symbols=OSI codes comma-separated (max 20). category=US_OPTION.",
    ),
    "option_bars": WebullEndpoint(
        name="option_bars",
        method="GET",
        path="/openapi/market-data/option/bars",
        verified=True,
        rate_limit_per_minute=60,
        required_params=("symbols", "category", "timespan"),
        docs_url="https://developer.webull.com/apis/docs/reference/option-historical-bars",
        notes="1 call/sec per App Key per docs. timespan: M1, M5, etc.",
    ),
    "option_tick": WebullEndpoint(
        name="option_tick",
        method="GET",
        path="/openapi/market-data/option/tick",
        verified=True,
        rate_limit_per_minute=60,
        required_params=("symbols", "category"),
        docs_url="https://developer.webull.com/apis/docs/reference/option-tick",
        notes="Tick-by-tick trades for option symbols.",
    ),
    # --- Trade API (paper-account orders). Paths from official Webull OpenAPI reference.
    # See docs/WEBULL_TRADE_ENDPOINT_AUDIT.md. Real-money LIVE remains disabled.
    "broker_account_list": WebullEndpoint(
        name="broker_account_list",
        method="GET",
        path="/openapi/account/list",
        verified=True,
        rate_limit_per_minute=20,
        docs_url="https://developer.webull.com/apis/docs/reference/account-list",
        notes="Retrieve account_id list. Set WEBULL_PAPER_ACCOUNT_ID explicitly.",
    ),
    "broker_account_balance": WebullEndpoint(
        name="broker_account_balance",
        method="GET",
        path="/openapi/assets/balance",
        verified=True,
        rate_limit_per_minute=30,
        required_params=("account_id",),
        docs_url="https://developer.webull.com/apis/docs/reference/account-balance",
        notes="Query param account_id.",
    ),
    "broker_account_positions": WebullEndpoint(
        name="broker_account_positions",
        method="GET",
        path="/openapi/assets/positions",
        verified=True,
        rate_limit_per_minute=30,
        required_params=("account_id",),
        docs_url="https://developer.webull.com/apis/docs/reference/account-position",
        notes="Query param account_id.",
    ),
    "broker_order_place": WebullEndpoint(
        name="broker_order_place",
        method="POST",
        path="/openapi/trade/order/place",
        verified=True,
        rate_limit_per_minute=60,
        docs_url="https://developer.webull.com/apis/docs/reference/common-order-place",
        notes="Body: {account_id, new_orders:[...]}. Options require LIMIT + legs.",
    ),
    "broker_order_cancel": WebullEndpoint(
        name="broker_order_cancel",
        method="POST",
        path="/openapi/trade/order/cancel",
        verified=True,
        rate_limit_per_minute=60,
        docs_url="https://developer.webull.com/apis/docs/reference/common-order-cancel",
        notes="Body: {account_id, client_order_id}.",
    ),
    "broker_order_detail": WebullEndpoint(
        name="broker_order_detail",
        method="GET",
        path="/openapi/trade/order/detail",
        verified=True,
        rate_limit_per_minute=30,
        required_params=("account_id", "client_order_id"),
        docs_url="https://developer.webull.com/apis/docs/reference/order-detail",
        notes="Query params account_id + client_order_id.",
    ),
    "broker_open_orders": WebullEndpoint(
        name="broker_open_orders",
        method="GET",
        path="/openapi/trade/order/open",
        verified=True,
        rate_limit_per_minute=30,
        required_params=("account_id",),
        docs_url="https://developer.webull.com/apis/docs/reference/order-open",
        notes="Query param account_id; optional page_size / last_client_order_id.",
    ),
}

WEBULL_BASE_URLS = {
    # Legacy UAT host (market-data era). Prefer "sandbox" for Trading API tests.
    "uat": "https://us-openapi-alb.uat.webullbroker.com",
    "sandbox": "https://api.sandbox.webull.com",
    "prod": "https://api.webull.com",
}


def get_endpoint(name: str) -> WebullEndpoint:
    if name not in WEBULL_ENDPOINTS:
        raise KeyError(f"Unknown Webull endpoint: {name}")
    return WEBULL_ENDPOINTS[name]


def require_verified(name: str) -> WebullEndpoint:
    ep = get_endpoint(name)
    if not ep.verified:
        from joker.data.webull_errors import OptionEndpointUnverified

        raise OptionEndpointUnverified(
            f"Endpoint '{name}' ({ep.path}) is not verified — {ep.notes}"
        )
    return ep


def endpoint_status_map() -> dict[str, str]:
    """Return verified/unverified status for all registered endpoints."""
    return {name: "verified" if ep.verified else "unverified" for name, ep in WEBULL_ENDPOINTS.items()}
