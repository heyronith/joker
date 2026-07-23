"""Webull OpenAPI options market-data client (no trading endpoints)."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

import httpx

from joker.config.settings import EnvSettings
from joker.data.webull_api import _ensure_spy
from joker.data.webull_config import ensure_live_trading_disabled
from joker.data.webull_endpoints import get_endpoint, require_verified
from joker.data.webull_errors import OptionEndpointUnverified, WebullApiError, WebullAuthResult
from joker.data.webull_http import WebullHttpClient
from joker.data.webull_option_symbols import build_osi_symbol
from joker.schemas.options_data import OptionContractMetadata, OptionSnapshot

__all__ = ["OptionEndpointUnverified", "HttpWebullOptionsMarketApi", "MockWebullOptionsMarketApi"]


class WebullOptionsMarketApi(Protocol):
    def authenticate(self) -> WebullAuthResult: ...

    def find_option_contracts(self, symbol: str, expiration: date) -> list[OptionContractMetadata]: ...

    def get_option_snapshot(self, contract: OptionContractMetadata) -> OptionSnapshot: ...

    def get_option_snapshots(
        self, contracts: list[OptionContractMetadata]
    ) -> list[OptionSnapshot]: ...

    def get_option_bars(
        self, contract: OptionContractMetadata, timeframe: str
    ) -> list[dict[str, Any]]: ...

    def get_option_ticks(
        self, contract: OptionContractMetadata
    ) -> list[dict[str, Any]]: ...


def _resolve_osi_symbol(contract: OptionContractMetadata) -> str:
    """Resolve OSI symbol required by Webull option snapshot endpoint."""
    if contract.contract_id and contract.contract_id.startswith("SPY"):
        return contract.contract_id
    if contract.instrument_id:
        return contract.instrument_id
    if contract.strike and contract.expiration and contract.option_type:
        return build_osi_symbol(
            contract.underlying_symbol,
            contract.expiration,
            contract.strike,
            contract.option_type,
        )
    raise WebullApiError(
        "Missing required identifier for option snapshot — provide OSI contract_id, "
        "instrument_id, or full contract metadata (symbol, expiry, strike, type)"
    )


def _parse_snapshot_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]
        return [payload]
    raise WebullApiError("Malformed option snapshot response")


class HttpWebullOptionsMarketApi:
    """HTTP client for Webull options market data — verified endpoints only."""

    MARKET_DATA_CALLS_ENABLED = True

    def __init__(
        self,
        env: EnvSettings,
        *,
        client: httpx.Client | None = None,
        http_client: WebullHttpClient | None = None,
        stock_api: Any | None = None,
    ) -> None:
        ensure_live_trading_disabled(env)
        self._env = env
        self._http = http_client or WebullHttpClient(env, client=client)
        self._stock_api = stock_api

    @property
    def CONTRACT_DISCOVERY_VERIFIED(self) -> bool:
        return get_endpoint("option_chain").verified

    @property
    def SNAPSHOT_VERIFIED(self) -> bool:
        return get_endpoint("option_snapshot").verified

    @property
    def BARS_VERIFIED(self) -> bool:
        return get_endpoint("option_bars").verified

    @property
    def TICKS_VERIFIED(self) -> bool:
        return get_endpoint("option_tick").verified

    def _ensure_auth(self) -> None:
        if self._http.access_token:
            return
        auth = self.authenticate()
        if not auth.success:
            raise WebullApiError(f"Auth failed: {auth.message}")

    def authenticate(self) -> WebullAuthResult:
        if self._env.webull_access_token:
            self._http.set_access_token(self._env.webull_access_token)
            return WebullAuthResult(success=True, token_present=True, message="Using configured token")
        return self._http.authenticate_legacy()

    def find_option_contracts(self, symbol: str, expiration: date) -> list[OptionContractMetadata]:
        _ensure_spy(symbol)
        require_verified("option_chain")
        self._ensure_auth()
        endpoint = get_endpoint("option_chain")
        path = endpoint.path.replace("{symbol}", symbol.upper())
        raise OptionEndpointUnverified(
            f"Option chain endpoint not verified — {endpoint.notes}"
        )

    def get_option_snapshot(self, contract: OptionContractMetadata) -> OptionSnapshot:
        from joker.data.options_normalizer import normalize_webull_option_snapshot

        _ensure_spy(contract.underlying_symbol)
        require_verified("option_snapshot")
        osi = _resolve_osi_symbol(contract)
        self._ensure_auth()
        payload = self._http.request_json(
            "option_snapshot",
            params={"symbols": osi, "category": "US_OPTION"},
        )
        rows = _parse_snapshot_rows(payload)
        if not rows:
            raise WebullApiError("Empty option snapshot response")
        data = rows[0]
        merged = contract.model_copy(
            update={
                "contract_id": data.get("symbol") or osi,
                "instrument_id": data.get("instrument_id") or contract.instrument_id,
            }
        )
        return normalize_webull_option_snapshot(merged, data)

    def get_option_snapshots(
        self, contracts: list[OptionContractMetadata]
    ) -> list[OptionSnapshot]:
        if not contracts:
            return []
        require_verified("option_snapshot")
        if len(contracts) == 1:
            return [self.get_option_snapshot(contracts[0])]
        symbols = ",".join(_resolve_osi_symbol(c) for c in contracts[:20])
        self._ensure_auth()
        payload = self._http.request_json(
            "option_snapshot",
            params={"symbols": symbols, "category": "US_OPTION"},
        )
        rows = _parse_snapshot_rows(payload)
        by_symbol = {str(r.get("symbol")): r for r in rows if r.get("symbol")}
        results: list[OptionSnapshot] = []
        from joker.data.options_normalizer import normalize_webull_option_snapshot

        for contract in contracts[:20]:
            osi = _resolve_osi_symbol(contract)
            row = by_symbol.get(osi)
            if row is None:
                raise WebullApiError(f"No snapshot row for {osi}")
            merged = contract.model_copy(
                update={
                    "contract_id": row.get("symbol") or osi,
                    "instrument_id": row.get("instrument_id") or contract.instrument_id,
                }
            )
            results.append(normalize_webull_option_snapshot(merged, row))
        return results

    def get_option_bars(
        self, contract: OptionContractMetadata, timeframe: str
    ) -> list[dict[str, Any]]:
        require_verified("option_bars")
        osi = _resolve_osi_symbol(contract)
        self._ensure_auth()
        payload = self._http.request_json(
            "option_bars",
            params={
                "symbols": osi,
                "category": "US_OPTION",
                "timespan": timeframe,
            },
        )
        if isinstance(payload, list):
            return payload
        rows = payload.get("data") or payload.get("bars") or []
        if not isinstance(rows, list):
            raise WebullApiError("Malformed option bars response")
        return rows

    def get_option_ticks(self, contract: OptionContractMetadata) -> list[dict[str, Any]]:
        require_verified("option_tick")
        osi = _resolve_osi_symbol(contract)
        self._ensure_auth()
        payload = self._http.request_json(
            "option_tick",
            params={"symbols": osi, "category": "US_OPTION"},
        )
        if isinstance(payload, list):
            return payload
        rows = payload.get("data") or payload.get("ticks") or []
        if not isinstance(rows, list):
            raise WebullApiError("Malformed option tick response")
        return rows


class MockWebullOptionsMarketApi:
    """Offline test double for Webull options data."""

    MARKET_DATA_CALLS_ENABLED = True
    CONTRACT_DISCOVERY_VERIFIED = True
    SNAPSHOT_VERIFIED = True
    BARS_VERIFIED = True
    TICKS_VERIFIED = True

    def __init__(
        self,
        *,
        contracts: list[OptionContractMetadata] | None = None,
        snapshots: dict[str, OptionSnapshot] | None = None,
        auth_success: bool = True,
        fail_discovery: WebullApiError | None = None,
        rate_limit_after: int | None = None,
    ) -> None:
        self._contracts = contracts or []
        self._snapshots = snapshots or {}
        self._auth_success = auth_success
        self._fail_discovery = fail_discovery
        self._rate_limit_after = rate_limit_after
        self._snapshot_calls = 0

    def authenticate(self) -> WebullAuthResult:
        if not self._auth_success:
            return WebullAuthResult(success=False, message="Mock auth failed")
        return WebullAuthResult(success=True, token_present=True, message="Mock authenticated")

    def find_option_contracts(self, symbol: str, expiration: date) -> list[OptionContractMetadata]:
        _ensure_spy(symbol)
        if self._fail_discovery:
            raise self._fail_discovery
        return [c for c in self._contracts if c.expiration == expiration]

    def get_option_snapshot(self, contract: OptionContractMetadata) -> OptionSnapshot:
        self._snapshot_calls += 1
        if self._rate_limit_after is not None and self._snapshot_calls > self._rate_limit_after:
            raise WebullApiError(
                "Rate limit exceeded",
                status_code=429,
                rate_limited=True,
            )
        key = contract.contract_id or ""
        if key not in self._snapshots:
            raise WebullApiError(f"No mock snapshot for {key}")
        return self._snapshots[key]

    def get_option_snapshots(
        self, contracts: list[OptionContractMetadata]
    ) -> list[OptionSnapshot]:
        return [self.get_option_snapshot(c) for c in contracts]

    def get_option_bars(
        self, contract: OptionContractMetadata, timeframe: str
    ) -> list[dict[str, Any]]:
        raise OptionEndpointUnverified("Option bars not available in mock")

    def get_option_ticks(self, contract: OptionContractMetadata) -> list[dict[str, Any]]:
        raise OptionEndpointUnverified("Option ticks not available in mock")
