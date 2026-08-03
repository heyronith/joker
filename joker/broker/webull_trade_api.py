"""Webull Trade API HTTP adapter — paper-account orders only."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from joker.config.settings import EnvSettings
from joker.data.webull_errors import WebullApiError
from joker.data.webull_http import WebullHttpClient
from joker.schemas.domain import OptionContract, OrderIntent


class WebullTradeConfigError(Exception):
    """Safe configuration error for paper-account trading."""


class WebullTradeApi(ABC):
    """Low-level trade endpoints. Implementations must never place real-money orders."""

    @abstractmethod
    def list_accounts(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def get_balance(self, account_id: str) -> dict[str, Any]:
        ...

    @abstractmethod
    def get_positions(self, account_id: str) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def place_order(self, account_id: str, new_orders: list[dict[str, Any]]) -> dict[str, Any]:
        ...

    def preview_order(
        self, account_id: str, new_orders: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Optional preview; paper API may not implement — live client requires it."""
        raise NotImplementedError("preview_order is not implemented for this trade API")

    @abstractmethod
    def cancel_order(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        ...

    @abstractmethod
    def get_order_detail(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        ...

    @abstractmethod
    def list_open_orders(
        self, account_id: str, *, page_size: int = 20
    ) -> list[dict[str, Any]]:
        ...


def ensure_paper_trading_allowed(
    env: EnvSettings, *, require_account_id: bool = True
) -> None:
    """Fail closed unless paper trading is explicitly enabled and live money is off."""
    if env.webull_live_trading_enabled:
        raise WebullTradeConfigError(
            "WEBULL_LIVE_TRADING_ENABLED must remain false. "
            "Real-money order placement is not enabled."
        )
    if not env.webull_paper_trading_enabled:
        raise WebullTradeConfigError(
            "WEBULL_PAPER_TRADING_ENABLED must be true to use Webull paper-account orders."
        )
    if require_account_id and (
        not env.webull_paper_account_id or not str(env.webull_paper_account_id).strip()
    ):
        raise WebullTradeConfigError(
            "WEBULL_PAPER_ACCOUNT_ID is required. Run `joker broker accounts` and set "
            "the paper/sandbox account id explicitly."
        )


def validate_webull_paper_trade_env(
    env: EnvSettings, *, require_account_id: bool = True
) -> None:
    """Credentials required for paper-account trading (device/pin optional)."""
    ensure_paper_trading_allowed(env, require_account_id=require_account_id)
    trade_env = env.trade_credentials_env()
    missing: list[str] = []
    if not trade_env.webull_app_key:
        missing.append("WEBULL_TRADE_APP_KEY (or WEBULL_APP_KEY)")
    if not trade_env.webull_app_secret:
        missing.append("WEBULL_TRADE_APP_SECRET (or WEBULL_APP_SECRET)")
    if not trade_env.webull_access_token:
        missing.append("WEBULL_TRADE_ACCESS_TOKEN (or WEBULL_ACCESS_TOKEN)")
    if missing:
        raise WebullTradeConfigError(
            "Webull paper trading credentials required: " + ", ".join(missing)
        )
    # Fail closed if trade env is still prod while paper trading is enabled and
    # the configured account looks like live cash — checked at CLI; here warn via env.
    api_env = (trade_env.webull_api_env or "").strip().lower()
    if api_env == "prod" and not env.webull_trade_app_key:
        # Allowed but unsafe — CLI warns. Keep API usable for users who insist.
        pass



def new_client_order_id() -> str:
    """Webull client_order_id max length is 32 characters."""
    from uuid import uuid4

    return uuid4().hex  # 32 chars


def build_option_limit_order_payload(
    intent: OrderIntent,
    *,
    client_order_id: str | None = None,
    account_id: str | None = None,
) -> dict[str, Any]:
    """Build a single-leg OPTION LIMIT order dict for preview/place new_orders[]."""
    if intent.order_type != "limit":
        raise WebullTradeConfigError(
            "Webull options only support LIMIT (or stop variants); market orders rejected."
        )
    if intent.limit_price is None or intent.limit_price <= 0:
        raise WebullTradeConfigError("limit_price is required for option orders")
    if intent.quantity < 1:
        raise WebullTradeConfigError("quantity must be >= 1")

    contract = intent.contract
    side = intent.side.upper()
    option_type = "CALL" if contract.option_type == "call" else "PUT"
    strike = f"{contract.strike:.2f}"
    qty = str(intent.quantity)
    cid = client_order_id or intent.intent_id or new_client_order_id()
    if len(cid) > 32:
        raise WebullTradeConfigError("client_order_id must be <= 32 characters")

    position_intent = getattr(intent, "position_intent", None)
    open_close = None
    if position_intent in {"BUY_TO_OPEN", "SELL_TO_OPEN"}:
        open_close = "OPEN"
    elif position_intent in {"BUY_TO_CLOSE", "SELL_TO_CLOSE"}:
        open_close = "CLOSE"

    leg: dict[str, Any] = {
        "side": side,
        "quantity": qty,
        "symbol": contract.symbol.upper(),
        "strike_price": strike,
        "option_expire_date": contract.expiration.isoformat(),
        "instrument_type": "OPTION",
        "option_type": option_type,
        "market": "US",
    }
    if open_close is not None:
        leg["open_close"] = open_close

    payload: dict[str, Any] = {
        "client_order_id": cid,
        "combo_type": "NORMAL",
        "order_type": "LIMIT",
        "limit_price": f"{intent.limit_price:.2f}",
        "quantity": qty,
        "option_strategy": "SINGLE",
        "side": side,
        "time_in_force": "DAY",
        "entrust_type": "QTY",
        "instrument_type": "OPTION",
        "market": "US",
        "symbol": contract.symbol.upper(),
        "legs": [leg],
    }
    if account_id:
        payload["account_id"] = account_id
    if position_intent:
        payload["position_intent"] = position_intent
    if open_close is not None:
        payload["open_close"] = open_close
    return payload


def map_webull_order_status(raw: str | None) -> str:
    """Map Webull status strings to BrokerOrder.status literals."""
    value = (raw or "").strip().upper()
    if value in {"FILLED", "FILLED_ALL", "COMPLETED"}:
        return "filled"
    if value in {"PARTIAL_FILLED", "PARTIALLY_FILLED", "PARTIAL"}:
        return "partially_filled"
    if value in {"CANCELLED", "CANCELED", "CANCEL"}:
        return "cancelled"
    if value in {"REJECTED", "FAILED", "EXPIRED"}:
        return "rejected"
    if value in {"PENDING", "SUBMITTED", "WORKING", "OPEN", ""}:
        return "open" if value else "pending"
    if value in {"PENDING_SUBMIT", "WAIT"}:
        return "pending"
    return "open"


class HttpWebullTradeApi(WebullTradeApi):
    """Signed HTTP trade client. Gated by paper-trading env flags."""

    PAPER_CALLS_ENABLED = True

    def __init__(
        self,
        env: EnvSettings,
        *,
        http: WebullHttpClient | None = None,
        client: httpx.Client | None = None,
        require_account_id: bool = True,
    ) -> None:
        validate_webull_paper_trade_env(env, require_account_id=require_account_id)
        if not self.PAPER_CALLS_ENABLED:
            raise WebullTradeConfigError("HttpWebullTradeApi paper calls are disabled")
        self._env = env
        trade_env = env.trade_credentials_env()
        self._trade_env = trade_env
        self._http = http or WebullHttpClient(trade_env, client=client)
        token = trade_env.webull_access_token
        if token:
            self._http.set_access_token(token)

    def close(self) -> None:
        """Close the owned Webull HTTP trade client."""
        close = getattr(self._http, "close", None)
        if callable(close):
            close()

    def list_accounts(self) -> list[dict[str, Any]]:
        payload = self._http.request_json("broker_account_list")
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("accounts", "data", "result"):
                nested = payload.get(key)
                if isinstance(nested, list):
                    return [row for row in nested if isinstance(row, dict)]
        raise WebullApiError("Unexpected account list response shape")

    def get_balance(self, account_id: str) -> dict[str, Any]:
        payload = self._http.request_json(
            "broker_account_balance",
            params={"account_id": account_id},
        )
        if isinstance(payload, dict):
            return payload
        raise WebullApiError("Unexpected balance response shape")

    def get_positions(self, account_id: str) -> list[dict[str, Any]]:
        payload = self._http.request_json(
            "broker_account_positions",
            params={"account_id": account_id},
        )
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            nested = payload.get("positions") or payload.get("data") or []
            if isinstance(nested, list):
                return [row for row in nested if isinstance(row, dict)]
        raise WebullApiError("Unexpected positions response shape")

    def place_order(self, account_id: str, new_orders: list[dict[str, Any]]) -> dict[str, Any]:
        ensure_paper_trading_allowed(self._env)
        if account_id != self._env.webull_paper_account_id:
            raise WebullTradeConfigError(
                "Refusing order: account_id does not match WEBULL_PAPER_ACCOUNT_ID"
            )
        body = {"account_id": account_id, "new_orders": new_orders}
        payload = self._http.request_json("broker_order_place", json_body=body)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
        raise WebullApiError("Unexpected place-order response shape")

    def preview_order(
        self, account_id: str, new_orders: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ensure_paper_trading_allowed(self._env)
        if account_id != self._env.webull_paper_account_id:
            raise WebullTradeConfigError(
                "Refusing preview: account_id does not match WEBULL_PAPER_ACCOUNT_ID"
            )
        body = {"account_id": account_id, "new_orders": new_orders}
        payload = self._http.request_json("broker_order_preview", json_body=body)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
        raise WebullApiError("Unexpected preview-order response shape")

    def cancel_order(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        ensure_paper_trading_allowed(self._env)
        if account_id != self._env.webull_paper_account_id:
            raise WebullTradeConfigError(
                "Refusing cancel: account_id does not match WEBULL_PAPER_ACCOUNT_ID"
            )
        body = {"account_id": account_id, "client_order_id": client_order_id}
        payload = self._http.request_json("broker_order_cancel", json_body=body)
        if isinstance(payload, dict):
            return payload
        return {"client_order_id": client_order_id, "raw": payload}

    def get_order_detail(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        payload = self._http.request_json(
            "broker_order_detail",
            params={"account_id": account_id, "client_order_id": client_order_id},
        )
        if isinstance(payload, dict):
            return payload
        raise WebullApiError("Unexpected order detail response shape")

    def list_open_orders(
        self, account_id: str, *, page_size: int = 20
    ) -> list[dict[str, Any]]:
        payload = self._http.request_json(
            "broker_open_orders",
            params={"account_id": account_id, "page_size": str(page_size)},
        )
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("orders", "data", "result", "open_orders"):
                nested = payload.get(key)
                if isinstance(nested, list):
                    return [row for row in nested if isinstance(row, dict)]
        return []


class MockWebullTradeApi(WebullTradeApi):
    """Offline test double — never hits the network."""

    def __init__(
        self,
        account_id: str = "PAPER_ACCT_TEST",
        *,
        account_label: str = "Paper Trading",
        account_class: str = "INDIVIDUAL_CASH",
    ) -> None:
        self.account_id = account_id
        self.account_label = account_label
        self.account_class = account_class
        self.placed: list[dict[str, Any]] = []
        self.previewed: list[dict[str, Any]] = []
        self.cancelled: list[str] = []
        self._orders: dict[str, dict[str, Any]] = {}
        self._positions: list[dict[str, Any]] = []
        self.balance_usd = 100_000.0
        self.buying_power_usd = 100_000.0
        self.net_liquidation_usd = 100_000.0
        self.preview_reject: str | None = None
        self.place_timeout: bool = False
        self.place_accepts_before_timeout: bool = False
        self.preview_cost_usd: float | None = None

    def list_accounts(self) -> list[dict[str, Any]]:
        return [
            {
                "account_id": self.account_id,
                "account_type": "CASH",
                "account_label": self.account_label,
                "account_class": self.account_class,
            }
        ]

    def get_balance(self, account_id: str) -> dict[str, Any]:
        return {
            "account_id": account_id,
            "total_cash": str(self.balance_usd),
            "buying_power": str(self.buying_power_usd),
            "total_net_liquidation_value": str(self.net_liquidation_usd),
        }

    def get_positions(self, account_id: str) -> list[dict[str, Any]]:
        return list(self._positions)

    def preview_order(
        self, account_id: str, new_orders: list[dict[str, Any]]
    ) -> dict[str, Any]:
        order = new_orders[0]
        self.previewed.append({"account_id": account_id, **order})
        if self.preview_reject:
            return {
                "accepted": False,
                "reject_code": self.preview_reject,
                "reject_message": self.preview_reject,
            }
        limit = float(order.get("limit_price") or 0)
        qty = float(order.get("quantity") or 0)
        cost = self.preview_cost_usd
        if cost is None:
            cost = limit * qty * 100.0
        return {
            "accepted": True,
            "estimated_cost": str(cost),
            "estimated_fees": "0.65",
            "buying_power_effect": str(-cost),
            "client_order_id": order.get("client_order_id"),
        }

    def place_order(self, account_id: str, new_orders: list[dict[str, Any]]) -> dict[str, Any]:
        order = new_orders[0]
        cid = str(order["client_order_id"])
        if self.place_timeout:
            if self.place_accepts_before_timeout:
                record = {
                    "account_id": account_id,
                    "client_order_id": cid,
                    "order_id": f"WB-{cid[:12]}",
                    "status": "SUBMITTED",
                    **order,
                }
                self._orders[cid] = record
                self.placed.append(record)
            raise TimeoutError("simulated place_order timeout")
        record = {
            "account_id": account_id,
            "client_order_id": cid,
            "order_id": f"WB-{cid[:12]}",
            "status": "FILLED",
            **order,
        }
        self._orders[cid] = record
        self.placed.append(record)
        if str(order.get("instrument_type", "")).upper() == "OPTION":
            legs = order.get("legs") if isinstance(order.get("legs"), list) else []
            leg = legs[0] if legs and isinstance(legs[0], dict) else {}
            if str(order.get("side", "")).upper() == "BUY":
                self._positions.append(
                    {
                        "position_id": f"POS-{cid[:10]}",
                        "instrument_type": "OPTION",
                        "symbol": order.get("symbol") or leg.get("symbol") or "SPY",
                        "quantity": order.get("quantity", "1"),
                        "cost_price": order.get("limit_price", "0"),
                        "option_strategy": "SINGLE",
                        "legs": [
                            {
                                "symbol": leg.get("symbol") or order.get("symbol") or "SPY",
                                "option_type": leg.get("option_type", "CALL"),
                                "option_expire_date": leg.get("option_expire_date"),
                                "option_exercise_price": leg.get("strike_price"),
                                "quantity": order.get("quantity", "1"),
                            }
                        ],
                    }
                )
        return record

    def cancel_order(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        self.cancelled.append(client_order_id)
        if client_order_id in self._orders:
            self._orders[client_order_id]["status"] = "CANCELLED"
        return {"account_id": account_id, "client_order_id": client_order_id, "status": "CANCELLED"}

    def get_order_detail(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        if client_order_id not in self._orders:
            raise WebullApiError(f"Order not found: {client_order_id}")
        return dict(self._orders[client_order_id])

    def list_open_orders(
        self, account_id: str, *, page_size: int = 20
    ) -> list[dict[str, Any]]:
        return [
            o
            for o in self._orders.values()
            if str(o.get("status", "")).upper() not in {"CANCELLED", "FILLED", "REJECTED"}
        ][:page_size]


def extract_cash_balance(balance_payload: dict[str, Any]) -> float:
    """Best-effort cash float from Webull balance payload variants."""
    candidates: list[Any] = []
    for key in (
        "total_cash_balance",
        "total_cash",
        "cash_balance",
        "cash",
        "available_cash",
        "buying_power",
        "totalCashValue",
        "total_net_liquidation_value",
    ):
        if key in balance_payload:
            candidates.append(balance_payload.get(key))
    nested = balance_payload.get("account")
    if isinstance(nested, dict):
        for key in (
            "total_cash_balance",
            "cash_balance",
            "buying_power",
            "total_cash",
        ):
            if key in nested:
                candidates.append(nested.get(key))
    assets = balance_payload.get("account_currency_assets")
    if isinstance(assets, list):
        for row in assets:
            if isinstance(row, dict):
                for key in ("cash_balance", "settled_cash", "buying_power"):
                    if key in row:
                        candidates.append(row.get(key))
    for raw in candidates:
        if raw is None or raw == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return 0.0


def account_looks_like_live_brokerage(
    row: dict[str, Any], *, api_env: str | None = None
) -> bool:
    """Heuristic: OpenAPI 'Individual Cash/Margin' on prod is live brokerage, not app paper."""
    env_name = (api_env or "").strip().lower()
    if env_name in {"sandbox", "uat", "test", "paper"}:
        # Sandbox accounts often reuse CASH/MARGIN labels but are simulated.
        return False
    label = str(row.get("account_label") or "").strip().lower()
    klass = str(row.get("account_class") or "").strip().upper()
    if "paper" in label or "sandbox" in label or "sim" in label:
        return False
    if label in {"individual cash", "individual margin"}:
        return True
    if klass in {"INDIVIDUAL_CASH", "INDIVIDUAL_MARGIN"}:
        return True
    return False

def position_from_webull_row(row: dict[str, Any]) -> tuple[OptionContract, int, float] | None:
    """Parse an OPTION position row into (contract, qty, avg_entry). Returns None if equity."""
    instrument = str(row.get("instrument_type", "")).upper()
    if instrument and instrument != "OPTION":
        return None
    legs = row.get("legs") if isinstance(row.get("legs"), list) else []
    leg = legs[0] if legs and isinstance(legs[0], dict) else row
    symbol = str(leg.get("symbol") or row.get("symbol") or "SPY").upper()
    opt_type_raw = str(leg.get("option_type") or "").upper()
    if opt_type_raw not in {"CALL", "PUT"}:
        return None
    expire = leg.get("option_expire_date") or row.get("option_expire_date")
    strike_raw = (
        leg.get("option_exercise_price")
        or leg.get("strike_price")
        or row.get("strike_price")
    )
    if not expire or strike_raw is None:
        return None
    from datetime import date as date_cls

    try:
        expiration = date_cls.fromisoformat(str(expire)[:10])
        strike = float(strike_raw)
        qty = int(float(row.get("quantity") or leg.get("quantity") or 0))
        avg = float(row.get("cost_price") or 0.0)
    except (TypeError, ValueError):
        return None
    if qty == 0:
        return None
    # V1 schemas only allow 0DTE contracts.
    if expiration != date_cls.today():
        return None
    contract = OptionContract(
        symbol=symbol,
        expiration=expiration,
        strike=strike,
        option_type="call" if opt_type_raw == "CALL" else "put",
        is_0dte=True,
    )
    return contract, abs(qty), avg
