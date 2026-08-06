"""Webull paper-account broker adapter.

Real-money LIVE_GATED placement remains disabled. This client only submits when
WEBULL_PAPER_TRADING_ENABLED=true and WEBULL_LIVE_TRADING_ENABLED=false.
"""

from __future__ import annotations

from typing import Any

from joker.broker.interface import BrokerClient, BrokerError
from joker.broker.webull_trade_api import (
    MockWebullTradeApi,
    WebullTradeApi,
    WebullTradeConfigError,
    build_option_limit_order_payload,
    ensure_paper_trading_allowed,
    extract_cash_balance,
    map_webull_order_status,
    new_client_order_id,
    position_from_webull_row,
    validate_webull_paper_trade_env,
)
from joker.config.settings import EnvSettings
from joker.schemas.domain import BrokerOrder, OrderIntent, Position


class WebullConfigError(Exception):
    pass


def validate_webull_env(env: EnvSettings) -> None:
    """Legacy validator used by older tests — requires full broker credential set."""
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
        raise WebullConfigError(
            "Webull credentials required: " + ", ".join(missing)
        )


class WebullClient(BrokerClient):
    """Paper-account Webull broker. Live money orders are never permitted here."""

    LIVE_CALLS_ENABLED = False  # real-money path stays off

    def __init__(
        self,
        env: EnvSettings,
        *,
        trade_api: WebullTradeApi | None = None,
    ) -> None:
        try:
            validate_webull_paper_trade_env(env)
            ensure_paper_trading_allowed(env)
        except WebullTradeConfigError as exc:
            raise BrokerError(str(exc)) from exc
        if env.webull_live_trading_enabled or self.LIVE_CALLS_ENABLED:
            raise BrokerError(
                "Refusing WebullClient: live money trading must remain disabled"
            )
        self._env = env
        self._account_id = str(env.webull_paper_account_id)
        from joker.runtime.cognitive_session import paper_account_identity

        self._account_identity = paper_account_identity(
            broker_kind="webull_paper", env=env
        )
        self._api = trade_api or _default_trade_api(env)
        self._orders: dict[str, BrokerOrder] = {}
        self._intent_by_order: dict[str, OrderIntent] = {}

    @property
    def account_identity(self) -> str:
        """Non-reversible durable identity; the raw account id stays private."""
        return self._account_identity

    def close(self) -> None:
        """Close the underlying Webull trade HTTP client when present."""
        close = getattr(self._api, "close", None)
        if callable(close):
            close()

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        self._assert_paper_only()
        client_order_id = new_client_order_id()
        try:
            payload = build_option_limit_order_payload(
                intent, client_order_id=client_order_id
            )
            response = self._api.place_order(self._account_id, [payload])
        except WebullTradeConfigError as exc:
            raise BrokerError(str(exc)) from exc
        except Exception as exc:
            raise BrokerError(f"Webull place order failed: {exc}") from exc

        status = map_webull_order_status(
            str(response.get("status") or response.get("order_status") or "SUBMITTED")
        )
        order = BrokerOrder(
            order_id=client_order_id,
            intent_id=intent.intent_id,
            status=status,  # type: ignore[arg-type]
            contract=intent.contract,
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
        )
        self._orders[client_order_id] = order
        self._intent_by_order[client_order_id] = intent
        return order

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self._assert_paper_only()
        try:
            self._api.cancel_order(self._account_id, order_id)
        except Exception as exc:
            raise BrokerError(f"Webull cancel failed: {exc}") from exc
        order = self._orders.get(order_id)
        if order is None:
            intent = self._intent_by_order.get(order_id)
            if intent is None:
                raise BrokerError(f"Unknown local order_id: {order_id}")
            order = BrokerOrder(
                order_id=order_id,
                intent_id=intent.intent_id,
                status="cancelled",
                contract=intent.contract,
                side=intent.side,
                quantity=intent.quantity,
                limit_price=intent.limit_price,
            )
        else:
            order.status = "cancelled"
        self._orders[order_id] = order
        return order

    def get_order(self, order_id: str) -> BrokerOrder | None:
        try:
            detail = self._api.get_order_detail(self._account_id, order_id)
        except Exception:
            return self._orders.get(order_id)
        local = self._orders.get(order_id)
        intent = self._intent_by_order.get(order_id)
        status = map_webull_order_status(
            str(detail.get("status") or detail.get("order_status") or "")
        )
        if local is not None:
            local.status = status  # type: ignore[assignment]
            return local
        if intent is None:
            return None
        order = BrokerOrder(
            order_id=order_id,
            intent_id=intent.intent_id,
            status=status,  # type: ignore[arg-type]
            contract=intent.contract,
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
        )
        self._orders[order_id] = order
        return order

    def list_open_orders(self) -> list[BrokerOrder]:
        try:
            rows = self._api.list_open_orders(self._account_id)
        except Exception:
            return [o for o in self._orders.values() if o.status in {"open", "pending"}]
        result: list[BrokerOrder] = []
        for row in rows:
            cid = str(row.get("client_order_id") or "")
            if not cid:
                continue
            local = self.get_order(cid)
            if local is not None and local.status in {"open", "pending"}:
                result.append(local)
        return result

    def list_positions(self) -> list[Position]:
        try:
            rows = self._api.get_positions(self._account_id)
        except Exception as exc:
            raise BrokerError(f"Webull positions failed: {exc}") from exc
        positions: list[Position] = []
        for row in rows:
            parsed = position_from_webull_row(row)
            if parsed is None:
                continue
            contract, qty, avg = parsed
            positions.append(
                Position(
                    position_id=str(row.get("position_id") or new_client_order_id()),
                    contract=contract,
                    quantity=qty,
                    avg_entry_price=avg,
                )
            )
        return positions

    def get_account_balance(self) -> float:
        try:
            payload = self._api.get_balance(self._account_id)
        except Exception as exc:
            raise BrokerError(f"Webull balance failed: {exc}") from exc
        return extract_cash_balance(payload)

    def get_daily_pnl(self) -> float:
        # Legacy float API — callers must use get_daily_pnl_available() for truthfulness.
        # Unavailable day PnL must not be treated as a real zero by ExecutionRuntime.
        return 0.0

    def get_daily_pnl_available(self) -> tuple[bool, float | None]:
        """Webull day-PnL field is not verified — report unavailable (do not fabricate)."""
        return False, None

    def list_accounts_raw(self) -> list[dict[str, Any]]:
        return self._api.list_accounts()

    def _assert_paper_only(self) -> None:
        try:
            ensure_paper_trading_allowed(self._env)
        except WebullTradeConfigError as exc:
            raise BrokerError(str(exc)) from exc
        if self.LIVE_CALLS_ENABLED or self._env.webull_live_trading_enabled:
            raise BrokerError("Live money Webull calls remain disabled")


def _default_trade_api(env: EnvSettings) -> WebullTradeApi:
    from joker.broker.webull_trade_api import HttpWebullTradeApi

    return HttpWebullTradeApi(env)


class MockWebullClient(BrokerClient):
    """Test double that never hits the network."""

    def __init__(self, account_id: str = "PAPER_ACCT_TEST") -> None:
        self._api = MockWebullTradeApi(account_id=account_id)
        self._orders: dict[str, BrokerOrder] = {}

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        raise BrokerError("MockWebullClient does not submit in tests by default")

    def cancel_order(self, order_id: str) -> BrokerOrder:
        raise BrokerError("Not implemented in mock")

    def get_order(self, order_id: str) -> BrokerOrder | None:
        return self._orders.get(order_id)

    def list_open_orders(self) -> list[BrokerOrder]:
        return []

    def list_positions(self) -> list[Position]:
        return []

    def get_account_balance(self) -> float:
        return 0.0

    def get_daily_pnl(self) -> float:
        available, value = self.get_daily_pnl_available()
        return 0.0 if not available or value is None else value

    def get_daily_pnl_available(self) -> tuple[bool, float | None]:
        """Mock has no day-PnL source — unavailable."""
        return (False, None)
