"""Broker interface and paper trading implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from joker.schemas.domain import BrokerOrder, Fill, OptionContract, OrderIntent, Position


class BrokerError(Exception):
    pass


class BrokerClient(ABC):
    @abstractmethod
    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> BrokerOrder:
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> BrokerOrder | None:
        ...

    @abstractmethod
    def list_open_orders(self) -> list[BrokerOrder]:
        ...

    @abstractmethod
    def list_positions(self) -> list[Position]:
        ...

    @abstractmethod
    def get_account_balance(self) -> float:
        ...

    @abstractmethod
    def get_daily_pnl(self) -> float:
        ...


class PaperBroker(BrokerClient):
    """Simulated broker with configurable slippage and spread."""

    def __init__(
        self,
        initial_balance: float = 25000.0,
        slippage_pct: float = 2.0,
        default_spread_pct: float = 5.0,
    ) -> None:
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.slippage_pct = slippage_pct
        self.default_spread_pct = default_spread_pct
        self._orders: dict[str, BrokerOrder] = {}
        self._fills: dict[str, Fill] = {}
        self._positions: dict[str, Position] = {}
        self._daily_pnl = 0.0
        self._used_order_ids: set[str] = set()

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        order_id = str(uuid4())
        if order_id in self._used_order_ids:
            raise BrokerError(f"Duplicate order ID rejected: {order_id}")
        self._used_order_ids.add(order_id)

        order = BrokerOrder(
            order_id=order_id,
            intent_id=intent.intent_id,
            status="open",
            contract=intent.contract,
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
        )
        self._orders[order_id] = order

        if intent.order_type == "limit" and intent.limit_price is not None:
            fill_price = self._simulate_fill_price(intent)
            if self._should_fill(intent, fill_price):
                self._apply_fill(order, fill_price)
        return order

    def _simulate_fill_price(self, intent: OrderIntent) -> float:
        base = intent.limit_price or 1.0
        slip = base * (self.slippage_pct / 100.0)
        return base + slip if intent.side == "buy" else base - slip

    def _should_fill(self, intent: OrderIntent, fill_price: float) -> bool:
        if intent.limit_price is None:
            return True
        if intent.side == "buy":
            return fill_price <= intent.limit_price
        return fill_price >= intent.limit_price

    def _apply_fill(self, order: BrokerOrder, fill_price: float) -> None:
        fill = Fill(
            order_id=order.order_id,
            price=fill_price,
            quantity=order.quantity,
            slippage_pct=self.slippage_pct,
        )
        self._fills[fill.fill_id] = fill
        order.status = "filled"
        cost = fill_price * order.quantity * 100
        if order.side == "buy":
            self.cash_balance -= cost
            pos_id = str(uuid4())
            self._positions[pos_id] = Position(
                position_id=pos_id,
                contract=order.contract,
                quantity=order.quantity,
                avg_entry_price=fill_price,
            )
        else:
            self.cash_balance += cost
            for pos in self._positions.values():
                if pos.is_open and pos.contract == order.contract:
                    pnl = (fill_price - pos.avg_entry_price) * order.quantity * 100
                    self._daily_pnl += pnl
                    pos.is_open = False
                    pos.closed_at = datetime.now(timezone.utc)
                    pos.realized_pnl_usd = pnl

    def cancel_order(self, order_id: str) -> BrokerOrder:
        order = self._orders.get(order_id)
        if order is None:
            raise BrokerError(f"Order not found: {order_id}")
        if order.status == "filled":
            raise BrokerError(f"Cannot cancel filled order: {order_id}")
        order.status = "cancelled"
        return order

    def get_order(self, order_id: str) -> BrokerOrder | None:
        return self._orders.get(order_id)

    def list_open_orders(self) -> list[BrokerOrder]:
        return [o for o in self._orders.values() if o.status == "open"]

    def list_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_account_balance(self) -> float:
        return self.cash_balance

    def get_daily_pnl(self) -> float:
        return self._daily_pnl
