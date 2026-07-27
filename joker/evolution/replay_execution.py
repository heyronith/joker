"""Isolated deterministic replay execution (no production broker)."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from joker.evolution.replay_market import ReplayEpisodeTruth


class ReplayExecutionError(RuntimeError):
    pass


@dataclass
class ReplayOrder:
    client_order_id: str
    contract_id: str
    side: str
    quantity: Decimal
    limit_price: Decimal | None
    status: str = "accepted"
    filled_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    fees: Decimal = Decimal("0")
    parent_order_id: str | None = None


@dataclass
class ReplayPosition:
    contract_id: str
    quantity: Decimal
    avg_price: Decimal
    realised_pnl: Decimal = Decimal("0")
    configuration_version_id: UUID | None = None
    position_lifecycle_id: str | None = None


@dataclass
class ReplayFill:
    fill_id: str
    client_order_id: str
    contract_id: str
    side: str
    quantity: Decimal
    price: Decimal
    fees: Decimal


@dataclass
class ReplayExecutionRuntime:
    """Deterministic paper execution for cognitive replay and shadow."""

    truth: ReplayEpisodeTruth
    slippage_bps: Decimal = Decimal("5")
    fee_per_contract: Decimal = Decimal("0.65")
    cash: Decimal = field(init=False)
    quotes: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: dict[str, ReplayOrder] = field(default_factory=dict)
    positions: dict[str, ReplayPosition] = field(default_factory=dict)
    fills: list[ReplayFill] = field(default_factory=list)
    ledger_events: list[dict[str, Any]] = field(default_factory=list)
    _submitted_keys: set[str] = field(default_factory=set)
    _allowed_contracts: set[str] = field(default_factory=set)
    _surface_locked: bool = False

    def __post_init__(self) -> None:
        self.cash = Decimal(self.truth.starting_cash)
        self.quotes = {k: dict(v) for k, v in self.truth.contract_quotes.items()}
        for seed in self.truth.starting_positions:
            self.positions[seed.contract_id] = ReplayPosition(
                contract_id=seed.contract_id,
                quantity=seed.quantity,
                avg_price=seed.avg_price,
                position_lifecycle_id=getattr(seed, "position_lifecycle_id", None),
            )
        for order_seed in getattr(self.truth, "starting_working_orders", ()) or ():
            self.orders[order_seed.client_order_id] = ReplayOrder(
                client_order_id=order_seed.client_order_id,
                contract_id=order_seed.contract_id,
                side=order_seed.side,
                quantity=order_seed.quantity,
                limit_price=order_seed.limit_price,
                status=order_seed.status,
                filled_qty=order_seed.filled_qty,
                parent_order_id=order_seed.parent_client_order_id,
            )
        self._allowed_contracts = set(self.quotes.keys())
        self._surface_locked = bool(self._allowed_contracts)

    def restore_state(
        self,
        *,
        cash: Decimal,
        orders: dict[str, ReplayOrder],
        positions: dict[str, ReplayPosition],
        fills: list[ReplayFill],
        submitted_keys: set[str],
    ) -> None:
        """Rebuild durable execution state after restart."""
        self.cash = Decimal(cash)
        self.orders = dict(orders)
        self.positions = dict(positions)
        self.fills = list(fills)
        self._submitted_keys = set(submitted_keys)

    def allow_contract(self, contract_id: str, *, bid: Decimal, ask: Decimal) -> None:
        mid = (bid + ask) / Decimal("2")
        self.quotes[contract_id] = {
            "bid": str(bid),
            "ask": str(ask),
            "mid": str(mid),
        }
        self._allowed_contracts.add(contract_id)

    def lock_surface(self, contract_ids: set[str]) -> None:
        self._allowed_contracts = set(contract_ids)
        self._surface_locked = True

    def validate_contract(self, contract_id: str) -> None:
        if self._surface_locked and contract_id not in self._allowed_contracts:
            raise ReplayExecutionError(f"contract_not_on_frozen_surface:{contract_id}")
        if contract_id not in self.quotes:
            raise ReplayExecutionError(f"missing_quote_for_contract:{contract_id}")

    def submit_order(
        self,
        *,
        client_order_id: str,
        contract_id: str,
        side: str,
        quantity: Decimal,
        limit_price: Decimal | None = None,
        idempotency_key: str | None = None,
        fill_fraction: Decimal = Decimal("1"),
        parent_order_id: str | None = None,
        configuration_version_id: UUID | None = None,
        lifecycle_id: str | None = None,
    ) -> ReplayOrder:
        key = idempotency_key or client_order_id
        if key in self._submitted_keys:
            existing = self.orders.get(client_order_id)
            if existing is not None:
                return existing
            raise ReplayExecutionError(f"duplicate_submit_without_order:{key}")
        self.validate_contract(contract_id)
        quote = self.quotes.get(contract_id)
        if quote is None:
            raise ReplayExecutionError(f"missing_quote_for_contract:{contract_id}")
        mid = Decimal(str(quote["mid"]))
        slip = mid * self.slippage_bps / Decimal("10000")
        fill_price = mid + slip if side == "buy" else mid - slip
        if limit_price is not None:
            if side == "buy" and fill_price > limit_price:
                order = ReplayOrder(
                    client_order_id=client_order_id,
                    contract_id=contract_id,
                    side=side,
                    quantity=quantity,
                    limit_price=limit_price,
                    status="rejected",
                    parent_order_id=parent_order_id,
                )
                self.orders[client_order_id] = order
                self._submitted_keys.add(key)
                self.ledger_events.append(
                    {"type": "order_rejected", "client_order_id": client_order_id}
                )
                return order
            if side == "sell" and fill_price < limit_price:
                order = ReplayOrder(
                    client_order_id=client_order_id,
                    contract_id=contract_id,
                    side=side,
                    quantity=quantity,
                    limit_price=limit_price,
                    status="rejected",
                    parent_order_id=parent_order_id,
                )
                self.orders[client_order_id] = order
                self._submitted_keys.add(key)
                self.ledger_events.append(
                    {"type": "order_rejected", "client_order_id": client_order_id}
                )
                return order

        fill_qty = (quantity * fill_fraction).quantize(Decimal("0.0001"))
        if fill_qty <= 0:
            order = ReplayOrder(
                client_order_id=client_order_id,
                contract_id=contract_id,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                status="cancelled",
                parent_order_id=parent_order_id,
            )
            self.orders[client_order_id] = order
            self._submitted_keys.add(key)
            return order

        fees = self.fee_per_contract * fill_qty
        status = "filled" if fill_qty == quantity else "partially_filled"
        order = ReplayOrder(
            client_order_id=client_order_id,
            contract_id=contract_id,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            status=status,
            filled_qty=fill_qty,
            avg_fill_price=fill_price,
            fees=fees,
            parent_order_id=parent_order_id,
        )
        self.orders[client_order_id] = order
        self._submitted_keys.add(key)
        fill = ReplayFill(
            fill_id=str(uuid4()),
            client_order_id=client_order_id,
            contract_id=contract_id,
            side=side,
            quantity=fill_qty,
            price=fill_price,
            fees=fees,
        )
        self.fills.append(fill)
        self._apply_fill(
            fill,
            configuration_version_id=configuration_version_id,
            lifecycle_id=lifecycle_id,
        )
        self.ledger_events.append(
            {
                "type": "fill",
                "fill_id": fill.fill_id,
                "client_order_id": client_order_id,
                "qty": str(fill_qty),
                "price": str(fill_price),
                "fees": str(fees),
            }
        )
        return order

    def cancel_order(self, client_order_id: str) -> ReplayOrder:
        order = self.orders.get(client_order_id)
        if order is None:
            raise ReplayExecutionError(f"unknown_order:{client_order_id}")
        if order.status in {"filled", "cancelled", "rejected"}:
            return order
        order.status = "cancelled"
        self.ledger_events.append(
            {"type": "order_cancelled", "client_order_id": client_order_id}
        )
        return order

    def replace_order(
        self,
        *,
        parent_order_id: str,
        client_order_id: str,
        quantity: Decimal | None = None,
        limit_price: Decimal | None = None,
        idempotency_key: str | None = None,
    ) -> ReplayOrder:
        parent = self.cancel_order(parent_order_id)
        remaining = parent.quantity - parent.filled_qty
        qty = quantity if quantity is not None else remaining
        return self.submit_order(
            client_order_id=client_order_id,
            contract_id=parent.contract_id,
            side=parent.side,
            quantity=qty,
            limit_price=limit_price if limit_price is not None else parent.limit_price,
            idempotency_key=idempotency_key or client_order_id,
            parent_order_id=parent_order_id,
        )

    def _apply_fill(
        self,
        fill: ReplayFill,
        *,
        configuration_version_id: UUID | None,
        lifecycle_id: str | None,
    ) -> None:
        pos = self.positions.get(fill.contract_id)
        multiplier = Decimal("100")
        if fill.side == "buy":
            notional = fill.price * fill.quantity * multiplier
            self.cash -= notional + fill.fees
            if pos is None or pos.quantity <= 0:
                self.positions[fill.contract_id] = ReplayPosition(
                    contract_id=fill.contract_id,
                    quantity=fill.quantity,
                    avg_price=fill.price,
                    configuration_version_id=configuration_version_id,
                    position_lifecycle_id=lifecycle_id,
                )
            else:
                total_qty = pos.quantity + fill.quantity
                pos.avg_price = (
                    (pos.avg_price * pos.quantity) + (fill.price * fill.quantity)
                ) / total_qty
                pos.quantity = total_qty
        else:
            if pos is None or pos.quantity <= 0:
                raise ReplayExecutionError(f"no_position_to_sell:{fill.contract_id}")
            sell_qty = min(pos.quantity, fill.quantity)
            pnl = ((fill.price - pos.avg_price) * sell_qty * multiplier) - fill.fees
            pos.realised_pnl += pnl
            pos.quantity -= sell_qty
            self.cash += fill.price * sell_qty * multiplier - fill.fees
            if pos.quantity <= 0:
                # Keep realised on closed position for reporting.
                pos.quantity = Decimal("0")

    def realised_pnl(self) -> Decimal:
        return sum((p.realised_pnl for p in self.positions.values()), Decimal("0"))

    def projection(self) -> dict[str, Any]:
        return {
            "cash": str(self.cash),
            "orders": {
                oid: {
                    "status": o.status,
                    "filled_qty": str(o.filled_qty),
                    "avg_fill_price": str(o.avg_fill_price) if o.avg_fill_price else None,
                    "contract_id": o.contract_id,
                    "side": o.side,
                }
                for oid, o in self.orders.items()
            },
            "positions": {
                cid: {
                    "quantity": str(p.quantity),
                    "avg_price": str(p.avg_price),
                    "realised_pnl": str(p.realised_pnl),
                    "configuration_version_id": (
                        str(p.configuration_version_id)
                        if p.configuration_version_id
                        else None
                    ),
                }
                for cid, p in self.positions.items()
            },
            "fills": [
                {
                    "fill_id": f.fill_id,
                    "client_order_id": f.client_order_id,
                    "qty": str(f.quantity),
                    "price": str(f.price),
                    "fees": str(f.fees),
                    "side": f.side,
                }
                for f in self.fills
            ],
        }
