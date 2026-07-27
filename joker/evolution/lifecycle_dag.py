"""Mutually exclusive lifecycle order DAG for episode resolution."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from joker.ledger.projector import OrderLifecycle, OrderStatus


ActionKind = Literal[
    "entry",
    "entry_replace",
    "add",
    "reduction",
    "exit",
    "exit_replace",
]


class LifecycleOrderNode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str
    parent_client_order_id: str | None = None
    originating_entry_client_order_id: str | None = None
    action_kind: ActionKind
    side: Literal["buy", "sell"]
    filled_quantity: Decimal
    average_fill_price: Decimal | None = None
    fees: Decimal = Decimal("0")


class LifecycleOrderGraph(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: tuple[LifecycleOrderNode, ...] = ()
    original_entry_ids: frozenset[str] = Field(default_factory=frozenset)
    entry_replacement_ids: frozenset[str] = Field(default_factory=frozenset)
    scale_in_ids: frozenset[str] = Field(default_factory=frozenset)
    reduction_ids: frozenset[str] = Field(default_factory=frozenset)
    terminal_exit_ids: frozenset[str] = Field(default_factory=frozenset)
    exit_replacement_ids: frozenset[str] = Field(default_factory=frozenset)
    findings: tuple[str, ...] = ()

    def categories_overlap(self) -> bool:
        buckets = (
            self.original_entry_ids,
            self.entry_replacement_ids,
            self.scale_in_ids,
            self.reduction_ids,
            self.terminal_exit_ids,
            self.exit_replacement_ids,
        )
        seen: set[str] = set()
        for bucket in buckets:
            if seen & bucket:
                return True
            seen |= set(bucket)
        return False


def build_lifecycle_order_graph(
    orders: list[OrderLifecycle],
    *,
    originating_entry_id: str | None,
    terminal_exit_id: str | None,
    lifecycle_id: str | None,
) -> LifecycleOrderGraph:
    """Classify each order into exactly one category via parent ancestry."""
    findings: list[str] = []
    by_id = {o.client_order_id: o for o in orders}
    filled = [
        o
        for o in orders
        if o.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}
    ]

    # Detect parent cycles / cross-lifecycle parents.
    for order in orders:
        parent = getattr(order, "parent_client_order_id", None)
        if not parent:
            continue
        if parent not in by_id:
            findings.append(f"missing_parent:{order.client_order_id}")
            continue
        parent_order = by_id[parent]
        parent_life = getattr(parent_order, "position_lifecycle_id", None)
        child_life = getattr(order, "position_lifecycle_id", None) or lifecycle_id
        if parent_life and child_life and parent_life != child_life:
            findings.append(f"cross_lifecycle_parent:{order.client_order_id}")
        # Cycle detection.
        seen: set[str] = set()
        cur: str | None = order.client_order_id
        while cur:
            if cur in seen:
                findings.append(f"order_parent_cycle:{order.client_order_id}")
                break
            seen.add(cur)
            parent_obj = by_id.get(cur)
            cur = getattr(parent_obj, "parent_client_order_id", None) if parent_obj else None

    # Resolve originating entry.
    origin = originating_entry_id
    if origin is None:
        roots = [
            o
            for o in filled
            if o.side == "buy" and not getattr(o, "parent_client_order_id", None)
        ]
        if roots:
            origin = roots[0].client_order_id

    original_entry: set[str] = set()
    entry_replace: set[str] = set()
    scale_in: set[str] = set()
    reductions: set[str] = set()
    terminal_exits: set[str] = set()
    exit_replace: set[str] = set()
    nodes: list[LifecycleOrderNode] = []

    def _is_ancestor(candidate: str, ancestor: str) -> bool:
        cur: str | None = candidate
        guard = 0
        while cur and guard < 64:
            if cur == ancestor:
                return True
            parent_obj = by_id.get(cur)
            cur = getattr(parent_obj, "parent_client_order_id", None) if parent_obj else None
            guard += 1
        return False

    for order in filled:
        oid = order.client_order_id
        parent = getattr(order, "parent_client_order_id", None)
        kind: ActionKind
        if order.side == "buy":
            if origin and oid == origin:
                kind = "entry"
                original_entry.add(oid)
            elif parent and origin and _is_ancestor(oid, origin):
                # Replacement of entry chain (not a fresh scale-in).
                kind = "entry_replace"
                entry_replace.add(oid)
            elif parent is None and origin and oid != origin:
                kind = "add"
                scale_in.add(oid)
            elif parent and origin and not _is_ancestor(parent, origin) and parent != origin:
                kind = "add"
                scale_in.add(oid)
            else:
                # Default: first unmatched buy is entry; subsequent unmatched buys scale-in.
                if not original_entry and origin is None:
                    kind = "entry"
                    original_entry.add(oid)
                    origin = oid
                elif parent:
                    kind = "entry_replace"
                    entry_replace.add(oid)
                else:
                    kind = "add"
                    scale_in.add(oid)
        else:  # sell
            if terminal_exit_id and oid == terminal_exit_id:
                if parent and parent != origin:
                    kind = "exit_replace"
                    exit_replace.add(oid)
                else:
                    kind = "exit"
                    terminal_exits.add(oid)
            elif parent and terminal_exit_id and _is_ancestor(oid, terminal_exit_id):
                kind = "exit_replace"
                exit_replace.add(oid)
            elif terminal_exit_id is None and not terminal_exits and order == filled[-1]:
                kind = "exit"
                terminal_exits.add(oid)
            else:
                kind = "reduction"
                reductions.add(oid)

        nodes.append(
            LifecycleOrderNode(
                client_order_id=oid,
                parent_client_order_id=parent,
                originating_entry_client_order_id=origin
                or getattr(order, "originating_entry_client_order_id", None),
                action_kind=kind,
                side="buy" if order.side == "buy" else "sell",
                filled_quantity=order.filled_qty,
                average_fill_price=order.avg_fill_price,
                fees=order.fees,
            )
        )

    # If no terminal exit classified but sells exist, promote last sell.
    if not terminal_exits and not exit_replace:
        sell_nodes = [n for n in nodes if n.side == "sell"]
        if sell_nodes:
            last = sell_nodes[-1]
            reductions.discard(last.client_order_id)
            terminal_exits.add(last.client_order_id)
            nodes = [
                n
                if n.client_order_id != last.client_order_id
                else n.model_copy(update={"action_kind": "exit"})
                for n in nodes
            ]

    graph = LifecycleOrderGraph(
        nodes=tuple(nodes),
        original_entry_ids=frozenset(original_entry),
        entry_replacement_ids=frozenset(entry_replace),
        scale_in_ids=frozenset(scale_in),
        reduction_ids=frozenset(reductions),
        terminal_exit_ids=frozenset(terminal_exits),
        exit_replacement_ids=frozenset(exit_replace),
        findings=tuple(dict.fromkeys(findings)),
    )
    if graph.categories_overlap():
        findings.append("lifecycle_category_overlap")
        graph = graph.model_copy(
            update={"findings": tuple(dict.fromkeys([*graph.findings, *findings]))}
        )
    return graph


def unique_fill_accounting(
    graph: LifecycleOrderGraph,
) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
    """Return (entry_qty, exit_qty, buy_cost, sell_proceeds, fees) without double-count."""
    seen: set[str] = set()
    entry_qty = Decimal("0")
    exit_qty = Decimal("0")
    buy_cost = Decimal("0")
    sell_proceeds = Decimal("0")
    fees = Decimal("0")
    for node in graph.nodes:
        if node.client_order_id in seen:
            continue
        seen.add(node.client_order_id)
        fees += node.fees
        px = node.average_fill_price or Decimal("0")
        notional = px * node.filled_quantity * Decimal("100")
        if node.action_kind in {"entry", "entry_replace", "add"}:
            entry_qty += node.filled_quantity
            buy_cost += notional
        elif node.action_kind in {"reduction", "exit", "exit_replace"}:
            exit_qty += node.filled_quantity
            sell_proceeds += notional
    return entry_qty, exit_qty, buy_cost, sell_proceeds, fees
