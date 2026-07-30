"""Authoritative order-action gateway — sole path for agent-originated submits."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Sequence
from uuid import UUID, uuid4

from joker.agents.cognitive.execution import (
    AuthoritativeMarketTruth,
    ProvenancedExecutionCommand,
    parse_contract_id,
)
from joker.cognition.exceptions import CognitiveValidationError
from joker.graph.context_hydrate import load_snapshot_truth
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.ledger.projector import OrderLifecycle, OrderStatus, PositionState
from joker.market.option_surface import OptionContractSnapshot
from joker.market.quality import DataQualityReport
from joker.market.snapshots import MarketSnapshot
from joker.runtime.execution_runtime import ExecutionCommand, contract_id_for
from joker.schemas.domain import BrokerOrder, OptionContract, OrderIntent

logger = logging.getLogger(__name__)

_WORKING_STATUSES = frozenset(
    {
        OrderStatus.SUBMITTED.value,
        OrderStatus.ACCEPTED.value,
        OrderStatus.PARTIALLY_FILLED.value,
        "submitted",
        "accepted",
        "partially_filled",
        "open",
        "WORKING",
        "working",
    }
)


class OrderActionKind(StrEnum):
    ENTRY = "entry"
    PROBE = "probe"
    ADD = "add"
    REDUCE = "reduce"
    EXIT = "exit"
    REPLACE = "replace"
    CANCEL = "cancel"


@dataclass(frozen=True)
class WorkingOrderTruth:
    """Typed working-order projection used for conflict and replacement checks."""

    client_order_id: str
    contract_id: str
    side: str
    requested_quantity: int
    filled_quantity: int
    remaining_quantity: int
    status: str
    proposal_id: str | None = None


@dataclass(frozen=True)
class OrderActionRequest:
    """Typed agent action routed through the authoritative gateway."""

    action: OrderActionKind
    snapshot_id: str
    contract_id: str
    side: Literal["buy", "sell"]
    quantity: int
    client_order_id: str
    limit_price: float | None = None
    order_type: Literal["limit", "market"] = "limit"
    max_quote_age_seconds: int | None = 60
    proposal_id: str | None = None
    decision_id: str | None = None
    strategy_id: str | None = None
    cycle_id: str | None = None
    replace_of_client_order_id: str | None = None
    position_lifecycle_id: str | None = None
    originating_entry_client_order_id: str | None = None
    allow_degraded_exit: bool = True
    degraded_exit_reason: str | None = None
    evidence_ids: tuple[str, ...] = ()
    broker_account_id: str = "default"


@dataclass
class OrderActionResult:
    """Gateway outcome — either a submitted broker order or a blocked reason."""

    submitted: bool
    client_order_id: str
    broker_order: BrokerOrder | None = None
    blocked_reason: str | None = None
    degraded_exit: bool = False
    degraded_exit_reason: str | None = None
    command: ExecutionCommand | None = None
    working_orders: tuple[WorkingOrderTruth, ...] = ()


def working_orders_from_projection(
    projection: Any | None,
    *,
    proposal_id_by_client_order: dict[str, str] | None = None,
) -> tuple[WorkingOrderTruth, ...]:
    """Extract typed working-order truth from a Task 1 ledger projection."""
    if projection is None:
        return ()
    orders = getattr(projection, "orders", None) or {}
    order_iter = orders.values() if isinstance(orders, dict) else orders
    out: list[WorkingOrderTruth] = []
    proposal_map = proposal_id_by_client_order or {}
    for order in order_iter:
        if isinstance(order, OrderLifecycle):
            status = order.status.value
            client_id = order.client_order_id
            contract_id = order.contract_id
            side = order.side
            requested = int(order.submitted_qty)
            filled = int(order.filled_qty)
        elif isinstance(order, dict):
            status = str(
                getattr(order.get("status"), "value", None) or order.get("status") or ""
            )
            client_id = str(order.get("client_order_id") or "")
            contract_id = str(order.get("contract_id") or "")
            side = str(order.get("side") or "")
            requested = int(order.get("submitted_qty") or order.get("quantity") or 0)
            filled = int(order.get("filled_qty") or order.get("filled_quantity") or 0)
        else:
            status = str(
                getattr(getattr(order, "status", None), "value", None)
                or getattr(order, "status", "")
                or ""
            )
            client_id = str(getattr(order, "client_order_id", "") or "")
            contract_id = str(getattr(order, "contract_id", "") or "")
            side = str(getattr(order, "side", "") or "")
            requested = int(getattr(order, "submitted_qty", 0) or getattr(order, "quantity", 0) or 0)
            filled = int(getattr(order, "filled_qty", 0) or getattr(order, "filled_quantity", 0) or 0)
        if status not in _WORKING_STATUSES:
            continue
        if not client_id:
            continue
        remaining = max(0, requested - filled)
        out.append(
            WorkingOrderTruth(
                client_order_id=client_id,
                contract_id=contract_id,
                side=side,
                requested_quantity=requested,
                filled_quantity=filled,
                remaining_quantity=remaining,
                status=status,
                proposal_id=proposal_map.get(client_id),
            )
        )
    return tuple(out)


def open_positions_from_projection(
    projection: Any | None,
) -> dict[str, PositionState | dict[str, Any]]:
    if projection is None:
        return {}
    positions = getattr(projection, "positions", None) or {}
    if isinstance(positions, dict):
        items = positions.items()
    else:
        items = ((getattr(p, "contract_id", None), p) for p in positions)
    open_pos: dict[str, PositionState | dict[str, Any]] = {}
    for key, pos in items:
        qty = getattr(pos, "quantity", None)
        if qty is None and isinstance(pos, dict):
            qty = pos.get("quantity") or pos.get("net_quantity")
        try:
            q = Decimal(str(qty)) if qty is not None else Decimal("0")
        except Exception:
            q = Decimal("0")
        if q == 0:
            continue
        cid = str(getattr(pos, "contract_id", None) or key or "")
        if isinstance(pos, dict):
            cid = str(pos.get("contract_id") or key or "")
        if cid:
            open_pos[cid] = pos
    return open_pos


def has_working_entry_order(working: Sequence[WorkingOrderTruth]) -> bool:
    return any(o.side == "buy" and o.remaining_quantity > 0 for o in working)


def ensure_order_action_gateway(deps: CognitiveGraphDeps) -> OrderActionGateway | None:
    """Lazily attach the authoritative gateway once ExecutionRuntime is available."""
    if deps.order_action_gateway is not None:
        return deps.order_action_gateway
    if deps.execution_runtime is None:
        return None
    deps.order_action_gateway = OrderActionGateway(deps)
    return deps.order_action_gateway


class OrderActionGateway:
    """Deterministic gateway: reload Task 1 truth → validate → ExecutionRuntime."""

    def __init__(
        self,
        deps: CognitiveGraphDeps,
        *,
        max_quantity: int = 20,
        paper_mode_required: bool = True,
    ) -> None:
        self._deps = deps
        self._max_quantity = max_quantity
        self._paper_mode_required = paper_mode_required

    async def submit(self, request: OrderActionRequest) -> OrderActionResult:
        if self._deps.execution_runtime is None:
            return OrderActionResult(
                submitted=False,
                client_order_id=request.client_order_id,
                blocked_reason="ExecutionRuntime unavailable",
            )

        if request.action == OrderActionKind.CANCEL:
            target = request.replace_of_client_order_id or request.client_order_id
            await self._deps.execution_runtime.cancel_order(client_order_id=target)
            if self._deps.objective_service is not None:
                await self._deps.objective_service.release_for_order(
                    client_order_id=target,
                    reason="cancelled",
                )
            return OrderActionResult(
                submitted=True,
                client_order_id=target,
                broker_order=None,
            )

        snapshot, data_quality, surface, _slice = await load_snapshot_truth(
            self._deps, request.snapshot_id
        )
        projection = None
        if self._deps.projection_loader is not None:
            projection = await self._deps.projection_loader()
        working = working_orders_from_projection(projection)
        open_positions = open_positions_from_projection(projection)

        try:
            command = self._validate_and_compile(
                request,
                snapshot=snapshot,
                data_quality=data_quality,
                surface=surface,
                working=working,
                open_positions=open_positions,
            )
        except CognitiveValidationError as exc:
            msg = str(exc).lower()
            degraded = (
                request.action in {OrderActionKind.EXIT, OrderActionKind.REDUCE}
                and request.allow_degraded_exit
                and any(
                    token in msg
                    for token in (
                        "data quality",
                        "quote age",
                        "option surface",
                        "usable",
                    )
                )
            )
            if degraded:
                reason = request.degraded_exit_reason or str(exc)
                logger.warning(
                    "degraded_exit_attempt",
                    extra={
                        "client_order_id": request.client_order_id,
                        "reason": reason,
                    },
                )
                # Re-validate position constraints only, then compile without DQ gate.
                command = self._compile_degraded_close(
                    request,
                    snapshot=snapshot,
                    open_positions=open_positions,
                    degraded_reason=reason,
                )
                order = await self._deps.execution_runtime.submit_execution_command(command)
                if self._deps.provenance_registry is not None:
                    from joker.persistence.cognitive_execution_provenance import (
                        ExecutionProvenanceRecord,
                    )

                    await self._deps.provenance_registry.record(
                        ExecutionProvenanceRecord(
                            client_order_id=command.client_order_id,
                            proposal_id=request.proposal_id,
                            decision_id=request.decision_id,
                            strategy_id=request.strategy_id,
                            cycle_id=request.cycle_id,
                            snapshot_id=request.snapshot_id,
                            contract_id=request.contract_id,
                            session_id=self._deps.session_id,
                            kind=request.action.value,
                            extra={
                                "degraded_exit": True,
                                "degraded_exit_reason": reason,
                            },
                        )
                    )
                return OrderActionResult(
                    submitted=True,
                    client_order_id=command.client_order_id,
                    broker_order=order,
                    degraded_exit=True,
                    degraded_exit_reason=reason,
                    command=command,
                    working_orders=working,
                )
            return OrderActionResult(
                submitted=False,
                client_order_id=request.client_order_id,
                blocked_reason=str(exc),
                working_orders=working,
            )

        if request.action == OrderActionKind.REPLACE:
            assert request.replace_of_client_order_id
            await self._deps.execution_runtime.cancel_order(
                client_order_id=request.replace_of_client_order_id
            )

        # Objective capital reservation before ENTRY/PROBE/ADD
        reserved = False
        if request.action in {
            OrderActionKind.ENTRY,
            OrderActionKind.PROBE,
            OrderActionKind.ADD,
        } and self._deps.objective_service is not None:
            try:
                obj_state = await self._deps.objective_service.get_state()
                premium = Decimal(str(command.intent.limit_price or 0)) * Decimal("100") * Decimal(
                    int(command.intent.quantity)
                )
                await self._deps.objective_service.reserve_for_order(
                    client_order_id=command.client_order_id,
                    estimated_premium_usd=premium,
                    objective_state_version=obj_state.version,
                )
                reserved = True
            except Exception as exc:
                return OrderActionResult(
                    submitted=False,
                    client_order_id=request.client_order_id,
                    blocked_reason=f"objective_reserve_failed: {exc}",
                    working_orders=working,
                )

        try:
            order = await self._deps.execution_runtime.submit_execution_command(command)
        except Exception as exc:
            if reserved and self._deps.objective_service is not None:
                await self._deps.objective_service.release_for_order(
                    client_order_id=command.client_order_id,
                    reason="broker_submission_failed",
                )
            raise
        if reserved and self._deps.objective_service is not None and order is not None:
            await self._deps.objective_service.associate_broker_order(
                client_order_id=command.client_order_id,
                broker_order_id=str(order.order_id),
            )
        if self._deps.provenance_registry is not None:
            from joker.evolution.lifecycle_id import make_position_lifecycle_id
            from joker.persistence.cognitive_execution_provenance import (
                ExecutionProvenanceRecord,
            )

            lifecycle_id = request.position_lifecycle_id
            originating = request.originating_entry_client_order_id
            parent = request.replace_of_client_order_id
            if request.action in {OrderActionKind.ENTRY, OrderActionKind.PROBE}:
                originating = originating or command.client_order_id
                lifecycle_id = lifecycle_id or make_position_lifecycle_id(
                    session_id=self._deps.session_id,
                    originating_entry_client_order_id=originating,
                    contract_id=request.contract_id,
                )
            elif parent:
                prior = await self._deps.provenance_registry.get_by_client_order_id(
                    parent
                )
                if prior is not None:
                    lifecycle_id = lifecycle_id or prior.position_lifecycle_id
                    originating = (
                        originating or prior.originating_entry_client_order_id
                    )
            if lifecycle_id is None and request.contract_id:
                prior = await self._deps.provenance_registry.get_latest_by_contract_id(
                    request.contract_id
                )
                if prior is not None:
                    lifecycle_id = prior.position_lifecycle_id
                    originating = prior.originating_entry_client_order_id

            await self._deps.provenance_registry.record(
                ExecutionProvenanceRecord(
                    client_order_id=command.client_order_id,
                    proposal_id=request.proposal_id,
                    decision_id=request.decision_id,
                    strategy_id=request.strategy_id,
                    cycle_id=request.cycle_id,
                    snapshot_id=request.snapshot_id,
                    contract_id=request.contract_id,
                    session_id=self._deps.session_id,
                    kind=request.action.value,
                    position_lifecycle_id=lifecycle_id,
                    originating_entry_client_order_id=originating,
                    parent_client_order_id=parent,
                    extra={
                        "replace_of": request.replace_of_client_order_id,
                        "position_lifecycle_id": lifecycle_id,
                        "originating_entry_client_order_id": originating,
                    },
                )
            )
        if request.proposal_id:
            self._deps.submitted_proposal_ids.add(str(request.proposal_id))
        return OrderActionResult(
            submitted=True,
            client_order_id=command.client_order_id,
            broker_order=order,
            command=command,
            working_orders=working,
        )

    def _validate_and_compile(
        self,
        request: OrderActionRequest,
        *,
        snapshot: MarketSnapshot,
        data_quality: DataQualityReport | None,
        surface,
        working: Sequence[WorkingOrderTruth],
        open_positions: dict[str, Any],
    ) -> ExecutionCommand:
        action = request.action
        if action in {
            OrderActionKind.ENTRY,
            OrderActionKind.PROBE,
            OrderActionKind.ADD,
            OrderActionKind.REPLACE,
        }:
            if data_quality is None or not data_quality.usable_for_execution:
                raise CognitiveValidationError(
                    "data quality does not permit execution"
                    + (
                        f" (severity={data_quality.severity.value})"
                        if data_quality is not None
                        else " (missing)"
                    )
                )
        if surface is None:
            raise CognitiveValidationError("referenced option surface does not exist")
        if (
            snapshot.option_surface_id is not None
            and surface.surface_id != snapshot.option_surface_id
        ):
            raise CognitiveValidationError(
                "option surface id does not match snapshot.option_surface_id"
            )

        contract = parse_contract_id(
            request.contract_id, trading_date=snapshot.trading_date
        )
        if contract.symbol != "SPY":
            raise CognitiveValidationError("only SPY contracts are supported")
        if contract.expiration != snapshot.trading_date:
            raise CognitiveValidationError(
                "contract expiry must equal the snapshot exchange trading date"
            )
        # Mark 0DTE only after trading-date comparison.
        if not (contract.expiration == snapshot.trading_date):
            raise CognitiveValidationError("only 0DTE contracts are supported")

        surface_by_id = {c.contract_id: c for c in surface.contracts}
        surface_row = surface_by_id.get(request.contract_id)
        if surface_row is None:
            raise CognitiveValidationError(
                f"contract_id {request.contract_id!r} is absent from the option surface"
            )
        self._validate_quote_age(
            request.max_quote_age_seconds,
            surface_row,
            now=datetime.now(timezone.utc),
        )

        if request.quantity <= 0:
            raise CognitiveValidationError("quantity must be positive")
        if request.quantity > self._max_quantity:
            raise CognitiveValidationError(
                f"quantity {request.quantity} exceeds max_quantity={self._max_quantity}"
            )

        if action in {OrderActionKind.ENTRY, OrderActionKind.PROBE}:
            if has_working_entry_order(working):
                raise CognitiveValidationError(
                    "conflicting working entry order exists; no additional entry"
                )
            if request.proposal_id and str(request.proposal_id) in self._deps.submitted_proposal_ids:
                raise CognitiveValidationError("proposal/decision has already been submitted")
            if request.proposal_id and any(
                o.proposal_id == str(request.proposal_id) for o in working
            ):
                raise CognitiveValidationError(
                    "conflicting active order exists for proposal"
                )
            if request.contract_id in open_positions and action == OrderActionKind.ENTRY:
                raise CognitiveValidationError(
                    "conflicting active position exists for contract"
                )
            if request.side != "buy":
                raise CognitiveValidationError("entry/probe requires buy side")

        if action == OrderActionKind.ADD:
            if request.side != "buy":
                raise CognitiveValidationError("ADD requires buy side")
            if request.contract_id not in open_positions:
                raise CognitiveValidationError("ADD requires an authoritative open position")
            if has_working_entry_order(
                [o for o in working if o.contract_id == request.contract_id]
            ):
                raise CognitiveValidationError(
                    "conflicting working order exists for ADD contract"
                )

        if action in {OrderActionKind.REDUCE, OrderActionKind.EXIT}:
            pos = open_positions.get(request.contract_id)
            if pos is None:
                raise CognitiveValidationError(
                    "REDUCE/EXIT requires an authoritative open position"
                )
            open_qty = int(
                getattr(pos, "quantity", None)
                or (pos.get("quantity") if isinstance(pos, dict) else 0)
                or 0
            )
            if request.side != "sell":
                raise CognitiveValidationError("REDUCE/EXIT requires sell side")
            if request.quantity > open_qty:
                raise CognitiveValidationError(
                    f"sell quantity {request.quantity} exceeds open quantity {open_qty}"
                )
            # Duplicate closing command: same client_order_id already known.
            if any(o.client_order_id == request.client_order_id for o in working):
                raise CognitiveValidationError("duplicate closing command")

        if action == OrderActionKind.REPLACE:
            prior = request.replace_of_client_order_id
            if not prior:
                raise CognitiveValidationError("replace requires replace_of_client_order_id")
            match = next((o for o in working if o.client_order_id == prior), None)
            if match is None:
                raise CognitiveValidationError(
                    "original order is not cancellable / not working"
                )
            if match.remaining_quantity <= 0:
                raise CognitiveValidationError("no remaining quantity to replace")
            if match.contract_id != request.contract_id:
                raise CognitiveValidationError(
                    "replacement contract must match original lifecycle"
                )
            if match.side != request.side:
                raise CognitiveValidationError(
                    "replacement side must match original lifecycle"
                )
            if request.quantity > match.remaining_quantity:
                raise CognitiveValidationError(
                    "replacement quantity exceeds remaining authoritative quantity"
                )

        intent = OrderIntent(
            intent_id=request.client_order_id,
            candidate_id=request.proposal_id or request.decision_id or request.client_order_id,
            contract=OptionContract(
                symbol=contract.symbol,
                expiration=contract.expiration,
                strike=contract.strike,
                option_type=contract.option_type,
                is_0dte=True,
            ),
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            limit_price=request.limit_price,
        )
        return ExecutionCommand(
            client_order_id=request.client_order_id,
            intent=intent,
            broker_account_id=request.broker_account_id,
        )

    def _compile_degraded_close(
        self,
        request: OrderActionRequest,
        *,
        snapshot: MarketSnapshot,
        open_positions: dict[str, Any],
        degraded_reason: str,
    ) -> ExecutionCommand:
        pos = open_positions.get(request.contract_id)
        if pos is None:
            raise CognitiveValidationError(
                "degraded exit blocked: no authoritative open position"
            )
        open_qty = int(
            getattr(pos, "quantity", None)
            or (pos.get("quantity") if isinstance(pos, dict) else 0)
            or 0
        )
        qty = min(request.quantity, open_qty)
        if qty <= 0:
            raise CognitiveValidationError("degraded exit blocked: zero open quantity")
        contract = parse_contract_id(
            request.contract_id, trading_date=snapshot.trading_date
        )
        intent = OrderIntent(
            intent_id=request.client_order_id,
            candidate_id=request.decision_id or request.client_order_id,
            contract=OptionContract(
                symbol=contract.symbol,
                expiration=contract.expiration,
                strike=contract.strike,
                option_type=contract.option_type,
                is_0dte=contract.expiration == snapshot.trading_date,
            ),
            side="sell",
            order_type=request.order_type,
            quantity=qty,
            limit_price=request.limit_price,
        )
        logger.info(
            "compiled_degraded_exit",
            extra={
                "client_order_id": request.client_order_id,
                "reason": degraded_reason,
                "quantity": qty,
            },
        )
        return ExecutionCommand(
            client_order_id=request.client_order_id,
            intent=intent,
            broker_account_id=request.broker_account_id,
        )

    @staticmethod
    def _validate_quote_age(
        max_quote_age_seconds: int | None,
        row: OptionContractSnapshot,
        *,
        now: datetime,
    ) -> None:
        if max_quote_age_seconds is None:
            return
        if row.quote_age_ms is not None:
            if row.quote_age_ms > max_quote_age_seconds * 1000:
                raise CognitiveValidationError(
                    f"quote age {row.quote_age_ms}ms exceeds proposal limit "
                    f"{max_quote_age_seconds}s"
                )
            return
        if row.quote_timestamp is not None:
            age = (now - row.quote_timestamp).total_seconds()
            if age > max_quote_age_seconds:
                raise CognitiveValidationError(
                    f"quote age {age:.1f}s exceeds proposal limit {max_quote_age_seconds}s"
                )


def provenanced_to_action_request(
    provenanced: ProvenancedExecutionCommand,
    *,
    action: OrderActionKind = OrderActionKind.ENTRY,
) -> OrderActionRequest:
    """Adapt a compiled entry/probe ProvenancedExecutionCommand into a gateway request."""
    cmd = provenanced.command
    intent = cmd.intent
    return OrderActionRequest(
        action=action,
        snapshot_id=provenanced.snapshot_id,
        contract_id=contract_id_for(intent.contract),
        side=intent.side,  # type: ignore[arg-type]
        quantity=int(intent.quantity),
        client_order_id=cmd.client_order_id,
        limit_price=intent.limit_price,
        order_type=intent.order_type,  # type: ignore[arg-type]
        max_quote_age_seconds=provenanced.max_quote_age_seconds,
        proposal_id=provenanced.proposal_id,
        decision_id=provenanced.decision_id,
        strategy_id=provenanced.strategy_id,
        cycle_id=provenanced.cycle_id,
        evidence_ids=provenanced.evidence_ids,
        broker_account_id=cmd.broker_account_id,
    )
