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
    estimate_id: str | None = None
    cycle_id: str | None = None
    replace_of_client_order_id: str | None = None
    position_lifecycle_id: str | None = None
    originating_entry_client_order_id: str | None = None
    allow_degraded_exit: bool = True
    degraded_exit_reason: str | None = None
    evidence_ids: tuple[str, ...] = ()
    broker_account_id: str = "default"
    # Factual horizon-start anchor for entry provenance (never a fill event).
    # Prefer decision-completed → execution-proposal → cognitive-cycle trigger.
    causation_event_id: str | None = None


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

        # Authoritative entry-permission gate — EXIT/REDUCE remain available.
        entry_actions = {
            OrderActionKind.ENTRY,
            OrderActionKind.PROBE,
            OrderActionKind.ADD,
            OrderActionKind.REPLACE,
        }
        perm = getattr(self._deps, "entry_permission", None)
        if (
            request.action in entry_actions
            and perm is not None
            and not bool(getattr(perm, "permitted", True))
        ):
            reasons = getattr(perm, "reasons", ()) or ()
            return OrderActionResult(
                submitted=False,
                client_order_id=request.client_order_id,
                blocked_reason=(
                    "entry_permission_blocked: " + (",".join(reasons) or "blocked")
                ),
                working_orders={},
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
                preview_block = await self._maybe_live_preview(command, request)
                if preview_block is not None:
                    return preview_block
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
                            causation_event_id=request.causation_event_id,
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

        # Kill switch is stronger than objective approval — block new risk only.
        # Checked before REPLACE cancel so we never tear down a working order first.
        if request.action in {
            OrderActionKind.ENTRY,
            OrderActionKind.PROBE,
            OrderActionKind.ADD,
            OrderActionKind.REPLACE,
        } and bool(getattr(self._deps, "kill_switch", False)):
            return OrderActionResult(
                submitted=False,
                client_order_id=request.client_order_id,
                blocked_reason="KILL_SWITCH",
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
                if (
                    obj_state.status
                    in {
                        "deadline_reached",
                        "truth_degraded",
                        "insufficient_historical_evidence",
                        "pending_confirmation",
                    }
                    or obj_state.entries_paused
                    or obj_state.feasibility_classification == "infeasible"
                    or obj_state.time_remaining_seconds <= 0
                    or getattr(obj_state, "truth_degraded", False)
                ):
                    return OrderActionResult(
                        submitted=False,
                        client_order_id=request.client_order_id,
                        blocked_reason=(
                            f"objective_gate:{obj_state.status}:"
                            f"paused={obj_state.entries_paused}:"
                            f"feasibility={obj_state.feasibility_classification}"
                        ),
                        working_orders=working,
                    )
                # Reload persisted estimate / historical summary; reprice vs current quote.
                svc = self._deps.objective_service
                estimate = None
                if request.estimate_id:
                    estimate = svc.get_strategy_estimate(request.estimate_id)
                elif request.strategy_id:
                    estimate = svc.get_latest_estimate_for_strategy(
                        strategy_id=request.strategy_id,
                        objective_id=obj_state.objective_id,
                    )
                if estimate is None or not estimate.valid:
                    objective_policy_pre = str(
                        getattr(svc, "objective_policy", "positive_ev_baseline")
                    )
                    if estimate is None or objective_policy_pre != "target_attainment":
                        return OrderActionResult(
                            submitted=False,
                            client_order_id=request.client_order_id,
                            blocked_reason="objective_estimate_missing_or_invalid",
                            working_orders=working,
                        )
                if str(estimate.objective_id) != str(obj_state.objective_id):
                    return OrderActionResult(
                        submitted=False,
                        client_order_id=request.client_order_id,
                        blocked_reason="objective_estimate_wrong_objective",
                        working_orders=working,
                    )
                if request.strategy_id and str(estimate.strategy_id) != str(
                    request.strategy_id
                ):
                    return OrderActionResult(
                        submitted=False,
                        client_order_id=request.client_order_id,
                        blocked_reason="objective_estimate_wrong_strategy",
                        working_orders=working,
                    )
                if estimate.historical_summary_id is not None:
                    summary = svc.get_historical_summary(estimate.historical_summary_id)
                    objective_policy = str(
                        getattr(svc, "objective_policy", "positive_ev_baseline")
                    )
                    # Target-attainment may proceed on ordinal/low-sample evidence;
                    # baseline still requires valid_for_ev historical summaries.
                    if (
                        objective_policy != "target_attainment"
                        and (summary is None or not summary.valid_for_ev)
                    ):
                        return OrderActionResult(
                            submitted=False,
                            client_order_id=request.client_order_id,
                            blocked_reason="objective_historical_summary_invalid",
                            working_orders=working,
                        )
                # Authoritative Task-1 quote — never treat proposed limit as quote truth.
                quote_loader = getattr(
                    self._deps, "current_option_quote_loader", None
                )
                if quote_loader is None:
                    return OrderActionResult(
                        submitted=False,
                        client_order_id=request.client_order_id,
                        blocked_reason="current_option_quote_loader_missing",
                        working_orders=working,
                    )
                current_quote = await quote_loader(str(request.contract_id))
                if current_quote is None:
                    return OrderActionResult(
                        submitted=False,
                        client_order_id=request.client_order_id,
                        blocked_reason="current_quote_missing",
                        working_orders=working,
                    )
                if not current_quote.usable_for_execution:
                    return OrderActionResult(
                        submitted=False,
                        client_order_id=request.client_order_id,
                        blocked_reason=(
                            "current_quote_unusable:"
                            + ",".join(current_quote.invalidation_reasons)
                        ),
                        working_orders=working,
                    )
                from dataclasses import replace as dc_replace

                from joker.objectives.execution_quote import (
                    execution_premium_from_quote,
                )
                from joker.objectives.repricing import reprice_long_option_estimate

                ask_premium = execution_premium_from_quote(current_quote)
                qty = int(command.intent.quantity)
                require_ev = bool(
                    getattr(svc, "require_positive_expected_value", True)
                )
                objective_policy = str(
                    getattr(svc, "objective_policy", "positive_ev_baseline")
                )
                if objective_policy == "target_attainment":
                    require_ev = False
                hist_settings = getattr(
                    self._deps, "historical_outcome_settings", None
                )
                max_change = Decimal("25")
                if hist_settings is not None:
                    max_change = Decimal(
                        str(hist_settings.max_premium_change_pct_for_repricing)
                    )
                max_age = int(getattr(self._deps, "max_quote_age_seconds", 30) or 30)
                max_spread = float(
                    getattr(self._deps, "max_relative_spread", 0.25) or 0.25
                )
                max_limit_above_ask_pct = Decimal("5.0")
                exec_settings = getattr(self._deps, "objective_execution_settings", None)
                if exec_settings is not None:
                    max_limit_above_ask_pct = Decimal(
                        str(exec_settings.maximum_buy_limit_above_ask_pct)
                    )
                # Snapshot mismatch forces explicit revalidation (never a silent pass).
                if str(estimate.snapshot_id) != str(request.snapshot_id):
                    if estimate.expected_value_usd is None:
                        return OrderActionResult(
                            submitted=False,
                            client_order_id=request.client_order_id,
                            blocked_reason="objective_estimate_snapshot_mismatch",
                            working_orders=working,
                        )
                # Reconcile proposed limit with authoritative ask for long buys.
                proposed_limit = command.intent.limit_price
                side = str(command.intent.side or "").lower()
                worst_case = ask_premium
                if side == "buy" and proposed_limit is not None:
                    limit_dec = Decimal(str(proposed_limit))
                    ceiling = (
                        ask_premium
                        * (Decimal("100") + max_limit_above_ask_pct)
                        / Decimal("100")
                    ).quantize(Decimal("0.0001"))
                    if limit_dec > ceiling:
                        return OrderActionResult(
                            submitted=False,
                            client_order_id=request.client_order_id,
                            blocked_reason=(
                                "buy_limit_exceeds_max_displacement_above_ask:"
                                f"limit={limit_dec}:ask={ask_premium}:"
                                f"ceiling={ceiling}"
                            ),
                            working_orders=working,
                        )
                    worst_case = max(ask_premium, limit_dec)
                    # Submit the same validated price used for EV and reservation.
                    command = dc_replace(
                        command,
                        intent=command.intent.model_copy(
                            update={"limit_price": float(worst_case)}
                        ),
                    )
                elif side == "buy":
                    worst_case = ask_premium

                premium_per = worst_case
                repriced = reprice_long_option_estimate(
                    estimate,
                    current_premium_per_contract_usd=premium_per,
                    quantity=qty,
                    request_snapshot_id=current_quote.snapshot_id,
                    quote_timestamp=current_quote.quote_timestamp,
                    max_premium_change_pct=max_change,
                    max_quote_age_seconds=max_age,
                    quote_age_seconds=current_quote.quote_age_seconds,
                    max_spread_pct=max_spread * 100.0,
                    current_spread_pct=float(current_quote.relative_spread) * 100.0,
                )
                ev = repriced.repriced_expected_value_usd
                if require_ev and (ev is None or ev <= 0 or not repriced.valid):
                    return OrderActionResult(
                        submitted=False,
                        client_order_id=request.client_order_id,
                        blocked_reason=(
                            "objective_repriced_ev_not_positive:"
                            + ",".join(repriced.invalidation_reasons)
                        ),
                        working_orders=working,
                    )
                # ADD must use incremental capital only (quantity on the command).
                if request.action == OrderActionKind.ADD and require_ev:
                    if ev is None or ev <= 0:
                        return OrderActionResult(
                            submitted=False,
                            client_order_id=request.client_order_id,
                            blocked_reason="objective_incremental_add_ev_not_positive",
                            working_orders=working,
                        )
                premium = premium_per * Decimal("100") * Decimal(qty)
                await svc.reserve_for_order(
                    client_order_id=command.client_order_id,
                    premium_per_contract_usd=premium_per,
                    quantity=qty,
                    estimated_premium_usd=premium,
                    objective_state_version=obj_state.version,
                    contract_id=request.contract_id,
                    position_lifecycle_id=request.position_lifecycle_id,
                )
                reserved = True
            except Exception as exc:
                return OrderActionResult(
                    submitted=False,
                    client_order_id=request.client_order_id,
                    blocked_reason=f"objective_reserve_failed: {exc}",
                    working_orders=working,
                )

        # Live broker preview — same validated limit/quantity as reservation & placement.
        preview_block = await self._maybe_live_preview(command, request)
        if preview_block is not None:
            if reserved and self._deps.objective_service is not None:
                await self._deps.objective_service.release_for_order(
                    client_order_id=command.client_order_id,
                    reason="broker_preview_rejected",
                )
            return preview_block

        try:
            order = await self._deps.execution_runtime.submit_execution_command(command)
        except Exception as exc:
            from joker.broker.interface import BrokerSubmissionUnknown

            if reserved and self._deps.objective_service is not None:
                if isinstance(exc, BrokerSubmissionUnknown):
                    logger.error(
                        "submission_unknown_retaining_reservation",
                        extra={"client_order_id": command.client_order_id},
                    )
                else:
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
                    causation_event_id=request.causation_event_id,
                    extra={
                        "replace_of": request.replace_of_client_order_id,
                        "position_lifecycle_id": lifecycle_id,
                        "originating_entry_client_order_id": originating,
                        "causation_event_id": request.causation_event_id,
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

        position_intent = _resolve_gateway_position_intent(
            action=action,
            side=request.side,
            contract_id=request.contract_id,
            open_positions=open_positions,
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
            position_intent=position_intent,
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
            position_intent="SELL_TO_CLOSE",
        )
        logger.info(
            "compiled_degraded_exit",
            extra={
                "client_order_id": request.client_order_id,
                "reason": degraded_reason,
                "quantity": qty,
                "position_intent": "SELL_TO_CLOSE",
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

    async def _maybe_live_preview(
        self,
        command: ExecutionCommand,
        request: OrderActionRequest,
    ) -> OrderActionResult | None:
        """Journal prepare → live preview → previewed. Sole journal owner."""
        from joker.broker.webull_live import WebullLiveClient
        from joker.persistence.broker_submission_journal import (
            BrokerSubmissionRecord,
            DuplicateSubmissionError,
            payload_hash,
        )

        broker = getattr(self._deps.execution_runtime, "_broker", None)
        if not isinstance(broker, WebullLiveClient):
            return None

        journal = broker.journal
        if journal is None and not getattr(broker, "_capture_only", False):
            return OrderActionResult(
                submitted=False,
                client_order_id=command.client_order_id,
                blocked_reason="live_journal_required",
            )

        positions = broker.list_positions()
        payload = broker.build_payload(
            command.intent,
            client_order_id=command.client_order_id,
            open_positions=positions,
        )
        if journal is not None:
            contract_id = (
                f"{command.intent.contract.symbol}:"
                f"{command.intent.contract.expiration.isoformat()}:"
                f"{command.intent.contract.strike}:"
                f"{command.intent.contract.option_type}"
            )
            try:
                journal.prepare(
                    BrokerSubmissionRecord(
                        client_order_id=command.client_order_id,
                        broker_mode="webull_live",
                        account_id_hash=broker.account_id_hash,
                        status="prepared",
                        session_id=self._deps.session_id,
                        cycle_id=request.cycle_id,
                        proposal_id=request.proposal_id,
                        decision_id=request.decision_id,
                        strategy_id=request.strategy_id,
                        position_lifecycle_id=request.position_lifecycle_id,
                        contract_id=contract_id,
                        side=command.intent.side,
                        position_intent=command.intent.position_intent,
                        quantity=command.intent.quantity,
                        limit_price=(
                            f"{command.intent.limit_price:.2f}"
                            if command.intent.limit_price is not None
                            else None
                        ),
                        payload_hash=payload_hash(payload),
                    )
                )
            except DuplicateSubmissionError as exc:
                return OrderActionResult(
                    submitted=False,
                    client_order_id=command.client_order_id,
                    blocked_reason=f"duplicate_submission: {exc}",
                )

        expected = None
        if command.intent.limit_price is not None:
            expected = (
                Decimal(str(command.intent.limit_price))
                * Decimal("100")
                * Decimal(command.intent.quantity)
            )
        try:
            preview = broker.preview_order(
                command.intent,
                client_order_id=command.client_order_id,
                open_positions=positions,
                expected_notional_usd=expected,
            )
        except Exception as exc:
            return OrderActionResult(
                submitted=False,
                client_order_id=command.client_order_id,
                blocked_reason=f"broker_preview_failed: {exc}",
            )

        if journal is not None:
            try:
                journal.transition(
                    account_id_hash=broker.account_id_hash,
                    client_order_id=command.client_order_id,
                    status="previewed",
                    preview_hash=preview.raw_response_hash,
                    extra_update={
                        "preview_accepted": preview.accepted,
                        "estimated_cost_usd": (
                            str(preview.estimated_cost_usd)
                            if preview.estimated_cost_usd is not None
                            else None
                        ),
                        "estimated_fees_usd": (
                            str(preview.estimated_fees_usd)
                            if preview.estimated_fees_usd is not None
                            else None
                        ),
                    },
                )
            except KeyError as exc:
                raise RuntimeError(
                    "missing journal row during live preview transition — fail closed"
                ) from exc

        if not preview.accepted:
            if journal is not None:
                journal.transition(
                    account_id_hash=broker.account_id_hash,
                    client_order_id=command.client_order_id,
                    status="rejected",
                    last_error_code=preview.rejection_code or "preview_rejected",
                )
            return OrderActionResult(
                submitted=False,
                client_order_id=command.client_order_id,
                blocked_reason=(
                    f"broker_preview_rejected: "
                    f"{preview.rejection_code or 'rejected'}"
                ),
            )
        return None


def _resolve_gateway_position_intent(
    *,
    action: OrderActionKind,
    side: str,
    contract_id: str,
    open_positions: dict[str, Any],
) -> str | None:
    from joker.broker.position_intent import resolve_position_intent
    from joker.schemas.domain import Position

    positions: list[Position] = []
    for cid, pos in (open_positions or {}).items():
        qty = int(
            getattr(pos, "quantity", None)
            or (pos.get("quantity") if isinstance(pos, dict) else 0)
            or 0
        )
        if qty <= 0:
            continue
        # Minimal stand-in — resolve_position_intent matches on contract_id helper.
        try:
            parsed = parse_contract_id(cid)
            positions.append(
                Position(
                    contract=OptionContract(
                        symbol=parsed.symbol,
                        expiration=parsed.expiration,
                        strike=parsed.strike,
                        option_type=parsed.option_type,
                        is_0dte=True,
                    ),
                    quantity=qty,
                    avg_entry_price=float(
                        getattr(pos, "avg_entry_price", None)
                        or (pos.get("avg_entry_price") if isinstance(pos, dict) else 0)
                        or 0
                    ),
                )
            )
        except Exception:
            continue
    action_name = "entry" if action in {
        OrderActionKind.ENTRY,
        OrderActionKind.PROBE,
        OrderActionKind.ADD,
    } else "exit"
    try:
        return resolve_position_intent(
            action=action_name,
            side=side,  # type: ignore[arg-type]
            contract_id=contract_id,
            open_positions=positions,
        )
    except Exception:
        # Paper brokers may not require intent; live client validates again.
        if action_name == "entry" and side == "buy":
            return "BUY_TO_OPEN"
        if action_name == "exit" and side == "sell":
            return "SELL_TO_CLOSE"
        return None


def provenanced_to_action_request(
    provenanced: ProvenancedExecutionCommand,
    *,
    action: OrderActionKind = OrderActionKind.ENTRY,
    causation_event_id: str | None = None,
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
        estimate_id=getattr(provenanced, "estimate_id", None),
        cycle_id=provenanced.cycle_id,
        evidence_ids=provenanced.evidence_ids,
        broker_account_id=cmd.broker_account_id,
        causation_event_id=causation_event_id
        or getattr(provenanced, "causation_event_id", None),
    )
