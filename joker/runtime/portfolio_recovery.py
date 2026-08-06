"""Deterministic portfolio recovery coordinator shared by graph and runtime."""

from __future__ import annotations

import json
import inspect
from dataclasses import dataclass
from typing import Any

from joker.objectives.decision_fingerprint import ObjectiveDecisionFingerprint
from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
    PortfolioComponentStatus,
    PortfolioExecutionComponentRecord,
    PortfolioExecutionOwner,
    PortfolioReoptimizationRequestRecord,
    stable_reoptimization_request_id,
)
from joker.runtime.execution_runtime import ExecutionRuntime
from joker.runtime.recovery_mode import RecoveryMode
from joker.time.clock import ExchangeClock


@dataclass(frozen=True)
class PortfolioRecoveryResolution:
    request: PortfolioReoptimizationRequestRecord | None
    blocked_reason: str | None = None
    terminalized: bool = False


class PortfolioRecoveryCoordinator:
    """Reusable deterministic continuation coordinator for portfolio work."""

    def __init__(
        self,
        *,
        execution_runtime: ExecutionRuntime,
        provenance_registry: CognitiveExecutionProvenanceRegistry,
        stable_owner: PortfolioExecutionOwner,
        clock: ExchangeClock,
        objective_service: Any | None = None,
        recovery_mode: RecoveryMode = RecoveryMode.NORMAL,
    ) -> None:
        self._execution_runtime = execution_runtime
        self._provenance_registry = provenance_registry
        self._owner = stable_owner
        self._clock = clock
        self._objective_service = objective_service
        self._recovery_mode = recovery_mode

    @property
    def owner(self) -> PortfolioExecutionOwner:
        return self._owner

    @property
    def recovery_mode(self) -> RecoveryMode:
        return self._recovery_mode

    async def projection(self) -> Any | None:
        return await self._execution_runtime.project_session()

    @staticmethod
    def working_client_order_ids(projection: Any | None) -> tuple[str, ...]:
        orders = getattr(projection, "orders", None) or {}
        values = orders.values() if isinstance(orders, dict) else list(orders or [])
        ids: list[str] = []
        for order in values:
            status = str(getattr(order, "status", "") or "").lower()
            if status not in {
                "submitted",
                "accepted",
                "partially_filled",
                "open",
                "pending",
                "working",
            }:
                continue
            client_order_id = str(getattr(order, "client_order_id", "") or "")
            if client_order_id:
                ids.append(client_order_id)
        return tuple(dict.fromkeys(ids))

    async def poll_working_orders(self, projection: Any | None = None) -> Any | None:
        current = projection if projection is not None else await self.projection()
        bridge_poller = getattr(self._execution_runtime, "_bridge_poll_order_status", None)
        for client_order_id in self.working_client_order_ids(current):
            if bridge_poller is None:
                await self._execution_runtime.poll_order_status(client_order_id)
                continue
            result = bridge_poller(client_order_id)
            if inspect.isawaitable(result):
                await result
        return current

    def _current_objective_payload(self, objective: Any | None) -> dict[str, Any]:
        if objective is None:
            return {}
        if hasattr(objective, "as_dict"):
            return dict(objective.as_dict())
        if hasattr(objective, "model_dump"):
            return dict(objective.model_dump(mode="json"))
        if hasattr(objective, "__dict__"):
            return json.loads(json.dumps(vars(objective), default=str))
        return json.loads(json.dumps(objective, default=str))

    def _current_fingerprint(
        self,
        objective_state: Any,
        *,
        working_order_count: int,
    ) -> ObjectiveDecisionFingerprint:
        permission = getattr(self._execution_runtime, "entry_permission", None)
        broker_eligible = not bool(getattr(self._execution_runtime, "kill_switch", False))
        if permission is not None:
            broker_eligible = broker_eligible and bool(getattr(permission, "permitted", False))
        broker = getattr(self._execution_runtime, "_broker", None)
        broker_identity = type(broker).__qualname__ if broker is not None else "unconfigured"
        reconciliation_eligible = self._execution_runtime.unresolved_reconciliation is None
        return ObjectiveDecisionFingerprint.from_state(
            objective_state,
            working_order_count=working_order_count,
            broker_identity=broker_identity,
            broker_eligible=broker_eligible,
            reconciliation_eligible=reconciliation_eligible,
        )

    async def persist_filled_continuation(
        self,
        component: PortfolioExecutionComponentRecord,
        *,
        latest_snapshot_id: str,
        filled_quantity: int | None = None,
        broker_order_id: str | None = None,
    ) -> PortfolioExecutionComponentRecord:
        reconciled_filled_quantity = (
            component.filled_quantity if filled_quantity is None else int(filled_quantity)
        )
        if reconciled_filled_quantity != component.authorized_quantity:
            raise ValueError("filled component quantity is not fully reconciled")
        if self._objective_service is None:
            raise RuntimeError("objective authority unavailable for post-fill continuation")
        if not latest_snapshot_id:
            raise RuntimeError("latest snapshot id is required for post-fill continuation")
        await self._objective_service.recompute_from_truth(now=self._clock.now())
        post_objective = await self._objective_service.get_state()
        post_projection = await self._execution_runtime.project_session()
        post_fingerprint = self._current_fingerprint(
            post_objective,
            working_order_count=len(self.working_client_order_ids(post_projection)),
        )
        exchange_time = self._clock.now().isoformat()
        return await self._provenance_registry.portfolio_executions.transition(
            component.authorized_position_tuple_id,
            owner=self._owner,
            status=PortfolioComponentStatus.FILLED,
            broker_order_id=broker_order_id,
            submitted_quantity=component.authorized_quantity,
            filled_quantity=reconciled_filled_quantity,
            last_reconciliation_timestamp=exchange_time,
            post_fill_objective_version=int(post_objective.version),
            post_fill_objective_fingerprint=post_fingerprint.canonical_json,
            post_fill_snapshot_id=latest_snapshot_id,
            post_fill_exchange_time=exchange_time,
            reconciled_filled_quantity=reconciled_filled_quantity,
            continuation_ready=True,
            extra_update={
                "post_fill_objective_version": int(post_objective.version),
                "post_fill_objective_fingerprint": post_fingerprint.canonical_json,
                "post_fill_snapshot_id": latest_snapshot_id,
                "post_fill_exchange_time": exchange_time,
                "reconciled_filled_quantity": reconciled_filled_quantity,
                "continuation_ready": True,
            },
        )

    async def sync_component_from_projection(
        self,
        component: PortfolioExecutionComponentRecord,
        *,
        projection: Any | None = None,
        latest_snapshot_id: str | None = None,
    ) -> PortfolioExecutionComponentRecord:
        if not self._owner.matches(component):
            raise PermissionError("portfolio component owner does not match runtime")
        order = None
        orders = (
            projection.get("orders")
            if isinstance(projection, dict)
            else getattr(projection, "orders", None)
        ) or {}
        if isinstance(orders, dict):
            order = orders.get(component.client_order_id)
        else:
            order = next(
                (
                    item
                    for item in orders
                    if str(getattr(item, "client_order_id", "")) == component.client_order_id
                ),
                None,
            )
        if order is None:
            return component
        raw_status = (
            order.get("status")
            if isinstance(order, dict)
            else getattr(order, "status", "")
        )
        status_value = str(getattr(raw_status, "value", raw_status) or "").lower()
        filled_quantity = int(
            (order.get("filled_qty") or order.get("filled_quantity") or 0)
            if isinstance(order, dict)
            else getattr(order, "filled_qty", 0) or getattr(order, "filled_quantity", 0)
        )
        status_map = {
            "submitted": PortfolioComponentStatus.SUBMITTED,
            "accepted": PortfolioComponentStatus.WORKING,
            "open": PortfolioComponentStatus.WORKING,
            "pending": PortfolioComponentStatus.WORKING,
            "working": PortfolioComponentStatus.WORKING,
            "partially_filled": PortfolioComponentStatus.PARTIALLY_FILLED,
            "filled": PortfolioComponentStatus.FILLED,
            "rejected": PortfolioComponentStatus.REJECTED,
            "cancelled": PortfolioComponentStatus.CANCELLED,
        }
        mapped_status = status_map.get(status_value)
        terminal_statuses = {
            PortfolioComponentStatus.FILLED,
            PortfolioComponentStatus.REJECTED,
            PortfolioComponentStatus.CANCELLED,
            PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
        }
        if mapped_status is None or component.status in terminal_statuses:
            return component
        if mapped_status == PortfolioComponentStatus.FILLED:
            if latest_snapshot_id is None:
                raise RuntimeError("latest snapshot id required to persist filled continuation")
            return await self.persist_filled_continuation(
                component,
                latest_snapshot_id=latest_snapshot_id,
                filled_quantity=filled_quantity,
                broker_order_id=component.broker_order_id,
            )
        return await self._provenance_registry.portfolio_executions.transition(
            component.authorized_position_tuple_id,
            owner=self._owner,
            status=mapped_status,
            broker_order_id=(
                str(getattr(order, "order_id", "") or getattr(order, "broker_order_id", "") or "")
                if not isinstance(order, dict)
                else str(order.get("order_id") or order.get("broker_order_id") or "")
            )
            or component.broker_order_id,
            submitted_quantity=component.authorized_quantity,
            filled_quantity=filled_quantity,
            last_reconciliation_timestamp=self._clock.now().isoformat(),
            latest_validation_snapshot_id=latest_snapshot_id
            or component.latest_validation_snapshot_id,
            submission_objective_version=component.submission_objective_version,
        )

    async def reconcile_owner_components(
        self,
        *,
        projection: Any | None = None,
        latest_snapshot_id: str | None = None,
        terminal_recovery_reason: str | None = None,
        objective_status: str | None = None,
        origin_run_id: str,
        state: dict[str, Any] | None = None,
    ) -> list[PortfolioExecutionComponentRecord]:
        """Synchronize every owned component and terminalize suffixes when allowed."""
        current_projection = projection if projection is not None else await self.projection()
        if latest_snapshot_id is None and state is not None:
            latest_snapshot_id = str(
                state.get("snapshot_id") or state.get("latest_known_snapshot_id") or ""
            ) or None
        resumable = await self._provenance_registry.portfolio_executions.list_resumable(
            session_id=self._owner.session_id,
            broker_account_identity=self._owner.broker_account_identity,
            trading_date=self._owner.trading_date,
        )
        decision_ids = sorted({record.target_portfolio_decision_id for record in resumable})
        updated: list[PortfolioExecutionComponentRecord] = []
        for decision_id in decision_ids:
            components = await self._provenance_registry.portfolio_executions.list_by_decision(
                decision_id,
                owner=self._owner,
            )
            for component in components:
                synced = await self.sync_component_from_projection(
                    component,
                    projection=current_projection,
                    latest_snapshot_id=latest_snapshot_id,
                )
                if synced != component:
                    updated.append(synced)
            if (
                terminal_recovery_reason
                and objective_status is not None
                and self._objective_service is not None
                and objective_status in {"deadline_reached", "target_reached"}
            ):
                resolution = await self.request_suffix_reoptimization(
                    decision_id=decision_id,
                    reason=terminal_recovery_reason,
                    origin_run_id=origin_run_id,
                    state=state,
                    terminal_recovery=True,
                    latest_snapshot_id=latest_snapshot_id,
                    objective_status=objective_status,
                    source_components=components,
                )
                if resolution.terminalized and resolution.request is not None:
                    stored = await self._provenance_registry.portfolio_reoptimizations.get(
                        resolution.request.request_id
                    )
                    if stored is not None:
                        updated.append(stored)
        return updated

    async def request_suffix_reoptimization(
        self,
        *,
        decision_id: str,
        reason: str,
        origin_run_id: str,
        start_component_index: int = 0,
        authorized_positions: list[dict[str, Any]] | None = None,
        state: dict[str, Any] | None = None,
        terminal_recovery: bool = False,
        latest_snapshot_id: str | None = None,
        objective_status: str | None = None,
        source_components: list[PortfolioExecutionComponentRecord] | None = None,
    ) -> PortfolioRecoveryResolution:
        components = (
            source_components
            if source_components is not None
            else await self._provenance_registry.portfolio_executions.list_by_decision(
                decision_id,
                owner=self._owner,
            )
        )
        if not components:
            return PortfolioRecoveryResolution(request=None)
        suffix_components = [
            component
            for component in components
            if component.component_index >= start_component_index
            and component.status in {
                PortfolioComponentStatus.AUTHORIZED,
                PortfolioComponentStatus.READY,
            }
        ]
        if not suffix_components:
            return PortfolioRecoveryResolution(request=None)
        resolved_at = self._clock.now().isoformat()
        for component in suffix_components:
            await self._provenance_registry.portfolio_executions.transition(
                component.authorized_position_tuple_id,
                owner=self._owner,
                status=PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
                failure_reoptimization_reason=reason,
                last_reconciliation_timestamp=resolved_at,
                extra_update={
                    "recovery_mode": self._recovery_mode.value,
                    "terminal_recovery_reason": reason,
                },
            )
        suffix_authorized_positions = (
            list(authorized_positions)
            if authorized_positions is not None
            else [
                {
                    "position_tuple_id": str(component.authorized_position_tuple_id),
                    "strategy_id": str(component.strategy_id),
                    "contract_id": str(component.contract_id),
                    "quantity": int(component.authorized_quantity),
                    "capital_allocation": str(component.capital_allocation),
                    "snapshot_id": str(component.original_decision_snapshot_id),
                    "objective_version": int(component.evaluated_objective_version),
                }
                for component in suffix_components
            ]
        )
        if latest_snapshot_id is None:
            latest_snapshot_id = str(
                (state or {}).get("snapshot_id") or (state or {}).get("latest_known_snapshot_id") or ""
            )
        if not latest_snapshot_id:
            latest_snapshot_id = max(
                (component.original_decision_snapshot_id for component in suffix_components),
                default="",
            )
        objective = (
            await self._objective_service.get_state()
            if self._objective_service is not None
            else None
        )
        if objective_status is None and objective is not None:
            objective_status = str(getattr(objective, "status", "unknown") or "unknown")
        objective_payload = self._current_objective_payload(objective)
        open_positions: tuple[dict[str, Any], ...] = ()
        projection = await self._execution_runtime.project_session()
        raw_positions = (
            projection.get("positions", {})
            if isinstance(projection, dict)
            else getattr(projection, "positions", {})
            if projection is not None
            else {}
        )
        position_values = raw_positions.values() if isinstance(raw_positions, dict) else raw_positions
        open_positions = tuple(
            json.loads(
                json.dumps(
                    position.model_dump(mode="json")
                    if hasattr(position, "model_dump")
                    else dict(vars(position))
                    if hasattr(position, "__dict__")
                    else position,
                    default=str,
                )
            )
            for position in position_values
        )
        remaining = tuple(component.authorized_position_tuple_id for component in suffix_components)
        if not remaining:
            return PortfolioRecoveryResolution(request=None)
        request_id = stable_reoptimization_request_id(
            session_id=self._owner.session_id,
            broker_account_identity=self._owner.broker_account_identity,
            trading_date=self._owner.trading_date,
            original_portfolio_decision_id=decision_id,
            remaining_authorized_tuple_ids=remaining,
        )
        record = PortfolioReoptimizationRequestRecord(
            request_id=request_id,
            session_id=self._owner.session_id,
            origin_run_id=origin_run_id,
            broker_account_identity=self._owner.broker_account_identity,
            trading_date=self._owner.trading_date,
            original_portfolio_decision_id=decision_id,
            already_filled_tuple_ids=tuple(
                component.authorized_position_tuple_id
                for component in components
                if component.status == PortfolioComponentStatus.FILLED
            ),
            open_positions=open_positions,
            remaining_authorized_tuple_ids=remaining,
            reason_codes=(reason,),
            latest_objective_state=objective_payload,
            latest_objective_version=int(getattr(objective, "version", 0) or 0),
            latest_snapshot_id=latest_snapshot_id,
            created_exchange_time=resolved_at,
            extra={
                "stable_owner": {
                    "session_id": self._owner.session_id,
                    "broker_account_identity": self._owner.broker_account_identity,
                    "trading_date": self._owner.trading_date,
                },
                "origin_run_id": origin_run_id,
                "original_authorized_positions": suffix_authorized_positions,
                "source_cycle_id": (state or {}).get("cycle_id"),
            },
        )
        if terminal_recovery:
            if self._recovery_mode is RecoveryMode.BROKER_ONLY or self._objective_service is None:
                return PortfolioRecoveryResolution(
                    request=None,
                    blocked_reason="objective_authority_missing_for_terminal_recovery",
                    terminalized=False,
                )
            resolved = await self._provenance_registry.portfolio_reoptimizations.resolve_terminal_recovery(
                record,
                resolved_at=resolved_at,
                resolved_by=self._execution_runtime.session_id,
                terminal_recovery_reason=reason,
                objective_status=str(objective_status or "unknown"),
            )
            return PortfolioRecoveryResolution(request=resolved, terminalized=True)
        stored = await self._provenance_registry.portfolio_reoptimizations.enqueue(record)
        return PortfolioRecoveryResolution(request=stored, terminalized=False)

    async def resolve_remaining_suffix(
        self,
        *,
        decision_id: str,
        authorized_positions: list[dict[str, Any]],
        reason: str,
        origin_run_id: str,
        state: dict[str, Any] | None = None,
        terminal_recovery: bool = False,
        latest_snapshot_id: str | None = None,
        objective_status: str | None = None,
    ) -> PortfolioRecoveryResolution:
        return await self.request_suffix_reoptimization(
            decision_id=decision_id,
            reason=reason,
            origin_run_id=origin_run_id,
            authorized_positions=authorized_positions,
            state=state,
            terminal_recovery=terminal_recovery,
            latest_snapshot_id=latest_snapshot_id,
            objective_status=objective_status,
            source_components=None,
        )
