"""Task-1 objective capital projector — verified fill/close lifecycle ownership.

Financial mutations must not depend on CognitiveAgentRuntime. This projector
subscribes to Task 1 domain events (or is invoked directly from
ExecutionRuntime) and applies idempotent exposure transitions.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from joker.events.schemas import DomainEvent, EventType
from joker.objectives.service import ObjectiveServiceError, SessionObjectiveService

logger = logging.getLogger(__name__)


class ObjectiveCapitalProjector:
    """Project verified broker/ledger lifecycle into SessionObjectiveService."""

    def __init__(self, service: SessionObjectiveService) -> None:
        self._service = service

    @property
    def service(self) -> SessionObjectiveService:
        return self._service

    async def handle_domain_event(self, event: DomainEvent) -> None:
        try:
            await self._handle(event)
        except ObjectiveServiceError as exc:
            # Fail closed: mark truth degraded and block new entries.
            self._service.mark_truth_degraded(
                True, reason=f"projection_failed:{event.event_type.value}:{exc}"
            )
            raise
        except Exception as exc:
            self._service.mark_truth_degraded(
                True, reason=f"projection_unexpected:{event.event_type.value}:{exc}"
            )
            raise

    async def _handle(self, event: DomainEvent) -> None:
        payload = dict(event.payload or {})
        client_order_id = str(payload.get("client_order_id") or "")
        dedupe = self._dedupe_key(event, payload)

        if event.event_type == EventType.ORDER_REJECTED:
            if not client_order_id:
                return
            await self._service.release_for_order(
                client_order_id=client_order_id,
                reason="rejected",
                dedupe_key=dedupe,
            )
            return

        if event.event_type == EventType.ORDER_CANCELLED:
            if not client_order_id:
                return
            await self._service.release_for_order(
                client_order_id=client_order_id,
                reason="cancelled",
                dedupe_key=dedupe,
            )
            return

        if event.event_type in {
            EventType.ORDER_FILLED,
            EventType.ORDER_PARTIALLY_FILLED,
        }:
            if not client_order_id:
                return
            qty = int(payload.get("qty") or payload.get("fill_qty") or 0)
            price = payload.get("price") or payload.get("fill_price")
            if qty <= 0 or price is None:
                raise ObjectiveServiceError(
                    "verified fill missing qty/price for objective projection"
                )
            remaining_qty = payload.get("remaining_quantity")
            if remaining_qty is None and event.event_type == EventType.ORDER_FILLED:
                remaining_qty = 0
            await self._service.apply_verified_fill(
                client_order_id=client_order_id,
                fill_quantity=qty,
                fill_price=Decimal(str(price)),
                remaining_working_quantity=(
                    int(remaining_qty) if remaining_qty is not None else None
                ),
                dedupe_key=dedupe,
                contract_id=str(payload.get("contract_id") or "") or None,
                open_position_count=(
                    int(payload["open_position_count"])
                    if payload.get("open_position_count") is not None
                    else None
                ),
            )
            return

        if event.event_type == EventType.POSITION_CLOSED:
            pnl = payload.get("realized_pnl") or payload.get("realised_pnl_usd") or 0
            closed_qty = int(payload.get("closed_quantity") or payload.get("qty") or 0)
            await self._service.reduce_position_exposure(
                client_order_id=client_order_id or None,
                contract_id=str(payload.get("contract_id") or "") or None,
                closed_quantity=closed_qty,
                realised_pnl_delta_usd=pnl,
                dedupe_key=dedupe,
                open_position_count=(
                    int(payload["open_position_count"])
                    if payload.get("open_position_count") is not None
                    else 0
                ),
                final_close=True,
            )
            return

        if event.event_type == EventType.POSITION_CHANGED:
            # Partial reduction when quantity decreases with realised PnL.
            if payload.get("reduction") or payload.get("closed_quantity"):
                closed_qty = int(payload.get("closed_quantity") or 0)
                pnl = payload.get("realized_pnl") or payload.get("realised_pnl_usd") or 0
                await self._service.reduce_position_exposure(
                    client_order_id=client_order_id or None,
                    contract_id=str(payload.get("contract_id") or "") or None,
                    closed_quantity=closed_qty,
                    realised_pnl_delta_usd=pnl,
                    dedupe_key=dedupe,
                    open_position_count=(
                        int(payload["open_position_count"])
                        if payload.get("open_position_count") is not None
                        else None
                    ),
                    final_close=False,
                )

    @staticmethod
    def _dedupe_key(event: DomainEvent, payload: dict[str, Any]) -> str:
        ledger_id = payload.get("ledger_event_id") or payload.get("idempotency_key")
        if ledger_id:
            return f"ledger:{ledger_id}"
        fill_id = payload.get("fill_id")
        if fill_id:
            return f"fill_id:{fill_id}"
        return (
            f"domain:{event.event_type.value}:{event.event_id}:"
            f"{payload.get('client_order_id')}:{payload.get('qty')}:"
            f"{payload.get('price')}"
        )


def subscribe_objective_projector(
    event_bus: Any,
    projector: ObjectiveCapitalProjector,
) -> None:
    """Subscribe Task 1 event bus handlers for objective capital projection."""
    for event_type in (
        EventType.ORDER_FILLED,
        EventType.ORDER_PARTIALLY_FILLED,
        EventType.ORDER_CANCELLED,
        EventType.ORDER_REJECTED,
        EventType.POSITION_CLOSED,
        EventType.POSITION_CHANGED,
    ):
        event_bus.subscribe(event_type, projector.handle_domain_event)
