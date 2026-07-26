"""Compile TradingEpisode artefacts from authoritative Task 1/2 persisted truth."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID, uuid4

from joker.evolution.hashing import hash_model
from joker.evolution.idempotency import episode_idempotency_key
from joker.evolution.repositories import (
    DecisionTraceRepository,
    TradingEpisodeRepository,
)
from joker.evolution.schemas import DecisionTraceSummary, TradingEpisode
from joker.ledger.projector import OrderLifecycle, OrderStatus, ProjectionState


class _ExecutionProjection(Protocol):
    async def project_session(self) -> ProjectionState: ...


class EpisodeCompiler:
    """Build immutable episodes from ledger/projection/cognitive provenance."""

    def __init__(
        self,
        episode_repo: TradingEpisodeRepository,
        trace_repo: DecisionTraceRepository | None = None,
    ) -> None:
        self._episodes = episode_repo
        self._traces = trace_repo

    async def compile_from_position_closed(
        self,
        *,
        session_id: str,
        run_id: str,
        trading_date: date,
        configuration_version_id: UUID,
        event_payload: dict[str, Any],
        event_id: str,
        execution: _ExecutionProjection,
        initial_snapshot_id: UUID | None = None,
        cognitive_artifact_ids: tuple[UUID, ...] = (),
        model_call_ids: tuple[UUID, ...] = (),
        data_quality_ids: tuple[UUID, ...] = (),
        option_surface_ids: tuple[UUID, ...] = (),
        source_event_ids: tuple[UUID, ...] = (),
        entry_cycle_id: str | None = None,
        proposal_id: UUID | None = None,
        decision_id: UUID | None = None,
        market_regime_tags: tuple[str, ...] = (),
    ) -> TradingEpisode:
        """Derive a closed-trade episode from POSITION_CLOSED + live projection."""
        projection = await execution.project_session()
        contract_id = str(event_payload.get("contract_id") or "")
        client_order_id = str(event_payload.get("client_order_id") or "")
        findings: list[str] = []
        completed = True

        if not contract_id:
            findings.append("missing_contract_id_in_event")
            completed = False

        position = projection.positions.get(contract_id)
        if position is not None and position.open:
            findings.append("position_still_open_in_projection")
            completed = False

        entry_orders, exit_orders = self._partition_orders(
            projection, contract_id=contract_id
        )
        if not entry_orders:
            findings.append("missing_entry_orders_in_projection")
            completed = False
        if not exit_orders and client_order_id:
            # Closing order may be the event's client_order_id.
            closer = projection.orders.get(client_order_id)
            if closer is not None:
                exit_orders = (closer,)

        entry_qty = sum((o.filled_qty for o in entry_orders), Decimal("0"))
        exit_qty = sum((o.filled_qty for o in exit_orders), Decimal("0"))
        remaining = entry_qty - exit_qty
        if remaining != 0:
            findings.append("quantity_identity_mismatch")
            completed = False

        entry_price = self._vwap(entry_orders)
        exit_price = self._vwap(exit_orders)
        realised = None
        if "realized_pnl" in event_payload and event_payload["realized_pnl"] is not None:
            realised = Decimal(str(event_payload["realized_pnl"]))
        elif position is not None:
            realised = position.realized_pnl
        else:
            findings.append("missing_realised_pnl")
            completed = False

        if initial_snapshot_id is None:
            findings.append("missing_initial_snapshot")
            completed = False
            initial_snapshot_id = uuid4()

        fees = Decimal("0")
        for order in (*entry_orders, *exit_orders):
            fees += order.fees

        lifecycle = f"{contract_id}:{entry_orders[0].client_order_id if entry_orders else client_order_id}"
        key = episode_idempotency_key(session_id, lifecycle, event_id)
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            entry_cycle_id=entry_cycle_id,
            proposal_id=proposal_id,
            decision_id=decision_id,
            initial_snapshot_id=initial_snapshot_id,
            terminal_snapshot_id=initial_snapshot_id,
            contract_id=contract_id or None,
            direction="none",
            action_class="closed_trade",
            entry_order_ids=tuple(o.client_order_id for o in entry_orders),
            exit_order_ids=tuple(o.client_order_id for o in exit_orders),
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=entry_qty,
            realised_pnl=realised,
            total_fees=fees,
            market_regime_tags=market_regime_tags,
            data_quality_ids=data_quality_ids,
            option_surface_ids=option_surface_ids,
            source_event_ids=source_event_ids or (UUID(event_id),)
            if _is_uuid(event_id)
            else source_event_ids,
            cognitive_artifact_ids=cognitive_artifact_ids,
            model_call_ids=model_call_ids,
            configuration_version_id=configuration_version_id,
            completed=completed,
            completeness_findings=tuple(findings),
            idempotency_key=key,
        )
        await self._episodes.append(episode)
        return episode

    async def compile_from_order_rejected_or_cancelled(
        self,
        *,
        session_id: str,
        run_id: str,
        trading_date: date,
        configuration_version_id: UUID,
        event_payload: dict[str, Any],
        event_id: str,
        action_class: str,
        execution: _ExecutionProjection,
        initial_snapshot_id: UUID | None = None,
        rejection_codes: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> TradingEpisode:
        projection = await execution.project_session()
        client_order_id = str(event_payload.get("client_order_id") or "")
        order = projection.orders.get(client_order_id)
        findings: list[str] = list(rejection_codes)
        if order is None:
            findings.append("order_missing_from_projection")
        elif action_class == "entry_rejected" and order.status is not OrderStatus.REJECTED:
            findings.append("projection_status_not_rejected")
        elif action_class == "entry_cancelled" and order.status is not OrderStatus.CANCELLED:
            findings.append("projection_status_not_cancelled")

        if initial_snapshot_id is None:
            findings.append("missing_initial_snapshot")
            initial_snapshot_id = uuid4()

        lifecycle = f"{action_class}:{client_order_id or event_id}"
        key = episode_idempotency_key(session_id, lifecycle, event_id)
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            initial_snapshot_id=initial_snapshot_id,
            action_class=action_class,  # type: ignore[arg-type]
            entry_order_ids=(client_order_id,) if client_order_id else (),
            quantity=Decimal("0"),
            configuration_version_id=configuration_version_id,
            completed=True,
            completeness_findings=tuple(findings),
            idempotency_key=key,
            contract_id=order.contract_id if order else None,
        )
        await self._episodes.append(episode)
        return episode

    async def compile_from_no_trade_cycle(
        self,
        *,
        session_id: str,
        run_id: str,
        trading_date: date,
        configuration_version_id: UUID,
        cycle_id: str,
        snapshot_id: UUID,
        event_id: str,
        outcome: str,
        decision_rationale: str = "",
        rejection_codes: tuple[str, ...] = (),
        confidence_values: dict[str, Decimal] | None = None,
        cognitive_artifact_ids: tuple[UUID, ...] = (),
        model_call_ids: tuple[UUID, ...] = (),
        data_quality_ids: tuple[UUID, ...] = (),
        option_surface_ids: tuple[UUID, ...] = (),
    ) -> TradingEpisode:
        findings: list[str] = []
        if not snapshot_id:
            findings.append("missing_snapshot")
        key = episode_idempotency_key(session_id, f"no_trade:{cycle_id}", event_id)
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            entry_cycle_id=cycle_id,
            initial_snapshot_id=snapshot_id,
            action_class="no_trade",
            quantity=Decimal("0"),
            configuration_version_id=configuration_version_id,
            cognitive_artifact_ids=cognitive_artifact_ids,
            model_call_ids=model_call_ids,
            data_quality_ids=data_quality_ids,
            option_surface_ids=option_surface_ids,
            completed=True,
            completeness_findings=tuple(findings),
            idempotency_key=key,
        )
        await self._episodes.append(episode)
        if self._traces is not None:
            summary = DecisionTraceSummary(
                episode_id=episode.episode_id,
                typed_conclusions=(outcome or "no_trade",),
                rejection_codes=rejection_codes,
                decision_rationale=decision_rationale,
                confidence_values=confidence_values or {},
                content_hash="",
            )
            summary = summary.model_copy(
                update={"content_hash": hash_model(summary, exclude={"created_at"})}
            )
            await self._traces.append(summary)
        return episode

    @staticmethod
    def _partition_orders(
        projection: ProjectionState, *, contract_id: str
    ) -> tuple[tuple[OrderLifecycle, ...], tuple[OrderLifecycle, ...]]:
        entries: list[OrderLifecycle] = []
        exits: list[OrderLifecycle] = []
        for order in projection.orders.values():
            if contract_id and order.contract_id != contract_id:
                continue
            if order.status not in {
                OrderStatus.FILLED,
                OrderStatus.PARTIALLY_FILLED,
            }:
                continue
            if order.side == "buy":
                entries.append(order)
            else:
                exits.append(order)
        return tuple(entries), tuple(exits)

    @staticmethod
    def _vwap(orders: tuple[OrderLifecycle, ...]) -> Decimal | None:
        qty = Decimal("0")
        notional = Decimal("0")
        for order in orders:
            if order.avg_fill_price is None or order.filled_qty <= 0:
                continue
            qty += order.filled_qty
            notional += order.avg_fill_price * order.filled_qty
        if qty <= 0:
            return None
        return (notional / qty).quantize(Decimal("0.0001"))


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except Exception:
        return False
