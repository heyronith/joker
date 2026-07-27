"""Compile TradingEpisode artefacts from authoritative Task 1/2 persisted truth."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID, uuid4

from joker.evolution.hashing import hash_model
from joker.evolution.idempotency import episode_idempotency_key
from joker.evolution.lifecycle import PositionLifecycleResolver
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
        *,
        lifecycle_resolver: PositionLifecycleResolver | None = None,
        provenance: Any | None = None,
        cycle_registry: Any | None = None,
    ) -> None:
        self._episodes = episode_repo
        self._traces = trace_repo
        self._lifecycle = lifecycle_resolver or PositionLifecycleResolver(
            provenance=provenance
        )
        self._provenance = provenance
        self._cycle_registry = cycle_registry

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
        terminal_snapshot_id: UUID | None = None,
        cognitive_artifact_ids: tuple[UUID, ...] = (),
        model_call_ids: tuple[UUID, ...] = (),
        data_quality_ids: tuple[UUID, ...] = (),
        option_surface_ids: tuple[UUID, ...] = (),
        source_event_ids: tuple[UUID, ...] = (),
        entry_cycle_id: str | None = None,
        proposal_id: UUID | None = None,
        decision_id: UUID | None = None,
        market_regime_tags: tuple[str, ...] = (),
        entry_decision_timestamp: Any | None = None,
        terminal_event_timestamp: Any | None = None,
    ) -> TradingEpisode:
        """Derive a closed-trade episode from POSITION_CLOSED + lifecycle resolution."""
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

        resolved = await self._lifecycle.resolve_closed_lifecycle(
            session_id=session_id,
            terminal_event_id=event_id,
            contract_id=contract_id,
            client_order_id=client_order_id or None,
            projection=projection,
            configuration_version_id=configuration_version_id,
            known_entry_cycle_id=entry_cycle_id,
            known_snapshot_id=initial_snapshot_id,
            known_terminal_snapshot_id=terminal_snapshot_id,
            position_lifecycle_id=str(event_payload.get("position_lifecycle_id") or "")
            or None,
            entry_decision_timestamp=entry_decision_timestamp,
            terminal_event_timestamp=terminal_event_timestamp,
        )
        findings.extend(resolved.findings)
        if resolved.legacy_inferred:
            findings.append("legacy_lifecycle_inference")
            completed = False
        entry_orders = resolved.entry_orders
        exit_orders = resolved.exit_orders
        reduction_orders = resolved.reduction_orders
        if not entry_orders:
            completed = False
        if not exit_orders and client_order_id:
            closer = projection.orders.get(client_order_id)
            if closer is not None:
                exit_orders = (closer,)

        entry_qty = resolved.quantity or sum(
            (o.filled_qty for o in entry_orders), Decimal("0")
        )
        # Quantity identity includes reductions + terminal exit.
        exit_qty = sum(
            (o.filled_qty for o in (*reduction_orders, *exit_orders)), Decimal("0")
        )
        if entry_qty != exit_qty:
            findings.append("quantity_identity_mismatch")
            completed = False

        entry_price = self._vwap(entry_orders)
        exit_price = self._vwap(tuple((*reduction_orders, *exit_orders)))
        realised = resolved.realised_pnl
        # Cross-check against lifecycle-specific projection when available;
        # never against cumulative multi-lifecycle contract P&L.
        lifecycle_projection_pnl = None
        if position is not None and getattr(position, "position_lifecycle_id", None):
            if position.position_lifecycle_id == resolved.position_lifecycle_id:
                lifecycle_projection_pnl = position.realized_pnl
        if realised is None:
            findings.append("missing_realised_pnl")
            completed = False
        elif (
            lifecycle_projection_pnl is not None
            and abs(realised - lifecycle_projection_pnl) > Decimal("0.01")
        ):
            findings.append("lifecycle_pnl_mismatch")
            completed = False

        snap = resolved.initial_snapshot_id
        terminal_snap = resolved.terminal_snapshot_id
        snapshot_status: str = "verified"
        if snap is None:
            findings.append("missing_initial_snapshot")
            completed = False
            snapshot_status = "missing"
        if terminal_snap is None:
            findings.append("missing_terminal_snapshot")
            completed = False
            if snapshot_status != "missing":
                snapshot_status = "missing"
        elif snap is not None and terminal_snap == snap:
            # One-frame horizon is only acceptable when explicitly single-snapshot;
            # still mark for replay expansion via event sequence.
            findings.append("terminal_equals_initial_snapshot")

        fees = resolved.total_fees
        if fees == 0:
            for order in (*entry_orders, *reduction_orders, *exit_orders):
                fees += order.fees

        lifecycle = resolved.position_lifecycle_id
        key = episode_idempotency_key(session_id, lifecycle, event_id)
        terminal_event_uuid = UUID(event_id) if _is_uuid(event_id) else None
        entry_event_id = None
        if resolved.fill_event_ids:
            entry_event_id = resolved.fill_event_ids[0]
        market_ids = source_event_ids
        if not market_ids and terminal_event_uuid is not None:
            ordered: list[UUID] = []
            if entry_event_id is not None:
                ordered.append(entry_event_id)
            for eid in resolved.fill_event_ids:
                if eid not in ordered:
                    ordered.append(eid)
            for eid in resolved.position_event_ids:
                if eid not in ordered:
                    ordered.append(eid)
            if terminal_event_uuid not in ordered:
                ordered.append(terminal_event_uuid)
            market_ids = tuple(ordered)
        entry_ts = resolved.entry_decision_timestamp or entry_decision_timestamp
        terminal_ts = resolved.terminal_event_timestamp or terminal_event_timestamp
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            entry_cycle_id=resolved.entry_cycle_id or entry_cycle_id,
            proposal_id=resolved.proposal_id or proposal_id,
            decision_id=resolved.decision_id or decision_id,
            initial_snapshot_id=snap,
            terminal_snapshot_id=terminal_snap,
            snapshot_identity_status=snapshot_status,  # type: ignore[arg-type]
            position_lifecycle_id=lifecycle,
            contract_id=contract_id or None,
            direction="bullish" if entry_orders else "none",
            action_class="closed_trade",
            entry_order_ids=tuple(o.client_order_id for o in entry_orders),
            exit_order_ids=tuple(
                o.client_order_id for o in (*reduction_orders, *exit_orders)
            ),
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=entry_qty,
            realised_pnl=realised,
            total_fees=fees,
            market_regime_tags=market_regime_tags,
            data_quality_ids=data_quality_ids,
            option_surface_ids=option_surface_ids,
            source_event_ids=source_event_ids or (
                (terminal_event_uuid,) if terminal_event_uuid is not None else ()
            ),
            entry_decision_event_id=entry_event_id,
            entry_decision_timestamp=entry_ts,
            terminal_event_id=terminal_event_uuid,
            terminal_event_timestamp=terminal_ts,
            market_event_ids=market_ids,
            cognitive_artifact_ids=cognitive_artifact_ids,
            model_call_ids=model_call_ids,
            configuration_version_id=resolved.configuration_version_id
            or configuration_version_id,
            completed=completed,
            completeness_findings=tuple(dict.fromkeys(findings)),
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
        completed = True
        if order is None:
            findings.append("order_missing_from_projection")
            completed = False
        elif action_class == "entry_rejected" and order.status is not OrderStatus.REJECTED:
            findings.append("projection_status_not_rejected")
            completed = False
        elif action_class == "entry_cancelled" and order.status is not OrderStatus.CANCELLED:
            findings.append("projection_status_not_cancelled")
            completed = False

        if initial_snapshot_id is None:
            findings.append("missing_initial_snapshot")
            completed = False
            snap_status = "missing"
        else:
            snap_status = "verified"

        if configuration_version_id is None:
            findings.append("missing_configuration_version")
            completed = False

        lifecycle = f"{action_class}:{client_order_id or event_id}"
        key = episode_idempotency_key(session_id, lifecycle, event_id)
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            initial_snapshot_id=initial_snapshot_id,
            snapshot_identity_status=snap_status,  # type: ignore[arg-type]
            action_class=action_class,  # type: ignore[arg-type]
            entry_order_ids=(client_order_id,) if client_order_id else (),
            quantity=Decimal("0"),
            configuration_version_id=configuration_version_id,
            completed=completed,
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
        snapshot_id: UUID | None,
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
        completed = True
        snap_status = "verified"
        if snapshot_id is None:
            findings.append("missing_snapshot")
            completed = False
            snap_status = "missing"

        # Prefer persisted Task 2 cycle registry / cognitive repos over event-only.
        if self._cycle_registry is not None and cycle_id:
            try:
                record = await self._cycle_registry.get(
                    session_id=session_id, graph_kind="decision", cycle_id=cycle_id
                )
            except Exception:
                record = None
            if record is not None:
                payload = record.payload or {}
                for key_name, bucket in (
                    ("world_model_id", "cognitive_artifact_ids"),
                    ("strategy_ids", "cognitive_artifact_ids"),
                    ("debate_id", "cognitive_artifact_ids"),
                    ("decision_id", "cognitive_artifact_ids"),
                    ("model_call_ids", "model_call_ids"),
                    ("data_quality_id", "data_quality_ids"),
                    ("option_surface_id", "option_surface_ids"),
                ):
                    raw = payload.get(key_name)
                    if raw is None:
                        continue
                    ids = raw if isinstance(raw, (list, tuple)) else [raw]
                    collected: list[UUID] = []
                    for item in ids:
                        try:
                            collected.append(UUID(str(item)))
                        except Exception:
                            continue
                    if bucket == "cognitive_artifact_ids" and collected:
                        cognitive_artifact_ids = tuple(
                            dict.fromkeys((*cognitive_artifact_ids, *collected))
                        )
                    elif bucket == "model_call_ids" and collected:
                        model_call_ids = tuple(
                            dict.fromkeys((*model_call_ids, *collected))
                        )
                    elif bucket == "data_quality_ids" and collected:
                        data_quality_ids = tuple(
                            dict.fromkeys((*data_quality_ids, *collected))
                        )
                    elif bucket == "option_surface_ids" and collected:
                        option_surface_ids = tuple(
                            dict.fromkeys((*option_surface_ids, *collected))
                        )
                if payload.get("confidence_values"):
                    confidence_values = {
                        str(k): Decimal(str(v))
                        for k, v in dict(payload["confidence_values"]).items()
                    }
                if payload.get("rejection_codes"):
                    rejection_codes = tuple(payload["rejection_codes"])

        key = episode_idempotency_key(session_id, f"no_trade:{cycle_id}", event_id)
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            entry_cycle_id=cycle_id,
            initial_snapshot_id=snapshot_id,
            snapshot_identity_status=snap_status,  # type: ignore[arg-type]
            action_class="no_trade",
            quantity=Decimal("0"),
            configuration_version_id=configuration_version_id,
            cognitive_artifact_ids=cognitive_artifact_ids,
            model_call_ids=model_call_ids,
            data_quality_ids=data_quality_ids,
            option_surface_ids=option_surface_ids,
            completed=completed,
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
