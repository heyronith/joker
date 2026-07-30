"""Task-1 SessionObjectiveService — durable capital truth and reservations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from joker.objectives.deadline import time_remaining_seconds
from joker.objectives.events import (
    BoundedOperatorEventProjection,
    ObjectiveOperatorEventType,
    make_objective_event,
)
from joker.objectives.repository import ObjectiveRepository
from joker.objectives.schemas import (
    CapitalReservation,
    SessionObjectiveDefinition,
    SessionObjectiveState,
    build_definition,
    state_to_context,
)

logger = logging.getLogger(__name__)


class ObjectiveServiceError(RuntimeError):
    """Fail-closed objective / capital gate."""


class SessionObjectiveService:
    """Owns session objective lifecycle, reservations, and recomputation."""

    def __init__(
        self,
        repository: ObjectiveRepository,
        *,
        exchange_tz: str = "America/New_York",
        operator_events: BoundedOperatorEventProjection | None = None,
        pause_entries_when_goal_met: bool = True,
        stop_new_entries_at_deadline: bool = True,
        require_positive_expected_value: bool = True,
        minimum_win_probability: float = 0.45,
    ) -> None:
        self._repo = repository
        self._exchange_tz = exchange_tz
        self._events = operator_events or BoundedOperatorEventProjection()
        self.pause_entries_when_goal_met = pause_entries_when_goal_met
        self.stop_new_entries_at_deadline = stop_new_entries_at_deadline
        self.require_positive_expected_value = require_positive_expected_value
        self.minimum_win_probability = minimum_win_probability
        self._objective_id: UUID | None = None
        self._broker_submission_seen = False
        self._reconciliation_unresolved = False

    @property
    def operator_events(self) -> BoundedOperatorEventProjection:
        return self._events

    def _emit(
        self,
        event_type: ObjectiveOperatorEventType,
        *,
        objective_id: UUID,
        session_id: str,
        reason_codes: tuple[str, ...] = (),
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        linked_ids: dict[str, str] | None = None,
    ) -> None:
        event = make_objective_event(
            event_type,
            objective_id=objective_id,
            session_id=session_id,
            reason_codes=reason_codes,
            before=before,
            after=after,
            linked_ids=linked_ids,
        )
        self._events.publish(event)
        self._repo.append_audit(
            audit_id=str(uuid4()),
            objective_id=objective_id,
            session_id=session_id,
            event_type=event_type.value,
            payload=event.sanitised_payload(),
            created_at=event.timestamp,
        )

    async def create_objective(
        self,
        *,
        session_id: str,
        authorised_capital_usd: Decimal | float,
        target_profit_pct: Decimal | float,
        deadline_exchange_time: datetime,
        max_concurrent_positions: int,
        accepted_total_loss_risk: bool,
        pause_entries_when_goal_met: bool | None = None,
    ) -> SessionObjectiveDefinition:
        if self._broker_submission_seen:
            raise ObjectiveServiceError(
                "cannot create/replace objective after first broker submission"
            )
        definition = build_definition(
            session_id=session_id,
            authorised_capital_usd=authorised_capital_usd,
            target_profit_pct=target_profit_pct,
            deadline_exchange_time=deadline_exchange_time,
            max_concurrent_positions=max_concurrent_positions,
            accepted_total_loss_risk=accepted_total_loss_risk,
            pause_entries_when_goal_met=(
                self.pause_entries_when_goal_met
                if pause_entries_when_goal_met is None
                else pause_entries_when_goal_met
            ),
        )
        # Store unarmed until confirm
        pending = definition.model_copy(update={"armed": False})
        # frozen model — rebuild
        pending = SessionObjectiveDefinition(
            objective_id=definition.objective_id,
            session_id=definition.session_id,
            authorised_capital_usd=definition.authorised_capital_usd,
            target_profit_pct=definition.target_profit_pct,
            target_profit_usd=definition.target_profit_usd,
            target_ending_equity_usd=definition.target_ending_equity_usd,
            deadline_exchange_time=definition.deadline_exchange_time,
            max_concurrent_positions=definition.max_concurrent_positions,
            pause_entries_when_goal_met=definition.pause_entries_when_goal_met,
            accepted_total_loss_risk=definition.accepted_total_loss_risk,
            created_at=definition.created_at,
            definition_version=definition.definition_version,
            armed=False,
            first_broker_submission_at=None,
        )
        self._repo.save_definition(pending)
        remaining = time_remaining_seconds(
            pending.deadline_exchange_time, exchange_tz=self._exchange_tz
        )
        state = SessionObjectiveState(
            objective_id=pending.objective_id,
            session_id=session_id,
            status="pending_confirmation",
            authorised_capital_usd=pending.authorised_capital_usd,
            target_profit_usd=pending.target_profit_usd,
            target_ending_equity_usd=pending.target_ending_equity_usd,
            reserved_capital_usd=Decimal("0.00"),
            available_capital_usd=pending.authorised_capital_usd,
            required_profit_remaining_usd=pending.target_profit_usd,
            time_remaining_seconds=remaining,
            max_concurrent_positions=pending.max_concurrent_positions,
            deadline_exchange_time=pending.deadline_exchange_time,
            version=1,
        )
        self._repo.append_state(state)
        self._objective_id = pending.objective_id
        self._emit(
            ObjectiveOperatorEventType.CREATED,
            objective_id=pending.objective_id,
            session_id=session_id,
            after={"status": "pending_confirmation"},
        )
        return pending

    async def confirm_objective(
        self, objective_id: UUID | str | None = None
    ) -> SessionObjectiveState:
        oid = UUID(str(objective_id or self._objective_id))
        definition = self._repo.get_definition(oid)
        if definition is None:
            raise ObjectiveServiceError("objective definition missing")
        if not definition.accepted_total_loss_risk:
            raise ObjectiveServiceError("total-loss acknowledgement required")
        armed = SessionObjectiveDefinition(
            objective_id=definition.objective_id,
            session_id=definition.session_id,
            authorised_capital_usd=definition.authorised_capital_usd,
            target_profit_pct=definition.target_profit_pct,
            target_profit_usd=definition.target_profit_usd,
            target_ending_equity_usd=definition.target_ending_equity_usd,
            deadline_exchange_time=definition.deadline_exchange_time,
            max_concurrent_positions=definition.max_concurrent_positions,
            pause_entries_when_goal_met=definition.pause_entries_when_goal_met,
            accepted_total_loss_risk=True,
            created_at=definition.created_at,
            definition_version=definition.definition_version,
            armed=True,
            first_broker_submission_at=definition.first_broker_submission_at,
        )
        self._repo.save_definition(armed)
        self._objective_id = armed.objective_id
        state = await self.recompute_from_truth(force_status="active")
        self._emit(
            ObjectiveOperatorEventType.CONFIRMED,
            objective_id=armed.objective_id,
            session_id=armed.session_id,
            after={"status": state.status, "version": state.version},
        )
        return state

    async def get_state(self) -> SessionObjectiveState:
        if self._objective_id is None:
            raise ObjectiveServiceError("objective state is missing")
        state = self._repo.latest_state(self._objective_id)
        if state is None:
            raise ObjectiveServiceError("objective state is missing")
        return state

    def get_sanitised_context(self) -> dict[str, Any]:
        state = self._repo.latest_state(self._objective_id) if self._objective_id else None
        if state is None:
            raise ObjectiveServiceError("objective state is missing")
        return state_to_context(state).model_dump_for_hash()

    def mark_reconciliation_unresolved(self, unresolved: bool) -> None:
        self._reconciliation_unresolved = unresolved

    async def load_or_recover(self, session_id: str) -> SessionObjectiveState | None:
        definition = self._repo.latest_definition_for_session(session_id)
        if definition is None:
            return None
        self._objective_id = definition.objective_id
        self._broker_submission_seen = definition.first_broker_submission_at is not None
        return await self.recompute_from_truth()

    async def recompute_from_truth(
        self,
        *,
        realised_pnl_usd: Decimal | float | None = None,
        unrealised_pnl_usd: Decimal | float | None = None,
        open_position_count: int | None = None,
        force_status: str | None = None,
        now: datetime | None = None,
    ) -> SessionObjectiveState:
        if self._objective_id is None:
            raise ObjectiveServiceError("objective state is missing")
        definition = self._repo.get_definition(self._objective_id)
        if definition is None:
            raise ObjectiveServiceError("objective definition missing")
        prev = self._repo.latest_state(self._objective_id)
        reserved = Decimal("0.00")
        for res in self._repo.list_open_reservations(self._objective_id):
            reserved += Decimal(str(res.reserved_usd))
        realised = (
            Decimal(str(realised_pnl_usd))
            if realised_pnl_usd is not None
            else (prev.realised_pnl_usd if prev else Decimal("0.00"))
        )
        unrealised = (
            Decimal(str(unrealised_pnl_usd))
            if unrealised_pnl_usd is not None
            else (prev.unrealised_pnl_usd if prev else Decimal("0.00"))
        )
        positions = (
            open_position_count
            if open_position_count is not None
            else (prev.open_position_count if prev else 0)
        )
        available = max(
            Decimal("0.00"),
            (definition.authorised_capital_usd - reserved).quantize(Decimal("0.01")),
        )
        remaining_profit = max(
            Decimal("0.00"),
            (definition.target_profit_usd - realised).quantize(Decimal("0.01")),
        )
        if definition.target_profit_usd > 0:
            progress = min(
                Decimal("200.00"),
                (realised / definition.target_profit_usd * Decimal("100")).quantize(
                    Decimal("0.01")
                ),
            )
        else:
            progress = Decimal("100.00") if realised >= 0 else Decimal("0.00")
        remaining_s = time_remaining_seconds(
            definition.deadline_exchange_time,
            now=now,
            exchange_tz=self._exchange_tz,
        )
        status = force_status or (prev.status if prev else "pending_confirmation")
        entries_paused = bool(prev.entries_paused) if prev else False
        stance = prev.current_stance if prev else "observe"
        feasibility = prev.feasibility_classification if prev else "unknown"
        est_p = prev.estimated_success_probability if prev else None

        if not definition.armed and status != "pending_confirmation":
            status = "pending_confirmation"

        if definition.armed and status == "pending_confirmation" and force_status is None:
            status = "active"

        if remaining_s <= 0 and self.stop_new_entries_at_deadline:
            status = "deadline_reached"
            stance = "deadline"
            entries_paused = True

        if realised >= definition.target_profit_usd and definition.pause_entries_when_goal_met:
            status = "target_reached"
            stance = "defend"
            entries_paused = True

        if available <= 0 and positions == 0 and reserved <= 0:
            if realised < definition.target_profit_usd and remaining_s > 0:
                status = "capital_exhausted"
                stance = "infeasible"
                entries_paused = True

        if force_status:
            status = force_status

        version = (prev.version + 1) if prev else 1
        state = SessionObjectiveState(
            objective_id=definition.objective_id,
            session_id=definition.session_id,
            status=status,  # type: ignore[arg-type]
            authorised_capital_usd=definition.authorised_capital_usd,
            target_profit_usd=definition.target_profit_usd,
            target_ending_equity_usd=definition.target_ending_equity_usd,
            reserved_capital_usd=reserved.quantize(Decimal("0.01")),
            available_capital_usd=available,
            realised_pnl_usd=realised.quantize(Decimal("0.01")),
            unrealised_pnl_usd=unrealised.quantize(Decimal("0.01")),
            progress_to_goal_pct=progress,
            required_profit_remaining_usd=remaining_profit,
            time_remaining_seconds=remaining_s,
            estimated_success_probability=est_p,
            feasibility_classification=feasibility,  # type: ignore[arg-type]
            current_stance=stance,  # type: ignore[arg-type]
            last_recomputed_at=datetime.now(timezone.utc),
            version=version,
            entries_paused=entries_paused,
            open_position_count=positions,
            max_concurrent_positions=definition.max_concurrent_positions,
            deadline_exchange_time=definition.deadline_exchange_time,
        )
        self._repo.append_state(state)
        self._emit(
            ObjectiveOperatorEventType.RECOMPUTED,
            objective_id=state.objective_id,
            session_id=state.session_id,
            after={
                "status": state.status,
                "available": str(state.available_capital_usd),
                "reserved": str(state.reserved_capital_usd),
                "progress": str(state.progress_to_goal_pct),
                "version": state.version,
            },
        )
        if state.status == "target_reached":
            self._emit(
                ObjectiveOperatorEventType.TARGET_REACHED,
                objective_id=state.objective_id,
                session_id=state.session_id,
            )
        if state.status == "deadline_reached":
            self._emit(
                ObjectiveOperatorEventType.DEADLINE_REACHED,
                objective_id=state.objective_id,
                session_id=state.session_id,
            )
        return state

    def _assert_entry_allowed(self, state: SessionObjectiveState) -> None:
        definition = self._repo.get_definition(state.objective_id)
        if definition is None or not definition.armed:
            raise ObjectiveServiceError("objective is unconfirmed")
        if state.status == "pending_confirmation":
            raise ObjectiveServiceError("objective is unconfirmed")
        if self._reconciliation_unresolved:
            raise ObjectiveServiceError("reconciliation is unresolved")
        if state.status == "deadline_reached" or (
            self.stop_new_entries_at_deadline and state.time_remaining_seconds <= 0
        ):
            raise ObjectiveServiceError("deadline has passed")
        if state.entries_paused or state.status == "target_reached":
            raise ObjectiveServiceError("entries are paused (target reached or paused)")
        if state.status in {"capital_exhausted", "stopped_by_user"}:
            raise ObjectiveServiceError(f"objective status blocks entries: {state.status}")
        if state.feasibility_classification == "infeasible":
            raise ObjectiveServiceError("feasibility is infeasible")
        if state.open_position_count >= state.max_concurrent_positions:
            raise ObjectiveServiceError("max concurrent positions reached")

    async def reserve_for_order(
        self,
        *,
        client_order_id: str,
        estimated_premium_usd: Decimal | float,
        objective_state_version: int,
    ) -> CapitalReservation:
        state = await self.get_state()
        if state.version != objective_state_version:
            raise ObjectiveServiceError("objective version is stale")
        self._assert_entry_allowed(state)
        premium = Decimal(str(estimated_premium_usd)).quantize(Decimal("0.01"))
        if premium <= 0:
            raise ObjectiveServiceError("estimated premium must be > 0")
        existing = self._repo.get_reservation_by_client_order(client_order_id)
        if existing is not None and existing.status in {"open", "partial", "converted"}:
            # Idempotent: return existing open reservation
            return existing
        if premium > state.available_capital_usd:
            raise ObjectiveServiceError(
                f"available capital insufficient: need {premium}, have {state.available_capital_usd}"
            )
        reservation = CapitalReservation(
            objective_id=state.objective_id,
            session_id=state.session_id,
            client_order_id=client_order_id,
            estimated_premium_usd=premium,
            reserved_usd=premium,
            status="open",
            objective_state_version=state.version,
        )
        # Optimistic bump
        new_reserved = (state.reserved_capital_usd + premium).quantize(Decimal("0.01"))
        new_available = (state.authorised_capital_usd - new_reserved).quantize(
            Decimal("0.01")
        )
        new_state = state.model_copy(
            update={
                "reserved_capital_usd": new_reserved,
                "available_capital_usd": max(Decimal("0.00"), new_available),
                "version": state.version + 1,
                "last_recomputed_at": datetime.now(timezone.utc),
            }
        )
        ok = self._repo.compare_and_swap_state_version(
            objective_id=state.objective_id,
            expected_version=state.version,
            new_state=new_state,
        )
        if not ok:
            raise ObjectiveServiceError("objective version is stale")
        self._repo.upsert_reservation(reservation)
        self._emit(
            ObjectiveOperatorEventType.CAPITAL_RESERVED,
            objective_id=state.objective_id,
            session_id=state.session_id,
            after={"reserved_usd": str(premium), "client_order_id": client_order_id},
            linked_ids={"client_order_id": client_order_id},
        )
        return reservation

    async def release_for_order(
        self,
        *,
        client_order_id: str,
        reason: str = "released",
    ) -> SessionObjectiveState:
        existing = self._repo.get_reservation_by_client_order(client_order_id)
        if existing is None:
            return await self.get_state()
        if existing.status == "released":
            return await self.get_state()
        updated = existing.model_copy(
            update={
                "status": "released",
                "reserved_usd": Decimal("0.00"),
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._repo.upsert_reservation(updated)
        self._emit(
            ObjectiveOperatorEventType.CAPITAL_RELEASED,
            objective_id=existing.objective_id,
            session_id=existing.session_id,
            reason_codes=(reason,),
            linked_ids={"client_order_id": client_order_id},
        )
        return await self.recompute_from_truth()

    async def associate_broker_order(
        self, *, client_order_id: str, broker_order_id: str
    ) -> None:
        existing = self._repo.get_reservation_by_client_order(client_order_id)
        if existing is None:
            return
        self._repo.upsert_reservation(
            existing.model_copy(
                update={
                    "broker_order_id": broker_order_id,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        )
        definition = self._repo.get_definition(existing.objective_id)
        if definition is not None and definition.first_broker_submission_at is None:
            armed = SessionObjectiveDefinition(
                objective_id=definition.objective_id,
                session_id=definition.session_id,
                authorised_capital_usd=definition.authorised_capital_usd,
                target_profit_pct=definition.target_profit_pct,
                target_profit_usd=definition.target_profit_usd,
                target_ending_equity_usd=definition.target_ending_equity_usd,
                deadline_exchange_time=definition.deadline_exchange_time,
                max_concurrent_positions=definition.max_concurrent_positions,
                pause_entries_when_goal_met=definition.pause_entries_when_goal_met,
                accepted_total_loss_risk=definition.accepted_total_loss_risk,
                created_at=definition.created_at,
                definition_version=definition.definition_version,
                armed=definition.armed,
                first_broker_submission_at=datetime.now(timezone.utc),
            )
            self._repo.save_definition(armed)
            self._broker_submission_seen = True

    async def record_verified_outcome(
        self,
        *,
        client_order_id: str | None = None,
        realised_pnl_delta_usd: Decimal | float = 0,
        convert_reservation: bool = False,
        partial_reserved_usd: Decimal | float | None = None,
        open_position_count: int | None = None,
    ) -> SessionObjectiveState:
        if client_order_id:
            existing = self._repo.get_reservation_by_client_order(client_order_id)
            if existing is not None:
                if convert_reservation:
                    reserved = (
                        Decimal(str(partial_reserved_usd))
                        if partial_reserved_usd is not None
                        else existing.reserved_usd
                    )
                    status = "partial" if partial_reserved_usd is not None else "converted"
                    self._repo.upsert_reservation(
                        existing.model_copy(
                            update={
                                "status": status,
                                "reserved_usd": Decimal(str(reserved)).quantize(
                                    Decimal("0.01")
                                ),
                                "updated_at": datetime.now(timezone.utc),
                            }
                        )
                    )
                elif partial_reserved_usd is not None:
                    self._repo.upsert_reservation(
                        existing.model_copy(
                            update={
                                "status": "partial",
                                "reserved_usd": Decimal(str(partial_reserved_usd)).quantize(
                                    Decimal("0.01")
                                ),
                                "updated_at": datetime.now(timezone.utc),
                            }
                        )
                    )
        state = await self.get_state()
        new_realised = (
            state.realised_pnl_usd + Decimal(str(realised_pnl_delta_usd))
        ).quantize(Decimal("0.01"))
        return await self.recompute_from_truth(
            realised_pnl_usd=new_realised,
            open_position_count=open_position_count,
        )

    async def pause_entries(self, *, reason: str = "paused") -> SessionObjectiveState:
        state = await self.get_state()
        paused = state.model_copy(
            update={
                "entries_paused": True,
                "status": "paused" if state.status == "active" else state.status,
                "version": state.version + 1,
                "last_recomputed_at": datetime.now(timezone.utc),
            }
        )
        self._repo.append_state(paused)
        self._emit(
            ObjectiveOperatorEventType.PAUSED,
            objective_id=paused.objective_id,
            session_id=paused.session_id,
            reason_codes=(reason,),
        )
        return paused

    async def resume_entries(self, *, reason: str = "resumed") -> SessionObjectiveState:
        state = await self.get_state()
        if state.status in {"deadline_reached", "target_reached", "capital_exhausted"}:
            raise ObjectiveServiceError(f"cannot resume from terminal status {state.status}")
        resumed = state.model_copy(
            update={
                "entries_paused": False,
                "status": "active",
                "version": state.version + 1,
                "last_recomputed_at": datetime.now(timezone.utc),
            }
        )
        self._repo.append_state(resumed)
        self._emit(
            ObjectiveOperatorEventType.RESUMED,
            objective_id=resumed.objective_id,
            session_id=resumed.session_id,
            reason_codes=(reason,),
        )
        return resumed

    async def update_feasibility(
        self,
        *,
        classification: str,
        estimated_success_probability: Decimal | None,
    ) -> SessionObjectiveState:
        state = await self.get_state()
        stance = state.current_stance
        status = state.status
        if classification == "infeasible":
            stance = "infeasible"
            status = "temporarily_infeasible"
            self._emit(
                ObjectiveOperatorEventType.INFEASIBLE,
                objective_id=state.objective_id,
                session_id=state.session_id,
            )
        updated = state.model_copy(
            update={
                "feasibility_classification": classification,
                "estimated_success_probability": estimated_success_probability,
                "current_stance": stance,
                "status": status,
                "version": state.version + 1,
                "last_recomputed_at": datetime.now(timezone.utc),
            }
        )
        self._repo.append_state(updated)
        self._emit(
            ObjectiveOperatorEventType.FEASIBILITY_ASSESSED,
            objective_id=updated.objective_id,
            session_id=updated.session_id,
            after={
                "classification": classification,
                "p": str(estimated_success_probability)
                if estimated_success_probability is not None
                else None,
            },
        )
        return updated
