"""Task-1 SessionObjectiveService — durable capital truth and exposures."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, TypeVar
from uuid import UUID, uuid4

from zoneinfo import ZoneInfo

from joker.objectives.deadline import time_remaining_seconds
from joker.objectives.events import (
    BoundedOperatorEventProjection,
    ObjectiveOperatorEventType,
    make_objective_event,
)
from joker.objectives.repository import ObjectivePersistenceBusyError, ObjectiveRepository
from joker.objectives.schemas import (
    CapitalExposure,
    SessionObjectiveDefinition,
    SessionObjectiveState,
    build_definition,
    premium_notional_usd,
    state_to_context,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _elapsed_seconds(
    *,
    confirmed_at: datetime | None,
    duration_seconds: int | None,
    time_remaining_seconds: int,
    now: datetime | None = None,
    exchange_tz: str = "America/New_York",
) -> int:
    """Elapsed wall time since confirmation; never mutates stored duration."""
    if duration_seconds is not None and duration_seconds >= 0:
        # Prefer remaining-based elapsed when duration is known (stable under clock skew).
        return max(0, int(duration_seconds) - max(0, int(time_remaining_seconds)))
    if confirmed_at is None:
        return 0
    tz = ZoneInfo(exchange_tz)
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    conf = confirmed_at
    if conf.tzinfo is None:
        conf = conf.replace(tzinfo=tz)
    return max(0, int((current.astimezone(tz) - conf.astimezone(tz)).total_seconds()))


class ObjectiveServiceError(RuntimeError):
    """Fail-closed objective / capital gate."""


class SessionObjectiveService:
    """Owns session objective lifecycle, exposures, and recomputation."""

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
        objective_policy: str = "positive_ev_baseline",
        shadow_baseline_enabled: bool = False,
    ) -> None:
        self._repo = repository
        self._exchange_tz = exchange_tz
        self._events = operator_events or BoundedOperatorEventProjection()
        self.pause_entries_when_goal_met = pause_entries_when_goal_met
        self.stop_new_entries_at_deadline = stop_new_entries_at_deadline
        self.require_positive_expected_value = require_positive_expected_value
        self.minimum_win_probability = minimum_win_probability
        self.objective_policy = objective_policy
        self.shadow_baseline_enabled = shadow_baseline_enabled
        self._objective_id: UUID | None = None
        self._broker_submission_seen = False
        self._reconciliation_unresolved = False
        self._truth_degraded = False
        # Cross-loop safe: CLI confirms on loop A, Task-1 runtime uses loop B.
        # Never bind an asyncio.Lock to the short-lived confirmation loop.
        self._write_lock = threading.RLock()

    async def _db_write(self, fn: Callable[[], T]) -> T:
        """Run objective mutations off the event loop under a process-local lock."""

        def _run() -> T:
            with self._write_lock:
                try:
                    return fn()
                except ObjectivePersistenceBusyError as exc:
                    # Fail closed with a service-level error; never swallow.
                    raise ObjectiveServiceError(str(exc)) from exc

        return await asyncio.to_thread(_run)

    async def _db_read(self, fn: Callable[[], T]) -> T:
        """Run objective reads off the event loop without holding the write lock."""

        def _run() -> T:
            try:
                return fn()
            except ObjectivePersistenceBusyError as exc:
                raise ObjectiveServiceError(str(exc)) from exc

        return await asyncio.to_thread(_run)

    def _latest_state_sync(self) -> SessionObjectiveState:
        """Read the durable latest state. Callers must already hold the lock."""
        if self._objective_id is None:
            raise ObjectiveServiceError("objective state is missing")
        state = self._repo.latest_state(self._objective_id)
        if state is None:
            raise ObjectiveServiceError("objective state is missing")
        return state

    @property
    def operator_events(self) -> BoundedOperatorEventProjection:
        return self._events

    @property
    def repository(self) -> ObjectiveRepository:
        """Public access to the objective persistence repository."""
        return self._repo

    @property
    def truth_degraded(self) -> bool:
        return self._truth_degraded or self._reconciliation_unresolved

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
        persist_audit: bool = True,
    ) -> dict[str, Any]:
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
        payload = event.sanitised_payload()
        audit_id = str(uuid4())
        if persist_audit:
            self._repo.append_audit(
                audit_id=audit_id,
                objective_id=objective_id,
                session_id=session_id,
                event_type=event_type.value,
                payload=payload,
                created_at=event.timestamp,
            )
        return {
            "audit_id": audit_id,
            "session_id": session_id,
            "event_type": event_type.value,
            "payload": payload,
            "created_at": event.timestamp,
        }

    def mark_truth_degraded(self, degraded: bool, *, reason: str = "") -> None:
        self._truth_degraded = bool(degraded)
        if degraded:
            logger.error(
                "objective_truth_degraded",
                extra={"reason": reason, "objective_id": str(self._objective_id)},
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
        def _body() -> SessionObjectiveDefinition:
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
                working_order_reservation_usd=Decimal("0.00"),
                filled_position_exposure_usd=Decimal("0.00"),
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

        return await self._db_write(_body)

    async def confirm_objective(
        self,
        objective_id: UUID | str | None = None,
        *,
        confirmed_at_exchange_time: datetime | None = None,
    ) -> SessionObjectiveState:
        def _arm() -> tuple[SessionObjectiveDefinition, datetime]:
            oid = UUID(str(objective_id or self._objective_id))
            definition = self._repo.get_definition(oid)
            if definition is None:
                raise ObjectiveServiceError("objective definition missing")
            if not definition.accepted_total_loss_risk:
                raise ObjectiveServiceError("total-loss acknowledgement required")
            tz = ZoneInfo(self._exchange_tz)
            confirmed_at = (
                definition.objective_confirmed_at_exchange_time
                or confirmed_at_exchange_time
                or datetime.now(tz)
            )
            if confirmed_at.tzinfo is None:
                raise ObjectiveServiceError(
                    "objective_confirmed_at_exchange_time must be timezone-aware"
                )
            duration = definition.objective_duration_seconds
            if duration is None:
                duration = int(
                    (
                        definition.deadline_exchange_time.astimezone(tz)
                        - confirmed_at.astimezone(tz)
                    ).total_seconds()
                )
            if duration <= 0:
                created = definition.created_at.astimezone(tz)
                deadline_local = definition.deadline_exchange_time.astimezone(tz)
                if created < deadline_local:
                    confirmed_at = created
                    duration = int((deadline_local - created).total_seconds())
                else:
                    confirmed_at = deadline_local - timedelta(seconds=1)
                    duration = 1
            if duration <= 0:
                raise ObjectiveServiceError(
                    "objective_duration_seconds must be > 0 at confirmation"
                )
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
                objective_confirmed_at_exchange_time=confirmed_at,
                objective_duration_seconds=int(duration),
                definition_version=definition.definition_version,
                armed=True,
                first_broker_submission_at=definition.first_broker_submission_at,
            )
            self._repo.save_definition(armed)
            self._objective_id = armed.objective_id
            return armed, confirmed_at

        armed, confirmed_at = await self._db_write(_arm)
        state = await self.recompute_from_truth(force_status="active", now=confirmed_at)

        def _confirm_audit() -> None:
            self._emit(
                ObjectiveOperatorEventType.CONFIRMED,
                objective_id=armed.objective_id,
                session_id=armed.session_id,
                after={
                    "status": state.status,
                    "version": state.version,
                    "objective_duration_seconds": armed.objective_duration_seconds,
                    "objective_confirmed_at_exchange_time": confirmed_at.isoformat(),
                },
            )

        await self._db_write(_confirm_audit)
        return state

    async def get_state(self) -> SessionObjectiveState:
        return await self._db_read(self._latest_state_sync)

    def get_sanitised_context(self) -> dict[str, Any]:
        """Sync helper for non-async callers; prefer ``aget_sanitised_context`` on loops."""
        state = self._repo.latest_state(self._objective_id) if self._objective_id else None
        if state is None:
            raise ObjectiveServiceError("objective state is missing")
        return state_to_context(state).model_dump_for_hash()

    async def aget_sanitised_context(self) -> dict[str, Any]:
        return await self._db_read(self.get_sanitised_context)

    def mark_reconciliation_unresolved(self, unresolved: bool) -> None:
        self._reconciliation_unresolved = unresolved
        if unresolved:
            self.mark_truth_degraded(True, reason="reconciliation_unresolved")

    async def load_or_recover(
        self, session_id: str, *, now: datetime | None = None
    ) -> SessionObjectiveState | None:
        def _load() -> SessionObjectiveDefinition | None:
            definition = self._repo.latest_definition_for_session(session_id)
            if definition is None:
                return None
            self._objective_id = definition.objective_id
            self._broker_submission_seen = definition.first_broker_submission_at is not None
            return definition

        definition = await self._db_write(_load)
        if definition is None:
            return None
        return await self.recompute_from_truth(now=now)

    def _sum_exposures(
        self, objective_id: UUID
    ) -> tuple[Decimal, Decimal, int]:
        working = Decimal("0.00")
        filled = Decimal("0.00")
        open_positions = 0
        for exp in self._repo.list_encumbering_exposures(objective_id):
            working += Decimal(str(exp.working_order_reservation_usd))
            filled += Decimal(str(exp.filled_exposure_usd))
            if exp.filled_quantity > 0 and exp.status in {
                "filled_position_exposure",
                "partial",
            }:
                open_positions += 1
        return (
            working.quantize(Decimal("0.01")),
            filled.quantize(Decimal("0.01")),
            open_positions,
        )

    def _build_state_from_exposures(
        self,
        definition: SessionObjectiveDefinition,
        prev: SessionObjectiveState | None,
        *,
        realised_pnl_usd: Decimal | None = None,
        unrealised_pnl_usd: Decimal | None = None,
        open_position_count: int | None = None,
        force_status: str | None = None,
        now: datetime | None = None,
        truth_degraded: bool | None = None,
    ) -> SessionObjectiveState:
        working, filled, derived_positions = self._sum_exposures(definition.objective_id)
        encumbered = (working + filled).quantize(Decimal("0.01"))
        available = max(
            Decimal("0.00"),
            (definition.authorised_capital_usd - encumbered).quantize(Decimal("0.01")),
        )
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
            else (
                derived_positions
                if prev is None
                else max(prev.open_position_count, derived_positions)
            )
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
        degraded = (
            self.truth_degraded
            if truth_degraded is None
            else bool(truth_degraded)
        )

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
        if available <= 0 and positions == 0 and encumbered <= 0:
            if realised < definition.target_profit_usd and remaining_s > 0:
                status = "capital_exhausted"
                stance = "infeasible"
                entries_paused = True
        if degraded:
            status = "truth_degraded"
            entries_paused = True
        if force_status:
            status = force_status

        version = (prev.version + 1) if prev else 1
        duration = definition.objective_duration_seconds
        confirmed_at = definition.objective_confirmed_at_exchange_time
        # Prefer durable definition values; fall back to previous state after restart.
        if duration is None and prev is not None:
            duration = prev.objective_duration_seconds
        if confirmed_at is None and prev is not None:
            confirmed_at = prev.objective_confirmed_at_exchange_time
        elapsed = _elapsed_seconds(
            confirmed_at=confirmed_at,
            duration_seconds=duration,
            time_remaining_seconds=remaining_s,
            now=now,
            exchange_tz=self._exchange_tz,
        )
        return SessionObjectiveState(
            objective_id=definition.objective_id,
            session_id=definition.session_id,
            status=status,  # type: ignore[arg-type]
            authorised_capital_usd=definition.authorised_capital_usd,
            target_profit_usd=definition.target_profit_usd,
            target_ending_equity_usd=definition.target_ending_equity_usd,
            working_order_reservation_usd=working,
            filled_position_exposure_usd=filled,
            reserved_capital_usd=encumbered,
            available_capital_usd=available,
            realised_pnl_usd=realised.quantize(Decimal("0.01")),
            unrealised_pnl_usd=unrealised.quantize(Decimal("0.01")),
            progress_to_goal_pct=progress,
            required_profit_remaining_usd=remaining_profit,
            time_remaining_seconds=remaining_s,
            objective_confirmed_at_exchange_time=confirmed_at,
            objective_duration_seconds=duration,
            elapsed_seconds=elapsed,
            estimated_success_probability=est_p,
            feasibility_classification=feasibility,  # type: ignore[arg-type]
            current_stance=stance,  # type: ignore[arg-type]
            last_recomputed_at=datetime.now(timezone.utc),
            version=version,
            entries_paused=entries_paused,
            open_position_count=positions,
            max_concurrent_positions=definition.max_concurrent_positions,
            deadline_exchange_time=definition.deadline_exchange_time,
            truth_degraded=degraded,
        )

    async def recompute_from_truth(
        self,
        *,
        realised_pnl_usd: Decimal | float | None = None,
        unrealised_pnl_usd: Decimal | float | None = None,
        open_position_count: int | None = None,
        force_status: str | None = None,
        now: datetime | None = None,
    ) -> SessionObjectiveState:
        def _body() -> SessionObjectiveState:
            if self._objective_id is None:
                raise ObjectiveServiceError("objective state is missing")
            definition = self._repo.get_definition(self._objective_id)
            if definition is None:
                raise ObjectiveServiceError("objective definition missing")

            def _build(
                next_version: int, prev: SessionObjectiveState | None
            ) -> SessionObjectiveState:
                state = self._build_state_from_exposures(
                    definition,
                    prev,
                    realised_pnl_usd=(
                        Decimal(str(realised_pnl_usd))
                        if realised_pnl_usd is not None
                        else None
                    ),
                    unrealised_pnl_usd=(
                        Decimal(str(unrealised_pnl_usd))
                        if unrealised_pnl_usd is not None
                        else None
                    ),
                    open_position_count=open_position_count,
                    force_status=force_status,
                    now=now,
                )
                return state.model_copy(update={"version": next_version})

            # Publish operator event before durable append so audit payload matches.
            # Persistence of RECOMPUTED audit happens inside the same SQLite txn.
            preview_prev = self._repo.latest_state(self._objective_id)
            preview = _build(
                ((preview_prev.version + 1) if preview_prev else 1),
                preview_prev,
            )
            audit = self._emit(
                ObjectiveOperatorEventType.RECOMPUTED,
                objective_id=preview.objective_id,
                session_id=preview.session_id,
                after={
                    "status": preview.status,
                    "available": str(preview.available_capital_usd),
                    "working": str(preview.working_order_reservation_usd),
                    "filled_exposure": str(preview.filled_position_exposure_usd),
                    "reserved": str(preview.reserved_capital_usd),
                    "version": preview.version,
                },
                persist_audit=False,
            )
            try:
                state = self._repo.append_next_state_atomic(
                    objective_id=self._objective_id,
                    build_state=_build,
                    audit=audit,
                )
            except ObjectivePersistenceBusyError as exc:
                raise ObjectiveServiceError(str(exc)) from exc
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

        return await self._db_write(_body)

    def _assert_entry_allowed(self, state: SessionObjectiveState) -> None:
        definition = self._repo.get_definition(state.objective_id)
        if definition is None or not definition.armed:
            raise ObjectiveServiceError("objective is unconfirmed")
        if state.status == "pending_confirmation":
            raise ObjectiveServiceError("objective is unconfirmed")
        if self.truth_degraded or state.truth_degraded or state.status == "truth_degraded":
            raise ObjectiveServiceError("objective truth is degraded")
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
        estimated_premium_usd: Decimal | float | None = None,
        premium_per_contract_usd: Decimal | float | None = None,
        quantity: int = 1,
        objective_state_version: int,
        contract_id: str | None = None,
        position_lifecycle_id: str | None = None,
    ) -> CapitalExposure:
        def _body() -> CapitalExposure:
            state = self._latest_state_sync()
            if state.version != objective_state_version:
                raise ObjectiveServiceError("objective version is stale")
            self._assert_entry_allowed(state)
            existing = self._repo.get_exposure_by_client_order(client_order_id)
            if existing is not None and existing.status in {
                "working_order_reservation",
                "partial",
                "filled_position_exposure",
            }:
                return existing

            if premium_per_contract_usd is not None:
                per = Decimal(str(premium_per_contract_usd))
                qty = max(1, int(quantity))
                working = premium_notional_usd(per, qty)
            elif estimated_premium_usd is not None:
                working = Decimal(str(estimated_premium_usd)).quantize(Decimal("0.01"))
                qty = max(1, int(quantity))
                per = (working / (Decimal("100") * Decimal(qty))).quantize(Decimal("0.01"))
            else:
                raise ObjectiveServiceError("premium required for reservation")
            if working <= 0:
                raise ObjectiveServiceError("estimated premium must be > 0")
            if working > state.available_capital_usd:
                raise ObjectiveServiceError(
                    "available capital insufficient: "
                    f"need {working}, have {state.available_capital_usd}"
                )

            exposure = CapitalExposure(
                objective_id=state.objective_id,
                session_id=state.session_id,
                client_order_id=client_order_id,
                contract_id=contract_id,
                position_lifecycle_id=position_lifecycle_id,
                estimated_premium_per_contract_usd=per,
                requested_quantity=qty,
                working_quantity=qty,
                filled_quantity=0,
                working_order_reservation_usd=working,
                filled_exposure_usd=Decimal("0.00"),
                status="working_order_reservation",
                objective_state_version=state.version,
            )
            definition = self._repo.get_definition(state.objective_id)
            assert definition is not None
            # Temporary upsert so sum includes this exposure for new state construction.
            # Atomic path writes both; we build new_state from projected sums.
            projected_working = state.working_order_reservation_usd + working
            projected_filled = state.filled_position_exposure_usd
            encumbered = (projected_working + projected_filled).quantize(Decimal("0.01"))
            available = max(
                Decimal("0.00"),
                (definition.authorised_capital_usd - encumbered).quantize(Decimal("0.01")),
            )
            new_state = state.model_copy(
                update={
                    "working_order_reservation_usd": projected_working.quantize(
                        Decimal("0.01")
                    ),
                    "filled_position_exposure_usd": projected_filled,
                    "reserved_capital_usd": encumbered,
                    "available_capital_usd": available,
                    "version": state.version + 1,
                    "last_recomputed_at": datetime.now(timezone.utc),
                }
            )
            audit = self._emit(
                ObjectiveOperatorEventType.CAPITAL_RESERVED,
                objective_id=state.objective_id,
                session_id=state.session_id,
                after={
                    "working_usd": str(working),
                    "client_order_id": client_order_id,
                },
                linked_ids={"client_order_id": client_order_id},
                persist_audit=False,
            )
            ok = self._repo.atomic_mutate_exposure(
                objective_id=state.objective_id,
                expected_version=state.version,
                exposure=exposure,
                new_state=new_state,
                audit=audit,
                dedupe_key=f"reserve:{client_order_id}",
            )
            if not ok:
                raise ObjectiveServiceError("objective version is stale")
            return exposure

        return await self._db_write(_body)

    async def release_for_order(
        self,
        *,
        client_order_id: str,
        reason: str = "released",
        dedupe_key: str | None = None,
    ) -> SessionObjectiveState:
        """Release remaining working-order reservation only (not filled exposure)."""

        def _body() -> SessionObjectiveState:
            existing = self._repo.get_exposure_by_client_order(client_order_id)
            if existing is None:
                return self._latest_state_sync()
            if (
                existing.status in {"released", "closed"}
                and existing.working_order_reservation_usd <= 0
            ):
                return self._latest_state_sync()
            state = self._latest_state_sync()
            key = dedupe_key or f"release_working:{client_order_id}:{reason}"
            if self._repo.has_projection_dedupe(key):
                return state

            updated = existing.model_copy(
                update={
                    "working_order_reservation_usd": Decimal("0.00"),
                    "working_quantity": 0,
                    "status": (
                        "filled_position_exposure"
                        if existing.filled_exposure_usd > 0
                        else "released"
                    ),
                    "updated_at": datetime.now(timezone.utc),
                    "objective_state_version": state.version,
                }
            )
            # Apply temporarily for sum — atomic write uses updated exposure.
            definition = self._repo.get_definition(state.objective_id)
            assert definition is not None
            # Compute new totals excluding released working portion.
            working = Decimal("0.00")
            filled = Decimal("0.00")
            for exp in self._repo.list_encumbering_exposures(state.objective_id):
                if exp.client_order_id == client_order_id:
                    working += updated.working_order_reservation_usd
                    filled += updated.filled_exposure_usd
                else:
                    working += exp.working_order_reservation_usd
                    filled += exp.filled_exposure_usd
            encumbered = (working + filled).quantize(Decimal("0.01"))
            available = max(
                Decimal("0.00"),
                (definition.authorised_capital_usd - encumbered).quantize(Decimal("0.01")),
            )
            new_state = state.model_copy(
                update={
                    "working_order_reservation_usd": working.quantize(Decimal("0.01")),
                    "filled_position_exposure_usd": filled.quantize(Decimal("0.01")),
                    "reserved_capital_usd": encumbered,
                    "available_capital_usd": available,
                    "version": state.version + 1,
                    "last_recomputed_at": datetime.now(timezone.utc),
                }
            )
            audit = self._emit(
                ObjectiveOperatorEventType.CAPITAL_RELEASED,
                objective_id=existing.objective_id,
                session_id=existing.session_id,
                reason_codes=(reason,),
                linked_ids={"client_order_id": client_order_id},
                persist_audit=False,
            )
            ok = self._repo.atomic_mutate_exposure(
                objective_id=state.objective_id,
                expected_version=state.version,
                exposure=updated,
                new_state=new_state,
                audit=audit,
                dedupe_key=key,
            )
            if not ok:
                raise ObjectiveServiceError("objective version is stale")
            return new_state

        return await self._db_write(_body)

    async def apply_verified_fill(
        self,
        *,
        client_order_id: str,
        fill_quantity: int,
        fill_price: Decimal | float,
        remaining_working_quantity: int | None = None,
        dedupe_key: str | None = None,
        contract_id: str | None = None,
        open_position_count: int | None = None,
    ) -> SessionObjectiveState:
        """Convert filled qty to cost-basis exposure; retain working for unfilled."""

        def _body() -> SessionObjectiveState:
            existing = self._repo.get_exposure_by_client_order(client_order_id)
            if existing is None:
                raise ObjectiveServiceError(
                    f"no exposure for client_order_id={client_order_id}"
                )
            state = self._latest_state_sync()
            key = dedupe_key or (
                f"fill:{client_order_id}:{fill_quantity}:{fill_price}:"
                f"{remaining_working_quantity}"
            )
            if self._repo.has_projection_dedupe(key):
                return state

            fill_qty = max(0, int(fill_quantity))
            price = Decimal(str(fill_price))
            fill_notional = premium_notional_usd(price, fill_qty)
            prior_filled_qty = int(existing.filled_quantity)
            new_filled_qty = prior_filled_qty + fill_qty
            # Weighted average fill price
            if prior_filled_qty > 0 and existing.average_fill_price is not None:
                prior_notional = premium_notional_usd(
                    existing.average_fill_price, prior_filled_qty
                )
                avg = (
                    (prior_notional + fill_notional)
                    / (Decimal("100") * Decimal(new_filled_qty))
                ).quantize(Decimal("0.0001"))
            else:
                avg = price
            filled_exposure = premium_notional_usd(avg, new_filled_qty)

            if remaining_working_quantity is not None:
                rem_qty = max(0, int(remaining_working_quantity))
            else:
                rem_qty = max(0, int(existing.working_quantity) - fill_qty)
            working_usd = (
                premium_notional_usd(
                    existing.estimated_premium_per_contract_usd, rem_qty
                )
                if rem_qty > 0
                else Decimal("0.00")
            )
            if rem_qty > 0 and filled_exposure > 0:
                status = "partial"
            elif filled_exposure > 0:
                status = "filled_position_exposure"
            elif working_usd > 0:
                status = "working_order_reservation"
            else:
                status = "released"

            updated = existing.model_copy(
                update={
                    "filled_quantity": new_filled_qty,
                    "working_quantity": rem_qty,
                    "average_fill_price": avg,
                    "filled_exposure_usd": filled_exposure,
                    "working_order_reservation_usd": working_usd,
                    "status": status,
                    "contract_id": contract_id or existing.contract_id,
                    "updated_at": datetime.now(timezone.utc),
                    "objective_state_version": state.version,
                }
            )
            definition = self._repo.get_definition(state.objective_id)
            assert definition is not None
            working = Decimal("0.00")
            filled = Decimal("0.00")
            for exp in self._repo.list_encumbering_exposures(state.objective_id):
                if exp.client_order_id == client_order_id:
                    working += updated.working_order_reservation_usd
                    filled += updated.filled_exposure_usd
                else:
                    working += exp.working_order_reservation_usd
                    filled += exp.filled_exposure_usd
            encumbered = (working + filled).quantize(Decimal("0.01"))
            available = max(
                Decimal("0.00"),
                (definition.authorised_capital_usd - encumbered).quantize(Decimal("0.01")),
            )
            positions = (
                open_position_count
                if open_position_count is not None
                else max(state.open_position_count, 1 if new_filled_qty > 0 else 0)
            )
            new_state = state.model_copy(
                update={
                    "working_order_reservation_usd": working.quantize(Decimal("0.01")),
                    "filled_position_exposure_usd": filled.quantize(Decimal("0.01")),
                    "reserved_capital_usd": encumbered,
                    "available_capital_usd": available,
                    "open_position_count": positions,
                    "version": state.version + 1,
                    "last_recomputed_at": datetime.now(timezone.utc),
                }
            )
            audit = self._emit(
                ObjectiveOperatorEventType.CAPITAL_RESERVED,
                objective_id=state.objective_id,
                session_id=state.session_id,
                reason_codes=("verified_fill",),
                after={
                    "filled_exposure_usd": str(filled_exposure),
                    "working_order_reservation_usd": str(working_usd),
                    "client_order_id": client_order_id,
                },
                linked_ids={"client_order_id": client_order_id},
                persist_audit=False,
            )
            ok = self._repo.atomic_mutate_exposure(
                objective_id=state.objective_id,
                expected_version=state.version,
                exposure=updated,
                new_state=new_state,
                audit=audit,
                dedupe_key=key,
            )
            if not ok:
                raise ObjectiveServiceError("objective version is stale")
            return new_state

        return await self._db_write(_body)

    async def reduce_position_exposure(
        self,
        *,
        client_order_id: str | None = None,
        contract_id: str | None = None,
        closed_quantity: int,
        realised_pnl_delta_usd: Decimal | float = 0,
        dedupe_key: str | None = None,
        open_position_count: int | None = None,
        final_close: bool = False,
    ) -> SessionObjectiveState:
        """Release proportional filled cost-basis exposure on reduce/close."""

        def _body() -> tuple[str, Any]:
            state = self._latest_state_sync()
            key = dedupe_key or (
                f"reduce:{client_order_id or contract_id}:{closed_quantity}:"
                f"{realised_pnl_delta_usd}:{final_close}"
            )
            if self._repo.has_projection_dedupe(key):
                return ("state", state)

            existing: CapitalExposure | None = None
            if client_order_id:
                existing = self._repo.get_exposure_by_client_order(client_order_id)
            if existing is None and contract_id:
                for exp in self._repo.list_encumbering_exposures(state.objective_id):
                    if exp.contract_id == contract_id and exp.filled_exposure_usd > 0:
                        existing = exp
                        break
            if existing is None:
                # No exposure row — realised PnL is still recorded, but via the
                # recompute path outside this transaction (see below).
                return (
                    "recompute",
                    (
                        state.realised_pnl_usd + Decimal(str(realised_pnl_delta_usd))
                    ).quantize(Decimal("0.01")),
                )

            closed_qty = max(0, int(closed_quantity))
            if existing.filled_quantity <= 0 or existing.average_fill_price is None:
                release = (
                    existing.filled_exposure_usd
                    if final_close
                    else Decimal("0.00")
                )
                remaining_qty = 0 if final_close else existing.filled_quantity
            else:
                release = premium_notional_usd(existing.average_fill_price, closed_qty)
                release = min(release, existing.filled_exposure_usd)
                remaining_qty = max(0, existing.filled_quantity - closed_qty)
                if final_close:
                    release = existing.filled_exposure_usd
                    remaining_qty = 0
            remaining_filled = (
                Decimal("0.00")
                if remaining_qty <= 0
                else premium_notional_usd(existing.average_fill_price or 0, remaining_qty)
            )
            status = (
                "closed"
                if remaining_qty <= 0 and existing.working_order_reservation_usd <= 0
                else (
                    "partial"
                    if remaining_qty > 0 and existing.working_order_reservation_usd > 0
                    else (
                        "filled_position_exposure"
                        if remaining_qty > 0
                        else (
                            "working_order_reservation"
                            if existing.working_order_reservation_usd > 0
                            else "closed"
                        )
                    )
                )
            )
            updated = existing.model_copy(
                update={
                    "filled_quantity": remaining_qty,
                    "filled_exposure_usd": remaining_filled,
                    "status": status,
                    "updated_at": datetime.now(timezone.utc),
                    "objective_state_version": state.version,
                }
            )
            definition = self._repo.get_definition(state.objective_id)
            assert definition is not None
            working = Decimal("0.00")
            filled = Decimal("0.00")
            for exp in self._repo.list_encumbering_exposures(state.objective_id):
                if exp.client_order_id == existing.client_order_id:
                    working += updated.working_order_reservation_usd
                    filled += updated.filled_exposure_usd
                else:
                    working += exp.working_order_reservation_usd
                    filled += exp.filled_exposure_usd
            encumbered = (working + filled).quantize(Decimal("0.01"))
            available = max(
                Decimal("0.00"),
                (definition.authorised_capital_usd - encumbered).quantize(Decimal("0.01")),
            )
            new_realised = (
                state.realised_pnl_usd + Decimal(str(realised_pnl_delta_usd))
            ).quantize(Decimal("0.01"))
            positions = (
                0
                if final_close or remaining_qty <= 0
                else (
                    open_position_count
                    if open_position_count is not None
                    else state.open_position_count
                )
            )
            new_state = self._build_state_from_exposures(
                definition,
                state.model_copy(
                    update={
                        "working_order_reservation_usd": working,
                        "filled_position_exposure_usd": filled,
                        "reserved_capital_usd": encumbered,
                        "available_capital_usd": available,
                        "realised_pnl_usd": new_realised,
                        "open_position_count": positions,
                    }
                ),
                realised_pnl_usd=new_realised,
                open_position_count=positions,
            )
            # Force version bump relative to current state
            new_state = new_state.model_copy(
                update={
                    "working_order_reservation_usd": working.quantize(Decimal("0.01")),
                    "filled_position_exposure_usd": filled.quantize(Decimal("0.01")),
                    "reserved_capital_usd": encumbered,
                    "available_capital_usd": available,
                    "version": state.version + 1,
                    "last_recomputed_at": datetime.now(timezone.utc),
                }
            )
            audit = self._emit(
                ObjectiveOperatorEventType.CAPITAL_RELEASED,
                objective_id=state.objective_id,
                session_id=state.session_id,
                reason_codes=("position_reduce" if not final_close else "position_close",),
                after={
                    "released_usd": str(release),
                    "remaining_filled_usd": str(remaining_filled),
                },
                linked_ids={
                    "client_order_id": existing.client_order_id,
                    **({"contract_id": contract_id} if contract_id else {}),
                },
                persist_audit=False,
            )
            ok = self._repo.atomic_mutate_exposure(
                objective_id=state.objective_id,
                expected_version=state.version,
                exposure=updated,
                new_state=new_state,
                audit=audit,
                dedupe_key=key,
            )
            if not ok:
                raise ObjectiveServiceError("objective version is stale")
            return ("state", new_state)

        kind, payload = await self._db_write(_body)
        if kind == "recompute":
            return await self.recompute_from_truth(
                realised_pnl_usd=payload,
                open_position_count=open_position_count,
            )
        return payload

    async def associate_broker_order(
        self, *, client_order_id: str, broker_order_id: str
    ) -> None:
        def _body() -> None:
            updated = self._repo.atomic_associate_broker_order(
                client_order_id=client_order_id,
                broker_order_id=broker_order_id,
            )
            if updated is None:
                return
            definition = self._repo.get_definition(updated.objective_id)
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
                    objective_confirmed_at_exchange_time=(
                        definition.objective_confirmed_at_exchange_time
                    ),
                    objective_duration_seconds=definition.objective_duration_seconds,
                    definition_version=definition.definition_version,
                    armed=definition.armed,
                    first_broker_submission_at=datetime.now(timezone.utc),
                )
                self._repo.save_definition(armed)
                self._broker_submission_seen = True

        await self._db_write(_body)

    async def record_verified_outcome(
        self,
        *,
        client_order_id: str | None = None,
        realised_pnl_delta_usd: Decimal | float = 0,
        convert_reservation: bool = False,
        partial_reserved_usd: Decimal | float | None = None,
        open_position_count: int | None = None,
        fill_quantity: int | None = None,
        fill_price: Decimal | float | None = None,
        remaining_working_quantity: int | None = None,
        dedupe_key: str | None = None,
        closed_quantity: int | None = None,
        final_close: bool = False,
        contract_id: str | None = None,
    ) -> SessionObjectiveState:
        """Compatibility facade over fill/reduce/close exposure transitions."""
        if convert_reservation and client_order_id and fill_quantity and fill_price is not None:
            return await self.apply_verified_fill(
                client_order_id=client_order_id,
                fill_quantity=fill_quantity,
                fill_price=fill_price,
                remaining_working_quantity=remaining_working_quantity,
                dedupe_key=dedupe_key,
                contract_id=contract_id,
                open_position_count=open_position_count,
            )
        if convert_reservation and client_order_id and partial_reserved_usd is not None:
            # Legacy partial path: interpret partial_reserved as remaining working.
            existing = await self._db_read(
                lambda: self._repo.get_exposure_by_client_order(client_order_id)
            )
            if existing is None:
                return await self.get_state()
            rem = Decimal(str(partial_reserved_usd))
            filled = max(
                Decimal("0.00"),
                existing.total_encumbered_usd - rem,
            )
            per = existing.estimated_premium_per_contract_usd
            filled_qty = (
                int((filled / (per * Decimal("100"))).to_integral_value())
                if per > 0
                else 0
            )
            return await self.apply_verified_fill(
                client_order_id=client_order_id,
                fill_quantity=max(1, filled_qty),
                fill_price=per,
                remaining_working_quantity=max(
                    0,
                    int((rem / (per * Decimal("100"))).to_integral_value())
                    if per > 0
                    else 0,
                ),
                dedupe_key=dedupe_key,
                open_position_count=open_position_count,
            )
        if convert_reservation and client_order_id and fill_price is None:
            # Full convert without explicit fill details — use reserved premium.
            existing = await self._db_read(
                lambda: self._repo.get_exposure_by_client_order(client_order_id)
            )
            if existing is None:
                return await self.get_state()
            qty = max(1, existing.working_quantity or existing.requested_quantity)
            return await self.apply_verified_fill(
                client_order_id=client_order_id,
                fill_quantity=qty,
                fill_price=existing.estimated_premium_per_contract_usd,
                remaining_working_quantity=0,
                dedupe_key=dedupe_key,
                open_position_count=open_position_count,
            )
        if closed_quantity is not None or final_close:
            return await self.reduce_position_exposure(
                client_order_id=client_order_id,
                contract_id=contract_id,
                closed_quantity=int(closed_quantity or 0),
                realised_pnl_delta_usd=realised_pnl_delta_usd,
                dedupe_key=dedupe_key,
                open_position_count=open_position_count,
                final_close=final_close,
            )
        state = await self.get_state()
        new_realised = (
            state.realised_pnl_usd + Decimal(str(realised_pnl_delta_usd))
        ).quantize(Decimal("0.01"))
        return await self.recompute_from_truth(
            realised_pnl_usd=new_realised,
            open_position_count=open_position_count,
        )

    def _append_state_with_audit(
        self,
        *,
        base: SessionObjectiveState,
        update: dict[str, Any],
        event_type: ObjectiveOperatorEventType,
        reason_codes: tuple[str, ...] = (),
        after: dict[str, Any] | None = None,
    ) -> SessionObjectiveState:
        """Append the next state version and its audit row in one transaction.

        Callers must already hold the write lock. The next version is allocated
        inside the transaction so concurrent writers cannot collide on
        ``UNIQUE (objective_id, version)``.
        """

        def _build(
            next_version: int, prev: SessionObjectiveState | None
        ) -> SessionObjectiveState:
            source = prev or base
            return source.model_copy(
                update={
                    **update,
                    "version": next_version,
                    "last_recomputed_at": datetime.now(timezone.utc),
                }
            )

        payload_after = dict(after or {})
        payload_after.setdefault("version", base.version + 1)
        audit = self._emit(
            event_type,
            objective_id=base.objective_id,
            session_id=base.session_id,
            reason_codes=reason_codes,
            after=payload_after,
            persist_audit=False,
        )
        return self._repo.append_next_state_atomic(
            objective_id=base.objective_id,
            build_state=_build,
            audit=audit,
        )

    async def pause_entries(self, *, reason: str = "paused") -> SessionObjectiveState:
        def _body() -> SessionObjectiveState:
            state = self._latest_state_sync()
            return self._append_state_with_audit(
                base=state,
                update={
                    "entries_paused": True,
                    "status": "paused" if state.status == "active" else state.status,
                },
                event_type=ObjectiveOperatorEventType.PAUSED,
                reason_codes=(reason,),
            )

        return await self._db_write(_body)

    async def resume_entries(self, *, reason: str = "resumed") -> SessionObjectiveState:
        def _body() -> SessionObjectiveState:
            state = self._latest_state_sync()
            if state.status in {
                "deadline_reached",
                "target_reached",
                "capital_exhausted",
                "truth_degraded",
            }:
                raise ObjectiveServiceError(
                    f"cannot resume from terminal status {state.status}"
                )
            return self._append_state_with_audit(
                base=state,
                update={"entries_paused": False, "status": "active"},
                event_type=ObjectiveOperatorEventType.RESUMED,
                reason_codes=(reason,),
            )

        return await self._db_write(_body)

    async def update_feasibility(
        self,
        *,
        classification: str,
        estimated_success_probability: Decimal | None,
    ) -> SessionObjectiveState:
        def _body() -> SessionObjectiveState:
            state = self._latest_state_sync()
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
            return self._append_state_with_audit(
                base=state,
                update={
                    "feasibility_classification": classification,
                    "estimated_success_probability": estimated_success_probability,
                    "current_stance": stance,
                    "status": status,
                },
                event_type=ObjectiveOperatorEventType.FEASIBILITY_ASSESSED,
                after={
                    "classification": classification,
                    "p": str(estimated_success_probability)
                    if estimated_success_probability is not None
                    else None,
                },
            )

        return await self._db_write(_body)

    # ------------------------------------------------------------------
    # Public persistence API (graph nodes must not touch _repo)
    # ------------------------------------------------------------------

    async def save_feasibility(self, assessment: Any) -> None:
        def _body() -> None:
            try:
                self._repo.save_feasibility(assessment)
            except ObjectivePersistenceBusyError as exc:
                raise ObjectiveServiceError(str(exc)) from exc

        await self._db_write(_body)

    async def save_strategy_estimate(self, estimate: Any) -> None:
        def _body() -> None:
            try:
                self._repo.save_strategy_estimate(estimate)
            except ObjectivePersistenceBusyError as exc:
                raise ObjectiveServiceError(str(exc)) from exc

        await self._db_write(_body)

    async def save_strategy_score(self, score: Any) -> None:
        def _body() -> None:
            try:
                self._repo.save_strategy_score(score)
            except ObjectivePersistenceBusyError as exc:
                raise ObjectiveServiceError(str(exc)) from exc

        await self._db_write(_body)

    async def get_strategy_estimate(self, estimate_id: UUID | str) -> Any | None:
        return await self._db_read(
            lambda: self._repo.get_strategy_estimate(estimate_id)
        )

    async def get_latest_estimate_for_strategy(
        self, *, strategy_id: UUID | str, objective_id: UUID | str
    ) -> Any | None:
        return await self._db_read(
            lambda: self._repo.get_latest_estimate_for_strategy(
                strategy_id=strategy_id, objective_id=objective_id
            )
        )

    async def get_historical_summary(self, summary_id: UUID | str) -> Any | None:
        return await self._db_read(
            lambda: self._repo.get_historical_summary(summary_id)
        )

    async def get_latest_historical_summary_for_strategy(
        self, *, strategy_id: UUID | str, snapshot_id: UUID | str
    ) -> Any | None:
        return await self._db_read(
            lambda: self._repo.get_latest_historical_summary_for_strategy(
                strategy_id=strategy_id, snapshot_id=snapshot_id
            )
        )

    async def mark_insufficient_historical_evidence(
        self, *, sample_count: int, minimum_required: int
    ) -> SessionObjectiveState:
        """Cold-start: keep observing; block ENTRY/PROBE/ADD until evidence exists."""

        def _body() -> SessionObjectiveState:
            state = self._latest_state_sync()
            if state.status in {
                "pending_confirmation",
                "deadline_reached",
                "stopped_by_user",
                "truth_degraded",
                "target_reached",
                "capital_exhausted",
            }:
                return state
            return self._append_state_with_audit(
                base=state,
                update={"status": "insufficient_historical_evidence"},
                event_type=ObjectiveOperatorEventType.FEASIBILITY_ASSESSED,
                reason_codes=("insufficient_historical_evidence",),
                after={
                    "sample_count": sample_count,
                    "minimum_required": minimum_required,
                },
            )

        return await self._db_write(_body)
