"""Task 1 event horizon loader for episode compilation and replay truth."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from joker.evolution.session_event_index import (
    SessionEventIndexRecord,
    SessionEventIndexRepository,
)


class Task1HorizonEvent(BaseModel):
    """One event in an ordered Task 1 horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    event_type: str
    exchange_timestamp: datetime
    sequence: int | None = None
    correlation_id: UUID | None = None
    cycle_id: str | None = None
    snapshot_id: UUID | None = None
    data_quality_id: UUID | None = None
    option_surface_id: UUID | None = None
    client_order_id: str | None = None
    contract_id: str | None = None
    position_lifecycle_id: str | None = None


class Task1EventHorizon(BaseModel):
    """Ordered horizon of Task 1 domain events between decision and terminal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    events: tuple[Task1HorizonEvent, ...] = ()
    market_event_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    snapshot_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    data_quality_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    option_surface_ids: tuple[UUID, ...] = Field(default_factory=tuple)


def _parse_uuid(value: str | None) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(str(value))
    except Exception:
        return None


class Task1EventHorizonLoader:
    """Load ordered event horizons from session_event_index + Task 1 repos."""

    def __init__(
        self,
        *,
        index_repo: SessionEventIndexRepository,
        snapshot_repo: Any | None = None,
        data_quality_repo: Any | None = None,
        option_surface_repo: Any | None = None,
    ) -> None:
        self._index = index_repo
        self._snapshots = snapshot_repo
        self._dq = data_quality_repo
        self._surfaces = option_surface_repo

    async def load(
        self,
        *,
        session_id: str,
        start_timestamp: datetime,
        end_timestamp: datetime,
        entry_decision_event_id: UUID | None = None,
        terminal_event_id: UUID | None = None,
    ) -> Task1EventHorizon:
        """Return ordered events for the entry→terminal horizon.

        Prefer the contiguous session sequence range between entry and terminal
        when both anchors resolve with sequences; otherwise fall back to the
        exchange-timestamp window (still merging explicit anchors).
        """
        if end_timestamp < start_timestamp:
            return Task1EventHorizon(session_id=session_id)

        entry_rec = None
        term_rec = None
        if entry_decision_event_id is not None:
            entry_rec = await self._index.get_by_event_id(str(entry_decision_event_id))
        if terminal_event_id is not None:
            term_rec = await self._index.get_by_event_id(str(terminal_event_id))

        records: list[SessionEventIndexRecord]
        if (
            entry_rec is not None
            and term_rec is not None
            and entry_rec.sequence is not None
            and term_rec.sequence is not None
        ):
            records = await self._index.list_sequence_range(
                session_id,
                start_sequence=int(entry_rec.sequence),
                end_sequence=int(term_rec.sequence),
            )
        else:
            records = await self._index.list_horizon(
                session_id,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
            if entry_rec is not None and not any(
                r.event_id == entry_rec.event_id for r in records
            ):
                records = [entry_rec, *records]
            if term_rec is not None and not any(
                r.event_id == term_rec.event_id for r in records
            ):
                records = [*records, term_rec]

        records = _sort_records(records)
        events: list[Task1HorizonEvent] = []
        market_ids: list[UUID] = []
        snapshot_ids: list[UUID] = []
        dq_ids: list[UUID] = []
        surface_ids: list[UUID] = []

        for rec in records:
            eid = _parse_uuid(rec.event_id)
            if eid is None:
                continue
            snap_id = _parse_uuid(rec.snapshot_id)
            dq_id = _parse_uuid(rec.data_quality_id)
            surface_id = _parse_uuid(rec.option_surface_id)

            if snap_id is None and self._snapshots is not None:
                snap_id = await self._resolve_snapshot_from_payload(rec)

            events.append(
                Task1HorizonEvent(
                    event_id=eid,
                    event_type=rec.event_type,
                    exchange_timestamp=rec.exchange_timestamp,
                    sequence=rec.sequence,
                    correlation_id=_parse_uuid(rec.correlation_id),
                    cycle_id=rec.cycle_id,
                    snapshot_id=snap_id,
                    data_quality_id=dq_id,
                    option_surface_id=surface_id,
                    client_order_id=rec.client_order_id,
                    contract_id=rec.contract_id,
                    position_lifecycle_id=rec.position_lifecycle_id,
                )
            )
            market_ids.append(eid)
            if snap_id is not None and snap_id not in snapshot_ids:
                snapshot_ids.append(snap_id)
            if dq_id is not None and dq_id not in dq_ids:
                dq_ids.append(dq_id)
            if surface_id is not None and surface_id not in surface_ids:
                surface_ids.append(surface_id)

        return Task1EventHorizon(
            session_id=session_id,
            events=tuple(events),
            market_event_ids=tuple(dict.fromkeys(market_ids)),
            snapshot_ids=tuple(snapshot_ids),
            data_quality_ids=tuple(dq_ids),
            option_surface_ids=tuple(surface_ids),
        )

    async def _resolve_snapshot_from_payload(
        self, rec: SessionEventIndexRecord
    ) -> UUID | None:
        payload_snap = _parse_uuid(rec.snapshot_id)
        if payload_snap is not None:
            return payload_snap
        if self._snapshots is None:
            return None
        get_fn = getattr(self._snapshots, "get_by_id", None)
        if get_fn is None:
            return None
        return None


def _sort_records(
    records: list[SessionEventIndexRecord],
) -> list[SessionEventIndexRecord]:
    return sorted(
        records,
        key=lambda r: (
            r.sequence is None,
            r.sequence if r.sequence is not None else 0,
            r.exchange_timestamp,
            r.event_id,
        ),
    )
