"""Material objective-truth fingerprints for optimization and submission."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from joker.objectives.schemas import SessionObjectiveState


@dataclass(frozen=True)
class ObjectiveDecisionFingerprint:
    objective_id: str
    available_capital_usd: Decimal
    reserved_capital_usd: Decimal
    working_order_reservation_usd: Decimal
    filled_position_exposure_usd: Decimal
    remaining_profit_gap_usd: Decimal
    realised_pnl_usd: Decimal
    deadline_exchange_time: str | None
    time_remaining_seconds: int
    deadline_reached: bool
    target_reached: bool
    entries_paused: bool
    truth_degraded: bool
    open_position_count: int
    working_order_count: int
    max_concurrent_positions: int
    broker_identity: str
    broker_eligible: bool
    reconciliation_eligible: bool

    @classmethod
    def from_state(
        cls,
        state: SessionObjectiveState,
        *,
        working_order_count: int,
        broker_identity: str,
        broker_eligible: bool,
        reconciliation_eligible: bool,
    ) -> ObjectiveDecisionFingerprint:
        return cls(
            objective_id=str(state.objective_id),
            available_capital_usd=state.available_capital_usd,
            reserved_capital_usd=state.reserved_capital_usd,
            working_order_reservation_usd=state.working_order_reservation_usd,
            filled_position_exposure_usd=state.filled_position_exposure_usd,
            remaining_profit_gap_usd=state.required_profit_remaining_usd,
            realised_pnl_usd=state.realised_pnl_usd,
            deadline_exchange_time=(
                state.deadline_exchange_time.isoformat()
                if state.deadline_exchange_time is not None
                else None
            ),
            time_remaining_seconds=int(state.time_remaining_seconds),
            deadline_reached=(
                state.status == "deadline_reached" or state.time_remaining_seconds <= 0
            ),
            target_reached=state.status == "target_reached",
            entries_paused=bool(state.entries_paused),
            truth_degraded=bool(state.truth_degraded),
            open_position_count=int(state.open_position_count),
            working_order_count=max(0, int(working_order_count)),
            max_concurrent_positions=int(state.max_concurrent_positions),
            broker_identity=str(broker_identity),
            broker_eligible=bool(broker_eligible),
            reconciliation_eligible=bool(reconciliation_eligible),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "available_capital_usd": str(self.available_capital_usd),
            "reserved_capital_usd": str(self.reserved_capital_usd),
            "working_order_reservation_usd": str(
                self.working_order_reservation_usd
            ),
            "filled_position_exposure_usd": str(
                self.filled_position_exposure_usd
            ),
            "remaining_profit_gap_usd": str(self.remaining_profit_gap_usd),
            "realised_pnl_usd": str(self.realised_pnl_usd),
            "deadline_exchange_time": self.deadline_exchange_time,
            "time_remaining_seconds": self.time_remaining_seconds,
            "deadline_reached": self.deadline_reached,
            "target_reached": self.target_reached,
            "entries_paused": self.entries_paused,
            "truth_degraded": self.truth_degraded,
            "open_position_count": self.open_position_count,
            "working_order_count": self.working_order_count,
            "max_concurrent_positions": self.max_concurrent_positions,
            "broker_identity": self.broker_identity,
            "broker_eligible": self.broker_eligible,
            "reconciliation_eligible": self.reconciliation_eligible,
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, value: str) -> ObjectiveDecisionFingerprint:
        payload = json.loads(value)
        return cls(
            objective_id=str(payload["objective_id"]),
            available_capital_usd=Decimal(payload["available_capital_usd"]),
            reserved_capital_usd=Decimal(payload["reserved_capital_usd"]),
            working_order_reservation_usd=Decimal(
                payload["working_order_reservation_usd"]
            ),
            filled_position_exposure_usd=Decimal(
                payload["filled_position_exposure_usd"]
            ),
            remaining_profit_gap_usd=Decimal(payload["remaining_profit_gap_usd"]),
            realised_pnl_usd=Decimal(payload["realised_pnl_usd"]),
            deadline_exchange_time=payload["deadline_exchange_time"],
            time_remaining_seconds=int(payload["time_remaining_seconds"]),
            deadline_reached=bool(payload["deadline_reached"]),
            target_reached=bool(payload["target_reached"]),
            entries_paused=bool(payload["entries_paused"]),
            truth_degraded=bool(payload["truth_degraded"]),
            open_position_count=int(payload["open_position_count"]),
            working_order_count=int(payload["working_order_count"]),
            max_concurrent_positions=int(payload["max_concurrent_positions"]),
            broker_identity=str(payload["broker_identity"]),
            broker_eligible=bool(payload["broker_eligible"]),
            reconciliation_eligible=bool(payload["reconciliation_eligible"]),
        )

    def material_differences(
        self,
        other: ObjectiveDecisionFingerprint,
        *,
        maximum_time_decay_seconds: int = 1,
    ) -> tuple[str, ...]:
        differences: list[str] = []
        left = self.as_dict()
        right = other.as_dict()
        for field_name in left:
            if field_name == "time_remaining_seconds":
                decay = self.time_remaining_seconds - other.time_remaining_seconds
                if decay < 0 or decay > maximum_time_decay_seconds:
                    differences.append(field_name)
                continue
            if left[field_name] != right[field_name]:
                differences.append(field_name)
        return tuple(differences)
