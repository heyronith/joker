"""Authoritative broker ↔ local projection reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Protocol

from joker.broker.account_truth import BrokerAccountTruth
from joker.broker.interface import BrokerClient
from joker.persistence.broker_submission_journal import (
    BrokerSubmissionRecord,
    SyncBrokerSubmissionJournal,
)
from joker.schemas.domain import BrokerOrder, Position

MismatchKind = Literal[
    "broker_order_missing_locally",
    "local_order_missing_at_broker",
    "broker_position_missing_locally",
    "local_position_missing_at_broker",
    "quantity_mismatch",
    "status_mismatch",
    "price_mismatch",
    "account_mismatch",
    "submission_unknown",
]


@dataclass(frozen=True)
class ReconciliationFinding:
    kind: MismatchKind
    severity: Literal["info", "warning", "critical"]
    client_order_id: str | None = None
    contract_id: str | None = None
    detail: str = ""
    broker_value: str | None = None
    local_value: str | None = None


@dataclass(frozen=True)
class ReconciliationReport:
    captured_at: datetime
    account_id_hash: str
    findings: tuple[ReconciliationFinding, ...]
    degraded: bool
    entries_blocked: bool
    unknown_submissions: int

    @property
    def clean(self) -> bool:
        return not self.findings and not self.degraded


class LocalOrderProjection(Protocol):
    def list_working_orders(self) -> list[Any]:
        ...

    def list_open_positions(self) -> list[Any]:
        ...


@dataclass
class BrokerReconciliationService:
    """Compare broker truth with local projections and submission journal."""

    broker: BrokerClient
    journal: SyncBrokerSubmissionJournal | None = None
    account_id_hash: str = ""
    findings_log: list[ReconciliationFinding] = field(default_factory=list)

    def reconcile(
        self,
        *,
        local_orders: list[BrokerOrder] | list[Any] | None = None,
        local_positions: list[Position] | list[Any] | None = None,
        account_truth: BrokerAccountTruth | None = None,
    ) -> ReconciliationReport:
        now = datetime.now(timezone.utc)
        findings: list[ReconciliationFinding] = []

        broker_orders = self.broker.list_open_orders()
        broker_positions = self.broker.list_positions()
        local_orders = list(local_orders or [])
        local_positions = list(local_positions or [])

        broker_order_ids = {
            str(getattr(o, "order_id", None) or getattr(o, "client_order_id", ""))
            for o in broker_orders
        }
        local_order_ids = {
            str(getattr(o, "order_id", None) or getattr(o, "client_order_id", ""))
            for o in local_orders
        }
        for oid in broker_order_ids - local_order_ids:
            if not oid:
                continue
            findings.append(
                ReconciliationFinding(
                    kind="broker_order_missing_locally",
                    severity="critical",
                    client_order_id=oid,
                    detail="Broker working order absent from local projection",
                )
            )
        for oid in local_order_ids - broker_order_ids:
            if not oid:
                continue
            findings.append(
                ReconciliationFinding(
                    kind="local_order_missing_at_broker",
                    severity="critical",
                    client_order_id=oid,
                    detail="Local working order absent at broker",
                )
            )

        broker_contracts = {_contract_key(p) for p in broker_positions}
        local_contracts = {_contract_key(p) for p in local_positions}
        for cid in broker_contracts - local_contracts:
            if not cid:
                continue
            findings.append(
                ReconciliationFinding(
                    kind="broker_position_missing_locally",
                    severity="critical",
                    contract_id=cid,
                    detail="Broker position absent locally",
                )
            )
        for cid in local_contracts - broker_contracts:
            if not cid:
                continue
            findings.append(
                ReconciliationFinding(
                    kind="local_position_missing_at_broker",
                    severity="critical",
                    contract_id=cid,
                    detail="Local position absent at broker",
                )
            )

        # Quantity mismatches for shared contracts.
        broker_qty = {_contract_key(p): int(getattr(p, "quantity", 0) or 0) for p in broker_positions}
        local_qty = {_contract_key(p): int(getattr(p, "quantity", 0) or 0) for p in local_positions}
        for cid in broker_qty.keys() & local_qty.keys():
            if broker_qty[cid] != local_qty[cid]:
                findings.append(
                    ReconciliationFinding(
                        kind="quantity_mismatch",
                        severity="critical",
                        contract_id=cid,
                        broker_value=str(broker_qty[cid]),
                        local_value=str(local_qty[cid]),
                    )
                )

        unknown = 0
        if self.journal is not None and self.account_id_hash:
            unknowns = self.journal.list_by_status(
                "submission_unknown", account_id_hash=self.account_id_hash
            )
            unknown = len(unknowns)
            for rec in unknowns:
                findings.append(
                    ReconciliationFinding(
                        kind="submission_unknown",
                        severity="critical",
                        client_order_id=rec.client_order_id,
                        detail="Ambiguous submission awaiting order-detail reconciliation",
                    )
                )
                self._try_resolve_unknown(rec)

        if account_truth is not None and self.account_id_hash:
            if account_truth.account_id_hash != self.account_id_hash:
                findings.append(
                    ReconciliationFinding(
                        kind="account_mismatch",
                        severity="critical",
                        detail="Account truth hash does not match configured live account",
                        broker_value=account_truth.account_id_hash,
                        local_value=self.account_id_hash,
                    )
                )

        critical = any(f.severity == "critical" for f in findings)
        self.findings_log.extend(findings)
        return ReconciliationReport(
            captured_at=now,
            account_id_hash=self.account_id_hash,
            findings=tuple(findings),
            degraded=critical,
            entries_blocked=critical,
            unknown_submissions=unknown,
        )

    def _try_resolve_unknown(self, rec: BrokerSubmissionRecord) -> None:
        get_order = getattr(self.broker, "get_order", None)
        if not callable(get_order) or self.journal is None:
            return
        order = get_order(rec.client_order_id)
        if order is None:
            return
        status = str(getattr(order, "status", "") or "")
        mapped = {
            "filled": "filled",
            "cancelled": "cancelled",
            "rejected": "rejected",
            "open": "accepted",
            "pending": "accepted",
        }.get(status)
        if mapped:
            self.journal.transition(
                account_id_hash=rec.account_id_hash,
                client_order_id=rec.client_order_id,
                status=mapped,  # type: ignore[arg-type]
                broker_order_id=str(getattr(order, "order_id", rec.client_order_id)),
            )


def _contract_key(pos: Any) -> str:
    contract = getattr(pos, "contract", None)
    if contract is None:
        return str(getattr(pos, "contract_id", "") or "")
    symbol = getattr(contract, "symbol", "SPY")
    expiration = getattr(contract, "expiration", None)
    strike = getattr(contract, "strike", None)
    option_type = getattr(contract, "option_type", "call")
    exp = expiration.isoformat() if hasattr(expiration, "isoformat") else str(expiration)
    return f"{symbol}:{exp}:{strike}:{option_type}"


def capital_reservation_release_allowed(status: str) -> bool:
    """Capital may release only after confirmed terminal broker outcomes."""
    return status in {"rejected", "cancelled", "reconciled"}
