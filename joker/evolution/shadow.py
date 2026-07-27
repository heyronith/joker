"""Shadow evaluation runtime — runs challenger cognitive graph without broker access."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.repositories import (
    ConfigurationVersionRepository,
    ShadowAssignmentRepository,
)
from joker.evolution.schemas import CognitiveConfigurationVersion, ShadowAssignment
from joker.evolution.shadow_ledger import ShadowLedger


class ShadowIsolationError(RuntimeError):
    """Raised if shadow path attempts broker or execution access."""


ChallengerRunner = Callable[
    [CognitiveConfigurationVersion, dict[str, Any]], Awaitable[dict[str, Any]]
]


@dataclass
class ShadowCycleResult:
    assignment_id: UUID
    challenger_version_id: UUID
    snapshot_id: str | None
    hypothetical_command: dict[str, Any]
    created_at: datetime


@dataclass
class ShadowRuntime:
    """Bounded-queue shadow worker with durable ledger recovery."""

    assignment_repo: ShadowAssignmentRepository
    policy_store: PolicyVersionStore | None = None
    queue_size: int = 128
    challenger_runner: ChallengerRunner | None = None
    ledger: ShadowLedger | None = None
    config_repo: ConfigurationVersionRepository | None = None
    replay_service: Any | None = None
    _queue: deque[dict[str, Any]] = field(default_factory=deque)
    _worker: asyncio.Task[None] | None = None
    _stopped: bool = True
    _results: list[ShadowCycleResult] = field(default_factory=list)
    _configs: dict[UUID, CognitiveConfigurationVersion] = field(default_factory=dict)
    _cursors: dict[str, str] = field(default_factory=dict)
    _cursor_keys: dict[str, tuple] = field(default_factory=dict)
    _seen_snapshots: dict[str, set[str]] = field(default_factory=dict)
    _restored_runtimes: dict[str, Any] = field(default_factory=dict)

    async def start(self) -> None:
        self._stopped = False
        if self._worker is None:
            self._worker = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopped = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def register_challenger(
        self,
        *,
        challenger: CognitiveConfigurationVersion,
        champion: CognitiveConfigurationVersion,
    ) -> ShadowAssignment:
        if self.policy_store is not None:
            ok, problems = await self.policy_store.verify_configuration_resolvable(
                challenger
            )
            if not ok:
                raise ShadowIsolationError(
                    "challenger not materialisable: " + ", ".join(problems)
                )
        assignment = ShadowAssignment(
            assignment_id=uuid4(),
            challenger_version_id=challenger.configuration_version_id,
            champion_version_id=champion.configuration_version_id,
            status="active",
        )
        await self.assignment_repo.append(assignment)
        self._configs[challenger.configuration_version_id] = challenger
        return assignment

    async def enqueue_snapshot(
        self,
        *,
        assignment_id: UUID,
        challenger_version_id: UUID,
        snapshot_id: str,
        payload: dict[str, Any],
        coalesce: bool = True,
        exchange_timestamp: datetime | None = None,
        event_sequence: int | None = None,
    ) -> bool:
        aid = str(assignment_id)
        # Authoritative monotonic cursor: (exchange_timestamp, event_sequence, snapshot_id).
        # Reject older-or-equal ordering keys; never compare UUID strings lexicographically.
        ts_key = (
            exchange_timestamp.astimezone(timezone.utc)
            if exchange_timestamp is not None
            else None
        )
        seq_key = event_sequence if event_sequence is not None else -1
        ordering_key = (ts_key, seq_key, snapshot_id)
        prior_key = self._cursor_keys.get(aid)
        if prior_key is not None:
            prior_ts, prior_seq, prior_sid = prior_key
            if ts_key is not None and prior_ts is not None:
                if ordering_key <= (prior_ts, prior_seq, prior_sid):
                    return False
            elif ts_key is None and prior_ts is not None:
                # Unseen older snapshot without timestamp cannot outrank timed cursor.
                return False
            elif snapshot_id == prior_sid:
                return False
        cursor = self._cursors.get(aid)
        if cursor is not None and snapshot_id == cursor:
            return False
        seen = self._seen_snapshots.get(aid)
        if seen is not None and snapshot_id in seen:
            return False
        if len(self._queue) >= self.queue_size:
            return False
        if coalesce:
            self._queue = deque(
                item
                for item in self._queue
                if not (
                    item.get("assignment_id") == aid
                    and item.get("kind") == "snapshot"
                )
            )
        self._queue.append(
            {
                "kind": "snapshot",
                "assignment_id": aid,
                "challenger_version_id": str(challenger_version_id),
                "snapshot_id": snapshot_id,
                "payload": {"snapshot_id": snapshot_id},
                "exchange_timestamp": (
                    exchange_timestamp.isoformat() if exchange_timestamp else None
                ),
                "event_sequence": event_sequence,
                "ordering_key": (
                    ts_key.isoformat() if ts_key is not None else None,
                    seq_key,
                    snapshot_id,
                ),
            }
        )
        return True

    def forbid_execution_runtime(self) -> None:
        raise ShadowIsolationError(
            "shadow challenger cannot access ExecutionRuntime or broker submission"
        )

    async def restore_from_ledger(self) -> None:
        """Reload active assignments, configs, and simulated execution after restart."""
        if self.ledger is not None:
            await self.ledger.initialize()
        from joker.evolution.shadow_restore import ShadowExecutionRestorer

        active = await self.assignment_repo.list_active()
        restorer = ShadowExecutionRestorer(self.ledger) if self.ledger else None
        for assignment in active:
            cfg_id = assignment.challenger_version_id
            if cfg_id not in self._configs and self.config_repo is not None:
                cfg = await self.config_repo.get_by_id(cfg_id)
                if cfg is not None:
                    self._configs[cfg_id] = cfg
            if restorer is None:
                continue
            restored = await restorer.restore_assignment(assignment)
            key = f"{cfg_id}:{assignment.assignment_id}"
            if self.replay_service is not None:
                self.replay_service.restore_shadow_runtime(key, restored.position_runtime)
                if restored.last_snapshot_id:
                    self.replay_service.set_shadow_cursor(key, restored.last_snapshot_id)
            else:
                self._restored_runtimes[key] = restored.position_runtime
            if restored.last_snapshot_id:
                self._cursors[str(assignment.assignment_id)] = restored.last_snapshot_id
                self._seen_snapshots.setdefault(str(assignment.assignment_id), set()).add(
                    restored.last_snapshot_id
                )
                cursor_payload = restored.cursor or {}
                ordering = cursor_payload.get("ordering_key")
                if ordering is not None:
                    ts_raw, seq, sid = ordering
                    ts_parsed = (
                        datetime.fromisoformat(ts_raw)
                        if isinstance(ts_raw, str)
                        else ts_raw
                    )
                    self._cursor_keys[str(assignment.assignment_id)] = (
                        ts_parsed,
                        seq,
                        sid,
                    )

    async def _loop(self) -> None:
        while not self._stopped:
            if not self._queue:
                await asyncio.sleep(0.01)
                continue
            item = self._queue.popleft()
            await self._run_shadow_cycle(item)

    async def _run_shadow_cycle(self, item: dict[str, Any]) -> None:
        challenger_id = UUID(item["challenger_version_id"])
        assignment_id = UUID(item["assignment_id"])
        challenger = self._configs.get(challenger_id)
        if challenger is None and self.config_repo is not None:
            challenger = await self.config_repo.get_by_id(challenger_id)
            if challenger is not None:
                self._configs[challenger_id] = challenger
        command: dict[str, Any]
        if self.challenger_runner is not None and challenger is not None:
            command = await self.challenger_runner(challenger, item)
            if command.get("broker_submit") or command.get("execution_runtime"):
                raise ShadowIsolationError("challenger runner attempted live execution")
            command = {**command, "shadow": True}
        else:
            command = {
                "action": "hypothetical_entry",
                "snapshot_id": item.get("snapshot_id"),
                "shadow": True,
                "challenger_version_id": str(challenger_id),
                "configuration_hash": (
                    challenger.content_hash if challenger is not None else None
                ),
                "ran_challenger_graph": False,
            }
        command_id = str(uuid4())
        created = datetime.now(timezone.utc).isoformat()
        await self.assignment_repo.append_hypothetical_command(
            command_id=command_id,
            assignment_id=item["assignment_id"],
            challenger_version_id=item["challenger_version_id"],
            payload=command,
            snapshot_id=item.get("snapshot_id"),
            created_at=created,
        )
        status = "traded" if command.get("traded") else "observed"
        if self.ledger is not None:
            await self.ledger.record_cycle(
                assignment_id=assignment_id,
                challenger_version_id=challenger_id,
                snapshot_id=str(item.get("snapshot_id") or ""),
                status=status,
                payload=command,
            )
            await self.ledger.save_checkpoint(
                assignment_id,
                last_snapshot_id=item.get("snapshot_id"),
                cursor={
                    "last_command_id": command_id,
                    "cash": (command.get("projection") or {}).get("cash"),
                    "realised_pnl": command.get("realised_pnl"),
                    "submitted_keys": list(
                        ((command.get("projection") or {}).get("orders") or {}).keys()
                    ),
                    "ordering_key": item.get("ordering_key"),
                    "exchange_timestamp": item.get("exchange_timestamp"),
                    "event_sequence": item.get("event_sequence"),
                },
            )
            self._cursors[str(assignment_id)] = str(item.get("snapshot_id") or "")
            ordering = item.get("ordering_key")
            if ordering is not None:
                ts_raw, seq, sid = ordering
                ts_parsed = (
                    datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else ts_raw
                )
                self._cursor_keys[str(assignment_id)] = (ts_parsed, seq, sid)
            self._seen_snapshots.setdefault(str(assignment_id), set()).add(
                str(item.get("snapshot_id") or "")
            )
            projection = command.get("projection") or {}
            positions = projection.get("positions") or {}
            for contract_id, pos in positions.items():
                qty = Decimal(str(pos.get("quantity", 0)))
                lifecycle = str(
                    pos.get("position_lifecycle_id")
                    or f"shadow:{assignment_id}:{contract_id}"
                )
                await self.ledger.upsert_position(
                    assignment_id=assignment_id,
                    challenger_version_id=challenger_id,
                    position_lifecycle_id=lifecycle,
                    contract_id=str(contract_id),
                    configuration_version_id=challenger_id,
                    quantity=qty,
                    average_price=Decimal(str(pos.get("avg_price", 0))),
                    realised_pnl=Decimal(str(pos.get("realised_pnl", 0))),
                    status="open" if qty > 0 else "closed",
                    last_snapshot_id=item.get("snapshot_id"),
                    payload=pos if isinstance(pos, dict) else {},
                )
            for oid, order in (projection.get("orders") or {}).items():
                await self.ledger.upsert_order(
                    assignment_id=assignment_id,
                    challenger_version_id=challenger_id,
                    client_order_id=str(oid),
                    contract_id=str(order.get("contract_id") or ""),
                    side=str(order.get("side") or "buy"),
                    quantity=Decimal(str(order.get("filled_qty") or 0)),
                    status=str(order.get("status") or "unknown"),
                    payload=order if isinstance(order, dict) else {},
                )
            for fill in projection.get("fills") or []:
                fill_id = str(fill.get("fill_id") or uuid4())
                await self.ledger.record_fill(
                    fill_id=fill_id,
                    client_order_id=str(fill.get("client_order_id") or fill_id),
                    assignment_id=assignment_id,
                    quantity=Decimal(str(fill.get("qty") or fill.get("quantity") or 0)),
                    price=Decimal(str(fill.get("price") or 0)),
                    fee=Decimal(str(fill.get("fees") or fill.get("fee") or 0)),
                    payload=fill if isinstance(fill, dict) else {},
                )
        self._results.append(
            ShadowCycleResult(
                assignment_id=assignment_id,
                challenger_version_id=challenger_id,
                snapshot_id=item.get("snapshot_id"),
                hypothetical_command=command,
                created_at=datetime.now(timezone.utc),
            )
        )

    @property
    def backlog(self) -> int:
        return len(self._queue)

    @property
    def results(self) -> list[ShadowCycleResult]:
        return list(self._results)
