"""Shadow evaluation runtime — challenger has no broker / ExecutionRuntime access."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from joker.evolution.repositories import ShadowAssignmentRepository
from joker.evolution.schemas import CognitiveConfigurationVersion, ShadowAssignment


class ShadowIsolationError(RuntimeError):
    """Raised if shadow path attempts broker or execution access."""


@dataclass
class ShadowCycleResult:
    assignment_id: UUID
    challenger_version_id: UUID
    snapshot_id: str | None
    hypothetical_command: dict[str, Any]
    created_at: datetime


@dataclass
class ShadowRuntime:
    """Bounded-queue shadow worker. Never submits orders."""

    assignment_repo: ShadowAssignmentRepository
    queue_size: int = 128
    _queue: deque[dict[str, Any]] = field(default_factory=deque)
    _worker: asyncio.Task[None] | None = None
    _stopped: bool = True
    _results: list[ShadowCycleResult] = field(default_factory=list)

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
        assignment = ShadowAssignment(
            assignment_id=uuid4(),
            challenger_version_id=challenger.configuration_version_id,
            champion_version_id=champion.configuration_version_id,
            status="active",
        )
        await self.assignment_repo.append(assignment)
        return assignment

    async def enqueue_snapshot(
        self,
        *,
        assignment_id: UUID,
        challenger_version_id: UUID,
        snapshot_id: str,
        payload: dict[str, Any],
        coalesce: bool = True,
    ) -> bool:
        if len(self._queue) >= self.queue_size:
            return False
        if coalesce:
            # Drop older entry snapshots for same assignment (not position events).
            self._queue = deque(
                item
                for item in self._queue
                if not (
                    item.get("assignment_id") == str(assignment_id)
                    and item.get("kind") == "snapshot"
                )
            )
        self._queue.append(
            {
                "kind": "snapshot",
                "assignment_id": str(assignment_id),
                "challenger_version_id": str(challenger_version_id),
                "snapshot_id": snapshot_id,
                "payload": payload,
            }
        )
        return True

    def forbid_execution_runtime(self) -> None:
        raise ShadowIsolationError(
            "shadow challenger cannot access ExecutionRuntime or broker submission"
        )

    async def _loop(self) -> None:
        while not self._stopped:
            if not self._queue:
                await asyncio.sleep(0.01)
                continue
            item = self._queue.popleft()
            await self._run_shadow_cycle(item)

    async def _run_shadow_cycle(self, item: dict[str, Any]) -> None:
        # Hypothetical command only — never broker.
        command = {
            "action": "hypothetical_entry",
            "snapshot_id": item.get("snapshot_id"),
            "shadow": True,
            "payload_keys": sorted(item.get("payload", {}).keys()),
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
        self._results.append(
            ShadowCycleResult(
                assignment_id=UUID(item["assignment_id"]),
                challenger_version_id=UUID(item["challenger_version_id"]),
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
