"""Shadow evaluation runtime — runs challenger cognitive graph without broker access."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.repositories import ShadowAssignmentRepository
from joker.evolution.schemas import CognitiveConfigurationVersion, ShadowAssignment


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
    """Bounded-queue shadow worker. Never submits orders."""

    assignment_repo: ShadowAssignmentRepository
    policy_store: PolicyVersionStore | None = None
    queue_size: int = 128
    challenger_runner: ChallengerRunner | None = None
    _queue: deque[dict[str, Any]] = field(default_factory=deque)
    _worker: asyncio.Task[None] | None = None
    _stopped: bool = True
    _results: list[ShadowCycleResult] = field(default_factory=list)
    _configs: dict[UUID, CognitiveConfigurationVersion] = field(default_factory=dict)

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
    ) -> bool:
        if len(self._queue) >= self.queue_size:
            return False
        if coalesce:
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
        challenger_id = UUID(item["challenger_version_id"])
        challenger = self._configs.get(challenger_id)
        command: dict[str, Any]
        if self.challenger_runner is not None and challenger is not None:
            # Real challenger cognitive execution — must not touch broker.
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
        self._results.append(
            ShadowCycleResult(
                assignment_id=UUID(item["assignment_id"]),
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
