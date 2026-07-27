"""Durable adversarial recovery checkpoints for crash-resume proofs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

_ADVERSARIAL_RECOVERY_DDL = """
CREATE TABLE IF NOT EXISTS adversarial_recovery_checkpoints (
  checkpoint_key TEXT PRIMARY KEY NOT NULL,
  experiment_id TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  scenario_version TEXT NOT NULL,
  configuration_version_id TEXT NOT NULL,
  sample_number INTEGER NOT NULL,
  crash_point TEXT,
  graph_thread_ids_json TEXT NOT NULL,
  cash TEXT NOT NULL,
  submitted_keys_json TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


class AdversarialRecoveryCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint_key: str
    experiment_id: UUID
    scenario_id: str
    scenario_version: str
    configuration_version_id: UUID
    sample_number: int
    crash_point: str | None = None
    graph_thread_ids: tuple[str, ...] = ()
    cash: str = "25000"
    submitted_keys: tuple[str, ...] = ()
    order_ids: tuple[str, ...] = ()
    fill_ids: tuple[str, ...] = ()
    model_call_ids: tuple[str, ...] = ()
    gateway_action_ids: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()
    extra: dict[str, Any] = Field(default_factory=dict)


class AdversarialRecoveryStore:
    """Persist adversarial recovery state keyed by experiment/scenario/config/sample."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    @staticmethod
    def checkpoint_key(
        experiment_id: UUID,
        scenario_id: str,
        scenario_version: str,
        configuration_version_id: UUID,
        sample_number: int,
    ) -> str:
        return (
            f"{experiment_id}:{scenario_id}:{scenario_version}:"
            f"{configuration_version_id}:{sample_number}"
        )

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_ADVERSARIAL_RECOVERY_DDL)
            await db.commit()

    async def save(self, checkpoint: AdversarialRecoveryCheckpoint) -> None:
        await self.initialize()
        payload = {
            "order_ids": list(checkpoint.order_ids),
            "fill_ids": list(checkpoint.fill_ids),
            "model_call_ids": list(checkpoint.model_call_ids),
            "gateway_action_ids": list(checkpoint.gateway_action_ids),
            "findings": list(checkpoint.findings),
            **checkpoint.extra,
        }
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO adversarial_recovery_checkpoints (
                    checkpoint_key, experiment_id, scenario_id, scenario_version,
                    configuration_version_id, sample_number, crash_point,
                    graph_thread_ids_json, cash, submitted_keys_json,
                    payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_key,
                    str(checkpoint.experiment_id),
                    checkpoint.scenario_id,
                    checkpoint.scenario_version,
                    str(checkpoint.configuration_version_id),
                    checkpoint.sample_number,
                    checkpoint.crash_point,
                    json.dumps(list(checkpoint.graph_thread_ids)),
                    checkpoint.cash,
                    json.dumps(list(checkpoint.submitted_keys)),
                    json.dumps(payload, default=str),
                    now,
                ),
            )
            await db.commit()

    async def load(self, checkpoint_key: str) -> AdversarialRecoveryCheckpoint | None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                SELECT checkpoint_key, experiment_id, scenario_id, scenario_version,
                       configuration_version_id, sample_number, crash_point,
                       graph_thread_ids_json, cash, submitted_keys_json, payload_json
                FROM adversarial_recovery_checkpoints
                WHERE checkpoint_key = ?
                """,
                (checkpoint_key,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        payload = json.loads(row[10])
        return AdversarialRecoveryCheckpoint(
            checkpoint_key=row[0],
            experiment_id=UUID(row[1]),
            scenario_id=row[2],
            scenario_version=row[3],
            configuration_version_id=UUID(row[4]),
            sample_number=int(row[5]),
            crash_point=row[6],
            graph_thread_ids=tuple(json.loads(row[7])),
            cash=row[8],
            submitted_keys=tuple(json.loads(row[9])),
            order_ids=tuple(payload.get("order_ids") or ()),
            fill_ids=tuple(payload.get("fill_ids") or ()),
            model_call_ids=tuple(payload.get("model_call_ids") or ()),
            gateway_action_ids=tuple(payload.get("gateway_action_ids") or ()),
            findings=tuple(payload.get("findings") or ()),
            extra={
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "order_ids",
                    "fill_ids",
                    "model_call_ids",
                    "gateway_action_ids",
                    "findings",
                }
            },
        )
