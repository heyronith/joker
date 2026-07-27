"""Durable per-episode/sample experiment results for exact-once resume."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite

from joker.evolution.hashing import stable_json_dumps
from joker.evolution.migrations import apply_task3_migrations

_EPISODE_RESULT_SQL = """
CREATE TABLE IF NOT EXISTS experiment_episode_results (
    idempotency_key TEXT PRIMARY KEY NOT NULL,
    experiment_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    configuration_version_id TEXT NOT NULL,
    sample_number INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exp_episode_exp
    ON experiment_episode_results (experiment_id, episode_id, sample_number);
"""


class ExperimentEpisodeResultStore:
    """Persist every experiment episode/sample result by idempotency key."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        apply_task3_migrations(self._db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_EPISODE_RESULT_SQL)
            await db.commit()
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    async def has_key(self, idempotency_key: str) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                "SELECT 1 FROM experiment_episode_results WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            return await cur.fetchone() is not None

    async def list_keys(self, experiment_id: UUID | str) -> set[str]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                "SELECT idempotency_key FROM experiment_episode_results WHERE experiment_id = ?",
                (str(experiment_id),),
            )
            rows = await cur.fetchall()
        return {r[0] for r in rows}

    async def append(
        self,
        *,
        idempotency_key: str,
        experiment_id: UUID | str,
        episode_id: UUID | str,
        configuration_version_id: UUID | str,
        sample_number: int,
        payload: dict[str, Any],
    ) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            try:
                await db.execute(
                    """
                    INSERT INTO experiment_episode_results (
                        idempotency_key, experiment_id, episode_id,
                        configuration_version_id, sample_number, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        idempotency_key,
                        str(experiment_id),
                        str(episode_id),
                        str(configuration_version_id),
                        sample_number,
                        stable_json_dumps(payload),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def get_payload(self, idempotency_key: str) -> dict[str, Any] | None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                "SELECT payload_json FROM experiment_episode_results WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            row = await cur.fetchone()
        return json.loads(row[0]) if row else None

    async def list_payloads_for_configuration(
        self,
        experiment_id: UUID | str,
        configuration_version_id: UUID | str,
    ) -> list[dict[str, Any]]:
        """Return payloads for one configuration — keys are content hashes, not IDs."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                SELECT payload_json FROM experiment_episode_results
                WHERE experiment_id = ? AND configuration_version_id = ?
                """,
                (str(experiment_id), str(configuration_version_id)),
            )
            rows = await cur.fetchall()
        return [json.loads(r[0]) for r in rows]
