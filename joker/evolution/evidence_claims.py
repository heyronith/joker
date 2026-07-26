"""Durable evolution evidence ownership claims."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

ClaimStatus = Literal["claimed", "consumed", "released", "invalidated"]

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS evolution_evidence_claims (
    evaluation_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    evolution_cycle_id TEXT NOT NULL,
    dataset_id TEXT,
    claim_status TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    released_at TEXT,
    claim_reason TEXT NOT NULL,
    reuse_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_cycle
    ON evolution_evidence_claims (evolution_cycle_id, claim_status);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_status
    ON evolution_evidence_claims (claim_status, claimed_at);
"""


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_id: UUID
    episode_id: UUID
    evolution_cycle_id: str
    dataset_id: UUID | None = None
    claim_status: ClaimStatus = "claimed"
    claimed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    released_at: datetime | None = None
    claim_reason: str = "automatic_cycle"
    reuse_reason: str | None = None


class EvidenceClaimStore:
    """Transactional ownership of evaluations for evolution cycles."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_SQL)
            await db.commit()

    async def claim_batch(
        self,
        *,
        evolution_cycle_id: str,
        claims: list[EvidenceClaim],
        minimum_count: int,
    ) -> tuple[bool, list[EvidenceClaim]]:
        """Atomically claim evaluations; roll back if minimum not met."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("BEGIN IMMEDIATE")
            inserted: list[EvidenceClaim] = []
            try:
                for claim in claims:
                    try:
                        await db.execute(
                            """
                            INSERT INTO evolution_evidence_claims (
                                evaluation_id, episode_id, evolution_cycle_id,
                                dataset_id, claim_status, claimed_at, released_at,
                                claim_reason, reuse_reason
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(claim.evaluation_id),
                                str(claim.episode_id),
                                claim.evolution_cycle_id,
                                str(claim.dataset_id) if claim.dataset_id else None,
                                claim.claim_status,
                                claim.claimed_at.isoformat(),
                                None,
                                claim.claim_reason,
                                claim.reuse_reason,
                            ),
                        )
                        inserted.append(claim)
                    except aiosqlite.IntegrityError:
                        continue
                if len(inserted) < minimum_count:
                    await db.execute("ROLLBACK")
                    return False, []
                await db.commit()
                return True, inserted
            except Exception:
                await db.execute("ROLLBACK")
                raise

    async def list_unclaimed_evaluation_ids(self) -> set[str]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                SELECT evaluation_id FROM evolution_evidence_claims
                WHERE claim_status IN ('claimed', 'consumed')
                """
            )
            rows = await cur.fetchall()
        return {r[0] for r in rows}

    async def mark_consumed(self, evolution_cycle_id: str) -> None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE evolution_evidence_claims
                SET claim_status = 'consumed'
                WHERE evolution_cycle_id = ? AND claim_status = 'claimed'
                """,
                (evolution_cycle_id,),
            )
            await db.commit()

    async def release_cycle(
        self, evolution_cycle_id: str, *, reason: str = "cycle_failed_pre_dataset"
    ) -> None:
        await self.initialize()
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE evolution_evidence_claims
                SET claim_status = 'released', released_at = ?, claim_reason = ?
                WHERE evolution_cycle_id = ? AND claim_status = 'claimed'
                """,
                (now, reason, evolution_cycle_id),
            )
            await db.commit()

    async def attach_dataset(self, evolution_cycle_id: str, dataset_id: UUID) -> None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE evolution_evidence_claims
                SET dataset_id = ?
                WHERE evolution_cycle_id = ?
                """,
                (str(dataset_id), evolution_cycle_id),
            )
            await db.commit()

    async def list_by_cycle(self, evolution_cycle_id: str) -> list[EvidenceClaim]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM evolution_evidence_claims
                WHERE evolution_cycle_id = ?
                """,
                (evolution_cycle_id,),
            )
            rows = await cur.fetchall()
        out: list[EvidenceClaim] = []
        for row in rows:
            out.append(
                EvidenceClaim(
                    evaluation_id=UUID(row["evaluation_id"]),
                    episode_id=UUID(row["episode_id"]),
                    evolution_cycle_id=row["evolution_cycle_id"],
                    dataset_id=UUID(row["dataset_id"]) if row["dataset_id"] else None,
                    claim_status=row["claim_status"],
                    claimed_at=datetime.fromisoformat(row["claimed_at"]),
                    released_at=(
                        datetime.fromisoformat(row["released_at"])
                        if row["released_at"]
                        else None
                    ),
                    claim_reason=row["claim_reason"],
                    reuse_reason=row["reuse_reason"],
                )
            )
        return out

    async def explicit_reuse(
        self,
        *,
        evaluation_id: UUID,
        evolution_cycle_id: str,
        episode_id: UUID,
        reuse_reason: str,
    ) -> EvidenceClaim:
        claim = EvidenceClaim(
            evaluation_id=evaluation_id,
            episode_id=episode_id,
            evolution_cycle_id=evolution_cycle_id,
            claim_status="claimed",
            claim_reason="explicit_reuse",
            reuse_reason=reuse_reason,
        )
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            # Delete prior released claim if present, allow audited reuse.
            await db.execute(
                "DELETE FROM evolution_evidence_claims WHERE evaluation_id = ?",
                (str(evaluation_id),),
            )
            await db.execute(
                """
                INSERT INTO evolution_evidence_claims (
                    evaluation_id, episode_id, evolution_cycle_id,
                    dataset_id, claim_status, claimed_at, released_at,
                    claim_reason, reuse_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(claim.evaluation_id),
                    str(claim.episode_id),
                    claim.evolution_cycle_id,
                    None,
                    claim.claim_status,
                    claim.claimed_at.isoformat(),
                    None,
                    claim.claim_reason,
                    claim.reuse_reason,
                ),
            )
            await db.commit()
        return claim
