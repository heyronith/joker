"""Durable append-only evolution evidence ownership claims."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

ClaimStatus = Literal["claimed", "consumed", "released", "invalidated"]

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS evolution_evidence_claims (
    claim_id TEXT PRIMARY KEY NOT NULL,
    evaluation_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    evolution_cycle_id TEXT NOT NULL,
    dataset_id TEXT,
    claim_status TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    released_at TEXT,
    claim_reason TEXT NOT NULL,
    reuse_reason TEXT,
    prior_claim_id TEXT,
    actor TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_cycle
    ON evolution_evidence_claims (evolution_cycle_id, claim_status);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_status
    ON evolution_evidence_claims (claim_status, claimed_at);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_evaluation
    ON evolution_evidence_claims (evaluation_id, claim_status);
"""


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: UUID = Field(default_factory=uuid4)
    evaluation_id: UUID
    episode_id: UUID
    evolution_cycle_id: str
    dataset_id: UUID | None = None
    claim_status: ClaimStatus = "claimed"
    claimed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    released_at: datetime | None = None
    claim_reason: str = "automatic_cycle"
    reuse_reason: str | None = None
    prior_claim_id: UUID | None = None
    actor: str | None = None


class EvidenceClaimStore:
    """Transactional ownership of evaluations for evolution cycles."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_SQL)
            await self._migrate_legacy(db)
            await db.commit()

    async def _migrate_legacy(self, db: aiosqlite.Connection) -> None:
        """Forward-only migrate evaluation_id PK schema → claim_id append-only."""
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='evolution_evidence_claims'"
        )
        if await cur.fetchone() is None:
            return
        cols = {
            row[1]: row
            for row in await (await db.execute("PRAGMA table_info(evolution_evidence_claims)")).fetchall()
        }
        for col, decl in (
            ("claim_id", "TEXT"),
            ("prior_claim_id", "TEXT"),
            ("actor", "TEXT"),
        ):
            if col not in cols:
                await db.execute(
                    f"ALTER TABLE evolution_evidence_claims ADD COLUMN {col} {decl}"
                )
        await db.execute(
            """
            UPDATE evolution_evidence_claims
            SET claim_id = evaluation_id || ':' || claimed_at
            WHERE claim_id IS NULL OR claim_id = ''
            """
        )
        # Re-read schema after ALTERs to decide whether PK rebuild is required.
        info = await (
            await db.execute("PRAGMA table_info(evolution_evidence_claims)")
        ).fetchall()
        pk_cols = [row[1] for row in info if row[5]]
        if "evaluation_id" in pk_cols and "claim_id" not in pk_cols:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS evolution_evidence_claims_v2 (
                    claim_id TEXT PRIMARY KEY NOT NULL,
                    evaluation_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    evolution_cycle_id TEXT NOT NULL,
                    dataset_id TEXT,
                    claim_status TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    released_at TEXT,
                    claim_reason TEXT NOT NULL,
                    reuse_reason TEXT,
                    prior_claim_id TEXT,
                    actor TEXT
                )
                """
            )
            await db.execute(
                """
                INSERT OR IGNORE INTO evolution_evidence_claims_v2 (
                    claim_id, evaluation_id, episode_id, evolution_cycle_id,
                    dataset_id, claim_status, claimed_at, released_at,
                    claim_reason, reuse_reason, prior_claim_id, actor
                )
                SELECT
                    COALESCE(NULLIF(claim_id, ''), evaluation_id || ':' || claimed_at),
                    evaluation_id, episode_id, evolution_cycle_id,
                    dataset_id, claim_status, claimed_at, released_at,
                    claim_reason, reuse_reason, prior_claim_id, actor
                FROM evolution_evidence_claims
                """
            )
            await db.execute("DROP TABLE evolution_evidence_claims")
            await db.execute(
                "ALTER TABLE evolution_evidence_claims_v2 RENAME TO evolution_evidence_claims"
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evidence_claims_cycle
                    ON evolution_evidence_claims (evolution_cycle_id, claim_status)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evidence_claims_status
                    ON evolution_evidence_claims (claim_status, claimed_at)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evidence_claims_evaluation
                    ON evolution_evidence_claims (evaluation_id, claim_status)
                """
            )

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
                    # Exclusive automatic claim: block if active automatic claim exists.
                    cur = await db.execute(
                        """
                        SELECT claim_id FROM evolution_evidence_claims
                        WHERE evaluation_id = ?
                          AND claim_status IN ('claimed', 'consumed')
                          AND claim_reason = 'automatic_cycle'
                        LIMIT 1
                        """,
                        (str(claim.evaluation_id),),
                    )
                    if await cur.fetchone() is not None:
                        continue
                    try:
                        await db.execute(
                            """
                            INSERT INTO evolution_evidence_claims (
                                claim_id, evaluation_id, episode_id, evolution_cycle_id,
                                dataset_id, claim_status, claimed_at, released_at,
                                claim_reason, reuse_reason, prior_claim_id, actor
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                str(claim.claim_id),
                                str(claim.evaluation_id),
                                str(claim.episode_id),
                                claim.evolution_cycle_id,
                                str(claim.dataset_id) if claim.dataset_id else None,
                                claim.claim_status,
                                claim.claimed_at.isoformat(),
                                None,
                                claim.claim_reason,
                                claim.reuse_reason,
                                str(claim.prior_claim_id) if claim.prior_claim_id else None,
                                claim.actor,
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
        """Return evaluation IDs that currently have an active claim."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                SELECT DISTINCT evaluation_id FROM evolution_evidence_claims
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

    def _row_to_claim(self, row: aiosqlite.Row) -> EvidenceClaim:
        claim_id_raw = row["claim_id"] if "claim_id" in row.keys() else None
        prior_raw = row["prior_claim_id"] if "prior_claim_id" in row.keys() else None
        actor = row["actor"] if "actor" in row.keys() else None
        return EvidenceClaim(
            claim_id=UUID(str(claim_id_raw)) if claim_id_raw else uuid4(),
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
            prior_claim_id=UUID(str(prior_raw)) if prior_raw else None,
            actor=actor,
        )

    async def list_by_cycle(self, evolution_cycle_id: str) -> list[EvidenceClaim]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM evolution_evidence_claims
                WHERE evolution_cycle_id = ?
                ORDER BY claimed_at ASC
                """,
                (evolution_cycle_id,),
            )
            rows = await cur.fetchall()
        return [self._row_to_claim(row) for row in rows]

    async def list_history(self, evaluation_id: UUID) -> list[EvidenceClaim]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM evolution_evidence_claims
                WHERE evaluation_id = ?
                ORDER BY claimed_at ASC
                """,
                (str(evaluation_id),),
            )
            rows = await cur.fetchall()
        return [self._row_to_claim(row) for row in rows]

    async def explicit_reuse(
        self,
        *,
        evaluation_id: UUID,
        evolution_cycle_id: str,
        episode_id: UUID,
        reuse_reason: str,
        actor: str = "operator",
        allow_reuse: bool = True,
    ) -> EvidenceClaim:
        """Append a new claim linked to the prior claim. Never deletes history."""
        if not allow_reuse:
            raise RuntimeError("explicit_reuse_requires_allow_reuse_flag")
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("BEGIN IMMEDIATE")
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM evolution_evidence_claims
                WHERE evaluation_id = ?
                ORDER BY claimed_at DESC
                LIMIT 1
                """,
                (str(evaluation_id),),
            )
            prior_row = await cur.fetchone()
            prior_claim_id = None
            if prior_row is not None:
                prior_claim_id = (
                    UUID(str(prior_row["claim_id"]))
                    if prior_row["claim_id"]
                    else None
                )
                # Leave prior row intact; only release if still active.
                if prior_row["claim_status"] in {"claimed", "consumed"}:
                    await db.execute(
                        """
                        UPDATE evolution_evidence_claims
                        SET claim_status = 'released', released_at = ?
                        WHERE claim_id = ?
                        """,
                        (
                            datetime.now(timezone.utc).isoformat(),
                            str(prior_row["claim_id"]),
                        ),
                    )
            claim = EvidenceClaim(
                evaluation_id=evaluation_id,
                episode_id=episode_id,
                evolution_cycle_id=evolution_cycle_id,
                claim_status="claimed",
                claim_reason="explicit_reuse",
                reuse_reason=reuse_reason,
                prior_claim_id=prior_claim_id,
                actor=actor,
            )
            await db.execute(
                """
                INSERT INTO evolution_evidence_claims (
                    claim_id, evaluation_id, episode_id, evolution_cycle_id,
                    dataset_id, claim_status, claimed_at, released_at,
                    claim_reason, reuse_reason, prior_claim_id, actor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(claim.claim_id),
                    str(claim.evaluation_id),
                    str(claim.episode_id),
                    claim.evolution_cycle_id,
                    None,
                    claim.claim_status,
                    claim.claimed_at.isoformat(),
                    None,
                    claim.claim_reason,
                    claim.reuse_reason,
                    str(claim.prior_claim_id) if claim.prior_claim_id else None,
                    claim.actor,
                ),
            )
            await db.commit()
        return claim
