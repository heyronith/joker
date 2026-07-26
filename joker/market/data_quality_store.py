"""Append-only repository for Task 1 DataQualityReport rows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import aiosqlite

from joker.market.quality import (
    DataQualityCode,
    DataQualityFinding,
    DataQualityReport,
    DataQualitySeverity,
)
from joker.persistence.migrations import apply_task1_migrations


class DataQualityRepository:
    """Persist and load exact DataQualityReport payloads by report_id.

    Uses the Task 1 ``data_quality_reports`` migration schema. Missing reports
    must never be fabricated as healthy — callers use ``unavailable_report``.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        apply_task1_migrations(self._db_path)
        self._initialized = True

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def save(
        self,
        report: DataQualityReport,
        *,
        session_id: str | None = None,
    ) -> UUID:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO data_quality_reports (
                    report_id, snapshot_id, session_id, severity,
                    usable_for_reasoning, usable_for_execution, payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(report.report_id),
                    str(report.snapshot_id) if report.snapshot_id else None,
                    session_id,
                    report.severity.value,
                    1 if report.usable_for_reasoning else 0,
                    1 if report.usable_for_execution else 0,
                    report.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
        return report.report_id

    async def get_by_id(self, report_id: UUID | str) -> DataQualityReport | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                "SELECT payload FROM data_quality_reports WHERE report_id = ?",
                (str(report_id),),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return DataQualityReport.model_validate_json(row[0])

    @staticmethod
    def unavailable_report(
        report_id: UUID | str,
        *,
        snapshot_id: UUID | str | None = None,
    ) -> DataQualityReport:
        """Fail-closed placeholder when the exact Task 1 report cannot be loaded."""
        return DataQualityReport(
            report_id=UUID(str(report_id)),
            snapshot_id=UUID(str(snapshot_id)) if snapshot_id else None,
            severity=DataQualitySeverity.CRITICAL,
            findings=(
                DataQualityFinding(
                    code=DataQualityCode.REPORT_UNAVAILABLE,
                    severity=DataQualitySeverity.CRITICAL,
                    message=(
                        f"data quality report {report_id} unavailable; "
                        "reasoning and execution blocked"
                    ),
                ),
            ),
            usable_for_reasoning=False,
            usable_for_execution=False,
        )
