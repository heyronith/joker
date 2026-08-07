"""Historical outcome persistence: async boundary, atomicity, contention."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiosqlite
import pytest

from joker.objectives.config import HistoricalOutcomeSettings
from joker.objectives.historical_outcomes import HistoricalOutcomeService
from joker.objectives.repository import (
    CrashInjected,
    ObjectivePersistenceBusyError,
    ObjectiveRepository,
)


def _cold_service(repo: ObjectiveRepository) -> HistoricalOutcomeService:
    """Evolution-disabled / no Task-3 history topology used by paper runs."""
    return HistoricalOutcomeService(
        repository=repo,
        settings=HistoricalOutcomeSettings(
            minimum_samples_for_ev=20,
            minimum_effective_sample_size=15,
        ),
        source_diagnostic_reason="no_task3_history",
    )


@pytest.mark.asyncio
async def test_sync_historical_persist_on_loop_exposes_50f774f_hazard(
    tmp_path: Path,
) -> None:
    """Pre-fix probe: sync historical saves on the event loop under contention."""
    db = tmp_path / "hazard.db"
    repo = ObjectiveRepository(db)
    svc = _cold_service(repo)
    stop = asyncio.Event()
    heartbeats: list[float] = []
    locked: list[BaseException] = []

    async def _writer() -> None:
        while not stop.is_set():
            async with aiosqlite.connect(db) as conn:
                await conn.execute("PRAGMA busy_timeout = 50")
                try:
                    await conn.execute("BEGIN IMMEDIATE")
                    await conn.execute(
                        "CREATE TABLE IF NOT EXISTS hist_hazard_noise(id INTEGER)"
                    )
                    await conn.execute("INSERT INTO hist_hazard_noise(id) VALUES (1)")
                    await asyncio.sleep(0.35)
                    await conn.commit()
                except Exception:
                    try:
                        await conn.rollback()
                    except Exception:
                        pass
            await asyncio.sleep(0.01)

    async def _beat() -> None:
        while not stop.is_set():
            heartbeats.append(time.monotonic())
            await asyncio.sleep(0.05)

    writer = asyncio.create_task(_writer())
    beat = asyncio.create_task(_beat())
    await asyncio.sleep(0.1)
    try:
        for _ in range(4):
            try:
                # Direct sync repository path — the 50f774f hazard shape.
                query_id = uuid4()
                from joker.objectives.historical_schemas import (
                    HistoricalLeakageReport,
                    HistoricalOutcomeQuery,
                    HistoricalOutcomeSummary,
                )

                query = HistoricalOutcomeQuery(
                    query_id=query_id,
                    objective_id=uuid4(),
                    strategy_id=uuid4(),
                    snapshot_id=uuid4(),
                    as_of_timestamp=datetime.now(timezone.utc),
                )
                summary = HistoricalOutcomeSummary(
                    summary_id=uuid4(),
                    query_id=query_id,
                    strategy_id=query.strategy_id,
                    snapshot_id=query.snapshot_id,
                    sample_count=0,
                    profitable_count=0,
                    losing_count=0,
                    flat_count=0,
                    minimum_similarity=query.minimum_similarity,
                    valid_for_ev=False,
                )
                report = HistoricalLeakageReport(
                    query_id=query_id, safe=True, notes=("hazard_probe",)
                )
                repo.save_historical_query(query)
                repo.save_historical_summary(summary)
                repo.save_leakage_report(report)
            except (sqlite3.OperationalError, ObjectivePersistenceBusyError) as exc:
                locked.append(exc)
            await asyncio.sleep(0)
    finally:
        stop.set()
        await asyncio.gather(writer, beat, return_exceptions=True)

    gaps = [b - a for a, b in zip(heartbeats, heartbeats[1:])]
    assert gaps, "heartbeat never advanced"
    assert locked or max(gaps) >= 1.0, (
        f"expected sync-on-loop hazard; locked={locked!r} max_gap={max(gaps):.3f}"
    )


@pytest.mark.asyncio
async def test_summarize_for_strategy_under_task1_contention(
    tmp_path: Path,
) -> None:
    """Cold-start summarize_for_strategy must stay off-loop and fully durable."""
    db = tmp_path / "hist_contention.db"
    repo = ObjectiveRepository(db)
    svc = _cold_service(repo)
    stop = asyncio.Event()
    heartbeats: list[float] = []

    async def _writer() -> None:
        while not stop.is_set():
            async with aiosqlite.connect(db) as conn:
                await conn.execute("PRAGMA busy_timeout = 500")
                try:
                    await conn.execute("BEGIN IMMEDIATE")
                    await conn.execute(
                        "CREATE TABLE IF NOT EXISTS hist_noise(id INTEGER)"
                    )
                    await conn.execute("INSERT INTO hist_noise(id) VALUES (1)")
                    await conn.commit()
                except Exception:
                    try:
                        await conn.rollback()
                    except Exception:
                        pass
            await asyncio.sleep(0.01)

    async def _beat() -> None:
        while not stop.is_set():
            heartbeats.append(time.monotonic())
            await asyncio.sleep(0.05)

    writer = asyncio.create_task(_writer())
    beat = asyncio.create_task(_beat())
    await asyncio.sleep(0.1)
    objective_id = uuid4()
    strategy_id = uuid4()
    snapshot_id = uuid4()
    try:
        summary = await svc.summarize_for_strategy(
            objective_id=objective_id,
            strategy_id=strategy_id,
            snapshot_id=snapshot_id,
            as_of_timestamp=datetime.now(timezone.utc),
            strategy_family="breakout_continuation",
            direction="bullish",
            session_phase="midday",
        )
        await asyncio.sleep(0.15)
    finally:
        stop.set()
        await asyncio.gather(writer, beat, return_exceptions=True)

    gaps = [b - a for a, b in zip(heartbeats, heartbeats[1:])]
    assert gaps, "heartbeat never advanced"
    assert max(gaps) < 1.5, f"event loop stalled: max heartbeat gap={max(gaps):.3f}s"

    assert summary.query_id is not None
    loaded_query = repo.get_historical_query(summary.query_id)
    loaded_summary = repo.get_historical_summary(summary.summary_id)
    loaded_report = repo.get_leakage_report(summary.query_id)
    assert loaded_query is not None
    assert loaded_summary is not None
    assert loaded_report is not None
    assert str(loaded_query.query_id) == str(summary.query_id)
    assert str(loaded_summary.query_id) == str(summary.query_id)
    assert str(loaded_report.query_id) == str(summary.query_id)
    assert str(loaded_summary.strategy_id) == str(strategy_id)
    assert str(loaded_summary.snapshot_id) == str(snapshot_id)


@pytest.mark.asyncio
async def test_historical_result_atomic_crash_leaves_no_partial(
    tmp_path: Path,
) -> None:
    """Crash/sustained failure must not leave query-only or query+summary rows."""
    db = tmp_path / "hist_atomic.db"
    repo = ObjectiveRepository(db)
    from joker.objectives.historical_schemas import (
        HistoricalLeakageReport,
        HistoricalOutcomeQuery,
        HistoricalOutcomeSummary,
    )

    for crash_point in (
        "after_historical_query",
        "after_historical_summary",
        "after_historical_leakage",
    ):
        query_id = uuid4()
        query = HistoricalOutcomeQuery(
            query_id=query_id,
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            as_of_timestamp=datetime.now(timezone.utc),
        )
        summary = HistoricalOutcomeSummary(
            summary_id=uuid4(),
            query_id=query_id,
            strategy_id=query.strategy_id,
            snapshot_id=query.snapshot_id,
            sample_count=0,
            profitable_count=0,
            losing_count=0,
            flat_count=0,
            minimum_similarity=query.minimum_similarity,
            valid_for_ev=False,
        )
        report = HistoricalLeakageReport(
            query_id=query_id, safe=True, notes=(f"crash:{crash_point}",)
        )

        def _hook(point: str, *, expected: str = crash_point) -> None:
            if point == expected:
                raise CrashInjected(expected)

        repo.set_crash_hook(_hook)
        with pytest.raises(CrashInjected):
            repo.persist_historical_result_atomic(query, summary, report)
        repo.set_crash_hook(None)

        assert repo.get_historical_query(query_id) is None
        assert repo.get_historical_summary(summary.summary_id) is None
        assert repo.get_leakage_report(query_id) is None

    # Sustained busy → exact diagnostic, no partial durable rows.
    # Hold the write lock for the entire assertion; do not auto-release during
    # the bounded busy-retry window (otherwise the write can succeed late).
    query_id = uuid4()
    query = HistoricalOutcomeQuery(
        query_id=query_id,
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=datetime.now(timezone.utc),
    )
    summary = HistoricalOutcomeSummary(
        summary_id=uuid4(),
        query_id=query_id,
        strategy_id=query.strategy_id,
        snapshot_id=query.snapshot_id,
        sample_count=0,
        profitable_count=0,
        losing_count=0,
        flat_count=0,
        minimum_similarity=query.minimum_similarity,
        valid_for_ev=False,
    )
    report = HistoricalLeakageReport(query_id=query_id, safe=True, notes=("busy",))
    holder = sqlite3.connect(db, timeout=1.0)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(ObjectivePersistenceBusyError):
            repo.persist_historical_result_atomic(query, summary, report)
    finally:
        holder.rollback()
        holder.close()
    assert repo.get_historical_query(query_id) is None
    assert repo.get_historical_summary(summary.summary_id) is None
    assert repo.get_leakage_report(query_id) is None


@pytest.mark.asyncio
async def test_summarize_busy_failure_does_not_return_undurable_success(
    tmp_path: Path,
) -> None:
    """Required durable persistence failure must not look like a successful result."""
    db = tmp_path / "hist_busy_svc.db"
    repo = ObjectiveRepository(db)
    svc = _cold_service(repo)

    holder = sqlite3.connect(db, timeout=1.0)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(ObjectivePersistenceBusyError):
            await asyncio.wait_for(
                svc.summarize_for_strategy(
                    objective_id=uuid4(),
                    strategy_id=uuid4(),
                    snapshot_id=uuid4(),
                    as_of_timestamp=datetime.now(timezone.utc),
                ),
                timeout=30,
            )
    finally:
        holder.rollback()
        holder.close()

    with sqlite3.connect(db) as conn:
        q = conn.execute("SELECT COUNT(*) FROM objective_historical_queries").fetchone()[0]
        s = conn.execute(
            "SELECT COUNT(*) FROM objective_historical_summaries"
        ).fetchone()[0]
        r = conn.execute(
            "SELECT COUNT(*) FROM objective_historical_leakage_reports"
        ).fetchone()[0]
    assert q == 0 and s == 0 and r == 0
