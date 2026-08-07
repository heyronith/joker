"""Objective persistence concurrency, atomicity, and event-loop safety."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import aiosqlite
import pytest

from joker.cli.graph_view import render_graph_event
from joker.cli.paper import agent_runtime_label, model_provider_label
from joker.config.settings import AppSettings
from joker.objectives.repository import (
    CrashInjected,
    ObjectivePersistenceBusyError,
    ObjectiveRepository,
)
from joker.objectives.service import ObjectiveServiceError, SessionObjectiveService

ET = ZoneInfo("America/New_York")


def _deadline(hours: float = 2.0) -> datetime:
    return datetime.now(tz=ET) + timedelta(hours=hours)


async def _armed(
    tmp_path: Path, *, name: str = "obj.db"
) -> tuple[SessionObjectiveService, ObjectiveRepository, Path]:
    db = tmp_path / name
    repo = ObjectiveRepository(db)
    svc = SessionObjectiveService(repo, exchange_tz="America/New_York")
    definition = await svc.create_objective(
        session_id="sess-concurrency",
        authorised_capital_usd=500,
        target_profit_pct=20,
        deadline_exchange_time=_deadline(),
        max_concurrent_positions=3,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    return svc, repo, db


@pytest.mark.asyncio
async def test_shared_sqlite_contention_recompute_stays_responsive(
    tmp_path: Path,
) -> None:
    """Production topology: Task-1 aiosqlite writer + objective recompute.

    Pre-fix behaviour blocked the event loop (sync sqlite busy wait) or
    raised ``database is locked`` into ``objective_unavailable``. Post-fix
    mutations must succeed under transient contention without stalling the loop.
    """
    svc, _repo, db = await _armed(tmp_path)
    stop = asyncio.Event()
    heartbeats: list[float] = []
    writer_errors: list[BaseException] = []

    async def _task1_writer() -> None:
        try:
            while not stop.is_set():
                async with aiosqlite.connect(db) as conn:
                    await conn.execute(f"PRAGMA busy_timeout = 500")
                    await conn.execute("BEGIN IMMEDIATE")
                    await conn.execute(
                        "CREATE TABLE IF NOT EXISTS contention_noise (id INTEGER)"
                    )
                    await conn.execute("INSERT INTO contention_noise(id) VALUES (1)")
                    await conn.commit()
                await asyncio.sleep(0.01)
        except Exception as exc:  # noqa: BLE001
            writer_errors.append(exc)

    async def _heartbeat() -> None:
        while not stop.is_set():
            heartbeats.append(time.monotonic())
            await asyncio.sleep(0.05)

    writer = asyncio.create_task(_task1_writer())
    beat = asyncio.create_task(_heartbeat())
    await asyncio.sleep(0.05)
    try:
        states = await asyncio.gather(
            *[svc.recompute_from_truth() for _ in range(8)]
        )
    finally:
        stop.set()
        await asyncio.gather(writer, beat, return_exceptions=True)

    versions = sorted(state.version for state in states)
    assert len(set(versions)) == len(versions)
    assert versions == list(range(versions[0], versions[0] + len(versions)))
    latest = await svc.get_state()
    assert latest.version == max(versions)
    # Event loop remained responsive (no multi-second stalls from sync busy waits).
    gaps = [b - a for a, b in zip(heartbeats, heartbeats[1:])]
    assert gaps, "heartbeat never advanced"
    assert max(gaps) < 1.5, f"event loop stalled: max heartbeat gap={max(gaps):.3f}s"
    assert writer_errors == [] or all(
        "locked" in str(exc).lower() or "busy" in str(exc).lower()
        for exc in writer_errors
    )


def test_confirm_on_loop_a_recompute_on_loop_b(tmp_path: Path) -> None:
    """CLI asyncio.run confirmation must not bind the service to that loop."""
    db = tmp_path / "cross_loop.db"
    repo = ObjectiveRepository(db)
    svc = SessionObjectiveService(repo, exchange_tz="America/New_York")

    async def _confirm() -> None:
        definition = await svc.create_objective(
            session_id="sess-cross",
            authorised_capital_usd=400,
            target_profit_pct=25,
            deadline_exchange_time=_deadline(),
            max_concurrent_positions=2,
            accepted_total_loss_risk=True,
        )
        await svc.confirm_objective(definition.objective_id)

    asyncio.run(_confirm())

    async def _recompute() -> int:
        state = await svc.recompute_from_truth()
        return state.version

    version = asyncio.run(_recompute())
    assert version >= 2


@pytest.mark.asyncio
async def test_concurrent_recompute_versions_are_unique_and_monotonic(
    tmp_path: Path,
) -> None:
    svc, repo, _db = await _armed(tmp_path)
    before = await svc.get_state()
    states = await asyncio.gather(
        *[svc.recompute_from_truth() for _ in range(12)]
    )
    versions = [state.version for state in states]
    assert len(versions) == len(set(versions))
    assert sorted(versions) == list(range(before.version + 1, before.version + 13))
    latest = await svc.get_state()
    assert latest.version == max(versions)

    audits = []
    with sqlite3.connect(repo.db_path) as conn:
        rows = conn.execute(
            """
            SELECT event_type, payload_json FROM objective_decision_audit
            WHERE objective_id=? AND event_type='objective.recomputed'
            ORDER BY created_at
            """,
            (str(latest.objective_id),),
        ).fetchall()
        audits = list(rows)
    assert len(audits) >= 12
    # Every recomputed state version has a matching audit payload version.
    import json

    audit_versions = {
        int(json.loads(payload)["after"]["version"]) for _, payload in audits
    }
    assert set(versions).issubset(audit_versions)


@pytest.mark.asyncio
async def test_sustained_lock_fails_closed_without_partial_mutation(
    tmp_path: Path,
) -> None:
    svc, repo, db = await _armed(tmp_path)
    before = await svc.get_state()
    holder = sqlite3.connect(db, timeout=1.0)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises((ObjectiveServiceError, ObjectivePersistenceBusyError)):
            await asyncio.wait_for(svc.recompute_from_truth(), timeout=30)
        after = repo.latest_state(before.objective_id)
        assert after is not None
        assert after.version == before.version
        with sqlite3.connect(db) as conn:
            audits = conn.execute(
                """
                SELECT COUNT(*) FROM objective_decision_audit
                WHERE objective_id=? AND event_type='objective.recomputed'
                  AND payload_json LIKE ?
                """,
                (
                    str(before.objective_id),
                    f'%"version": {before.version + 1}%',
                ),
            ).fetchone()[0]
        assert audits == 0
    finally:
        holder.rollback()
        holder.close()


@pytest.mark.asyncio
async def test_recompute_state_and_audit_commit_atomically_under_crash(
    tmp_path: Path,
) -> None:
    svc, repo, db = await _armed(tmp_path)
    before = await svc.get_state()

    def _crash(point: str) -> None:
        if point == "after_state_append":
            raise CrashInjected(point)

    repo.set_crash_hook(_crash)
    with pytest.raises(CrashInjected):
        await svc.recompute_from_truth()
    repo.set_crash_hook(None)

    after = repo.latest_state(before.objective_id)
    assert after is not None
    assert after.version == before.version
    with sqlite3.connect(db) as conn:
        state_rows = conn.execute(
            """
            SELECT COUNT(*) FROM session_objective_state_versions
            WHERE objective_id=? AND version=?
            """,
            (str(before.objective_id), before.version + 1),
        ).fetchone()[0]
        audit_rows = conn.execute(
            """
            SELECT COUNT(*) FROM objective_decision_audit
            WHERE objective_id=? AND event_type='objective.recomputed'
              AND payload_json LIKE ?
            """,
            (str(before.objective_id), f'%"version": {before.version + 1}%'),
        ).fetchone()[0]
    assert state_rows == 0
    assert audit_rows == 0


@pytest.mark.asyncio
async def test_recompute_does_not_block_event_loop(tmp_path: Path) -> None:
    svc, _repo, _db = await _armed(tmp_path)
    progressed = asyncio.Event()

    async def _progress() -> None:
        await asyncio.sleep(0)
        progressed.set()

    progress = asyncio.create_task(_progress())
    await svc.recompute_from_truth()
    await asyncio.wait_for(progress, timeout=1.0)
    assert progressed.is_set()


def test_structured_objective_unavailable_is_operator_visible() -> None:
    rendered = render_graph_event(
        "graph.cycle.completed",
        {
            "cycle_id": "283510ef-bef3-4af4-8328-428af271a0eb",
            "outcome": "completed",
            "decision_action": None,
            "execution_command_ids": [],
            "error_codes": ["objective_unavailable"],
            "errors": [
                {
                    "code": "objective_unavailable",
                    "node": "validate_trigger",
                    "message": "database is locked",
                    "recoverable": False,
                }
            ],
        },
        view="compact",
    )
    assert "OBJECTIVE ERROR" in rendered
    assert "node=validate_trigger" in rendered
    assert "database is locked" in rendered


def test_cli_model_provider_labels_are_truthful() -> None:
    app = AppSettings()
    app = app.model_copy(
        update={
            "agents": app.agents.model_copy(
                update={"runtime": "cognitive_graph", "mock_agents": False}
            ),
            "models": app.models.model_copy(
                update={
                    "ollama": app.models.ollama.model_copy(update={"enabled": True}),
                    "openai": app.models.openai.model_copy(update={"enabled": False}),
                }
            ),
        }
    )
    assert "OpenAI council" not in agent_runtime_label(app)
    assert "cognitive graph" in agent_runtime_label(app)
    label = model_provider_label(app)
    assert "Ollama=enabled" in label or "Ollama=disabled" in label
    assert "OpenAI=disabled" in label


@pytest.mark.asyncio
async def test_sync_downstream_objective_persistence_exposes_824a7ca_hazard(
    tmp_path: Path,
) -> None:
    """Pre-fix probe: sync sqlite3 feasibility/estimate/score saves on the loop.

    At ``824a7ca`` graph callers invoked sync ``SessionObjectiveService`` save
    methods directly on the cognitive asyncio loop. Under Task-1 contention that
    either stalls the loop for multi-second busy waits or surfaces raw
    ``sqlite3.OperationalError``. This probe recreates that hazard shape.
    """
    from uuid import uuid4

    from joker.objectives.schemas import (
        GoalFeasibilityAssessment,
        ObjectiveStrategyScore,
        StrategyObjectiveEstimate,
    )

    svc, repo, db = await _armed(tmp_path, name="hazard.db")
    state = await svc.get_state()
    stop = asyncio.Event()
    heartbeats: list[float] = []
    locked: list[BaseException] = []

    async def _task1_writer() -> None:
        while not stop.is_set():
            async with aiosqlite.connect(db) as conn:
                await conn.execute("PRAGMA busy_timeout = 50")
                try:
                    await conn.execute("BEGIN IMMEDIATE")
                    await conn.execute(
                        "CREATE TABLE IF NOT EXISTS hazard_noise(id INTEGER)"
                    )
                    await conn.execute("INSERT INTO hazard_noise(id) VALUES (1)")
                    # Hold the write lock long enough that a sync busy-wait on the
                    # event loop produces a visible heartbeat stall.
                    await asyncio.sleep(0.35)
                    await conn.commit()
                except Exception:
                    try:
                        await conn.rollback()
                    except Exception:
                        pass
            await asyncio.sleep(0.01)

    async def _heartbeat() -> None:
        while not stop.is_set():
            heartbeats.append(time.monotonic())
            await asyncio.sleep(0.05)

    writer = asyncio.create_task(_task1_writer())
    beat = asyncio.create_task(_heartbeat())
    await asyncio.sleep(0.1)
    snap = uuid4()
    strategy_id = uuid4()
    try:
        for _ in range(6):
            try:
                # Direct repo writes on the event loop — the 824a7ca hazard.
                repo.save_feasibility(
                    GoalFeasibilityAssessment(
                        objective_id=state.objective_id,
                        snapshot_id=snap,
                        classification="medium",
                        required_return_remaining_pct=Decimal("10"),
                        required_profit_remaining_usd=Decimal("50"),
                        time_remaining_seconds=3600,
                    )
                )
                repo.save_strategy_estimate(
                    StrategyObjectiveEstimate(
                        strategy_id=strategy_id,
                        objective_id=state.objective_id,
                        snapshot_id=snap,
                        capital_required_usd=Decimal("100"),
                        maximum_loss_usd=Decimal("100"),
                        calculation_method="hazard_probe",
                        valid=False,
                    )
                )
                repo.save_strategy_score(
                    ObjectiveStrategyScore(
                        objective_id=state.objective_id,
                        strategy_id=strategy_id,
                        snapshot_id=snap,
                        maximum_loss_usd=Decimal("100"),
                        capital_required_usd=Decimal("100"),
                        valid=False,
                        is_no_trade=True,
                    )
                )
            except (sqlite3.OperationalError, ObjectivePersistenceBusyError) as exc:
                locked.append(exc)
            await asyncio.sleep(0)
    finally:
        stop.set()
        await asyncio.gather(writer, beat, return_exceptions=True)

    gaps = [b - a for a, b in zip(heartbeats, heartbeats[1:])]
    assert gaps, "heartbeat never advanced"
    # Either the loop stalled on sync busy waits or a lock error escaped —
    # both are the unsafe 824a7ca behaviours this suite must remain sensitive to.
    assert locked or max(gaps) >= 1.0, (
        f"expected sync-on-loop hazard; locked={locked!r} max_gap={max(gaps):.3f}"
    )


@pytest.mark.asyncio
async def test_downstream_objective_persistence_under_task1_contention(
    tmp_path: Path,
) -> None:
    """Feasibility / estimate / score persistence must stay off the event loop."""
    from uuid import uuid4

    from joker.objectives.schemas import (
        GoalFeasibilityAssessment,
        ObjectiveStrategyScore,
        StrategyObjectiveEstimate,
    )

    svc, repo, db = await _armed(tmp_path, name="downstream.db")
    state = await svc.get_state()
    stop = asyncio.Event()
    heartbeats: list[float] = []
    errors: list[BaseException] = []

    async def _task1_writer() -> None:
        while not stop.is_set():
            async with aiosqlite.connect(db) as conn:
                await conn.execute("PRAGMA busy_timeout = 500")
                try:
                    await conn.execute("BEGIN IMMEDIATE")
                    await conn.execute(
                        "CREATE TABLE IF NOT EXISTS downstream_noise(id INTEGER)"
                    )
                    await conn.execute("INSERT INTO downstream_noise(id) VALUES (1)")
                    await conn.commit()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    try:
                        await conn.rollback()
                    except Exception:
                        pass
            await asyncio.sleep(0.01)

    async def _heartbeat() -> None:
        while not stop.is_set():
            heartbeats.append(time.monotonic())
            await asyncio.sleep(0.05)

    writer = asyncio.create_task(_task1_writer())
    beat = asyncio.create_task(_heartbeat())
    await asyncio.sleep(0.1)
    snap = uuid4()
    strategy_id = uuid4()
    try:
        for i in range(5):
            await svc.save_feasibility(
                GoalFeasibilityAssessment(
                    objective_id=state.objective_id,
                    snapshot_id=snap,
                    classification="medium",
                    required_return_remaining_pct=Decimal("10"),
                    required_profit_remaining_usd=Decimal("50"),
                    time_remaining_seconds=3600,
                    assumptions=(f"pass-{i}",),
                )
            )
            estimate = StrategyObjectiveEstimate(
                strategy_id=strategy_id,
                objective_id=state.objective_id,
                snapshot_id=snap,
                capital_required_usd=Decimal("100"),
                maximum_loss_usd=Decimal("100"),
                calculation_method="downstream_contention",
                valid=False,
                uncertainty_reasons=("no_history",),
            )
            await svc.save_strategy_estimate(estimate)
            await svc.save_strategy_score(
                ObjectiveStrategyScore(
                    objective_id=state.objective_id,
                    strategy_id=strategy_id,
                    snapshot_id=snap,
                    estimate_id=estimate.estimate_id,
                    maximum_loss_usd=Decimal("100"),
                    capital_required_usd=Decimal("100"),
                    valid=False,
                    is_no_trade=True,
                )
            )
        await asyncio.sleep(0.2)
    finally:
        stop.set()
        await asyncio.gather(writer, beat, return_exceptions=True)

    gaps = [b - a for a, b in zip(heartbeats, heartbeats[1:])]
    assert gaps, "heartbeat never advanced"
    assert max(gaps) < 1.5, f"event loop stalled: max heartbeat gap={max(gaps):.3f}s"
    assert not any(isinstance(exc, sqlite3.OperationalError) for exc in errors)

    with sqlite3.connect(db) as conn:
        feas = conn.execute(
            "SELECT COUNT(*) FROM objective_feasibility_assessments WHERE objective_id=?",
            (str(state.objective_id),),
        ).fetchone()[0]
        ests = conn.execute(
            "SELECT COUNT(*) FROM objective_strategy_estimates WHERE objective_id=?",
            (str(state.objective_id),),
        ).fetchone()[0]
        scores = conn.execute(
            "SELECT COUNT(*) FROM objective_strategy_scores WHERE objective_id=?",
            (str(state.objective_id),),
        ).fetchone()[0]
    assert feas >= 1
    assert ests >= 1
    assert scores >= 1
    loaded = await svc.get_latest_estimate_for_strategy(
        strategy_id=strategy_id, objective_id=state.objective_id
    )
    assert loaded is not None
    assert loaded.calculation_method == "downstream_contention"
    listed = repo.list_strategy_scores_for_snapshot(
        objective_id=state.objective_id, snapshot_id=snap
    )
    assert listed
