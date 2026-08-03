"""Unit tests for Task 3 event horizon index and loader."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from joker.evolution.event_horizon import Task1EventHorizonLoader
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.session_event_index import (
    SessionEventIndexRecord,
    SessionEventIndexRepository,
)


@pytest.mark.asyncio
async def test_session_event_index_round_trip(tmp_path):
    db = tmp_path / "idx.db"
    apply_task3_migrations(db)
    repo = SessionEventIndexRepository(str(db))
    await repo.initialize()
    ts = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    eid = str(uuid4())
    inserted = await repo.record(
        SessionEventIndexRecord(
            event_id=eid,
            session_id="sess-a",
            event_type="market_snapshot_created",
            exchange_timestamp=ts,
            sequence=1,
            snapshot_id=str(uuid4()),
        )
    )
    assert inserted is True
    dup = await repo.record(
        SessionEventIndexRecord(
            event_id=eid,
            session_id="sess-a",
            event_type="market_snapshot_created",
            exchange_timestamp=ts,
        )
    )
    assert dup is False
    rows = await repo.list_horizon(
        "sess-a",
        start_timestamp=ts - timedelta(minutes=1),
        end_timestamp=ts + timedelta(minutes=1),
    )
    assert len(rows) == 1
    assert rows[0].event_id == eid


@pytest.mark.asyncio
async def test_task1_event_horizon_loader_ordering(tmp_path):
    db = tmp_path / "horizon.db"
    apply_task3_migrations(db)
    repo = SessionEventIndexRepository(str(db))
    await repo.initialize()
    loader = Task1EventHorizonLoader(index_repo=repo)
    session_id = "sess-b"
    base = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    e1, e2, e3 = uuid4(), uuid4(), uuid4()
    for seq, eid, offset in (
        (2, e2, timedelta(minutes=2)),
        (1, e1, timedelta(minutes=1)),
        (3, e3, timedelta(minutes=3)),
    ):
        await repo.record(
            SessionEventIndexRecord(
                event_id=str(eid),
                session_id=session_id,
                event_type="order_submitted",
                exchange_timestamp=base + offset,
                sequence=seq,
            )
        )
    horizon = await loader.load(
        session_id=session_id,
        start_timestamp=base,
        end_timestamp=base + timedelta(minutes=5),
    )
    assert [ev.event_id for ev in horizon.events] == [e1, e2, e3]
    assert horizon.market_event_ids == (e1, e2, e3)


@pytest.mark.asyncio
async def test_activation_completed_false_when_history_missing(tmp_path):
    from joker.evolution.champion_registry import ChampionRegistry
    from joker.evolution.improvement import ImprovementProposalService
    from joker.evolution.repositories import build_evolution_repositories
    from joker.evolution.schemas import PromptPatch, PromotionDecision
    from joker.evolution.decision import EvolutionDecisionService
    import aiosqlite

    apply_task3_migrations(tmp_path / "act.db")
    repos = build_evolution_repositories(tmp_path / "act.db")
    for r in repos.values():
        await r.initialize()
    registry = ChampionRegistry(tmp_path / "act.db")
    champion = await registry.bootstrap_champion()
    improvement = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, challenger = await improvement.propose(
        parent_champion=champion,
        training_dataset_ids=(uuid4(),),
        weakness="x",
        hypothesis="y",
        patch=PromptPatch(
            role="falsifier",
            parent_prompt_version_id=uuid4(),
            replacement_template="t",
            change_rationale="r",
        ),
    )
    decision = PromotionDecision(
        experiment_id=uuid4(),
        challenger_version_id=challenger.configuration_version_id,
        champion_version_id=champion.configuration_version_id,
        deterministic_eligible=True,
        agent_action="promote",
        strategic_rationale="test",
        final_status="promoted",
        idempotency_key=f"missing-hist-{uuid4()}",
    )
    await repos["promotions"].append(decision)
    service = EvolutionDecisionService(
        repos["promotions"],
        repos["configurations"],
        registry,
        activation_repo=repos["activations"],
    )
    await registry.promote(
        challenger=challenger,
        expected_champion_id=champion.configuration_version_id,
        reason="agent_promote",
        experiment_id=decision.experiment_id,
        promotion_decision_id=decision.promotion_decision_id,
    )
    async with aiosqlite.connect(tmp_path / "act.db") as db:
        await db.execute("DELETE FROM champion_history")
        await db.commit()

    async def _no_repair(**kwargs: object) -> bool:
        return False

    registry.repair_promotion_history_if_missing = _no_repair  # type: ignore[method-assign]
    await service.apply_persisted_decision(
        promotion_decision_id=decision.promotion_decision_id
    )
    activation = await repos["activations"].get_by_decision_id(
        decision.promotion_decision_id
    )
    assert activation is not None
    assert activation.completed is False
    assert "history_transition_missing" in activation.failure_codes
