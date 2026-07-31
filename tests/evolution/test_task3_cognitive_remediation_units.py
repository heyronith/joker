"""Unit proofs for Task 3 cognitive validation remediation."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.decision import EvolutionDecisionService
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.replay import entry_thread_id, position_thread_id
from joker.evolution.replay_store import ReplayExecutionStore, replay_key
from joker.evolution.replay_truth import ReplayTruthLoadError, ReplayTruthLoader
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import (
    ChampionActivationRecord,
    PromotionDecision,
    TradingEpisode,
)
from joker.evolution.champion_registry import ChampionRegistry


def test_replay_key_includes_experiment_id():
    exp_a = uuid4()
    exp_b = uuid4()
    ep = uuid4()
    cfg = uuid4()
    assert replay_key(exp_a, ep, cfg, 1) != replay_key(exp_b, ep, cfg, 1)
    assert str(exp_a) in replay_key(exp_a, ep, cfg, 1)


def test_two_experiments_use_distinct_graph_threads():
    exp_a = uuid4()
    exp_b = uuid4()
    ep = uuid4()
    cfg = uuid4()
    assert entry_thread_id(exp_a, ep, cfg, 1) != entry_thread_id(exp_b, ep, cfg, 1)
    assert position_thread_id(exp_a, ep, cfg, 1, 2) != position_thread_id(
        exp_b, ep, cfg, 1, 2
    )


@pytest.mark.asyncio
async def test_second_experiment_does_not_load_first_checkpoint(tmp_path):
    store = ReplayExecutionStore(tmp_path / "r.db")
    await store.initialize()
    exp_a = uuid4()
    exp_b = uuid4()
    ep = uuid4()
    cfg = uuid4()
    key_a = replay_key(exp_a, ep, cfg, 1)
    await store.save_checkpoint(
        key=key_a,
        experiment_id=str(exp_a),
        episode_id=str(ep),
        configuration_version_id=str(cfg),
        sample_number=1,
        status="completed",
        frame_index=3,
        cash=Decimal("100"),
        realised_pnl=Decimal("5"),
        orders={},
        fills=[],
        positions={},
        submitted_keys=set(),
        entry_decision_completed=True,
        extra={"final_result_persisted": True},
    )
    key_b = replay_key(exp_b, ep, cfg, 1)
    assert await store.load_checkpoint(key_b) is None
    loaded_a = await store.load_checkpoint(key_a)
    assert loaded_a is not None
    assert loaded_a["status"] == "completed"


@pytest.mark.asyncio
async def test_activation_repairs_challenger_status(tmp_path):
    from joker.evolution.improvement import ImprovementProposalService
    from joker.evolution.schemas import PromptPatch

    apply_task3_migrations(tmp_path / "a.db")
    repos = build_evolution_repositories(tmp_path / "a.db")
    for r in repos.values():
        await r.initialize()
    registry = ChampionRegistry(tmp_path / "a.db")
    champion = await registry.bootstrap_champion()
    improvement = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, challenger = await improvement.propose(
        parent_champion=champion,
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
        idempotency_key=f"act-{uuid4()}",
    )
    await repos["promotions"].append(decision)
    service = EvolutionDecisionService(
        repos["promotions"],
        repos["configurations"],
        registry,
        activation_repo=repos["activations"],
    )
    await service.apply_persisted_decision(
        promotion_decision_id=decision.promotion_decision_id
    )
    await repos["configurations"].mark_status(
        challenger.configuration_version_id, "challenger"
    )
    await service.apply_persisted_decision(
        promotion_decision_id=decision.promotion_decision_id
    )
    repaired = await repos["configurations"].get_by_id(
        challenger.configuration_version_id
    )
    assert repaired is not None
    assert repaired.status == "champion"
    activation = await repos["activations"].get_by_decision_id(
        decision.promotion_decision_id
    )
    assert activation is not None
    assert activation.completed is True


@pytest.mark.asyncio
async def test_activation_completed_retry_is_noop(tmp_path):
    from joker.evolution.improvement import ImprovementProposalService
    from joker.evolution.schemas import PromptPatch

    apply_task3_migrations(tmp_path / "b.db")
    repos = build_evolution_repositories(tmp_path / "b.db")
    for r in repos.values():
        await r.initialize()
    registry = ChampionRegistry(tmp_path / "b.db")
    champion = await registry.bootstrap_champion()
    improvement = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, challenger = await improvement.propose(
        parent_champion=champion,
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
        idempotency_key=f"act2-{uuid4()}",
    )
    await repos["promotions"].append(decision)
    service = EvolutionDecisionService(
        repos["promotions"],
        repos["configurations"],
        registry,
        activation_repo=repos["activations"],
    )
    await service.apply_persisted_decision(
        promotion_decision_id=decision.promotion_decision_id
    )
    history_before = await registry.compare_champion_history(limit=20)
    await service.apply_persisted_decision(
        promotion_decision_id=decision.promotion_decision_id
    )
    history_after = await registry.compare_champion_history(limit=20)
    assert len(history_after) == len(history_before)


@pytest.mark.asyncio
async def test_episode_persists_authoritative_event_horizon():
    entry_ts = datetime(2026, 7, 1, 14, 30, tzinfo=timezone.utc)
    term_ts = datetime(2026, 7, 1, 15, 45, tzinfo=timezone.utc)
    entry_eid = uuid4()
    term_eid = uuid4()
    ep = TradingEpisode(
        session_id="s",
        run_id="r",
        trading_date=entry_ts.date(),
        initial_snapshot_id=uuid4(),
        terminal_snapshot_id=uuid4(),
        action_class="closed_trade",
        configuration_version_id=uuid4(),
        quantity=Decimal("1"),
        realised_pnl=Decimal("1"),
        completed=True,
        idempotency_key=f"h-{uuid4()}",
        entry_decision_event_id=entry_eid,
        entry_decision_timestamp=entry_ts,
        terminal_event_id=term_eid,
        terminal_event_timestamp=term_ts,
        market_event_ids=(entry_eid, term_eid),
    )
    assert ep.terminal_event_timestamp != ep.entry_decision_timestamp
    assert ep.market_event_ids == (entry_eid, term_eid)


@pytest.mark.asyncio
async def test_replay_terminal_time_uses_position_closed_event(tmp_path):
    class Snap:
        def __init__(self, sid, ts):
            self.snapshot_id = sid
            self.exchange_timestamp = ts
            self.exchange_time = ts
            self.underlying = type(
                "U",
                (),
                {
                    "bid": Decimal("500"),
                    "ask": Decimal("500.1"),
                    "last": Decimal("500"),
                },
            )()
            self.data_quality_id = uuid4()
            self.option_surface_id = None
            self.contracts = ()

        @property
        def timestamp(self):
            return self.exchange_timestamp

    class SnapRepo:
        def __init__(self):
            self._items = {}

        async def get_by_id(self, sid):
            return self._items.get(sid)

        async def list_between(self, *a, **k):
            return []

        async def list_by_session(self, session_id):
            return list(self._items.values())

    repo = SnapRepo()
    sid0 = uuid4()
    sid1 = uuid4()
    snap_ts = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    event_ts = datetime(2026, 7, 1, 15, 30, tzinfo=timezone.utc)
    repo._items[sid0] = Snap(sid0, datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc))
    repo._items[sid1] = Snap(sid1, event_ts)
    loader = ReplayTruthLoader(
        snapshot_repo=repo, allow_synthetic_starting_cash=True, session_starting_cash=Decimal("1000")
    )
    ep = TradingEpisode(
        session_id="s",
        run_id="r",
        trading_date=snap_ts.date(),
        initial_snapshot_id=sid0,
        terminal_snapshot_id=sid1,
        action_class="closed_trade",
        configuration_version_id=uuid4(),
        quantity=Decimal("1"),
        realised_pnl=Decimal("1"),
        completed=True,
        idempotency_key=f"t-{uuid4()}",
        entry_decision_event_id=uuid4(),
        entry_decision_timestamp=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
        terminal_event_id=uuid4(),
        terminal_event_timestamp=event_ts,
        market_event_ids=(),
    )
    truth = await loader.load_for_episode(ep)
    assert truth.terminal_event_timestamp == event_ts
    assert truth.terminal_event_timestamp != snap_ts


@pytest.mark.asyncio
async def test_replay_truth_empty_horizon_fails_closed() -> None:
    """Empty authoritative windows may use diagnostic frames only when marked incomplete."""
    from datetime import datetime, timezone

    from joker.evolution.replay_truth import ReplayTruthLoader
    from joker.evolution.schemas import TradingEpisode

    class Snap:
        def __init__(self, sid, ts):
            self.snapshot_id = sid
            self.exchange_timestamp = ts
            self.data_quality_id = uuid4()
            self.underlying = type("U", (), {"bid": Decimal("1"), "ask": Decimal("1"), "last": Decimal("1")})()
            self.option_surface_id = None

        @property
        def timestamp(self):
            return self.exchange_timestamp

    class SnapRepo:
        def __init__(self):
            self._items = {}

        async def get_by_id(self, sid):
            return self._items.get(sid)

        async def list_between(self, *a, **k):
            return []

    repo = SnapRepo()
    sid0 = uuid4()
    sid1 = uuid4()
    start_ts = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    end_ts = datetime(2026, 7, 1, 15, 30, tzinfo=timezone.utc)
    repo._items[sid0] = Snap(sid0, start_ts)
    repo._items[sid1] = Snap(sid1, end_ts)
    loader = ReplayTruthLoader(
        snapshot_repo=repo,
        allow_synthetic_starting_cash=True,
        session_starting_cash=Decimal("1000"),
    )
    ep = TradingEpisode(
        session_id="s",
        run_id="r",
        trading_date=start_ts.date(),
        initial_snapshot_id=sid0,
        terminal_snapshot_id=sid1,
        action_class="closed_trade",
        configuration_version_id=uuid4(),
        quantity=Decimal("1"),
        realised_pnl=Decimal("1"),
        completed=True,
        idempotency_key=f"t-{uuid4()}",
        entry_decision_event_id=uuid4(),
        entry_decision_timestamp=start_ts,
        terminal_event_id=uuid4(),
        terminal_event_timestamp=end_ts,
    )
    truth = await loader.load_for_episode(ep)
    assert truth.authoritative_horizon_complete is False
    assert "historical_ev_eligible=false" in truth.horizon_integrity_findings
    assert "promotion_eligible=false" in truth.horizon_integrity_findings
    assert "truth_degraded=true" in truth.horizon_integrity_findings
