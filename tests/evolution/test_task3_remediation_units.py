"""Unit coverage for Task 3 remediation workstreams A–F."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from joker.cognition.memory_policy import select_memories
from joker.cognition.prompt_overrides import (
    get_active_debate_policy,
    pinned_configuration_overrides,
)
from joker.evolution.checkpointers import EvolutionCheckpointerOwner
from joker.evolution.configuration_applicator import ConfigurationApplicator
from joker.evolution.episode_compiler import EpisodeCompiler
from joker.evolution.lifecycle import PositionLifecycleResolver
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.replay_execution import ReplayExecutionError, ReplayExecutionRuntime
from joker.evolution.replay_market import ReplayEpisodeTruth
from joker.evolution.replay_position_runtime import ReplayPositionRuntime
from joker.evolution.repositories import build_evolution_repositories
from joker.ledger.projector import OrderLifecycle, OrderStatus, PositionState, ProjectionState
from tests.evolution.projection_helpers import (
    FakeExecutionProjection,
    closed_trade_projection,
)


@pytest.mark.asyncio
async def test_evolution_checkpointers_open_and_close(tmp_path) -> None:
    db = tmp_path / "task3.db"
    owner = EvolutionCheckpointerOwner(db)
    savers = await owner.open_all()
    assert savers.evaluation is not None
    for path in owner.paths().values():
        assert path.exists()
    await owner.close_all()
    await owner.close_all()  # idempotent


@pytest.mark.asyncio
async def test_two_round_trips_same_contract_create_two_independent_episodes(
    tmp_path,
) -> None:
    db = tmp_path / "life.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    await repos["episodes"].initialize()
    compiler = EpisodeCompiler(repos["episodes"])
    cfg = uuid4()
    contract = "SPY:2026-07-01:500:call"
    life1 = "s:e1:SPY:2026-07-01:500:call"
    life2 = "s:e2:SPY:2026-07-01:500:call"
    orders = {
        "e1": OrderLifecycle(
            client_order_id="e1",
            status=OrderStatus.FILLED,
            submitted_qty=Decimal("1"),
            filled_qty=Decimal("1"),
            avg_fill_price=Decimal("1.00"),
            side="buy",
            contract_id=contract,
            fees=Decimal("0.65"),
            position_lifecycle_id=life1,
            originating_entry_client_order_id="e1",
        ),
        "x1": OrderLifecycle(
            client_order_id="x1",
            status=OrderStatus.FILLED,
            submitted_qty=Decimal("1"),
            filled_qty=Decimal("1"),
            avg_fill_price=Decimal("1.20"),
            side="sell",
            contract_id=contract,
            fees=Decimal("0.65"),
            position_lifecycle_id=life1,
            originating_entry_client_order_id="e1",
            parent_client_order_id="e1",
        ),
        "e2": OrderLifecycle(
            client_order_id="e2",
            status=OrderStatus.FILLED,
            submitted_qty=Decimal("2"),
            filled_qty=Decimal("2"),
            avg_fill_price=Decimal("1.05"),
            side="buy",
            contract_id=contract,
            fees=Decimal("1.30"),
            position_lifecycle_id=life2,
            originating_entry_client_order_id="e2",
        ),
        "x2": OrderLifecycle(
            client_order_id="x2",
            status=OrderStatus.FILLED,
            submitted_qty=Decimal("2"),
            filled_qty=Decimal("2"),
            avg_fill_price=Decimal("0.95"),
            side="sell",
            contract_id=contract,
            fees=Decimal("1.30"),
            position_lifecycle_id=life2,
            originating_entry_client_order_id="e2",
            parent_client_order_id="e2",
        ),
    }
    projection = ProjectionState(
        orders=orders,
        positions={
            contract: PositionState(
                contract_id=contract,
                quantity=Decimal("0"),
                avg_price=Decimal("1.05"),
                realized_pnl=Decimal("-22.60"),
                open=False,
            )
        },
    )
    exec_proj = FakeExecutionProjection(projection)
    snap1, snap2 = uuid4(), uuid4()
    ep1 = await compiler.compile_from_position_closed(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=cfg,
        event_payload={
            "contract_id": contract,
            "client_order_id": "x1",
            "realized_pnl": "18.70",
        },
        event_id=str(uuid4()),
        execution=exec_proj,
        initial_snapshot_id=snap1,
            terminal_snapshot_id=uuid4(),
    )
    ep2 = await compiler.compile_from_position_closed(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=cfg,
        event_payload={
            "contract_id": contract,
            "client_order_id": "x2",
            "realized_pnl": "-22.60",
        },
        event_id=str(uuid4()),
        execution=exec_proj,
        initial_snapshot_id=snap2,
            terminal_snapshot_id=uuid4(),
    )
    assert ep1.idempotency_key != ep2.idempotency_key
    assert ep1.quantity == Decimal("1")
    assert ep2.quantity == Decimal("2")
    assert ep1.entry_order_ids == ("e1",)
    assert ep2.entry_order_ids == ("e2",)
    assert ep1.total_fees == Decimal("1.30")
    assert ep2.total_fees == Decimal("2.60")


@pytest.mark.asyncio
async def test_missing_snapshot_creates_incomplete_episode_without_random_id(
    tmp_path,
) -> None:
    db = tmp_path / "miss.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    await repos["episodes"].initialize()
    compiler = EpisodeCompiler(repos["episodes"])
    contract = "SPY:2026-07-01:500:call"
    ep = await compiler.compile_from_position_closed(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=uuid4(),
        event_payload={"contract_id": contract, "client_order_id": "exit-1"},
        event_id=str(uuid4()),
        execution=FakeExecutionProjection(
            closed_trade_projection(contract_id=contract, realised_pnl=Decimal("50"))
        ),
        initial_snapshot_id=None,
            terminal_snapshot_id=None,
    )
    assert ep.completed is False
    assert "missing_initial_snapshot" in ep.completeness_findings
    assert ep.initial_snapshot_id is None
    assert ep.snapshot_identity_status == "missing"


@pytest.mark.asyncio
async def test_replay_wrong_contract_does_not_receive_historical_pnl() -> None:
    truth = ReplayEpisodeTruth(
        starting_cash=__import__("decimal").Decimal("100000"),
        episode_id=uuid4(),
        initial_snapshot_id=uuid4(),
            terminal_snapshot_id=uuid4(),
        contract_quotes={
            "GOOD": {"bid": "1.00", "ask": "1.02", "mid": "1.01"},
        },
    )
    runtime = ReplayExecutionRuntime(truth=truth)
    runtime.lock_surface({"GOOD"})
    with pytest.raises(ReplayExecutionError):
        runtime.submit_order(
            client_order_id="bad",
            contract_id="BAD",
            side="buy",
            quantity=Decimal("1"),
        )


@pytest.mark.asyncio
async def test_replay_no_trade_has_zero_position_and_pnl() -> None:
    truth = ReplayEpisodeTruth(
        starting_cash=__import__("decimal").Decimal("100000"),
        episode_id=uuid4(),
        initial_snapshot_id=uuid4(),
            terminal_snapshot_id=uuid4(),
        contract_quotes={"C": {"bid": "1.00", "ask": "1.02", "mid": "1.01"}},
    )
    pos = ReplayPositionRuntime(
        execution=ReplayExecutionRuntime(truth=truth),
        configuration_version_id=uuid4(),
    )
    out = pos.simulate_entry_from_meta(action="no_trade", contract_id="C")
    assert out["traded"] is False
    assert pos.execution.realised_pnl() == Decimal("0")
    assert not any(p.quantity > 0 for p in pos.execution.positions.values())


@pytest.mark.asyncio
async def test_replay_partial_fill_and_replace() -> None:
    truth = ReplayEpisodeTruth(
        starting_cash=__import__("decimal").Decimal("100000"),
        episode_id=uuid4(),
        initial_snapshot_id=uuid4(),
            terminal_snapshot_id=uuid4(),
        contract_quotes={"C": {"bid": "1.00", "ask": "1.02", "mid": "1.01"}},
    )
    rt = ReplayExecutionRuntime(truth=truth)
    first = rt.submit_order(
        client_order_id="o1",
        contract_id="C",
        side="buy",
        quantity=Decimal("2"),
        fill_fraction=Decimal("0.5"),
    )
    assert first.status == "partially_filled"
    replaced = rt.replace_order(
        parent_order_id="o1",
        client_order_id="o2",
        quantity=Decimal("1"),
    )
    assert replaced.filled_qty == Decimal("1")


@pytest.mark.asyncio
async def test_replay_restart_does_not_duplicate_fill() -> None:
    truth = ReplayEpisodeTruth(
        starting_cash=__import__("decimal").Decimal("100000"),
        episode_id=uuid4(),
        initial_snapshot_id=uuid4(),
            terminal_snapshot_id=uuid4(),
        contract_quotes={"C": {"bid": "1.00", "ask": "1.02", "mid": "1.01"}},
    )
    rt = ReplayExecutionRuntime(truth=truth)
    key = "idem-1"
    a = rt.submit_order(
        client_order_id="o1",
        contract_id="C",
        side="buy",
        quantity=Decimal("1"),
        idempotency_key=key,
    )
    b = rt.submit_order(
        client_order_id="o1",
        contract_id="C",
        side="buy",
        quantity=Decimal("1"),
        idempotency_key=key,
    )
    assert a is b
    assert len(rt.fills) == 1


@pytest.mark.asyncio
async def test_context_policy_changes_context_budget() -> None:
    from joker.cognition.prompt_overrides import get_active_context_policy

    with pinned_configuration_overrides(
        configuration_version_id=str(uuid4()),
        prompt_overrides={},
        role_profiles={},
        context_policy={"max_1m_bars": 5, "max_context_characters": 1000},
    ):
        policy = get_active_context_policy()
        assert policy is not None
        assert policy["max_1m_bars"] == 5
        assert policy["max_context_characters"] == 1000


def test_memory_policy_changes_memory_selection() -> None:
    class M:
        def __init__(self, regime: str, is_contradiction: bool = False):
            self.regime = regime
            self.is_contradiction = is_contradiction

    memories = [M("a"), M("b"), M("a", True), M("a")]
    with pinned_configuration_overrides(
        configuration_version_id=str(uuid4()),
        prompt_overrides={},
        role_profiles={},
        memory_policy={
            "max_memories": 2,
            "include_contradictions": False,
            "regime_matching": True,
        },
    ):
        selected = select_memories(memories, regime="a")
    assert len(selected) == 2
    assert all(m.regime == "a" and not m.is_contradiction for m in selected)


def test_debate_policy_round_limit_visible() -> None:
    with pinned_configuration_overrides(
        configuration_version_id=str(uuid4()),
        prompt_overrides={},
        role_profiles={},
        debate_policy={"maximum_rounds": 1},
    ):
        assert get_active_debate_policy()["maximum_rounds"] == 1


@pytest.mark.asyncio
async def test_configuration_applicator_loads_all_policies(tmp_path) -> None:
    db = tmp_path / "pol.db"
    store = PolicyVersionStore(db)
    await store.initialize()
    cfg = await store.bootstrap_defaults()
    applied = await ConfigurationApplicator(store).apply(cfg)
    assert applied.context_policy
    assert applied.memory_policy
    assert applied.debate_policy
    assert applied.routing_policy
    assert applied.escalation_policy
    assert applied.prompt_overrides


@pytest.mark.asyncio
async def test_lifecycle_pnl_matches_fills_and_fees() -> None:
    resolver = PositionLifecycleResolver()
    contract = "SPY:2026-07-01:500:call"
    projection = ProjectionState(
        orders={
            "e1": OrderLifecycle(
                client_order_id="e1",
                status=OrderStatus.FILLED,
                submitted_qty=Decimal("1"),
                filled_qty=Decimal("1"),
                avg_fill_price=Decimal("1.00"),
                side="buy",
                contract_id=contract,
                fees=Decimal("0.65"),
            ),
            "x1": OrderLifecycle(
                client_order_id="x1",
                status=OrderStatus.FILLED,
                submitted_qty=Decimal("1"),
                filled_qty=Decimal("1"),
                avg_fill_price=Decimal("1.50"),
                side="sell",
                contract_id=contract,
                fees=Decimal("0.65"),
            ),
        },
        positions={},
    )
    resolved = await resolver.resolve_closed_lifecycle(
        session_id="s",
        terminal_event_id="t",
        contract_id=contract,
        client_order_id="x1",
        projection=projection,
        known_snapshot_id=uuid4(),
    )
    assert resolved.realised_pnl == Decimal("50") - Decimal("1.30")
    assert resolved.total_fees == Decimal("1.30")
