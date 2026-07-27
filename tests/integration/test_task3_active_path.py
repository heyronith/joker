"""Task 3 active-path: Task1 surface → projection-backed episode → shadow."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.episode_compiler import EpisodeCompiler
from joker.evolution.improvement import ImprovementProposalService
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import PromptPatch
from joker.evolution.shadow import ShadowRuntime
from joker.evaluation.graph import EvaluationGraphRunner
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID
from tests.evolution.projection_helpers import (
    FakeExecutionProjection,
    closed_trade_projection,
)

ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_task3_active_path_with_task1_task2_surface(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "active.db"
    apply_task3_migrations(db)
    broker = PaperBroker(slippage_pct=0)
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id="task3-active",
            broker_account_id="paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
    )
    await supervisor.start(start_agent=False)
    market = supervisor.market_runtime
    assert market is not None

    for i in range(3):
        ts = start + timedelta(minutes=i, seconds=5)
        clock.set_now(ts)
        await market.ingest_underlying_quote(
            symbol="SPY",
            bid=Decimal("499.90"),
            ask=Decimal("500.10"),
            last=Decimal("500") + Decimal(i),
            source_timestamp=ts,
            received_timestamp=ts,
        )
    await market.ingest_option_quotes(
        [
            {
                "contract_id": CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "1.00",
                "ask": "1.20",
                "last": "1.10",
                "quote_timestamp": start + timedelta(minutes=3),
                "is_0dte": True,
            }
        ]
    )
    tick = await market.tick(now=start + timedelta(minutes=3, seconds=3))
    assert tick.snapshot is not None
    assert tick.quality is not None
    assert tick.quality.usable_for_execution is True

    repos = build_evolution_repositories(db)
    for repo in repos.values():
        await repo.initialize()
    registry = ChampionRegistry(db)
    champion = await registry.bootstrap_champion()
    compiler = EpisodeCompiler(repos["episodes"], repos["traces"])
    evaluator = EvaluationGraphRunner(repos["evaluations"])

    event_id = str(uuid4())
    episode = await compiler.compile_from_position_closed(
        session_id="task3-active",
        run_id="run",
        trading_date=date(2026, 7, 1),
        configuration_version_id=champion.configuration_version_id,
        event_payload={
            "contract_id": CONTRACT_ID,
            "client_order_id": "exit-1",
            "realized_pnl": "30",
        },
        event_id=event_id,
        execution=FakeExecutionProjection(
            closed_trade_projection(
                contract_id=CONTRACT_ID,
                entry_price=Decimal("1.10"),
                exit_price=Decimal("1.40"),
                realised_pnl=Decimal("30"),
            )
        ),
        initial_snapshot_id=tick.snapshot.snapshot_id,
            terminal_snapshot_id=uuid4(),
    )
    assert episode.completed is True
    evaluation = await evaluator.evaluate(episode)
    assert evaluation.episode_id == episode.episode_id

    improvement = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    proposal, challenger = await improvement.propose(
        parent_champion=champion,
        weakness="execution_quality",
        hypothesis="emphasize spread discipline",
        patch=PromptPatch(
            role="meta_decision",
            parent_prompt_version_id=uuid4(),
            replacement_template="Prefer tighter spreads when confidence is moderate.",
            change_rationale="fill quality",
        ),
    )
    ok, problems = await registry.policy_store.verify_configuration_resolvable(challenger)
    assert ok, problems

    shadow = ShadowRuntime(repos["shadow"], policy_store=registry.policy_store)
    await shadow.start()
    assignment = await shadow.register_challenger(challenger=challenger, champion=champion)
    await shadow.enqueue_snapshot(
        assignment_id=assignment.assignment_id,
        challenger_version_id=challenger.configuration_version_id,
        snapshot_id=str(tick.snapshot.snapshot_id),
        payload={"snapshot_id": str(tick.snapshot.snapshot_id)},
    )
    import asyncio

    await asyncio.sleep(0.05)
    assert shadow.results
    current = await registry.get_current_champion()
    assert current.configuration_version_id == champion.configuration_version_id
    assert proposal.status == "registered"
    await shadow.stop()
    await registry.close()
    await supervisor.shutdown()
