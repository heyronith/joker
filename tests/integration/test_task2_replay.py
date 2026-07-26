"""Task 2 integration replay with FakeModelProvider canned outputs."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.cognition.schemas import (
    AgentEvidence,
    AgentRole,
    DebateReview,
    EntryPlan,
    EvidenceReference,
    ExecutionLeg,
    ExecutionPlan,
    ExecutionProposal,
    ExitPlan,
    InvalidationPlan,
    MarketDirection,
    MetaDecision,
    MetaDecisionAction,
    PatternHypothesis,
    StrategyHypothesis,
    StrategyLegCandidate,
)
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import EventType, make_event
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.models.schemas import utc_now
from joker.runtime.cognitive_agent_runtime import CognitiveAgentRuntime
from joker.runtime.execution_runtime import ExecutionCommand
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.schemas.domain import OptionContract, OrderIntent

ET = ZoneInfo("America/New_York")


def _ref(snapshot_id) -> EvidenceReference:
    return EvidenceReference(
        snapshot_id=snapshot_id,
        source_type="underlying",
        source_id="SPY",
        observed_at=utc_now(),
        value_summary="test ref",
    )


def _register_canned(fake: FakeModelProvider, snapshot_id, cycle_id: str) -> uuid4:
    sid = snapshot_id
    session = "sess-t2"
    mc = uuid4()

    for role in (
        AgentRole.MARKET_STRUCTURE,
        AgentRole.VOLATILITY,
        AgentRole.OPTIONS_MICROSTRUCTURE,
        AgentRole.TEMPORAL_CONTEXT,
        AgentRole.ANOMALY,
    ):
        fake.set_canned_for_role(
            role.value,
            AgentEvidence(
                session_id=session,
                snapshot_id=sid,
                prompt_version="2.0.0",
                model_call_id=mc,
                cycle_id=cycle_id,
                agent_role=role,
                claim=f"{role.value} claim",
                direction=MarketDirection.BULLISH,
                confidence=0.7,
                supporting_references=(_ref(sid),),
            ),
        )

    fake.set_canned_for_role(
        "pattern_miner",
        PatternHypothesis(
            session_id=session,
            snapshot_id=sid,
            prompt_version="2.0.0",
            model_call_id=mc,
            cycle_id=cycle_id,
            name="breakout",
            description="test pattern",
            direction=MarketDirection.BULLISH,
            expected_horizon_seconds=300,
            novelty_score=0.5,
            confidence=0.6,
            agent_role=AgentRole.PATTERN_MINER,
            supporting_evidence_ids=(),
        ),
    )
    fake.set_canned_for_role(
        "sequence_analyst",
        PatternHypothesis(
            session_id=session,
            snapshot_id=sid,
            prompt_version="2.0.0",
            model_call_id=mc,
            cycle_id=cycle_id,
            name="sequence",
            description="seq pattern",
            direction=MarketDirection.BULLISH,
            expected_horizon_seconds=300,
            novelty_score=0.4,
            confidence=0.55,
            agent_role=AgentRole.SEQUENCE_ANALYST,
        ),
    )
    fake.set_canned_for_role(
        "analogy_retriever",
        PatternHypothesis(
            session_id=session,
            snapshot_id=sid,
            prompt_version="2.0.0",
            model_call_id=mc,
            cycle_id=cycle_id,
            name="analogy",
            description="weak analogy",
            direction=MarketDirection.BULLISH,
            expected_horizon_seconds=300,
            novelty_score=0.2,
            confidence=0.4,
            agent_role=AgentRole.ANALOGY_RETRIEVER,
        ),
    )

    strategy_id = uuid4()
    leg = StrategyLegCandidate(
        contract_id="SPY:2026-07-01:500.0:call",
        side="buy",
        option_type="call",
        strike=Decimal("500"),
        quantity=1,
        rationale="test leg",
    )
    strategy = StrategyHypothesis(
        session_id=session,
        snapshot_id=sid,
        strategy_id=strategy_id,
        prompt_version="2.0.0",
        model_call_id=mc,
        cycle_id=cycle_id,
        name="bullish",
        market_thesis="bullish test",
        direction=MarketDirection.BULLISH,
        candidate_legs=(leg,),
        entry_plan=EntryPlan(entry_style="immediate", preferred_order_type="limit"),
        execution_plan=ExecutionPlan(
            max_quote_age_seconds=60,
            partial_fill_policy="wait",
            replacement_policy="none",
        ),
        exit_plan=ExitPlan(stop_conditions=("stop",)),
        invalidation_plan=InvalidationPlan(conditions=("inv",)),
        expected_horizon_seconds=600,
        confidence=0.65,
        novelty_score=0.5,
        agent_role=AgentRole.BULLISH_INVENTOR,
    )
    for role in ("bullish_inventor", "bearish_inventor", "neutral_advocate"):
        fake.set_canned_for_role(role, strategy)

    for role in (
        "strategy_advocate",
        "falsifier",
        "historical_critic",
        "execution_critic",
        "alternative_explanation",
    ):
        fake.set_canned_for_role(
            role,
            DebateReview(
                strategy_id=strategy_id,
                snapshot_id=sid,
                cycle_id=cycle_id,
                reviewer_role=AgentRole(role),
                verdict="support",
                confidence=0.6,
                prompt_version="2.0.0",
                model_call_id=mc,
            ),
        )

    decision_id = uuid4()
    fake.set_canned_for_role(
        "meta_decision",
        MetaDecision(
            session_id=session,
            snapshot_id=sid,
            decision_id=decision_id,
            prompt_version="2.0.0",
            model_call_id=mc,
            cycle_id=cycle_id,
            action=MetaDecisionAction.EXECUTE,
            selected_strategy_id=strategy_id,
            confidence=0.7,
            rationale_summary="execute test",
        ),
    )

    proposal_id = uuid4()
    fake.set_canned_for_role(
        "entry_tactician",
        ExecutionProposal(
            proposal_id=proposal_id,
            decision_id=decision_id,
            strategy_id=strategy_id,
            session_id=session,
            cycle_id=cycle_id,
            snapshot_id=sid,
            action="execute",
            legs=(
                ExecutionLeg(
                    contract_id="SPY:2026-07-01:500.0:call",
                    side="buy",
                    quantity=1,
                    limit_price=Decimal("1.10"),
                    sequence_order=0,
                    max_quote_age_seconds=60,
                    replacement_policy="none",
                    partial_fill_policy="wait",
                ),
            ),
            order_type="limit",
            time_in_force="day",
            entry_rationale="test entry",
            prompt_version="2.0.0",
            model_call_id=mc,
        ),
    )
    return strategy_id


@pytest.mark.asyncio
async def test_task2_replay_perception_to_execute_provenance(tmp_path) -> None:
    async def _run() -> None:
        start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
        db = tmp_path / "joker.db"
        broker = PaperBroker(slippage_pct=0)
        session_id = "sess-t2"
        cycle_id = "cycle-t2"

        supervisor = SessionSupervisor(
            broker=broker,
            config=SessionSupervisorConfig(
                db_path=db,
                session_id=session_id,
                broker_account_id="paper",
                market=MarketRuntimeConfig(
                    min_option_contracts=1,
                    underlying_stale_seconds=3600,
                    option_stale_seconds=3600,
                ),
            ),
        )
        await supervisor.start()
        assert supervisor.market_runtime is not None
        assert supervisor.execution_runtime is not None

        market = supervisor.market_runtime
        for i in range(3):
            ts = start + timedelta(minutes=i, seconds=5)
            await market.ingest_underlying_quote(
                symbol="SPY",
                bid=Decimal("499.90"),
                ask=Decimal("500.10"),
                last=Decimal("500") + Decimal(i),
                source_timestamp=ts,
                received_timestamp=ts,
            )
        tick = await market.tick(now=start + timedelta(minutes=3, seconds=3))
        assert tick.snapshot is not None
        snapshot_id = tick.snapshot.snapshot_id

        fake = FakeModelProvider(available=True)
        _register_canned(fake, snapshot_id, cycle_id)
        profiles = {
            name: profile.model_copy(update={"provider": "fake", "model": "fake-model"})
            for name, profile in default_model_profiles().items()
        }
        models_config = ModelsConfig(
            profiles=profiles,
        )
        models_config = models_config.model_copy(
            update={
                "ollama": models_config.ollama.model_copy(update={"enabled": False}),
                "openai": models_config.openai.model_copy(update={"enabled": False}),
            }
        )
        registry = ModelRegistry(models_config, providers={"fake": fake})
        router = ModelRouter(registry, session_id=session_id)
        submitted: list[ExecutionCommand] = []

        snapshot_repo = SnapshotRepository(db)
        await snapshot_repo.initialize()

        async def submit_callback(provenanced) -> object:
            submitted.append(provenanced.command)
            return await supervisor.execution_runtime.submit_execution_command(
                provenanced.command
            )

        deps = CognitiveGraphDeps(
            router=router,
            config=CognitiveGraphSettings(),
            session_id=session_id,
            run_id=session_id,
            snapshot_repo=snapshot_repo,
            submit_callback=submit_callback,
            event_bus=supervisor.event_bus,
            execution_runtime=supervisor.execution_runtime,
        )
        graph = build_cognitive_graph(deps)
        state = initial_cycle_state(
            session_id=session_id,
            run_id=session_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(snapshot_id),
        )
        result = await graph.ainvoke(state)
        assert result.get("execution_command_id") is not None
        assert len(submitted) == 1
        assert submitted[0].intent.contract.symbol == "SPY"
        assert result.get("meta_decision") is not None
        assert result["meta_decision"].action == MetaDecisionAction.EXECUTE

        # Position hold cycle via runtime
        runtime = CognitiveAgentRuntime(
            session_id=session_id,
            run_id=session_id,
            router=router,
            config=CognitiveGraphSettings(),
            graph_deps=deps,
            registry=registry,
        )
        await runtime.start()
        await runtime.on_event(
            make_event(
                EventType.POSITION_OPENED,
                session_id=session_id,
                source="test",
                exchange_timestamp=start,
                payload={
                    "position_id": "pos-1",
                    "contract_id": "SPY:2026-07-01:500.0:call",
                    "snapshot_id": str(snapshot_id),
                    "original_strategy_id": str(result["meta_decision"].selected_strategy_id),
                },
            )
        )
        await asyncio.sleep(0.1)
        await runtime.shutdown()

        # Exit and verify ledger PnL path exists
        contract = OptionContract(
            symbol="SPY",
            expiration=date(2026, 7, 1),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        )
        sell_intent = OrderIntent(
            candidate_id="exit-1",
            contract=contract,
            side="sell",
            order_type="limit",
            quantity=1,
            limit_price=1.20,
        )
        await supervisor.execution_runtime.submit_execution_command(
            ExecutionCommand(client_order_id="exit-1", intent=sell_intent)
        )
        projected = await supervisor.execution_runtime.project_session()
        assert projected.orders

        await supervisor.shutdown()

    await _run()
