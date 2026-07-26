"""Unit tests for Task 2 cognitive agents."""

from __future__ import annotations

from uuid import uuid4

import pytest

from joker.agents.cognitive import (
    BullishInventorAgent,
    ExecutionCommandCompiler,
    ExecutionProposalValidator,
    MarketStructureAgent,
    MetaDecisionAgent,
    is_novel_strategy_name,
    run_agent_with_optional_data_request,
    run_debate_panel,
    run_discovery_swarm,
    run_perception_swarm,
    select_pattern_hypotheses,
    validate_meta_decision,
)
from joker.cognition.exceptions import CognitiveValidationError
from joker.cognition.schemas import (
    AgentDataRequest,
    AgentRole,
    DebateReview,
    MetaDecisionAction,
    PatternHypothesis,
)
from joker.cognition.tools import InMemoryCognitiveReadTools
from joker.models import FakeModelProvider, ModelResponseEmpty

from tests.agents.cognitive.conftest import (
    CYCLE_ID,
    SESSION_ID,
    SNAPSHOT_ID,
    QueueingFakeProvider,
    make_agent_evidence,
    make_context,
    make_debate_review,
    make_execution_proposal,
    make_market_snapshot,
    make_meta_decision,
    make_pattern_hypothesis,
    make_router,
    make_strategy_hypothesis,
)


@pytest.mark.asyncio
async def test_market_structure_agent_with_canned_output() -> None:
    provider = FakeModelProvider()
    provider.set_canned_for_role(
        AgentRole.MARKET_STRUCTURE.value,
        make_agent_evidence(agent_role=AgentRole.MARKET_STRUCTURE),
    )
    router = make_router(provider)
    context = make_context(role=AgentRole.MARKET_STRUCTURE)

    evidence = await MarketStructureAgent().run(context, router)

    assert evidence.agent_role == AgentRole.MARKET_STRUCTURE
    assert evidence.session_id == SESSION_ID
    assert evidence.cycle_id == CYCLE_ID
    assert evidence.snapshot_id == SNAPSHOT_ID
    assert len(provider.calls) == 1
    assert provider.calls[0].request.idempotency_key


@pytest.mark.asyncio
async def test_run_perception_swarm_gathers_five_evidence_items() -> None:
    provider = FakeModelProvider()
    for role in (
        AgentRole.MARKET_STRUCTURE,
        AgentRole.VOLATILITY,
        AgentRole.OPTIONS_MICROSTRUCTURE,
        AgentRole.TEMPORAL_CONTEXT,
        AgentRole.ANOMALY,
    ):
        provider.set_canned_for_role(
            role.value,
            make_agent_evidence(agent_role=role),
        )

    router = make_router(provider)
    context = make_context(role=AgentRole.MARKET_STRUCTURE)
    evidence_items = await run_perception_swarm(router, context)

    assert len(evidence_items) == 5
    roles = {item.agent_role for item in evidence_items}
    assert roles == {
        AgentRole.MARKET_STRUCTURE,
        AgentRole.VOLATILITY,
        AgentRole.OPTIONS_MICROSTRUCTURE,
        AgentRole.TEMPORAL_CONTEXT,
        AgentRole.ANOMALY,
    }


@pytest.mark.asyncio
async def test_run_agent_with_optional_data_request_round_trip() -> None:
    provider = QueueingFakeProvider()
    data_request = AgentDataRequest(
        request_type="bars",
        parameters={"timeframe": "1m", "limit": 5},
        reason="Need recent 1m bars to confirm structure",
    )
    provider.queue_for_role(
        AgentRole.MARKET_STRUCTURE.value,
        [
            make_agent_evidence(
                agent_role=AgentRole.MARKET_STRUCTURE,
                requires_more_data=True,
                data_request=data_request,
            ),
            make_agent_evidence(
                agent_role=AgentRole.MARKET_STRUCTURE,
                claim="Confirmed range after supplemental bars",
                requires_more_data=False,
            ),
        ],
    )

    tools = InMemoryCognitiveReadTools()
    tools.seed_snapshot(make_market_snapshot())
    router = make_router(provider)
    context = make_context(role=AgentRole.MARKET_STRUCTURE)
    agent = MarketStructureAgent()

    evidence = await run_agent_with_optional_data_request(agent, context, router, tools)

    assert evidence.requires_more_data is False
    assert evidence.claim == "Confirmed range after supplemental bars"
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_fake_provider_requires_canned_output_per_role() -> None:
    provider = FakeModelProvider()
    router = make_router(provider)
    context = make_context(role=AgentRole.MARKET_STRUCTURE)

    with pytest.raises(ModelResponseEmpty):
        await MarketStructureAgent().run(context, router)


def test_select_pattern_hypotheses_dedupes_and_bounds() -> None:
    hypotheses = (
        make_pattern_hypothesis(name="alpha", novelty_score=0.9, confidence=0.8),
        make_pattern_hypothesis(name="alpha", novelty_score=0.95, confidence=0.9),
        make_pattern_hypothesis(name="beta", novelty_score=0.5, confidence=0.9),
        make_pattern_hypothesis(name="gamma", novelty_score=0.8, confidence=0.7),
    )
    selected = select_pattern_hypotheses(hypotheses, max_hypotheses=2)
    assert len(selected) == 2
    names = {h.name for h in selected}
    assert "alpha" in names
    assert len(names) == 2


@pytest.mark.asyncio
async def test_bullish_inventor_produces_novel_strategy_name() -> None:
    strategy = make_strategy_hypothesis(name="failed_breakout_reclaim_call")
    provider = FakeModelProvider()
    provider.set_canned_for_role(AgentRole.BULLISH_INVENTOR.value, strategy)
    router = make_router(provider)
    context = make_context(role=AgentRole.BULLISH_INVENTOR)

    result = await BullishInventorAgent().run(context, router)

    assert result.name == "failed_breakout_reclaim_call"
    assert is_novel_strategy_name(
        result.name,
        {"opening_range_breakout", "vwap_reclaim"},
    )


@pytest.mark.asyncio
async def test_meta_decision_agent_calls_model_and_validates_strategy() -> None:
    strategy = make_strategy_hypothesis()
    decision = make_meta_decision(strategy_id=strategy.strategy_id)
    provider = FakeModelProvider()
    provider.set_canned_for_role(AgentRole.META_DECISION.value, decision)
    router = make_router(provider)
    context = make_context(role=AgentRole.META_DECISION)

    result = await MetaDecisionAgent().decide(
        context,
        router,
        strategies=[strategy],
        reviews=[],
    )

    assert result.action == MetaDecisionAction.EXECUTE
    assert result.selected_strategy_id == strategy.strategy_id
    validate_meta_decision(result, [strategy])


def test_validate_meta_decision_rejects_unknown_strategy() -> None:
    strategy = make_strategy_hypothesis()
    decision = make_meta_decision(strategy_id=uuid4())
    with pytest.raises(CognitiveValidationError):
        validate_meta_decision(decision, [strategy])


@pytest.mark.asyncio
async def test_run_debate_panel_returns_five_reviews() -> None:
    strategy = make_strategy_hypothesis()
    provider = FakeModelProvider()
    for role in (
        AgentRole.STRATEGY_ADVOCATE,
        AgentRole.FALSIFIER,
        AgentRole.HISTORICAL_CRITIC,
        AgentRole.EXECUTION_CRITIC,
        AgentRole.ALTERNATIVE_EXPLANATION,
    ):
        provider.set_canned_for_role(
            role.value,
            make_debate_review(role=role, strategy_id=strategy.strategy_id),
        )

    router = make_router(provider)
    context = make_context(role=AgentRole.STRATEGY_ADVOCATE)
    reviews = await run_debate_panel(strategy, context, router)

    assert len(reviews) == 5
    assert all(isinstance(r, DebateReview) for r in reviews)
    assert {r.reviewer_role for r in reviews} == {
        AgentRole.STRATEGY_ADVOCATE,
        AgentRole.FALSIFIER,
        AgentRole.HISTORICAL_CRITIC,
        AgentRole.EXECUTION_CRITIC,
        AgentRole.ALTERNATIVE_EXPLANATION,
    }


def test_execution_validator_and_compiler() -> None:
    strategy = make_strategy_hypothesis()
    decision = make_meta_decision(strategy_id=strategy.strategy_id)
    proposal = make_execution_proposal(
        decision_id=decision.decision_id,
        strategy_id=strategy.strategy_id,
    )

    ExecutionProposalValidator().validate(proposal, trading_mode="PAPER")

    provenanced = ExecutionCommandCompiler().compile(
        proposal,
        evidence_ids=(uuid4(),),
    )
    assert provenanced.command.intent.contract.symbol == "SPY"
    assert provenanced.command.intent.quantity == 1
    assert provenanced.strategy_id == str(strategy.strategy_id)
    assert len(provenanced.evidence_ids) == 1
