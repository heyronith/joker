"""Shared fixtures for cognitive agent tests."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID, uuid4

from joker.cognition.context import ContextAssembler
from joker.cognition.schemas import (
    AgentDataRequest,
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
    OrderManagementAction,
    OrderManagementDecision,
    PatternHypothesis,
    PositionAction,
    PositionThesisVersion,
    StrategyHypothesis,
    StrategyLegCandidate,
)
from joker.market.snapshots import MarketSnapshot, UnderlyingSnapshot
from typing import Any

from joker.models import FakeModelProvider, ModelRegistry, ModelRouter, ModelsConfig
from joker.models.schemas import ModelRequest

SESSION_ID = "test-session"
CYCLE_ID = "cycle-1"
NOW = datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc)
SNAPSHOT_ID = uuid4()
MODEL_CALL_ID = uuid4()
PROMPT_VERSION = "2.0.0"


def make_market_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id=SNAPSHOT_ID,
        exchange_time=NOW,
        trading_date=date(2026, 7, 25),
        underlying=UnderlyingSnapshot(
            symbol="SPY",
            exchange_time=NOW,
            last=Decimal("550.0"),
        ),
        data_quality_id=uuid4(),
    )


def make_context(*, role: AgentRole):
    snapshot = make_market_snapshot()
    return ContextAssembler().assemble(
        agent_role=role,
        session_id=SESSION_ID,
        cycle_id=CYCLE_ID,
        snapshot=snapshot,
    )


def make_evidence_ref() -> EvidenceReference:
    return EvidenceReference(
        snapshot_id=SNAPSHOT_ID,
        source_type="underlying",
        source_id="SPY",
        observed_at=NOW,
        value_summary="SPY last near 550",
    )


def cycle_fields() -> dict:
    return {
        "session_id": SESSION_ID,
        "snapshot_id": SNAPSHOT_ID,
        "cycle_id": CYCLE_ID,
        "prompt_version": PROMPT_VERSION,
        "model_call_id": MODEL_CALL_ID,
    }


def make_agent_evidence(**kwargs) -> AgentEvidence:
    ref = make_evidence_ref()
    defaults = {
        **cycle_fields(),
        "agent_role": AgentRole.MARKET_STRUCTURE,
        "claim": "Range-bound structure between 548 and 552",
        "direction": MarketDirection.UNCERTAIN,
        "confidence": 0.65,
        "supporting_references": (ref,),
    }
    defaults.update(kwargs)
    return AgentEvidence(**defaults)


def make_pattern_hypothesis(**kwargs) -> PatternHypothesis:
    defaults = {
        **cycle_fields(),
        "name": "compression_breakout_setup",
        "description": "Tight range with rising volume",
        "direction": MarketDirection.BULLISH,
        "expected_horizon_seconds": 900,
        "novelty_score": 0.7,
        "confidence": 0.6,
        "agent_role": AgentRole.PATTERN_MINER,
    }
    defaults.update(kwargs)
    return PatternHypothesis(**defaults)


def make_strategy_plans():
    return {
        "entry_plan": EntryPlan(entry_style="immediate", preferred_order_type="limit"),
        "execution_plan": ExecutionPlan(
            max_quote_age_seconds=30,
            partial_fill_policy="wait",
            replacement_policy="none",
        ),
        "exit_plan": ExitPlan(stop_conditions=("thesis_invalidated",)),
        "invalidation_plan": InvalidationPlan(conditions=("break_below_support",)),
    }


def make_strategy_hypothesis(**kwargs) -> StrategyHypothesis:
    plans = make_strategy_plans()
    defaults = {
        **cycle_fields(),
        "name": "failed_breakout_reclaim_call",
        "market_thesis": "Failed breakout reclaim favors bullish 0DTE call",
        "direction": MarketDirection.BULLISH,
        "expected_horizon_seconds": 1800,
        "confidence": 0.55,
        "novelty_score": 0.8,
        "agent_role": AgentRole.BULLISH_INVENTOR,
        "candidate_legs": (
            StrategyLegCandidate(
                contract_id="SPY:2026-07-25:550.0:call",
                side="buy",
                option_type="call",
                strike=Decimal("550.0"),
                quantity=1,
                rationale="ATM call on reclaim",
            ),
        ),
        **plans,
    }
    defaults.update(kwargs)
    return StrategyHypothesis(**defaults)


def make_debate_review(*, role: AgentRole, strategy_id: UUID, **kwargs) -> DebateReview:
    defaults = {
        "strategy_id": strategy_id,
        "snapshot_id": SNAPSHOT_ID,
        "cycle_id": CYCLE_ID,
        "reviewer_role": role,
        "verdict": "support",
        "confidence": 0.6,
        "prompt_version": PROMPT_VERSION,
        "model_call_id": MODEL_CALL_ID,
    }
    defaults.update(kwargs)
    return DebateReview(**defaults)


def make_meta_decision(*, strategy_id: UUID | None = None, **kwargs) -> MetaDecision:
    defaults = {
        **cycle_fields(),
        "action": MetaDecisionAction.EXECUTE,
        "selected_strategy_id": strategy_id,
        "confidence": 0.7,
        "rationale_summary": "Debate supports execution with manageable execution risk",
    }
    defaults.update(kwargs)
    return MetaDecision(**defaults)


def make_execution_proposal(*, decision_id: UUID, strategy_id: UUID, **kwargs) -> ExecutionProposal:
    defaults = {
        "decision_id": decision_id,
        "strategy_id": strategy_id,
        "session_id": SESSION_ID,
        "cycle_id": CYCLE_ID,
        "snapshot_id": SNAPSHOT_ID,
        "action": "execute",
        "legs": (
            ExecutionLeg(
                contract_id="SPY:2026-07-25:550.0:call",
                side="buy",
                quantity=1,
                limit_price=Decimal("1.05"),
                sequence_order=1,
                max_quote_age_seconds=30,
                replacement_policy="none",
                partial_fill_policy="wait",
            ),
        ),
        "order_type": "limit",
        "time_in_force": "day",
        "entry_rationale": "Limit entry at mid after reclaim confirmation",
        "prompt_version": PROMPT_VERSION,
        "model_call_id": MODEL_CALL_ID,
    }
    defaults.update(kwargs)
    return ExecutionProposal(**defaults)


def make_order_management_decision(**kwargs) -> OrderManagementDecision:
    defaults = {
        **cycle_fields(),
        "client_order_id": "cog-order-1",
        "action": OrderManagementAction.CONTINUE_WAITING,
        "rationale_summary": "Quote still inside acceptable spread",
    }
    defaults.update(kwargs)
    return OrderManagementDecision(**defaults)


def make_position_thesis(**kwargs) -> PositionThesisVersion:
    defaults = {
        "position_id": "pos-1",
        "contract_id": "SPY:2026-07-25:550.0:call",
        "session_id": SESSION_ID,
        "snapshot_id": SNAPSHOT_ID,
        "original_strategy_id": uuid4(),
        "current_thesis": "Reclaim thesis still valid",
        "recommended_action": PositionAction.HOLD,
        "confidence": 0.6,
        "prompt_version": PROMPT_VERSION,
        "model_call_id": MODEL_CALL_ID,
    }
    defaults.update(kwargs)
    return PositionThesisVersion(**defaults)


class QueueingFakeProvider(FakeModelProvider):
    """Fake provider that dequeues canned outputs per role for multi-step tests."""

    def __init__(self) -> None:
        super().__init__()
        self._queues: dict[str, list[Any]] = {}

    def queue_for_role(self, role: str, items: list[Any]) -> None:
        self._queues[role] = list(items)

    def _lookup_raw(self, request: ModelRequest) -> Any:
        queue = self._queues.get(request.role)
        if queue:
            return queue.pop(0)
        return super()._lookup_raw(request)


def make_router(provider: FakeModelProvider | None = None) -> ModelRouter:
    fake = provider or FakeModelProvider()
    registry = ModelRegistry(ModelsConfig(), providers={"ollama": fake, "openai": fake})
    return ModelRouter(registry, session_id=SESSION_ID)
