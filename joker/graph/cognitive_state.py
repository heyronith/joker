"""Task 2 cognitive graph state contract (spec section 18)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from typing_extensions import TypedDict

from joker.cognition.context import ContextPackage
from joker.cognition.schemas import (
    AgentDataRequest,
    AgentEvidence,
    CognitiveError,
    DebateReview,
    ExecutionProposal,
    GraphNodeTrace,
    MarketWorldModel,
    MetaDecision,
    PatternHypothesis,
    PositionThesisVersion,
    StrategyHypothesis,
)
from joker.graph.reducers import (
    merge_errors,
    merge_evidence,
    merge_hypotheses,
    merge_reviews,
    merge_strategies,
    merge_traces,
)
from joker.market.quality import DataQualityReport


class CognitiveGraphState(TypedDict, total=False):
    """Dedicated Task 2 LangGraph state — no full snapshots or option surfaces."""

    run_id: str
    session_id: str
    cycle_id: str

    trigger_event_id: str
    trigger_event_type: str
    snapshot_id: str
    latest_known_snapshot_id: str
    started_at: datetime

    context_ref: str
    evidence: Annotated[list[AgentEvidence], merge_evidence]
    world_model: MarketWorldModel | None

    hypotheses: Annotated[list[PatternHypothesis], merge_hypotheses]
    strategies: Annotated[list[StrategyHypothesis], merge_strategies]
    reviews: Annotated[list[DebateReview], merge_reviews]

    meta_decision: MetaDecision | None
    execution_proposal: ExecutionProposal | None
    execution_command_id: str | None
    execution_result_ref: str | None

    pending_evidence_requests: list[AgentDataRequest]
    pending_hypothesis_ids: list[str]

    node_trace: Annotated[list[GraphNodeTrace], merge_traces]
    errors: Annotated[list[CognitiveError], merge_errors]

    # Graph control (bounded loops — not trading decisions)
    debate_round: int
    strategy_switch_count: int
    selected_strategy_ids: list[str]
    stale_decision: bool

    # Runtime-only hydrated context (not persisted to checkpoints)
    _context_package: ContextPackage | None
    _data_quality: DataQualityReport | None
    _option_surface_id: str | None
    _feasibility_assessment: dict | None
    _feasibility_inputs: dict | None
    _strategy_scores: list | None
    _strategy_estimates: list | None
    _historical_summaries: list | None
    _historical_sample_count: int | None
    _historical_minimum_required: int | None
    _no_valid_strategy: bool
    _sizing_decision: dict | None
    _objective_context: dict | None
    _objective_policy: str | None
    _objective_session: dict | None
    _block_new_entries: bool
    _meta_decision_override: str | None

    # Target-attainment authority channels (must be declared or LangGraph drops them)
    _target_attainment_decision: dict | None
    _target_attainment_action: str | None
    _target_attainment_strategy_id: str | None
    _target_attainment_contract_id: str | None
    _target_attainment_quantity: int | None
    _target_attainment_objective_version: int | None
    _target_attainment_snapshot_id: str | None
    _target_attainment_authoritative: bool
    _meta_target_review: dict | None
    _full_chain_universe: dict | None
    _contract_selection_specs: list | None
    _contract_outcomes: list | None
    _shared_underlying_scenario_grid: dict | None
    _quantity_grid: list | None
    _portfolio_grid: list | None
    _target_portfolio_decision: dict | None
    _provisional_target_portfolio_decision: dict | None
    _target_authorized_positions: list | None
    _portfolio_review_context: dict | None
    _portfolio_debate_reviews: list | None
    _portfolio_review_finalization: dict | None
    _execution_command_ids: list[str] | None

    # Position graph channels
    _position_id: str | None
    _contract_id: str | None
    _original_strategy_id: str | None
    _original_strategy: StrategyHypothesis | None
    _prior_thesis: PositionThesisVersion | None
    _position_thesis: PositionThesisVersion | None
    _position_decision: PositionThesisVersion | None
    _position_projection: dict | None
    _order_projection: dict | None
    _trading_date: str | None
    _position_critic_notes: dict | None
    _position_action: str | None
    _position_command_id: str | None
