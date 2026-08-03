"""Task 2 cognitive schemas (sections 7.1–7.9)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "2.0"

Side = Literal["buy", "sell"]
OptionType = Literal["call", "put"]
EvidenceSourceType = Literal[
    "underlying",
    "bar_1m",
    "bar_5m",
    "option_contract",
    "option_surface",
    "data_quality",
    "ledger_order",
    "ledger_position",
    "prior_cognitive_artifact",
]
DebateVerdict = Literal[
    "support",
    "oppose",
    "request_revision",
    "request_more_evidence",
    "execution_concern",
    "insufficient_information",
]
ExecutionProposalAction = Literal["execute", "probe"]
OrderType = Literal["limit", "market"]
OrderManagementAction = Literal[
    "continue_waiting",
    "cancel",
    "replace",
    "reduce_quantity",
    "abandon",
]
GraphNodeStatus = Literal["started", "completed", "failed", "skipped"]
CognitiveRuntimeStatus = Literal["healthy", "degraded", "unavailable", "shutting_down"]
AnomalySeverity = Literal["low", "medium", "high"]
ConflictResolutionStatus = Literal["unresolved", "partially_resolved", "resolved"]
EntryStyle = Literal["immediate", "scaled", "conditional"]


def _require_tz_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


def _validate_confidence(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return value


class AgentRole(StrEnum):
    """Distinct cognitive agent roles in the Task 2 graph."""

    MARKET_STRUCTURE = "market_structure"
    VOLATILITY = "volatility"
    OPTIONS_MICROSTRUCTURE = "options_microstructure"
    TEMPORAL_CONTEXT = "temporal_context"
    ANOMALY = "anomaly"

    PATTERN_MINER = "pattern_miner"
    SEQUENCE_ANALYST = "sequence_analyst"
    ANALOGY_RETRIEVER = "analogy_retriever"

    BULLISH_INVENTOR = "bullish_inventor"
    BEARISH_INVENTOR = "bearish_inventor"
    NEUTRAL_ADVOCATE = "neutral_advocate"

    STRATEGY_ADVOCATE = "strategy_advocate"
    FALSIFIER = "falsifier"
    HISTORICAL_CRITIC = "historical_critic"
    EXECUTION_CRITIC = "execution_critic"
    ALTERNATIVE_EXPLANATION = "alternative_explanation"

    WORLD_MODEL_SYNTHESISER = "world_model_synthesiser"

    META_DECISION = "meta_decision"
    ENTRY_TACTICIAN = "entry_tactician"
    ORDER_MANAGER = "order_manager"
    POSITION_THESIS = "position_thesis"
    POSITION_DECISION = "position_decision"


class MarketDirection(StrEnum):
    """Directional or volatility-regime labels for evidence and hypotheses."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_COMPRESSION = "volatility_compression"
    UNCERTAIN = "uncertain"


class MetaDecisionAction(StrEnum):
    """Meta-decision routing actions."""

    EXECUTE = "execute"
    PROBE = "probe"
    DELAY = "delay"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"
    SWITCH_STRATEGY = "switch_strategy"
    ABANDON = "abandon"


class PositionAction(StrEnum):
    """Position-management actions."""

    HOLD = "hold"
    ADD = "add"
    REDUCE = "reduce"
    EXIT = "exit"
    CANCEL_WORKING_ORDER = "cancel_working_order"
    REPLACE_WORKING_ORDER = "replace_working_order"
    UPDATE_THESIS = "update_thesis"


class CognitiveArtifactType(StrEnum):
    """Persisted cognitive artifact discriminator."""

    AGENT_EVIDENCE = "agent_evidence"
    MARKET_WORLD_MODEL = "market_world_model"
    PATTERN_HYPOTHESIS = "pattern_hypothesis"
    STRATEGY_HYPOTHESIS = "strategy_hypothesis"
    DEBATE_REVIEW = "debate_review"
    META_DECISION = "meta_decision"
    EXECUTION_PROPOSAL = "execution_proposal"
    POSITION_THESIS_VERSION = "position_thesis_version"
    ORDER_MANAGEMENT_DECISION = "order_management_decision"


class ModelCallStatus(StrEnum):
    """Lifecycle status for idempotent model-call records."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactBase(BaseModel):
    """Common metadata shared by versioned cognitive artefacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    session_id: str
    snapshot_id: UUID
    prompt_version: str
    model_call_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_tz_aware(value)

    @field_validator("session_id", "prompt_version")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class CycleArtifactBase(ArtifactBase):
    """Cognitive artefact tied to a decision cycle."""

    cycle_id: str

    @field_validator("cycle_id")
    @classmethod
    def _non_empty_cycle(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cycle_id must be a non-empty string")
        return value


class EvidenceReference(BaseModel):
    """Typed pointer to a Task 1 or prior cognitive fact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference_id: UUID = Field(default_factory=uuid4)
    snapshot_id: UUID
    source_type: EvidenceSourceType
    source_id: str
    field_name: str | None = None
    observed_at: datetime
    value_summary: str

    @field_validator("observed_at")
    @classmethod
    def _aware_observed_at(cls, value: datetime) -> datetime:
        return _require_tz_aware(value)

    @field_validator("source_id", "value_summary")
    @classmethod
    def _non_empty_ref(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class AgentDataRequest(BaseModel):
    """Bounded read-only data request from an agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_type: Literal[
        "bars",
        "option_surface_slice",
        "data_quality",
        "ledger_order",
        "ledger_position",
        "active_hypotheses",
    ]
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str

    @field_validator("reason")
    @classmethod
    def _non_empty_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must be a non-empty string")
        return value


class AgentEvidence(CycleArtifactBase):
    """Perception or analysis evidence with auditable references."""

    evidence_id: UUID = Field(default_factory=uuid4)
    agent_role: AgentRole
    claim: str
    direction: MarketDirection | None = None
    confidence: float
    supporting_references: tuple[EvidenceReference, ...] = ()
    contradicting_references: tuple[EvidenceReference, ...] = ()
    invalidation_conditions: tuple[str, ...] = ()
    expected_horizon_seconds: int | None = None
    uncertainty_sources: tuple[str, ...] = ()
    requires_more_data: bool = False
    data_request: AgentDataRequest | None = None

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return _validate_confidence(value)

    @model_validator(mode="after")
    def _directional_requires_support(self) -> Self:
        if self.direction is not None and self.direction != MarketDirection.UNCERTAIN:
            if not self.supporting_references:
                raise ValueError(
                    "directional claims require at least one supporting reference"
                )
        return self


class RegimeHypothesis(BaseModel):
    """Regime label synthesized into the world model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    regime_id: UUID = Field(default_factory=uuid4)
    label: str
    direction: MarketDirection
    confidence: float
    supporting_evidence_ids: tuple[UUID, ...] = ()
    rationale: str

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return _validate_confidence(value)


class MarketStructureAssessment(BaseModel):
    """Market structure synthesis from perception evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment_id: UUID = Field(default_factory=uuid4)
    primary_direction: MarketDirection
    structure_summary: str
    key_levels: tuple[str, ...] = ()
    trend_strength: float | None = None
    range_bound: bool = False
    supporting_evidence_ids: tuple[UUID, ...] = ()
    confidence: float

    @field_validator("confidence", "trend_strength")
    @classmethod
    def _optional_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _validate_confidence(value)


class VolatilityAssessment(BaseModel):
    """Volatility regime assessment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment_id: UUID = Field(default_factory=uuid4)
    state: MarketDirection
    implied_vs_realized: str | None = None
    summary: str
    supporting_evidence_ids: tuple[UUID, ...] = ()
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return _validate_confidence(value)


class OptionsMicrostructureAssessment(BaseModel):
    """Options liquidity and microstructure assessment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment_id: UUID = Field(default_factory=uuid4)
    liquidity_summary: str
    spread_conditions: str
    skew_summary: str | None = None
    supporting_evidence_ids: tuple[UUID, ...] = ()
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return _validate_confidence(value)


class TemporalAssessment(BaseModel):
    """Session timing and 0DTE temporal context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment_id: UUID = Field(default_factory=uuid4)
    session_phase: str
    minutes_to_close: int | None = None
    time_decay_context: str
    supporting_evidence_ids: tuple[UUID, ...] = ()
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return _validate_confidence(value)


class MarketAnomaly(BaseModel):
    """Detected anomaly requiring explicit acknowledgement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    anomaly_id: UUID = Field(default_factory=uuid4)
    description: str
    severity: AnomalySeverity
    supporting_evidence_ids: tuple[UUID, ...] = ()


class EvidenceConflict(BaseModel):
    """Unresolved or partially resolved evidence disagreement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conflict_id: UUID = Field(default_factory=uuid4)
    claim_a: str
    claim_b: str
    evidence_ids_a: tuple[UUID, ...] = ()
    evidence_ids_b: tuple[UUID, ...] = ()
    resolution_status: ConflictResolutionStatus = "unresolved"


class MarketWorldModel(CycleArtifactBase):
    """Shared blackboard synthesizing perception evidence."""

    world_model_id: UUID = Field(default_factory=uuid4)
    regime_hypotheses: tuple[RegimeHypothesis, ...] = ()
    market_structure: MarketStructureAssessment
    volatility_state: VolatilityAssessment
    options_state: OptionsMicrostructureAssessment
    temporal_state: TemporalAssessment
    anomalies: tuple[MarketAnomaly, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()
    evidence_conflicts: tuple[EvidenceConflict, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    overall_uncertainty: float
    synthesizer_model_call_id: UUID

    @field_validator("overall_uncertainty")
    @classmethod
    def _uncertainty_range(cls, value: float) -> float:
        return _validate_confidence(value)


class PatternHypothesis(CycleArtifactBase):
    """Discovery output describing a candidate market pattern."""

    hypothesis_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    direction: MarketDirection
    expected_sequence: tuple[str, ...] = ()
    expected_horizon_seconds: int
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    invalidation_conditions: tuple[str, ...] = ()
    novelty_score: float
    confidence: float
    open_questions: tuple[str, ...] = ()
    agent_role: AgentRole

    @field_validator("novelty_score", "confidence")
    @classmethod
    def _score_range(cls, value: float) -> float:
        return _validate_confidence(value)


class StrategyLegCandidate(BaseModel):
    """Proposed strategy leg referencing a real contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    leg_id: UUID = Field(default_factory=uuid4)
    contract_id: str
    side: Side
    option_type: OptionType
    strike: Decimal
    quantity: int
    rationale: str
    max_acceptable_spread_pct: float | None = None

    @field_validator("quantity")
    @classmethod
    def _positive_qty(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("quantity must be positive")
        return value


class EntryPlan(BaseModel):
    """Entry timing and order-style preferences."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_style: EntryStyle
    preferred_order_type: OrderType
    limit_price: Decimal | None = None
    entry_conditions: tuple[str, ...] = ()
    timing_notes: str = ""


class ExecutionPlan(BaseModel):
    """Execution sequencing and fill policies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequencing: tuple[str, ...] = ()
    max_quote_age_seconds: int
    partial_fill_policy: str
    replacement_policy: str


class ExitPlan(BaseModel):
    """Profit-taking and adverse exit conditions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profit_taking_conditions: tuple[str, ...] = ()
    stop_conditions: tuple[str, ...] = ()
    time_based_exit: str | None = None


class InvalidationPlan(BaseModel):
    """Thesis invalidation monitoring plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conditions: tuple[str, ...] = ()
    monitoring_frequency_seconds: int | None = None


class StrategyHypothesis(CycleArtifactBase):
    """Competing strategy proposal from an inventor agent."""

    strategy_id: UUID = Field(default_factory=uuid4)
    source_hypothesis_ids: tuple[UUID, ...] = ()
    name: str
    market_thesis: str
    direction: MarketDirection
    # Required for positive-EV historical comparison when objectives are armed.
    # Legacy records may omit it; compilers/objective graph fail closed if absent.
    strategy_family: str | None = None
    candidate_legs: tuple[StrategyLegCandidate, ...] = ()
    entry_plan: EntryPlan
    execution_plan: ExecutionPlan
    exit_plan: ExitPlan
    invalidation_plan: InvalidationPlan
    expected_favourable_path: tuple[str, ...] = ()
    expected_adverse_path: tuple[str, ...] = ()
    expected_horizon_seconds: int
    confidence: float
    novelty_score: float
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    agent_role: AgentRole

    @field_validator("confidence", "novelty_score")
    @classmethod
    def _score_range(cls, value: float) -> float:
        return _validate_confidence(value)


class DebateReview(BaseModel):
    """Adversarial review of a strategy hypothesis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_id: UUID = Field(default_factory=uuid4)
    strategy_id: UUID
    snapshot_id: UUID
    cycle_id: str
    reviewer_role: AgentRole
    verdict: DebateVerdict
    claims: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    identified_failure_modes: tuple[str, ...] = ()
    required_revisions: tuple[str, ...] = ()
    invalidation_conditions: tuple[str, ...] = ()
    confidence: float
    prompt_version: str
    model_call_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return _validate_confidence(value)

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_tz_aware(value)


class MetaDecision(CycleArtifactBase):
    """Meta-decision routing output — not a majority vote."""

    decision_id: UUID = Field(default_factory=uuid4)
    action: MetaDecisionAction
    selected_strategy_id: UUID | None = None
    alternate_strategy_ids: tuple[UUID, ...] = ()
    confidence: float
    rationale_summary: str
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    review_ids: tuple[UUID, ...] = ()
    required_conditions: tuple[str, ...] = ()
    requested_evidence: tuple[AgentDataRequest, ...] = ()
    exploration_reason: str | None = None

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return _validate_confidence(value)


class ExecutionLeg(BaseModel):
    """Single leg in an execution proposal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    leg_id: UUID = Field(default_factory=uuid4)
    contract_id: str
    side: Side
    quantity: int
    limit_price: Decimal | None = None
    sequence_order: int
    max_quote_age_seconds: int
    replacement_policy: str
    partial_fill_policy: str

    @field_validator("quantity")
    @classmethod
    def _positive_qty(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("quantity must be positive")
        return value


class ExecutionProposal(BaseModel):
    """Typed entry proposal compiled toward Task 1 ExecutionCommand."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_id: UUID = Field(default_factory=uuid4)
    decision_id: UUID
    strategy_id: UUID
    session_id: str
    cycle_id: str
    snapshot_id: UUID
    action: ExecutionProposalAction
    legs: tuple[ExecutionLeg, ...]
    order_type: OrderType
    time_in_force: str
    cancel_after_seconds: int | None = None
    entry_rationale: str
    invalidation_conditions: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[UUID, ...] = ()
    prompt_version: str
    model_call_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_tz_aware(value)


class PositionThesisVersion(BaseModel):
    """Immutable versioned position thesis — never overwritten."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    thesis_version_id: UUID = Field(default_factory=uuid4)
    schema_version: str = SCHEMA_VERSION
    position_id: str
    contract_id: str
    session_id: str
    snapshot_id: UUID
    prior_version_id: UUID | None = None
    original_strategy_id: UUID
    current_thesis: str
    expected_path: tuple[str, ...] = ()
    invalidation_conditions: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    recommended_action: PositionAction
    recommended_quantity: int | None = None
    recommended_limit_price: Decimal | None = None
    confidence: float
    prompt_version: str
    model_call_id: UUID
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return _validate_confidence(value)

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_tz_aware(value)


class OrderManagementDecision(CycleArtifactBase):
    """Working-order management proposal — no broker access."""

    decision_id: UUID = Field(default_factory=uuid4)
    agent_role: AgentRole = AgentRole.ORDER_MANAGER
    client_order_id: str
    proposal_id: UUID | None = None
    action: OrderManagementAction
    new_limit_price: Decimal | None = None
    new_quantity: int | None = None
    rationale_summary: str
    supporting_evidence_ids: tuple[UUID, ...] = ()


class GraphNodeTrace(BaseModel):
    """Deterministic graph node execution trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: UUID = Field(default_factory=uuid4)
    node_name: str
    agent_role: AgentRole | None = None
    started_at: datetime
    completed_at: datetime | None = None
    status: GraphNodeStatus
    artifact_ids: tuple[UUID, ...] = ()
    error_code: str | None = None

    @field_validator("started_at", "completed_at")
    @classmethod
    def _aware_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _require_tz_aware(value)


class CognitiveError(BaseModel):
    """Graph-state error record (distinct from exceptions.CognitiveError)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_id: UUID = Field(default_factory=uuid4)
    node_name: str | None = None
    agent_role: AgentRole | None = None
    error_code: str
    message: str
    recoverable: bool = True
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context: dict[str, Any] | None = None

    @field_validator("occurred_at")
    @classmethod
    def _aware_occurred_at(cls, value: datetime) -> datetime:
        return _require_tz_aware(value)


class CognitiveRuntimeHealth(BaseModel):
    """Cognitive runtime health snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: CognitiveRuntimeStatus
    local_provider_available: bool
    remote_provider_available: bool
    active_decision_cycles: int
    active_position_cycles: int
    queued_events: int
    last_success_at: datetime | None = None
    last_error: CognitiveError | None = None

    @field_validator("last_success_at")
    @classmethod
    def _aware_last_success(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _require_tz_aware(value)


class PromptSpec(BaseModel):
    """Immutable versioned prompt specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_id: str
    version: str
    agent_role: AgentRole
    system_template: str
    output_schema_name: str
    required_context_schema: str
    created_at: datetime
    content_hash: str

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        return _require_tz_aware(value)
