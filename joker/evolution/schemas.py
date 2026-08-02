"""Task 3 evolution domain schemas (immutable, append-only artefacts)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "3.0.0"

_FORBIDDEN_COT_KEYS = frozenset(
    {
        "chain_of_thought",
        "chain-of-thought",
        "scratchpad",
        "private_reasoning",
        "raw_transcript",
        "hidden_reasoning",
        "internal_monologue",
    }
)

PROHIBITED_MUTATION_TARGETS = frozenset(
    {
        "execution_validators",
        "order_action_gateway",
        "task1_market_truth",
        "ledger_calculations",
        "broker_code",
        "position_quantity_bounds",
        "live_money_flags",
        "database_migrations",
        "credential_handling",
        "filesystem_permissions",
        "python_import_paths",
        "shell_commands",
        "ci_configuration",
        "safety_limits",
    }
)


def assert_no_chain_of_thought(payload: dict[str, Any] | None) -> None:
    """Reject payloads that attempt to persist hidden reasoning."""
    if not payload:
        return

    def _walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()
                if key_l in _FORBIDDEN_COT_KEYS:
                    raise ValueError(
                        f"hidden chain-of-thought key forbidden at {path}.{key}"
                    )
                _walk(value, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for i, value in enumerate(obj):
                _walk(value, f"{path}[{i}]")

    _walk(payload)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_tz(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Typed cognitive patches (permitted mutation surface)
# ---------------------------------------------------------------------------


class PromptPatch(_Frozen):
    patch_type: Literal["prompt"] = "prompt"
    role: str
    parent_prompt_version_id: UUID
    replacement_template: str
    change_rationale: str

    @field_validator("replacement_template", "role", "change_rationale")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value


class EvidencePriorityChange(_Frozen):
    evidence_type: str
    priority_delta: int


class ContextPolicyPatch(_Frozen):
    patch_type: Literal["context_policy"] = "context_policy"
    role: str
    token_budget_delta: int | None = None
    evidence_priority_changes: tuple[EvidencePriorityChange, ...] = ()
    recency_policy: str | None = None


class RoutingPolicyPatch(_Frozen):
    patch_type: Literal["routing_policy"] = "routing_policy"
    role: str
    preferred_profile: str | None = None
    escalation_profile: str | None = None
    escalation_conditions: tuple[str, ...] = ()


class DebatePolicyPatch(_Frozen):
    patch_type: Literal["debate_policy"] = "debate_policy"
    minimum_reviews: int | None = None
    maximum_rounds: int | None = None
    dissent_required: bool | None = None
    unresolved_conflict_action: str | None = None


class MemoryPolicyPatch(_Frozen):
    patch_type: Literal["memory_policy"] = "memory_policy"
    max_memories: int | None = None
    recency_weight: Decimal | None = None
    regime_match_boost: Decimal | None = None
    diversity_constraint: bool | None = None
    include_contradictions: bool | None = None
    token_allocation: int | None = None


class EscalationPolicyPatch(_Frozen):
    patch_type: Literal["escalation_policy"] = "escalation_policy"
    role: str
    escalate_on_unresolved_conflict: bool | None = None
    escalate_on_low_confidence: bool | None = None
    min_confidence_threshold: Decimal | None = None


CognitivePatch = (
    PromptPatch
    | ContextPolicyPatch
    | RoutingPolicyPatch
    | DebatePolicyPatch
    | MemoryPolicyPatch
    | EscalationPolicyPatch
)


# ---------------------------------------------------------------------------
# Core artefacts
# ---------------------------------------------------------------------------


class TradingEpisode(_Frozen):
    episode_id: UUID = Field(default_factory=uuid4)
    schema_version: str = SCHEMA_VERSION
    session_id: str
    run_id: str
    trading_date: date
    entry_cycle_id: str | None = None
    parent_strategy_id: UUID | None = None
    proposal_id: UUID | None = None
    decision_id: UUID | None = None
    initial_snapshot_id: UUID | None = None
    terminal_snapshot_id: UUID | None = None
    snapshot_identity_status: Literal["verified", "missing"] = "verified"
    position_lifecycle_id: str | None = None
    contract_id: str | None = None
    direction: Literal["bullish", "bearish", "neutral", "none"] = "none"
    action_class: Literal[
        "no_trade",
        "entry_rejected",
        "entry_cancelled",
        "closed_trade",
        "open_at_session_end",
    ]
    entry_order_ids: tuple[str, ...] = ()
    position_action_ids: tuple[str, ...] = ()
    exit_order_ids: tuple[str, ...] = ()
    entry_price: Decimal | None = None
    exit_price: Decimal | None = None
    quantity: Decimal = Decimal("0")
    realised_pnl: Decimal | None = None
    max_favourable_excursion: Decimal | None = None
    max_adverse_excursion: Decimal | None = None
    holding_seconds: int | None = None
    entry_slippage: Decimal | None = None
    exit_slippage: Decimal | None = None
    total_fees: Decimal | None = None
    market_regime_tags: tuple[str, ...] = ()
    strategy_family: str | None = None
    pattern_ids: tuple[UUID, ...] = ()
    option_type: str | None = None
    session_phase: str | None = None
    volatility_bucket: str | None = None
    liquidity_bucket: str | None = None
    data_quality_ids: tuple[UUID, ...] = ()
    option_surface_ids: tuple[UUID, ...] = ()
    source_event_ids: tuple[UUID, ...] = ()
    entry_decision_event_id: UUID | None = None
    entry_decision_timestamp: datetime | None = None
    terminal_event_id: UUID | None = None
    terminal_event_timestamp: datetime | None = None
    market_event_ids: tuple[UUID, ...] = ()
    cognitive_artifact_ids: tuple[UUID, ...] = ()
    model_call_ids: tuple[UUID, ...] = ()
    prompt_version_ids: tuple[UUID, ...] = ()
    configuration_version_id: UUID
    completed: bool = False
    completeness_findings: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=_utc_now)
    idempotency_key: str = ""

    @field_validator("created_at", "entry_decision_timestamp", "terminal_event_timestamp")
    @classmethod
    def _tz(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _require_tz(value)

    @model_validator(mode="after")
    def _validate_closed_trade(self) -> TradingEpisode:
        assert_no_chain_of_thought(self.model_dump(mode="json"))
        if self.action_class == "closed_trade" and self.completed:
            if self.quantity != 0 and self.realised_pnl is None:
                raise ValueError("closed_trade requires realised_pnl when completed")
        return self


class DecisionTraceSummary(_Frozen):
    summary_id: UUID = Field(default_factory=uuid4)
    episode_id: UUID
    schema_version: str = SCHEMA_VERSION
    typed_conclusions: tuple[str, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()
    confidence_values: dict[str, Decimal] = Field(default_factory=dict)
    critic_objections: tuple[str, ...] = ()
    debate_votes: tuple[str, ...] = ()
    decision_rationale: str = ""
    requested_evidence: tuple[str, ...] = ()
    rejection_codes: tuple[str, ...] = ()
    model_metadata: dict[str, str] = Field(default_factory=dict)
    prompt_version: str = ""
    latency_ms: int | None = None
    token_usage: int | None = None
    content_hash: str = ""
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("created_at")
    @classmethod
    def _tz(cls, value: datetime) -> datetime:
        return _require_tz(value)

    @model_validator(mode="after")
    def _no_cot(self) -> DecisionTraceSummary:
        assert_no_chain_of_thought(self.model_dump(mode="json"))
        return self


class EpisodeEvaluation(_Frozen):
    evaluation_id: UUID = Field(default_factory=uuid4)
    episode_id: UUID
    evaluator_version: str
    outcome_quality: Decimal | None = None
    risk_adjusted_outcome: Decimal | None = None
    calibration_score: Decimal | None = None
    thesis_quality: Decimal | None = None
    evidence_grounding_score: Decimal | None = None
    debate_quality: Decimal | None = None
    decision_consistency_score: Decimal | None = None
    execution_quality: Decimal | None = None
    position_management_score: Decimal | None = None
    efficiency_score: Decimal | None = None
    avoidable_error_codes: tuple[str, ...] = ()
    safety_violation_codes: tuple[str, ...] = ()
    integrity_violation_codes: tuple[str, ...] = ()
    counterfactual_summary_ids: tuple[UUID, ...] = ()
    evaluator_agent_artifact_ids: tuple[UUID, ...] = ()
    deterministic_metrics: dict[str, Decimal | int | str | bool] = Field(
        default_factory=dict
    )
    valid: bool = True
    invalid_reasons: tuple[str, ...] = ()
    configuration_version_id: UUID | None = None
    content_hash: str = ""
    idempotency_key: str = ""
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("created_at")
    @classmethod
    def _tz(cls, value: datetime) -> datetime:
        return _require_tz(value)

    @model_validator(mode="after")
    def _no_cot(self) -> EpisodeEvaluation:
        assert_no_chain_of_thought(self.model_dump(mode="json"))
        return self


class PromptVersionRecord(_Frozen):
    prompt_version_id: UUID = Field(default_factory=uuid4)
    role: str
    template: str
    parent_prompt_version_id: UUID | None = None
    content_hash: str
    created_by: Literal["bootstrap", "agent", "human"] = "bootstrap"
    created_at: datetime = Field(default_factory=_utc_now)


class _PolicyVersion(_Frozen):
    version_id: UUID = Field(default_factory=uuid4)
    content: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""
    created_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _no_cot(self) -> _PolicyVersion:
        assert_no_chain_of_thought(self.content)
        return self


class ContextPolicyVersion(_PolicyVersion):
    pass


class MemoryPolicyVersion(_PolicyVersion):
    pass


class DebatePolicyVersion(_PolicyVersion):
    pass


class RoutingPolicyVersion(_PolicyVersion):
    pass


class EscalationPolicyVersion(_PolicyVersion):
    pass


class CognitiveConfigurationVersion(_Frozen):
    configuration_version_id: UUID = Field(default_factory=uuid4)
    parent_version_id: UUID | None = None
    status: Literal[
        "draft",
        "challenger",
        "shadow",
        "eligible",
        "champion",
        "rejected",
        "rolled_back",
        "retired",
    ] = "draft"
    prompt_versions: dict[str, UUID] = Field(default_factory=dict)
    role_model_profiles: dict[str, str] = Field(default_factory=dict)
    context_policy_version_id: UUID
    memory_policy_version_id: UUID
    debate_policy_version_id: UUID
    routing_policy_version_id: UUID
    escalation_policy_version_id: UUID
    # Dataset provenance for leakage-safe historical EV (fail closed when unknown).
    training_dataset_ids: tuple[UUID, ...] = ()
    challenger_dataset_ids: tuple[UUID, ...] = ()
    evaluation_dataset_ids: tuple[UUID, ...] = ()
    content_hash: str
    created_by: Literal["bootstrap", "agent", "human"] = "bootstrap"
    proposer_artifact_id: UUID | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    scope_key: str = "default"

    @field_validator("created_at")
    @classmethod
    def _tz(cls, value: datetime) -> datetime:
        return _require_tz(value)


class ImprovementProposal(_Frozen):
    proposal_id: UUID = Field(default_factory=uuid4)
    parent_champion_version_id: UUID
    weakness: str
    supporting_episode_ids: tuple[UUID, ...] = ()
    supporting_evaluation_ids: tuple[UUID, ...] = ()
    hypothesis: str
    proposed_change: dict[str, Any]
    expected_benefit: str
    expected_risks: tuple[str, ...] = ()
    metrics_to_improve: tuple[str, ...] = ()
    metrics_must_not_regress: tuple[str, ...] = ()
    required_evaluation_slices: tuple[str, ...] = ()
    suggested_min_sample_size: int = 20
    rollback_indicators: tuple[str, ...] = ()
    proposer_model_call_id: UUID | None = None
    proposer_prompt_version: str = ""
    content_hash: str = ""
    status: Literal["draft", "registered", "rejected", "experimenting", "resolved"] = (
        "draft"
    )
    idempotency_key: str = ""
    created_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _validate_surface(self) -> ImprovementProposal:
        assert_no_chain_of_thought(self.proposed_change)
        target = str(self.proposed_change.get("mutation_target", "")).lower()
        if target in PROHIBITED_MUTATION_TARGETS:
            raise ValueError(f"prohibited mutation target: {target}")
        for key in PROHIBITED_MUTATION_TARGETS:
            if key in self.proposed_change:
                raise ValueError(f"prohibited mutation key present: {key}")
        return self


class ExperimentDefinition(_Frozen):
    experiment_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID | None = None
    champion_version_id: UUID
    challenger_version_id: UUID
    dataset_id: UUID
    evaluation_windows: tuple[str, ...] = ()
    holdout_windows: tuple[str, ...] = ()
    adversarial_scenario_ids: tuple[str, ...] = ()
    random_seed: int = 42
    replay_engine_version: str = "3.0.0"
    model_provider_versions: dict[str, str] = Field(default_factory=dict)
    maximum_cost_gbp: Decimal = Decimal("25.00")
    maximum_duration_seconds: int = 3600
    required_metrics: tuple[str, ...] = ()
    non_regression_thresholds: dict[str, Decimal] = Field(default_factory=dict)
    status: Literal[
        "pending", "running", "completed", "failed", "cancelled"
    ] = "pending"
    recovery_cursor: str | None = None
    content_hash: str = ""
    created_at: datetime = Field(default_factory=_utc_now)


class ExperimentSliceResult(_Frozen):
    slice_name: str
    metrics: dict[str, Decimal | int | str | bool] = Field(default_factory=dict)
    episode_count: int = 0
    missing_episode_count: int = 0
    confidence_intervals: dict[str, tuple[Decimal, Decimal]] = Field(
        default_factory=dict
    )


class ExperimentResult(_Frozen):
    result_id: UUID = Field(default_factory=uuid4)
    experiment_id: UUID
    per_slice_results: tuple[ExperimentSliceResult, ...] = ()
    aggregate_metrics: dict[str, Decimal | int | str | bool] = Field(
        default_factory=dict
    )
    confidence_intervals: dict[str, tuple[Decimal, Decimal]] = Field(
        default_factory=dict
    )
    bootstrap_summaries: dict[str, tuple[Decimal, ...]] = Field(default_factory=dict)
    safety_failures: tuple[str, ...] = ()
    data_integrity_failures: tuple[str, ...] = ()
    cost_gbp: Decimal | None = None
    latency_ms_total: int = 0
    model_call_counts: dict[str, int] = Field(default_factory=dict)
    missing_episodes: tuple[UUID, ...] = ()
    replay_divergences: tuple[str, ...] = ()
    champion_metrics: dict[str, Decimal | int | str | bool] = Field(
        default_factory=dict
    )
    challenger_metrics: dict[str, Decimal | int | str | bool] = Field(
        default_factory=dict
    )
    eligibility_outcome: bool = False
    gate_rejection_codes: tuple[str, ...] = ()
    content_hash: str = ""
    created_at: datetime = Field(default_factory=_utc_now)


class PromotionDecision(_Frozen):
    promotion_decision_id: UUID = Field(default_factory=uuid4)
    experiment_id: UUID
    challenger_version_id: UUID
    champion_version_id: UUID
    deterministic_eligible: bool
    deterministic_gate_codes: tuple[str, ...] = ()
    agent_action: Literal[
        "promote", "reject", "extend_shadow", "request_more_evidence"
    ]
    strategic_rationale: str
    accepted_tradeoffs: tuple[str, ...] = ()
    unresolved_risks: tuple[str, ...] = ()
    final_status: Literal[
        "promoted", "rejected", "pending_evidence", "blocked_by_gate"
    ]
    idempotency_key: str = ""
    created_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _agent_cannot_override_gate(self) -> PromotionDecision:
        if not self.deterministic_eligible and self.agent_action == "promote":
            raise ValueError("agent cannot promote when deterministic eligibility is false")
        if not self.deterministic_eligible and self.final_status == "promoted":
            raise ValueError("final_status promoted requires deterministic eligibility")
        return self


class RollbackRecord(_Frozen):
    rollback_id: UUID = Field(default_factory=uuid4)
    rolled_back_version_id: UUID
    restored_version_id: UUID
    trigger: str
    trigger_metrics: dict[str, Decimal | int | str | bool] = Field(default_factory=dict)
    initiator: Literal["deterministic", "agent", "human"]
    affected_episode_ids: tuple[UUID, ...] = ()
    detection_timestamp: datetime
    completion_timestamp: datetime | None = None
    active_cycles_retained_original_config: bool = True
    recovery_status: Literal["pending", "completed", "failed"] = "pending"
    idempotency_key: str = ""
    content_hash: str = ""

    @field_validator("detection_timestamp")
    @classmethod
    def _tz(cls, value: datetime) -> datetime:
        return _require_tz(value)


class ChampionTransition(_Frozen):
    transition_id: UUID = Field(default_factory=uuid4)
    scope_key: str = "default"
    previous_version_id: UUID | None
    new_version_id: UUID
    reason: str
    experiment_id: UUID | None = None
    promotion_decision_id: UUID | None = None
    activated_at: datetime = Field(default_factory=_utc_now)
    content_hash: str = ""


class ChampionActivationRecord(_Frozen):
    activation_id: UUID = Field(default_factory=uuid4)
    promotion_decision_id: UUID
    experiment_id: UUID
    challenger_version_id: UUID
    previous_champion_version_id: UUID
    registry_applied: bool = False
    history_verified: bool = False
    configuration_status_applied: bool = False
    completed: bool = False
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    failure_codes: tuple[str, ...] = ()

    @field_validator("created_at", "updated_at")
    @classmethod
    def _tz(cls, value: datetime) -> datetime:
        return _require_tz(value)


class ReplayFrameCheckpoint(_Frozen):
    replay_key: str
    frame_index: int
    snapshot_id: UUID
    order_management_completed: bool = False
    position_graph_completed: bool = False
    action_submitted: bool = False
    execution_checkpointed: bool = False
    position_thread_id: str | None = None
    order_management_thread_ids: tuple[str, ...] = ()
    model_call_ids: tuple[UUID, ...] = ()


class DriftObservation(_Frozen):
    observation_id: UUID = Field(default_factory=uuid4)
    configuration_version_id: UUID
    dimension: str
    baseline_value: Decimal | int | str | bool | None = None
    observed_value: Decimal | int | str | bool | None = None
    severity: Literal["info", "warning", "critical"] = "warning"
    evidence: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _no_cot(self) -> DriftObservation:
        assert_no_chain_of_thought(self.evidence)
        return self


class ShadowAssignment(_Frozen):
    assignment_id: UUID = Field(default_factory=uuid4)
    challenger_version_id: UUID
    champion_version_id: UUID
    status: Literal["active", "paused", "stopped"] = "active"
    created_at: datetime = Field(default_factory=_utc_now)


class EvolutionCycleRecord(_Frozen):
    cycle_id: str
    session_id: str
    status: Literal[
        "pending", "running", "completed", "failed", "abandoned", "blocked"
    ]
    stage: str
    payload: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _no_cot(self) -> EvolutionCycleRecord:
        assert_no_chain_of_thought(self.payload)
        return self


class EvaluationDataset(_Frozen):
    dataset_id: UUID = Field(default_factory=uuid4)
    builder_version: str = "3.0.0"
    construction_timestamp: datetime = Field(default_factory=_utc_now)
    time_start: datetime | None = None
    time_end: datetime | None = None
    episode_ids: tuple[UUID, ...] = ()
    partition_map: dict[str, tuple[UUID, ...]] = Field(default_factory=dict)
    regime_distribution: dict[str, int] = Field(default_factory=dict)
    outcome_distribution: dict[str, int] = Field(default_factory=dict)
    configuration_distribution: dict[str, int] = Field(default_factory=dict)
    data_quality_distribution: dict[str, int] = Field(default_factory=dict)
    source_db_hashes: dict[str, str] = Field(default_factory=dict)
    random_seed: int = 42
    exclusion_reasons: tuple[str, ...] = ()
    leakage_audit: tuple[str, ...] = ()
    content_hash: str = ""
    created_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _no_overlap(self) -> EvaluationDataset:
        seen: set[UUID] = set()
        for partition, ids in self.partition_map.items():
            for eid in ids:
                if eid in seen:
                    raise ValueError(
                        f"episode {eid} appears in multiple partitions including {partition}"
                    )
                seen.add(eid)
        return self


class MemoryLessonEntry(_Frozen):
    lesson_id: UUID = Field(default_factory=uuid4)
    lesson_type: Literal[
        "episode_lesson",
        "failure_pattern",
        "successful_evidence_pattern",
        "regime_specific_lesson",
        "execution_lesson",
        "position_management_lesson",
        "debate_lesson",
        "calibration_lesson",
    ]
    content: str
    source_episode_ids: tuple[UUID, ...] = ()
    source_evaluation_ids: tuple[UUID, ...] = ()
    confidence: Decimal = Decimal("0.5")
    applicability_conditions: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    created_configuration_version_id: UUID
    last_validated_date: date | None = None
    decay_policy: str = "linear_30d"
    content_hash: str = ""
    created_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _no_cot(self) -> MemoryLessonEntry:
        payload = {"content": self.content}
        assert_no_chain_of_thought(payload)
        lowered = self.content.lower()
        for key in _FORBIDDEN_COT_KEYS:
            if key in lowered:
                raise ValueError("memory content must not embed chain-of-thought")
        return self
