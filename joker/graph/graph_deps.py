"""Shared dependencies injected into cognitive LangGraph nodes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver

from joker.cognition.context import ContextAssembler
from joker.config.settings import CognitiveGraphSettings
from joker.events.bus import InProcessAsyncEventBus
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.quality import DataQualityReport
from joker.market.data_quality_store import DataQualityRepository
from joker.market.snapshots import MarketSnapshot, SnapshotRepository
from joker.models.router import ModelRouter
from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
)
from joker.persistence.cognitive_repositories import (
    DebateRepository,
    DecisionRepository,
    EvidenceRepository,
    HypothesisRepository,
    ModelCallRepository,
    OrderManagementRepository,
    PositionThesisRepository,
    StrategyRepository,
    WorldModelRepository,
)
from joker.runtime.execution_runtime import ExecutionRuntime
from joker.time.clock import ExchangeClock


SubmitCallback = Callable[..., Awaitable[Any]]
DataQualityLoader = Callable[
    [UUID, MarketSnapshot],
    Awaitable[DataQualityReport | None],
]
ProjectionLoader = Callable[[], Awaitable[Any]]
ObjectiveStateLoader = Callable[[], Awaitable[Any]]


@dataclass
class CognitiveGraphDeps:
    """Runtime services available to graph nodes — no broker polling."""

    router: ModelRouter
    config: CognitiveGraphSettings
    session_id: str
    run_id: str
    context_assembler: ContextAssembler = field(default_factory=ContextAssembler)
    snapshot_repo: SnapshotRepository | None = None
    option_surface_repo: OptionSurfaceRepository | None = None
    data_quality_repo: DataQualityRepository | None = None
    evidence_repo: EvidenceRepository | None = None
    world_model_repo: WorldModelRepository | None = None
    hypothesis_repo: HypothesisRepository | None = None
    strategy_repo: StrategyRepository | None = None
    debate_repo: DebateRepository | None = None
    decision_repo: DecisionRepository | None = None
    position_thesis_repo: PositionThesisRepository | None = None
    order_management_repo: OrderManagementRepository | None = None
    model_call_repo: ModelCallRepository | None = None
    execution_runtime: ExecutionRuntime | None = None
    submit_callback: SubmitCallback | None = None
    event_bus: InProcessAsyncEventBus | None = None
    clock: ExchangeClock | None = None
    db_path: Path | None = None
    checkpointer: BaseCheckpointSaver | None = None
    data_quality_loader: DataQualityLoader | None = None
    projection_loader: ProjectionLoader | None = None
    provenance_registry: CognitiveExecutionProvenanceRegistry | None = None
    order_action_gateway: Any | None = None
    cycle_registry: Any | None = None
    order_management_action_repo: Any | None = None
    submitted_proposal_ids: set[str] = field(default_factory=set)
    # Goal-driven objective dependencies (required when objective.enabled)
    objective_service: Any | None = None
    objective_state_loader: ObjectiveStateLoader | None = None
    feasibility_engine: Any | None = None
    objective_strategy_scorer: Any | None = None
    capital_sizer: Any | None = None

    def limits_dict(self) -> dict[str, int]:
        return {
            "max_agent_data_requests_per_invocation": self.config.max_agent_data_requests,
            "max_debate_rounds": self.config.max_debate_rounds,
            "max_strategy_switches": self.config.max_strategy_switches,
            "max_strategy_candidates": self.config.max_strategy_candidates,
            "max_hypotheses_per_cycle": self.config.max_hypotheses_per_cycle,
            "max_cycle_seconds": self.config.max_cycle_seconds,
        }

    def require_objective_dependencies(self) -> None:
        """Fail closed when a goal-aware production profile lacks objective wiring."""
        missing: list[str] = []
        if self.objective_service is None:
            missing.append("objective_service")
        if self.objective_state_loader is None:
            missing.append("objective_state_loader")
        if self.feasibility_engine is None:
            missing.append("feasibility_engine")
        if self.objective_strategy_scorer is None:
            missing.append("objective_strategy_scorer")
        if self.capital_sizer is None:
            missing.append("capital_sizer")
        if missing:
            raise RuntimeError(
                "objective-enabled production path missing CognitiveGraphDeps: "
                + ", ".join(missing)
            )
