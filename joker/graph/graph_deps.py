"""Shared dependencies injected into cognitive LangGraph nodes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from joker.cognition.context import ContextAssembler
from joker.config.settings import CognitiveGraphSettings
from joker.events.bus import InProcessAsyncEventBus
from joker.market.snapshots import SnapshotRepository
from joker.models.router import ModelRouter
from joker.persistence.cognitive_repositories import (
    DebateRepository,
    DecisionRepository,
    EvidenceRepository,
    HypothesisRepository,
    StrategyRepository,
    WorldModelRepository,
)
from joker.runtime.execution_runtime import ExecutionRuntime
from joker.time.clock import ExchangeClock


SubmitCallback = Callable[..., Awaitable[Any]]


@dataclass
class CognitiveGraphDeps:
    """Runtime services available to graph nodes — no broker polling."""

    router: ModelRouter
    config: CognitiveGraphSettings
    session_id: str
    run_id: str
    context_assembler: ContextAssembler = field(default_factory=ContextAssembler)
    snapshot_repo: SnapshotRepository | None = None
    evidence_repo: EvidenceRepository | None = None
    world_model_repo: WorldModelRepository | None = None
    hypothesis_repo: HypothesisRepository | None = None
    strategy_repo: StrategyRepository | None = None
    debate_repo: DebateRepository | None = None
    decision_repo: DecisionRepository | None = None
    execution_runtime: ExecutionRuntime | None = None
    submit_callback: SubmitCallback | None = None
    event_bus: InProcessAsyncEventBus | None = None
    clock: ExchangeClock | None = None
    db_path: Path | None = None

    def limits_dict(self) -> dict[str, int]:
        return {
            "max_agent_data_requests_per_invocation": self.config.max_agent_data_requests,
            "max_debate_rounds": self.config.max_debate_rounds,
            "max_strategy_switches": self.config.max_strategy_switches,
            "max_strategy_candidates": self.config.max_strategy_candidates,
            "max_hypotheses_per_cycle": self.config.max_hypotheses_per_cycle,
            "max_cycle_seconds": self.config.max_cycle_seconds,
        }
