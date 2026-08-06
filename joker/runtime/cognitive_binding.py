"""Bind cognitive graph deps to a started Task 1 bridge."""

from __future__ import annotations

from typing import Any

from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.data_quality_store import DataQualityRepository
from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
)
from joker.runtime.compatibility import CompatibilityLivePaperBridge


def bind_cognitive_graph_to_task1(
    deps: CognitiveGraphDeps,
    bridge: CompatibilityLivePaperBridge,
    *,
    data_quality_repo: DataQualityRepository | None = None,
    provenance_registry: CognitiveExecutionProvenanceRegistry | None = None,
) -> None:
    """Wire ExecutionRuntime and loaders after ``bridge.start()``.

    SessionSupervisor creates ExecutionRuntime inside ``start()``, so cognitive
    deps must be rebound after startup — never before.
    """
    execution = bridge.supervisor.execution_runtime
    if execution is None:
        raise RuntimeError(
            "ExecutionRuntime is None after task1_bridge.start(); "
            "cognitive position/order actions cannot submit"
        )

    async def _submit_callback(provenanced: Any) -> Any:
        runtime = bridge.supervisor.execution_runtime
        if runtime is None:
            raise RuntimeError("ExecutionRuntime not available")
        if deps.provenance_registry is not None:
            from joker.persistence.cognitive_execution_provenance import (
                ExecutionProvenanceRecord,
            )
            from joker.runtime.execution_runtime import contract_id_for

            contract_id = contract_id_for(provenanced.command.intent.contract)
            await deps.provenance_registry.record(
                ExecutionProvenanceRecord(
                    client_order_id=provenanced.command.client_order_id,
                    proposal_id=str(getattr(provenanced, "proposal_id", "") or None),
                    decision_id=str(getattr(provenanced, "decision_id", "") or None),
                    strategy_id=str(getattr(provenanced, "strategy_id", "") or None),
                    cycle_id=str(getattr(provenanced, "cycle_id", "") or None),
                    snapshot_id=str(getattr(provenanced, "snapshot_id", "") or None),
                    contract_id=contract_id,
                    session_id=deps.session_id,
                    kind="entry",
                    causation_event_id=str(
                        getattr(provenanced, "causation_event_id", None) or ""
                    )
                    or None,
                )
            )
        return await runtime.submit_execution_command(provenanced.command)

    async def _projection_loader() -> Any:
        runtime = bridge.supervisor.execution_runtime
        if runtime is None:
            return None
        return await runtime.project_session()

    dq_repo = data_quality_repo or bridge.supervisor.data_quality_repository

    async def _data_quality_loader(report_id, snapshot):
        if dq_repo is None:
            return None
        return await dq_repo.get_by_id(report_id)

    deps.execution_runtime = execution
    deps.broker_account_identity = execution.broker_account_identity
    deps.submit_callback = _submit_callback
    deps.projection_loader = _projection_loader
    deps.data_quality_loader = _data_quality_loader
    if dq_repo is not None:
        deps.data_quality_repo = dq_repo
    deps.event_bus = bridge.supervisor.event_bus
    if provenance_registry is not None:
        deps.provenance_registry = provenance_registry
    if bridge.supervisor.option_surface_repository is not None:
        deps.option_surface_repo = bridge.supervisor.option_surface_repository
    if bridge.supervisor.snapshot_repository is not None:
        deps.snapshot_repo = bridge.supervisor.snapshot_repository
    deps.clock = bridge.supervisor.clock

    from joker.objectives.execution_quote import build_current_option_quote_loader
    from joker.runtime.order_action_gateway import OrderActionGateway

    max_age = int(getattr(deps, "max_quote_age_seconds", 30) or 30)
    max_spread = float(getattr(deps, "max_relative_spread", 0.25) or 0.25)
    deps.current_option_quote_loader = build_current_option_quote_loader(
        deps,
        max_quote_age_seconds=max_age,
        max_relative_spread=max_spread,
    )

    async def _current_dq_loader(report_id):
        if dq_repo is None:
            return None
        return await dq_repo.get_by_id(report_id)

    deps.current_data_quality_loader = _current_dq_loader
    deps.order_action_gateway = OrderActionGateway(deps)
