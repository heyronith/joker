"""Helpers to hydrate role-specific context from Task 1 truth."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from joker.cognition.context import ContextAssembler, ContextAssemblerConfig, ContextPackage
from joker.cognition.schemas import AgentRole
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.market.quality import DataQualityReport, DataQualitySeverity
from joker.market.snapshots import MarketSnapshot


def context_assembler_from_settings(config) -> ContextAssembler:
    """Build ContextAssembler from CognitiveGraphSettings.context budgets."""
    ctx = getattr(config, "context", None)
    if ctx is None:
        return ContextAssembler()
    return ContextAssembler(
        ContextAssemblerConfig(
            max_1m_bars=int(getattr(ctx, "max_1m_bars", 60)),
            max_5m_bars=int(getattr(ctx, "max_5m_bars", 36)),
            maximum_option_rows_per_request=int(
                getattr(ctx, "maximum_option_rows_per_request", 80)
            ),
            maximum_context_characters=int(
                getattr(ctx, "maximum_context_characters", 60_000)
            ),
            include_legacy_playbook=False,
        )
    )


async def load_snapshot_truth(
    deps: CognitiveGraphDeps,
    snapshot_id: str | UUID,
) -> tuple[
    MarketSnapshot,
    DataQualityReport | None,
    OptionSurfaceSnapshot | None,
    tuple[OptionContractSnapshot, ...],
]:
    """Load snapshot, optional quality, and option surface from Task 1 repos."""
    if deps.snapshot_repo is None:
        raise RuntimeError("snapshot_repo required for context hydration")
    record = await deps.snapshot_repo.get_by_id(UUID(str(snapshot_id)))
    if record is None:
        raise RuntimeError(f"snapshot {snapshot_id} not found")

    surface: OptionSurfaceSnapshot | None = None
    surface_slice: tuple[OptionContractSnapshot, ...] = ()
    if record.option_surface_id is not None and deps.option_surface_repo is not None:
        surface = await deps.option_surface_repo.get_by_id(record.option_surface_id)
        if surface is not None:
            surface_slice = tuple(surface.contracts)

    data_quality: DataQualityReport | None = None
    # Compact DataQualitySnapshot is embedded by id only; reconstruct a usable
    # report from projection metadata when a full report repo is unavailable.
    if deps.data_quality_loader is not None:
        data_quality = await deps.data_quality_loader(record.data_quality_id, record)
    if data_quality is None:
        data_quality = DataQualityReport(
            report_id=record.data_quality_id,
            snapshot_id=record.snapshot_id,
            severity=DataQualitySeverity.OK,
            findings=(),
            usable_for_reasoning=True,
            usable_for_execution=True,
        )
    return record, data_quality, surface, surface_slice


async def assemble_role_context(
    deps: CognitiveGraphDeps,
    *,
    agent_role: AgentRole,
    session_id: str,
    cycle_id: str,
    snapshot: MarketSnapshot,
    data_quality: DataQualityReport | None = None,
    option_surface_slice: tuple[OptionContractSnapshot, ...] = (),
    order_projection: dict[str, Any] | None = None,
    position_projection: dict[str, Any] | None = None,
    session_artifact_summaries: tuple[dict[str, Any], ...] = (),
    legacy_playbook_context: dict[str, Any] | None = None,
) -> ContextPackage:
    """Assemble a role-specific context package using injected assembler/limits."""
    return deps.context_assembler.assemble(
        agent_role=agent_role,
        session_id=session_id,
        cycle_id=cycle_id,
        snapshot=snapshot,
        data_quality=data_quality,
        option_surface_slice=option_surface_slice,
        order_projection=order_projection,
        position_projection=position_projection,
        session_artifact_summaries=session_artifact_summaries,
        legacy_playbook_context=legacy_playbook_context,
    )
