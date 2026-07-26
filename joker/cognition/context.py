"""Role-specific context assembly from Task 1 truth."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from joker.cognition.exceptions import ContextAssemblyError
from joker.cognition.schemas import AgentRole
from joker.market.bars import MarketBar
from joker.market.option_surface import OptionContractSnapshot
from joker.market.quality import DataQualityReport
from joker.market.snapshots import MarketSnapshot, UnderlyingSnapshot


class ContextTruncationRecord(BaseModel):
    """Records what was removed to satisfy context budgets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_name: str
    original_count: int
    retained_count: int
    reason: str


class ContextPackage(BaseModel):
    """Typed, hashable context for a single agent invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    context_id: str
    agent_role: AgentRole
    session_id: str
    cycle_id: str
    snapshot_id: UUID
    assembled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    source_snapshot_id: UUID
    source_data_quality_id: UUID | None = None
    source_option_surface_id: UUID | None = None
    source_feature_snapshot_id: UUID | None = None
    source_order_ids: tuple[str, ...] = ()
    source_position_ids: tuple[str, ...] = ()
    source_artifact_ids: tuple[UUID, ...] = ()

    underlying: UnderlyingSnapshot | None = None
    bars_1m: tuple[MarketBar, ...] = ()
    bars_5m: tuple[MarketBar, ...] = ()
    data_quality: DataQualityReport | None = None
    option_surface_slice: tuple[OptionContractSnapshot, ...] = ()
    order_projection: dict[str, Any] | None = None
    position_projection: dict[str, Any] | None = None
    session_artifact_summaries: tuple[dict[str, Any], ...] = ()
    legacy_playbook_context: dict[str, Any] | None = None

    truncations: tuple[ContextTruncationRecord, ...] = ()
    context_hash: str
    serialized_size_chars: int

    def to_payload(self) -> dict[str, Any]:
        """Serialize for model provider context_payload."""
        return json.loads(self.model_dump_json())


@dataclass
class ContextAssemblerConfig:
    """Configurable context budgets."""

    max_1m_bars: int = 60
    max_5m_bars: int = 36
    maximum_option_rows_per_request: int = 80
    maximum_context_characters: int = 60_000
    include_legacy_playbook: bool = False


@dataclass
class ContextAssembler:
    """Build role-specific context packages without trade recommendations."""

    config: ContextAssemblerConfig = field(default_factory=ContextAssemblerConfig)

    def assemble(
        self,
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
        """Assemble a bounded context package for the given role."""
        if snapshot.snapshot_id is None:
            raise ContextAssemblyError("snapshot_id is required")

        truncations: list[ContextTruncationRecord] = []
        bars_1m = snapshot.bars_1m
        bars_5m = snapshot.bars_5m

        if len(bars_1m) > self.config.max_1m_bars:
            truncations.append(
                ContextTruncationRecord(
                    field_name="bars_1m",
                    original_count=len(bars_1m),
                    retained_count=self.config.max_1m_bars,
                    reason="max_1m_bars budget",
                )
            )
            bars_1m = bars_1m[-self.config.max_1m_bars :]

        if len(bars_5m) > self.config.max_5m_bars:
            truncations.append(
                ContextTruncationRecord(
                    field_name="bars_5m",
                    original_count=len(bars_5m),
                    retained_count=self.config.max_5m_bars,
                    reason="max_5m_bars budget",
                )
            )
            bars_5m = bars_5m[-self.config.max_5m_bars :]

        surface_slice = option_surface_slice
        if len(surface_slice) > self.config.maximum_option_rows_per_request:
            truncations.append(
                ContextTruncationRecord(
                    field_name="option_surface_slice",
                    original_count=len(surface_slice),
                    retained_count=self.config.maximum_option_rows_per_request,
                    reason="maximum_option_rows_per_request budget",
                )
            )
            surface_slice = surface_slice[: self.config.maximum_option_rows_per_request]

        include_underlying = True
        include_bars = True
        include_surface = False
        include_orders = False
        include_positions = False
        include_artifacts = False
        include_legacy = False

        if agent_role in {
            AgentRole.MARKET_STRUCTURE,
            AgentRole.VOLATILITY,
            AgentRole.TEMPORAL_CONTEXT,
            AgentRole.ANOMALY,
        }:
            include_bars = True
        elif agent_role == AgentRole.OPTIONS_MICROSTRUCTURE:
            include_surface = True
        elif agent_role in {
            AgentRole.PATTERN_MINER,
            AgentRole.SEQUENCE_ANALYST,
            AgentRole.ANALOGY_RETRIEVER,
        }:
            include_artifacts = True
        elif agent_role in {
            AgentRole.BULLISH_INVENTOR,
            AgentRole.BEARISH_INVENTOR,
            AgentRole.NEUTRAL_ADVOCATE,
        }:
            include_surface = True
            include_orders = True
            include_positions = True
            include_artifacts = True
        elif agent_role in {
            AgentRole.STRATEGY_ADVOCATE,
            AgentRole.FALSIFIER,
            AgentRole.HISTORICAL_CRITIC,
            AgentRole.EXECUTION_CRITIC,
            AgentRole.ALTERNATIVE_EXPLANATION,
            AgentRole.META_DECISION,
        }:
            include_artifacts = True
            include_surface = agent_role == AgentRole.EXECUTION_CRITIC
        elif agent_role == AgentRole.ENTRY_TACTICIAN:
            include_surface = True
            include_orders = True
        elif agent_role == AgentRole.ORDER_MANAGER:
            include_orders = True
            include_surface = True
        elif agent_role in {AgentRole.POSITION_THESIS, AgentRole.POSITION_DECISION}:
            include_positions = True
            include_orders = True
            include_artifacts = True

        if self.config.include_legacy_playbook and legacy_playbook_context:
            include_legacy = True

        package_body = {
            "agent_role": agent_role.value,
            "session_id": session_id,
            "cycle_id": cycle_id,
            "snapshot_id": str(snapshot.snapshot_id),
            "underlying": snapshot.underlying.model_dump(mode="json")
            if include_underlying
            else None,
            "bars_1m": [b.model_dump(mode="json") for b in bars_1m] if include_bars else [],
            "bars_5m": [b.model_dump(mode="json") for b in bars_5m] if include_bars else [],
            "data_quality": data_quality.model_dump(mode="json") if data_quality else None,
            "option_surface_slice": [c.model_dump(mode="json") for c in surface_slice]
            if include_surface
            else [],
            "order_projection": order_projection if include_orders else None,
            "position_projection": position_projection if include_positions else None,
            "session_artifact_summaries": list(session_artifact_summaries)
            if include_artifacts
            else [],
            "legacy_playbook_context": legacy_playbook_context if include_legacy else None,
            "truncations": [t.model_dump(mode="json") for t in truncations],
        }

        serialized = json.dumps(package_body, sort_keys=True, separators=(",", ":"))
        size_chars = len(serialized)

        if size_chars > self.config.maximum_context_characters:
            truncations.append(
                ContextTruncationRecord(
                    field_name="serialized_payload",
                    original_count=size_chars,
                    retained_count=self.config.maximum_context_characters,
                    reason="maximum_context_characters budget",
                )
            )
            raise ContextAssemblyError(
                f"context size {size_chars} exceeds maximum_context_characters="
                f"{self.config.maximum_context_characters}"
            )

        context_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        context_id = f"{session_id}:{cycle_id}:{agent_role.value}:{context_hash[:16]}"

        return ContextPackage(
            context_id=context_id,
            agent_role=agent_role,
            session_id=session_id,
            cycle_id=cycle_id,
            snapshot_id=snapshot.snapshot_id,
            source_snapshot_id=snapshot.snapshot_id,
            source_data_quality_id=snapshot.data_quality_id,
            source_option_surface_id=snapshot.option_surface_id,
            source_feature_snapshot_id=snapshot.feature_snapshot_id,
            source_order_ids=self._source_order_ids(order_projection, include_orders),
            source_position_ids=self._source_position_ids(
                position_projection, include_positions
            ),
            source_artifact_ids=self._artifact_ids_from_summaries(
                session_artifact_summaries if include_artifacts else ()
            ),
            underlying=snapshot.underlying if include_underlying else None,
            bars_1m=bars_1m if include_bars else (),
            bars_5m=bars_5m if include_bars else (),
            data_quality=data_quality,
            option_surface_slice=surface_slice if include_surface else (),
            order_projection=order_projection if include_orders else None,
            position_projection=position_projection if include_positions else None,
            session_artifact_summaries=session_artifact_summaries if include_artifacts else (),
            legacy_playbook_context=legacy_playbook_context if include_legacy else None,
            truncations=tuple(truncations),
            context_hash=context_hash,
            serialized_size_chars=size_chars,
        )

    def _artifact_ids_from_summaries(
        self, summaries: tuple[dict[str, Any], ...]
    ) -> tuple[UUID, ...]:
        ids: list[UUID] = []
        for summary in summaries:
            raw = summary.get("artifact_id")
            if raw is None:
                continue
            ids.append(raw if isinstance(raw, UUID) else UUID(str(raw)))
        return tuple(ids)

    def _source_order_ids(
        self, order_projection: dict[str, Any] | None, include: bool
    ) -> tuple[str, ...]:
        if not include or order_projection is None:
            return ()
        oid = order_projection.get("client_order_id")
        return (oid,) if isinstance(oid, str) else ()

    def _source_position_ids(
        self, position_projection: dict[str, Any] | None, include: bool
    ) -> tuple[str, ...]:
        if not include or position_projection is None:
            return ()
        pid = position_projection.get("position_id")
        return (pid,) if isinstance(pid, str) else ()
