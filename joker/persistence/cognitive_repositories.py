"""Typed repository facades over CognitiveArtifactStore."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar
from uuid import UUID

from pydantic import BaseModel

from joker.cognition.artifacts import (
    CognitiveArtifactRecord,
    CognitiveArtifactStore,
    ModelCallRecord,
    artifact_record_from_payload,
)
from joker.cognition.exceptions import (
    ArtifactConflictError,
    ArtifactNotFoundError,
    ArtifactPersistenceError,
)
from joker.cognition.schemas import (
    AgentEvidence,
    CognitiveArtifactType,
    DebateReview,
    ExecutionProposal,
    MarketWorldModel,
    MetaDecision,
    OrderManagementDecision,
    PatternHypothesis,
    PositionThesisVersion,
    StrategyHypothesis,
)

T = TypeVar("T", bound=BaseModel)


class _TypedArtifactRepository:
    """Generic append-only repository for one artefact type."""

    def __init__(
        self,
        store: CognitiveArtifactStore,
        artifact_type: CognitiveArtifactType,
        model_type: type[T],
        id_field: str,
    ) -> None:
        self._store = store
        self._artifact_type = artifact_type
        self._model_type = model_type
        self._id_field = id_field

    @property
    def store(self) -> CognitiveArtifactStore:
        return self._store

    async def initialize(self) -> None:
        await self._store.initialize()

    def _extract_id(self, payload: T) -> UUID:
        value = getattr(payload, self._id_field)
        if isinstance(value, UUID):
            return value
        return UUID(str(value))

    async def append(self, payload: T, **envelope_kwargs) -> UUID:
        """Persist a typed artefact."""
        artifact_id = self._extract_id(payload)
        session_id = envelope_kwargs.pop("session_id", None) or getattr(
            payload, "session_id", None
        )
        snapshot_id = envelope_kwargs.pop("snapshot_id", None) or getattr(
            payload, "snapshot_id", None
        )
        if session_id is None or snapshot_id is None:
            raise ArtifactPersistenceError(
                f"session_id and snapshot_id required to append {self._artifact_type.value}"
            )
        record = artifact_record_from_payload(
            artifact_id=artifact_id,
            artifact_type=self._artifact_type,
            payload=payload,
            session_id=str(session_id),
            snapshot_id=snapshot_id if isinstance(snapshot_id, UUID) else UUID(str(snapshot_id)),
            cycle_id=envelope_kwargs.pop("cycle_id", None) or getattr(payload, "cycle_id", None),
            agent_role=envelope_kwargs.pop("agent_role", None)
            or getattr(payload, "agent_role", None),
            prompt_version=envelope_kwargs.pop("prompt_version", None)
            or getattr(payload, "prompt_version", None),
            model_call_id=envelope_kwargs.pop("model_call_id", None)
            or getattr(payload, "model_call_id", None),
            **envelope_kwargs,
        )
        return await self._store.append_artifact(record)

    def _deserialize(self, record: CognitiveArtifactRecord) -> T:
        return self._model_type.model_validate_json(record.payload_json)

    async def get_by_id(self, artifact_id: UUID | str) -> T | None:
        record = await self._store.get_by_id(artifact_id)
        if record is None or record.artifact_type != self._artifact_type:
            return None
        return self._deserialize(record)

    async def require_by_id(self, artifact_id: UUID | str) -> T:
        item = await self.get_by_id(artifact_id)
        if item is None:
            raise ArtifactNotFoundError(
                f"{self._artifact_type.value} artifact_id={artifact_id} not found"
            )
        return item

    async def list_by_session(self, session_id: str, *, limit: int | None = None) -> list[T]:
        records = await self._store.list_by_session(
            session_id, artifact_type=self._artifact_type, limit=limit
        )
        return [self._deserialize(r) for r in records]

    async def list_by_cycle(
        self, session_id: str, cycle_id: str, *, limit: int | None = None
    ) -> list[T]:
        records = await self._store.list_by_cycle(
            session_id, cycle_id, artifact_type=self._artifact_type, limit=limit
        )
        return [self._deserialize(r) for r in records]

    async def list_by_snapshot(
        self, snapshot_id: UUID | str, *, limit: int | None = None
    ) -> list[T]:
        records = await self._store.list_by_snapshot(
            snapshot_id, artifact_type=self._artifact_type, limit=limit
        )
        return [self._deserialize(r) for r in records]


class EvidenceRepository(_TypedArtifactRepository):
    """Repository for AgentEvidence artefacts."""

    def __init__(self, store: CognitiveArtifactStore | str | Path) -> None:
        resolved = (
            store if isinstance(store, CognitiveArtifactStore) else CognitiveArtifactStore(store)
        )
        super().__init__(
            resolved,
            CognitiveArtifactType.AGENT_EVIDENCE,
            AgentEvidence,
            "evidence_id",
        )


class WorldModelRepository(_TypedArtifactRepository):
    """Repository for MarketWorldModel artefacts."""

    def __init__(self, store: CognitiveArtifactStore | str | Path) -> None:
        resolved = (
            store if isinstance(store, CognitiveArtifactStore) else CognitiveArtifactStore(store)
        )
        super().__init__(
            resolved,
            CognitiveArtifactType.MARKET_WORLD_MODEL,
            MarketWorldModel,
            "world_model_id",
        )


class HypothesisRepository(_TypedArtifactRepository):
    """Repository for PatternHypothesis artefacts."""

    def __init__(self, store: CognitiveArtifactStore | str | Path) -> None:
        resolved = (
            store if isinstance(store, CognitiveArtifactStore) else CognitiveArtifactStore(store)
        )
        super().__init__(
            resolved,
            CognitiveArtifactType.PATTERN_HYPOTHESIS,
            PatternHypothesis,
            "hypothesis_id",
        )


class StrategyRepository(_TypedArtifactRepository):
    """Repository for StrategyHypothesis artefacts."""

    def __init__(self, store: CognitiveArtifactStore | str | Path) -> None:
        resolved = (
            store if isinstance(store, CognitiveArtifactStore) else CognitiveArtifactStore(store)
        )
        super().__init__(
            resolved,
            CognitiveArtifactType.STRATEGY_HYPOTHESIS,
            StrategyHypothesis,
            "strategy_id",
        )


class DebateRepository(_TypedArtifactRepository):
    """Repository for DebateReview artefacts."""

    def __init__(self, store: CognitiveArtifactStore | str | Path) -> None:
        resolved = (
            store if isinstance(store, CognitiveArtifactStore) else CognitiveArtifactStore(store)
        )
        super().__init__(
            resolved,
            CognitiveArtifactType.DEBATE_REVIEW,
            DebateReview,
            "review_id",
        )


class DecisionRepository:
    """Repository for meta-decisions and execution proposals."""

    def __init__(self, store: CognitiveArtifactStore | str | Path) -> None:
        resolved = (
            store if isinstance(store, CognitiveArtifactStore) else CognitiveArtifactStore(store)
        )
        self._meta = _TypedArtifactRepository(
            resolved,
            CognitiveArtifactType.META_DECISION,
            MetaDecision,
            "decision_id",
        )
        self._proposals = _TypedArtifactRepository(
            resolved,
            CognitiveArtifactType.EXECUTION_PROPOSAL,
            ExecutionProposal,
            "proposal_id",
        )

    @property
    def store(self) -> CognitiveArtifactStore:
        return self._meta.store

    async def initialize(self) -> None:
        await self._meta.initialize()

    async def append_meta(self, payload: MetaDecision, **kwargs) -> UUID:
        return await self._meta.append(payload, **kwargs)

    async def append_proposal(self, payload: ExecutionProposal, **kwargs) -> UUID:
        return await self._proposals.append(payload, **kwargs)

    async def get_meta_by_id(self, decision_id: UUID | str) -> MetaDecision | None:
        return await self._meta.get_by_id(decision_id)

    async def get_proposal_by_id(self, proposal_id: UUID | str) -> ExecutionProposal | None:
        return await self._proposals.get_by_id(proposal_id)

    async def list_meta_by_session(self, session_id: str) -> list[MetaDecision]:
        return await self._meta.list_by_session(session_id)

    async def list_meta_by_cycle(self, session_id: str, cycle_id: str) -> list[MetaDecision]:
        return await self._meta.list_by_cycle(session_id, cycle_id)


class PositionThesisRepository(_TypedArtifactRepository):
    """Repository for PositionThesisVersion artefacts."""

    def __init__(self, store: CognitiveArtifactStore | str | Path) -> None:
        resolved = (
            store if isinstance(store, CognitiveArtifactStore) else CognitiveArtifactStore(store)
        )
        super().__init__(
            resolved,
            CognitiveArtifactType.POSITION_THESIS_VERSION,
            PositionThesisVersion,
            "thesis_version_id",
        )


class OrderManagementRepository(_TypedArtifactRepository):
    """Repository for OrderManagementDecision artefacts."""

    def __init__(self, store: CognitiveArtifactStore | str | Path) -> None:
        resolved = (
            store if isinstance(store, CognitiveArtifactStore) else CognitiveArtifactStore(store)
        )
        super().__init__(
            resolved,
            CognitiveArtifactType.ORDER_MANAGEMENT_DECISION,
            OrderManagementDecision,
            "decision_id",
        )


class ModelCallRepository:
    """Thin facade over model-call telemetry."""

    def __init__(self, store: CognitiveArtifactStore | str | Path) -> None:
        self._store = (
            store if isinstance(store, CognitiveArtifactStore) else CognitiveArtifactStore(store)
        )

    @property
    def store(self) -> CognitiveArtifactStore:
        return self._store

    async def initialize(self) -> None:
        await self._store.initialize()

    async def append(self, record: ModelCallRecord) -> UUID:
        return await self._store.append_model_call(record)

    async def get_by_idempotency(self, idempotency_key: str) -> ModelCallRecord | None:
        return await self._store.get_model_call_by_idempotency(idempotency_key)

    async def get_by_id(self, request_id: UUID | str) -> ModelCallRecord | None:
        return await self._store.get_model_call_by_id(request_id)

    async def mark_complete(self, request_id: UUID | str, **kwargs) -> ModelCallRecord:
        return await self._store.mark_model_call_complete(request_id, **kwargs)

    async def mark_failed(self, request_id: UUID | str, **kwargs) -> ModelCallRecord:
        return await self._store.mark_model_call_failed(request_id, **kwargs)
