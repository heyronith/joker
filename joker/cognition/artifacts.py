"""Append-only cognitive artifact and model-call persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from joker.cognition.exceptions import (
    ArtifactConflictError,
    ArtifactNotFoundError,
    ArtifactPersistenceError,
    ModelCallNotFoundError,
    ModelCallStateError,
)
from joker.cognition.schemas import (
    SCHEMA_VERSION,
    AgentRole,
    CognitiveArtifactType,
    ModelCallStatus,
)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS cognitive_artifacts (
    artifact_id TEXT PRIMARY KEY NOT NULL,
    artifact_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    session_id TEXT NOT NULL,
    cycle_id TEXT,
    snapshot_id TEXT NOT NULL,
    agent_role TEXT,
    prompt_version TEXT,
    model_call_id TEXT,
    parent_artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cognitive_artifacts_session
    ON cognitive_artifacts (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_artifacts_cycle
    ON cognitive_artifacts (session_id, cycle_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_artifacts_snapshot
    ON cognitive_artifacts (snapshot_id, created_at);

CREATE TABLE IF NOT EXISTS model_calls (
    request_id TEXT PRIMARY KEY NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    escalation_source TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    error_code TEXT,
    validated_output_artifact_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_calls_session
    ON model_calls (session_id, started_at);
CREATE INDEX IF NOT EXISTS idx_model_calls_snapshot
    ON model_calls (snapshot_id, started_at);
"""


class CognitiveArtifactRecord(BaseModel):
    """Stored cognitive artefact envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: UUID
    artifact_type: CognitiveArtifactType
    schema_version: str = SCHEMA_VERSION
    session_id: str
    cycle_id: str | None = None
    snapshot_id: UUID
    agent_role: AgentRole | None = None
    prompt_version: str | None = None
    model_call_id: UUID | None = None
    parent_artifact_ids: tuple[UUID, ...] = ()
    payload_json: str
    created_at: datetime


class ModelCallRecord(BaseModel):
    """Idempotent model-call telemetry record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: UUID
    idempotency_key: str
    session_id: str
    cycle_id: str
    snapshot_id: UUID
    agent_role: AgentRole
    prompt_id: str
    prompt_version: str
    provider: str | None = None
    model: str | None = None
    status: ModelCallStatus
    attempt_count: int = 1
    escalation_source: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_code: str | None = None
    validated_output_artifact_id: UUID | None = None


def build_model_call_idempotency_key(
    *,
    session_id: str,
    cycle_id: str,
    snapshot_id: UUID,
    node_name: str,
    agent_role: AgentRole,
    prompt_version: str,
    context_hash: str,
    attempt_level: int,
) -> str:
    """Derive a deterministic idempotency key for graph recovery."""
    parts = (
        session_id,
        cycle_id,
        str(snapshot_id),
        node_name,
        agent_role.value,
        prompt_version,
        context_hash,
        str(attempt_level),
    )
    return "|".join(parts)


def artifact_record_from_payload(
    *,
    artifact_id: UUID,
    artifact_type: CognitiveArtifactType,
    payload: BaseModel,
    session_id: str,
    snapshot_id: UUID,
    cycle_id: str | None = None,
    agent_role: AgentRole | None = None,
    prompt_version: str | None = None,
    model_call_id: UUID | None = None,
    parent_artifact_ids: tuple[UUID, ...] = (),
    created_at: datetime | None = None,
) -> CognitiveArtifactRecord:
    """Build a storage envelope from a typed payload model."""
    schema_version = getattr(payload, "schema_version", SCHEMA_VERSION)
    return CognitiveArtifactRecord(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        schema_version=schema_version,
        session_id=session_id,
        cycle_id=cycle_id,
        snapshot_id=snapshot_id,
        agent_role=agent_role,
        prompt_version=prompt_version,
        model_call_id=model_call_id,
        parent_artifact_ids=parent_artifact_ids,
        payload_json=payload.model_dump_json(),
        created_at=created_at or datetime.now(timezone.utc),
    )


class CognitiveArtifactStore:
    """Append-only aiosqlite store for cognitive artefacts and model calls."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def initialize(self) -> None:
        """Create tables if missing."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_SQL)
            await db.commit()
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def append_artifact(self, record: CognitiveArtifactRecord) -> UUID:
        """Append an artefact; reject duplicate IDs."""
        await self._ensure_initialized()
        existing = await self.get_by_id(record.artifact_id)
        if existing is not None:
            if existing.payload_json == record.payload_json:
                return record.artifact_id
            raise ArtifactConflictError(
                f"artifact_id={record.artifact_id} already exists with different payload"
            )
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    INSERT INTO cognitive_artifacts (
                        artifact_id, artifact_type, schema_version, session_id,
                        cycle_id, snapshot_id, agent_role, prompt_version,
                        model_call_id, parent_artifact_ids_json, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.artifact_id),
                        record.artifact_type.value,
                        record.schema_version,
                        record.session_id,
                        record.cycle_id,
                        str(record.snapshot_id),
                        record.agent_role.value if record.agent_role else None,
                        record.prompt_version,
                        str(record.model_call_id) if record.model_call_id else None,
                        json.dumps([str(x) for x in record.parent_artifact_ids]),
                        record.payload_json,
                        record.created_at.isoformat(),
                    ),
                )
                await db.commit()
        except aiosqlite.IntegrityError as exc:
            raise ArtifactConflictError(
                f"failed to append artifact_id={record.artifact_id}"
            ) from exc
        except Exception as exc:
            raise ArtifactPersistenceError(f"artifact append failed: {exc}") from exc
        return record.artifact_id

    def _row_to_artifact_record(self, row: aiosqlite.Row) -> CognitiveArtifactRecord:
        parent_ids = json.loads(row["parent_artifact_ids_json"] or "[]")
        return CognitiveArtifactRecord(
            artifact_id=UUID(row["artifact_id"]),
            artifact_type=CognitiveArtifactType(row["artifact_type"]),
            schema_version=row["schema_version"],
            session_id=row["session_id"],
            cycle_id=row["cycle_id"],
            snapshot_id=UUID(row["snapshot_id"]),
            agent_role=AgentRole(row["agent_role"]) if row["agent_role"] else None,
            prompt_version=row["prompt_version"],
            model_call_id=UUID(row["model_call_id"]) if row["model_call_id"] else None,
            parent_artifact_ids=tuple(UUID(x) for x in parent_ids),
            payload_json=row["payload_json"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    async def get_by_id(self, artifact_id: UUID | str) -> CognitiveArtifactRecord | None:
        """Fetch artefact envelope by ID."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM cognitive_artifacts WHERE artifact_id = ?",
                (str(artifact_id),),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_artifact_record(row)

    async def list_by_session(
        self,
        session_id: str,
        *,
        artifact_type: CognitiveArtifactType | None = None,
        limit: int | None = None,
    ) -> list[CognitiveArtifactRecord]:
        """List artefacts for a session, newest first."""
        return await self._list(
            "session_id = ?",
            (session_id,),
            artifact_type=artifact_type,
            limit=limit,
        )

    async def list_by_cycle(
        self,
        session_id: str,
        cycle_id: str,
        *,
        artifact_type: CognitiveArtifactType | None = None,
        limit: int | None = None,
    ) -> list[CognitiveArtifactRecord]:
        """List artefacts for a session cycle."""
        return await self._list(
            "session_id = ? AND cycle_id = ?",
            (session_id, cycle_id),
            artifact_type=artifact_type,
            limit=limit,
        )

    async def list_by_snapshot(
        self,
        snapshot_id: UUID | str,
        *,
        artifact_type: CognitiveArtifactType | None = None,
        limit: int | None = None,
    ) -> list[CognitiveArtifactRecord]:
        """List artefacts for a market snapshot."""
        return await self._list(
            "snapshot_id = ?",
            (str(snapshot_id),),
            artifact_type=artifact_type,
            limit=limit,
        )

    async def _list(
        self,
        where_clause: str,
        params: tuple[Any, ...],
        *,
        artifact_type: CognitiveArtifactType | None,
        limit: int | None,
    ) -> list[CognitiveArtifactRecord]:
        await self._ensure_initialized()
        query = f"SELECT * FROM cognitive_artifacts WHERE {where_clause}"
        args: list[Any] = list(params)
        if artifact_type is not None:
            query += " AND artifact_type = ?"
            args.append(artifact_type.value)
        query += " ORDER BY created_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            args.append(limit)
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, args)
            rows = await cursor.fetchall()
        return [self._row_to_artifact_record(row) for row in rows]

    async def append_model_call(self, record: ModelCallRecord) -> UUID:
        """Record an in-progress or completed model call."""
        await self._ensure_initialized()
        existing = await self.get_model_call_by_idempotency(record.idempotency_key)
        if existing is not None:
            if existing.model_dump_json() == record.model_dump_json():
                return record.request_id
            raise ArtifactConflictError(
                f"idempotency_key={record.idempotency_key!r} already used"
            )
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    INSERT INTO model_calls (
                        request_id, idempotency_key, session_id, cycle_id, snapshot_id,
                        agent_role, prompt_id, prompt_version, provider, model, status,
                        attempt_count, escalation_source, started_at, finished_at,
                        latency_ms, input_tokens, output_tokens, error_code,
                        validated_output_artifact_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.request_id),
                        record.idempotency_key,
                        record.session_id,
                        record.cycle_id,
                        str(record.snapshot_id),
                        record.agent_role.value,
                        record.prompt_id,
                        record.prompt_version,
                        record.provider,
                        record.model,
                        record.status.value,
                        record.attempt_count,
                        record.escalation_source,
                        record.started_at.isoformat(),
                        record.finished_at.isoformat() if record.finished_at else None,
                        record.latency_ms,
                        record.input_tokens,
                        record.output_tokens,
                        record.error_code,
                        str(record.validated_output_artifact_id)
                        if record.validated_output_artifact_id
                        else None,
                    ),
                )
                await db.commit()
        except aiosqlite.IntegrityError as exc:
            raise ArtifactConflictError(
                f"failed to append model call request_id={record.request_id}"
            ) from exc
        except Exception as exc:
            raise ArtifactPersistenceError(f"model call append failed: {exc}") from exc
        return record.request_id

    def _row_to_model_call(self, row: aiosqlite.Row) -> ModelCallRecord:
        return ModelCallRecord(
            request_id=UUID(row["request_id"]),
            idempotency_key=row["idempotency_key"],
            session_id=row["session_id"],
            cycle_id=row["cycle_id"],
            snapshot_id=UUID(row["snapshot_id"]),
            agent_role=AgentRole(row["agent_role"]),
            prompt_id=row["prompt_id"],
            prompt_version=row["prompt_version"],
            provider=row["provider"],
            model=row["model"],
            status=ModelCallStatus(row["status"]),
            attempt_count=row["attempt_count"],
            escalation_source=row["escalation_source"],
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=datetime.fromisoformat(row["finished_at"])
            if row["finished_at"]
            else None,
            latency_ms=row["latency_ms"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            error_code=row["error_code"],
            validated_output_artifact_id=UUID(row["validated_output_artifact_id"])
            if row["validated_output_artifact_id"]
            else None,
        )

    async def get_model_call_by_idempotency(
        self, idempotency_key: str
    ) -> ModelCallRecord | None:
        """Fetch model call by idempotency key."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM model_calls WHERE idempotency_key = ?",
                (idempotency_key,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_model_call(row)

    async def get_model_call_by_id(self, request_id: UUID | str) -> ModelCallRecord | None:
        """Fetch model call by request ID."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM model_calls WHERE request_id = ?",
                (str(request_id),),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_model_call(row)

    async def mark_model_call_complete(
        self,
        request_id: UUID | str,
        *,
        provider: str | None = None,
        model: str | None = None,
        latency_ms: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        validated_output_artifact_id: UUID | None = None,
        finished_at: datetime | None = None,
    ) -> ModelCallRecord:
        """Mark a model call completed with telemetry."""
        record = await self.get_model_call_by_id(request_id)
        if record is None:
            raise ModelCallNotFoundError(f"model call request_id={request_id} not found")
        if record.status == ModelCallStatus.COMPLETED:
            return record
        if record.status == ModelCallStatus.FAILED:
            raise ModelCallStateError(
                f"cannot complete failed model call request_id={request_id}"
            )
        done_at = finished_at or datetime.now(timezone.utc)
        updated = record.model_copy(
            update={
                "status": ModelCallStatus.COMPLETED,
                "provider": provider or record.provider,
                "model": model or record.model,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "validated_output_artifact_id": validated_output_artifact_id,
                "finished_at": done_at,
            }
        )
        await self._update_model_call(updated)
        return updated

    async def mark_model_call_failed(
        self,
        request_id: UUID | str,
        *,
        error_code: str,
        latency_ms: int | None = None,
        finished_at: datetime | None = None,
    ) -> ModelCallRecord:
        """Mark a model call failed."""
        record = await self.get_model_call_by_id(request_id)
        if record is None:
            raise ModelCallNotFoundError(f"model call request_id={request_id} not found")
        if record.status == ModelCallStatus.COMPLETED:
            raise ModelCallStateError(
                f"cannot fail completed model call request_id={request_id}"
            )
        done_at = finished_at or datetime.now(timezone.utc)
        updated = record.model_copy(
            update={
                "status": ModelCallStatus.FAILED,
                "error_code": error_code,
                "latency_ms": latency_ms,
                "finished_at": done_at,
            }
        )
        await self._update_model_call(updated)
        return updated

    async def _update_model_call(self, record: ModelCallRecord) -> None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE model_calls SET
                    provider = ?, model = ?, status = ?, attempt_count = ?,
                    escalation_source = ?, finished_at = ?, latency_ms = ?,
                    input_tokens = ?, output_tokens = ?, error_code = ?,
                    validated_output_artifact_id = ?
                WHERE request_id = ?
                """,
                (
                    record.provider,
                    record.model,
                    record.status.value,
                    record.attempt_count,
                    record.escalation_source,
                    record.finished_at.isoformat() if record.finished_at else None,
                    record.latency_ms,
                    record.input_tokens,
                    record.output_tokens,
                    record.error_code,
                    str(record.validated_output_artifact_id)
                    if record.validated_output_artifact_id
                    else None,
                    str(record.request_id),
                ),
            )
            await db.commit()
