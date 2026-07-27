"""Append-only Task 3 repositories with idempotent inserts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import UUID

import aiosqlite
from pydantic import BaseModel

from joker.evolution.hashing import content_hash, hash_model, stable_json_dumps
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.schemas import (
    ChampionActivationRecord,
    ChampionTransition,
    CognitiveConfigurationVersion,
    DecisionTraceSummary,
    DriftObservation,
    EpisodeEvaluation,
    EvaluationDataset,
    EvolutionCycleRecord,
    ExperimentDefinition,
    ExperimentResult,
    ImprovementProposal,
    MemoryLessonEntry,
    PromotionDecision,
    RollbackRecord,
    ShadowAssignment,
    TradingEpisode,
    assert_no_chain_of_thought,
)

T = TypeVar("T", bound=BaseModel)


class EvolutionRepository:
    """Shared initialize + short-lived aiosqlite access for Task 3 tables."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        apply_task3_migrations(self._db_path)
        self._initialized = True

    async def close(self) -> None:
        """No long-lived connection; present for lifecycle symmetry."""
        self._initialized = False

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def _insert_ignore(
        self,
        sql: str,
        params: tuple[Any, ...],
    ) -> bool:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            try:
                await db.execute(sql, params)
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def _fetchone(self, sql: str, params: tuple[Any, ...]) -> aiosqlite.Row | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(sql, params)
            return await cur.fetchone()

    async def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(sql, params)
            return list(await cur.fetchall())


class TradingEpisodeRepository(EvolutionRepository):
    async def append(self, episode: TradingEpisode) -> bool:
        assert_no_chain_of_thought(episode.model_dump(mode="json"))
        payload = episode.model_dump(mode="json")
        ch = episode.idempotency_key or content_hash(stable_json_dumps(payload))
        return await self._insert_ignore(
            """
            INSERT INTO trading_episodes (
                episode_id, idempotency_key, session_id, trading_date,
                configuration_version_id, action_class, completed, content_hash,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(episode.episode_id),
                ch,
                episode.session_id,
                episode.trading_date.isoformat(),
                str(episode.configuration_version_id),
                episode.action_class,
                1 if episode.completed else 0,
                hash_model(episode, exclude={"created_at"}),
                stable_json_dumps(payload),
                episode.created_at.isoformat(),
            ),
        )

    async def get_by_id(self, episode_id: UUID | str) -> TradingEpisode | None:
        row = await self._fetchone(
            "SELECT payload_json FROM trading_episodes WHERE episode_id = ?",
            (str(episode_id),),
        )
        return TradingEpisode.model_validate_json(row["payload_json"]) if row else None

    async def get_by_hash(self, content_hash_value: str) -> TradingEpisode | None:
        row = await self._fetchone(
            "SELECT payload_json FROM trading_episodes WHERE content_hash = ?",
            (content_hash_value,),
        )
        return TradingEpisode.model_validate_json(row["payload_json"]) if row else None

    async def list_by_session(self, session_id: str) -> list[TradingEpisode]:
        rows = await self._fetchall(
            "SELECT payload_json FROM trading_episodes WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        return [TradingEpisode.model_validate_json(r["payload_json"]) for r in rows]

    async def list_by_trading_date(self, trading_date: date) -> list[TradingEpisode]:
        rows = await self._fetchall(
            "SELECT payload_json FROM trading_episodes WHERE trading_date = ? ORDER BY created_at",
            (trading_date.isoformat(),),
        )
        return [TradingEpisode.model_validate_json(r["payload_json"]) for r in rows]

    async def list_by_configuration(
        self, configuration_version_id: UUID | str
    ) -> list[TradingEpisode]:
        rows = await self._fetchall(
            """
            SELECT payload_json FROM trading_episodes
            WHERE configuration_version_id = ? ORDER BY created_at
            """,
            (str(configuration_version_id),),
        )
        return [TradingEpisode.model_validate_json(r["payload_json"]) for r in rows]

    async def list_completed(self, *, limit: int = 500) -> list[TradingEpisode]:
        rows = await self._fetchall(
            """
            SELECT payload_json FROM trading_episodes
            WHERE completed = 1 ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        )
        return [TradingEpisode.model_validate_json(r["payload_json"]) for r in rows]


class EpisodeEvaluationRepository(EvolutionRepository):
    async def append(self, evaluation: EpisodeEvaluation) -> bool:
        assert_no_chain_of_thought(evaluation.model_dump(mode="json"))
        key = evaluation.idempotency_key or str(evaluation.evaluation_id)
        return await self._insert_ignore(
            """
            INSERT INTO episode_evaluations (
                evaluation_id, idempotency_key, episode_id, configuration_version_id,
                content_hash, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(evaluation.evaluation_id),
                key,
                str(evaluation.episode_id),
                str(evaluation.configuration_version_id)
                if evaluation.configuration_version_id
                else None,
                evaluation.content_hash or hash_model(evaluation, exclude={"created_at"}),
                evaluation.model_dump_json(),
                evaluation.created_at.isoformat(),
            ),
        )

    async def get_by_id(self, evaluation_id: UUID | str) -> EpisodeEvaluation | None:
        row = await self._fetchone(
            "SELECT payload_json FROM episode_evaluations WHERE evaluation_id = ?",
            (str(evaluation_id),),
        )
        return EpisodeEvaluation.model_validate_json(row["payload_json"]) if row else None

    async def list_by_episode(self, episode_id: UUID | str) -> list[EpisodeEvaluation]:
        rows = await self._fetchall(
            "SELECT payload_json FROM episode_evaluations WHERE episode_id = ?",
            (str(episode_id),),
        )
        return [EpisodeEvaluation.model_validate_json(r["payload_json"]) for r in rows]


class DecisionTraceRepository(EvolutionRepository):
    async def append(self, summary: DecisionTraceSummary) -> bool:
        assert_no_chain_of_thought(summary.model_dump(mode="json"))
        return await self._insert_ignore(
            """
            INSERT INTO decision_trace_summaries (
                summary_id, episode_id, content_hash, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(summary.summary_id),
                str(summary.episode_id),
                summary.content_hash or hash_model(summary, exclude={"created_at"}),
                summary.model_dump_json(),
                summary.created_at.isoformat(),
            ),
        )

    async def get_by_episode(self, episode_id: UUID | str) -> DecisionTraceSummary | None:
        row = await self._fetchone(
            """
            SELECT payload_json FROM decision_trace_summaries
            WHERE episode_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            (str(episode_id),),
        )
        return DecisionTraceSummary.model_validate_json(row["payload_json"]) if row else None


class ConfigurationVersionRepository(EvolutionRepository):
    async def append(self, version: CognitiveConfigurationVersion) -> bool:
        return await self._insert_ignore(
            """
            INSERT INTO cognitive_configuration_versions (
                configuration_version_id, parent_version_id, status, content_hash,
                scope_key, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(version.configuration_version_id),
                str(version.parent_version_id) if version.parent_version_id else None,
                version.status,
                version.content_hash,
                version.scope_key,
                version.model_dump_json(),
                version.created_at.isoformat(),
            ),
        )

    async def get_by_id(
        self, configuration_version_id: UUID | str
    ) -> CognitiveConfigurationVersion | None:
        row = await self._fetchone(
            """
            SELECT payload_json FROM cognitive_configuration_versions
            WHERE configuration_version_id = ?
            """,
            (str(configuration_version_id),),
        )
        return (
            CognitiveConfigurationVersion.model_validate_json(row["payload_json"])
            if row
            else None
        )

    async def get_by_hash(
        self, content_hash_value: str
    ) -> CognitiveConfigurationVersion | None:
        row = await self._fetchone(
            """
            SELECT payload_json FROM cognitive_configuration_versions
            WHERE content_hash = ?
            """,
            (content_hash_value,),
        )
        return (
            CognitiveConfigurationVersion.model_validate_json(row["payload_json"])
            if row
            else None
        )

    async def mark_status(
        self, configuration_version_id: UUID | str, status: str
    ) -> CognitiveConfigurationVersion | None:
        current = await self.get_by_id(configuration_version_id)
        if current is None:
            return None
        updated = current.model_copy(update={"status": status})
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE cognitive_configuration_versions
                SET status = ?, payload_json = ?
                WHERE configuration_version_id = ?
                """,
                (status, updated.model_dump_json(), str(configuration_version_id)),
            )
            await db.commit()
        return updated

    async def list_by_status(
        self, status: str, *, scope_key: str = "default"
    ) -> list[CognitiveConfigurationVersion]:
        rows = await self._fetchall(
            """
            SELECT payload_json FROM cognitive_configuration_versions
            WHERE status = ? AND scope_key = ? ORDER BY created_at
            """,
            (status, scope_key),
        )
        return [
            CognitiveConfigurationVersion.model_validate_json(r["payload_json"])
            for r in rows
        ]


class DatasetRepository(EvolutionRepository):
    async def append(self, dataset: EvaluationDataset) -> bool:
        inserted = await self._insert_ignore(
            """
            INSERT INTO evaluation_datasets (
                dataset_id, content_hash, payload_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                str(dataset.dataset_id),
                dataset.content_hash or hash_model(dataset, exclude={"created_at"}),
                dataset.model_dump_json(),
                dataset.created_at.isoformat(),
            ),
        )
        if not inserted:
            return False
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            for partition, ids in dataset.partition_map.items():
                for eid in ids:
                    await db.execute(
                        """
                        INSERT OR IGNORE INTO dataset_episode_membership
                        (dataset_id, episode_id, partition_name)
                        VALUES (?, ?, ?)
                        """,
                        (str(dataset.dataset_id), str(eid), partition),
                    )
            await db.commit()
        return True

    async def get_by_id(self, dataset_id: UUID | str) -> EvaluationDataset | None:
        row = await self._fetchone(
            "SELECT payload_json FROM evaluation_datasets WHERE dataset_id = ?",
            (str(dataset_id),),
        )
        return EvaluationDataset.model_validate_json(row["payload_json"]) if row else None


class ImprovementProposalRepository(EvolutionRepository):
    async def append(self, proposal: ImprovementProposal) -> bool:
        key = proposal.idempotency_key or str(proposal.proposal_id)
        return await self._insert_ignore(
            """
            INSERT INTO improvement_proposals (
                proposal_id, idempotency_key, parent_champion_version_id, status,
                content_hash, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(proposal.proposal_id),
                key,
                str(proposal.parent_champion_version_id),
                proposal.status,
                proposal.content_hash or hash_model(proposal, exclude={"created_at"}),
                proposal.model_dump_json(),
                proposal.created_at.isoformat(),
            ),
        )

    async def get_by_id(self, proposal_id: UUID | str) -> ImprovementProposal | None:
        row = await self._fetchone(
            "SELECT payload_json FROM improvement_proposals WHERE proposal_id = ?",
            (str(proposal_id),),
        )
        return ImprovementProposal.model_validate_json(row["payload_json"]) if row else None

    async def list_pending(self) -> list[ImprovementProposal]:
        rows = await self._fetchall(
            """
            SELECT payload_json FROM improvement_proposals
            WHERE status IN ('draft', 'registered') ORDER BY created_at
            """
        )
        return [ImprovementProposal.model_validate_json(r["payload_json"]) for r in rows]


class ExperimentRepository(EvolutionRepository):
    async def append_definition(self, definition: ExperimentDefinition) -> bool:
        return await self._insert_ignore(
            """
            INSERT INTO experiment_definitions (
                experiment_id, status, champion_version_id, challenger_version_id,
                dataset_id, recovery_cursor, content_hash, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(definition.experiment_id),
                definition.status,
                str(definition.champion_version_id),
                str(definition.challenger_version_id),
                str(definition.dataset_id),
                definition.recovery_cursor,
                definition.content_hash or hash_model(definition, exclude={"created_at"}),
                definition.model_dump_json(),
                definition.created_at.isoformat(),
            ),
        )

    async def get_definition(self, experiment_id: UUID | str) -> ExperimentDefinition | None:
        row = await self._fetchone(
            "SELECT payload_json FROM experiment_definitions WHERE experiment_id = ?",
            (str(experiment_id),),
        )
        return ExperimentDefinition.model_validate_json(row["payload_json"]) if row else None

    async def mark_status(
        self,
        experiment_id: UUID | str,
        status: str,
        *,
        recovery_cursor: str | None = None,
    ) -> ExperimentDefinition | None:
        current = await self.get_definition(experiment_id)
        if current is None:
            return None
        updated = current.model_copy(
            update={"status": status, "recovery_cursor": recovery_cursor}
        )
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE experiment_definitions
                SET status = ?, recovery_cursor = ?, payload_json = ?
                WHERE experiment_id = ?
                """,
                (
                    status,
                    recovery_cursor,
                    updated.model_dump_json(),
                    str(experiment_id),
                ),
            )
            await db.commit()
        return updated

    async def list_resumable(self) -> list[ExperimentDefinition]:
        rows = await self._fetchall(
            """
            SELECT payload_json FROM experiment_definitions
            WHERE status IN ('pending', 'running') ORDER BY created_at
            """
        )
        return [ExperimentDefinition.model_validate_json(r["payload_json"]) for r in rows]

    async def append_result(self, result: ExperimentResult) -> bool:
        return await self._insert_ignore(
            """
            INSERT INTO experiment_results (
                result_id, experiment_id, content_hash, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(result.result_id),
                str(result.experiment_id),
                result.content_hash or hash_model(result, exclude={"created_at"}),
                result.model_dump_json(),
                result.created_at.isoformat(),
            ),
        )

    async def get_result(self, experiment_id: UUID | str) -> ExperimentResult | None:
        row = await self._fetchone(
            "SELECT payload_json FROM experiment_results WHERE experiment_id = ?",
            (str(experiment_id),),
        )
        return ExperimentResult.model_validate_json(row["payload_json"]) if row else None


class PromotionDecisionRepository(EvolutionRepository):
    async def append(self, decision: PromotionDecision) -> bool:
        key = decision.idempotency_key or str(decision.promotion_decision_id)
        return await self._insert_ignore(
            """
            INSERT INTO promotion_decisions (
                promotion_decision_id, idempotency_key, experiment_id, final_status,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(decision.promotion_decision_id),
                key,
                str(decision.experiment_id),
                decision.final_status,
                decision.model_dump_json(),
                decision.created_at.isoformat(),
            ),
        )

    async def get_by_id(self, decision_id: UUID | str) -> PromotionDecision | None:
        row = await self._fetchone(
            "SELECT payload_json FROM promotion_decisions WHERE promotion_decision_id = ?",
            (str(decision_id),),
        )
        return PromotionDecision.model_validate_json(row["payload_json"]) if row else None

    async def get_by_experiment(
        self, experiment_id: UUID | str
    ) -> PromotionDecision | None:
        row = await self._fetchone(
            """
            SELECT payload_json FROM promotion_decisions
            WHERE experiment_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            (str(experiment_id),),
        )
        return PromotionDecision.model_validate_json(row["payload_json"]) if row else None


class DriftRepository(EvolutionRepository):
    async def append(self, observation: DriftObservation) -> bool:
        return await self._insert_ignore(
            """
            INSERT INTO drift_observations (
                observation_id, configuration_version_id, dimension, severity,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(observation.observation_id),
                str(observation.configuration_version_id),
                observation.dimension,
                observation.severity,
                observation.model_dump_json(),
                observation.created_at.isoformat(),
            ),
        )

    async def list_by_configuration(
        self, configuration_version_id: UUID | str
    ) -> list[DriftObservation]:
        rows = await self._fetchall(
            """
            SELECT payload_json FROM drift_observations
            WHERE configuration_version_id = ? ORDER BY created_at
            """,
            (str(configuration_version_id),),
        )
        return [DriftObservation.model_validate_json(r["payload_json"]) for r in rows]


class RollbackRepository(EvolutionRepository):
    async def append(self, record: RollbackRecord) -> bool:
        key = record.idempotency_key or str(record.rollback_id)
        return await self._insert_ignore(
            """
            INSERT INTO rollback_records (
                rollback_id, idempotency_key, rolled_back_version_id,
                restored_version_id, recovery_status, payload_json, detection_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.rollback_id),
                key,
                str(record.rolled_back_version_id),
                str(record.restored_version_id),
                record.recovery_status,
                record.model_dump_json(),
                record.detection_timestamp.isoformat(),
            ),
        )

    async def get_by_id(self, rollback_id: UUID | str) -> RollbackRecord | None:
        row = await self._fetchone(
            "SELECT payload_json FROM rollback_records WHERE rollback_id = ?",
            (str(rollback_id),),
        )
        return RollbackRecord.model_validate_json(row["payload_json"]) if row else None

    async def list_pending(self) -> list[RollbackRecord]:
        rows = await self._fetchall(
            """
            SELECT payload_json FROM rollback_records
            WHERE recovery_status = 'pending' ORDER BY detection_timestamp
            """
        )
        return [RollbackRecord.model_validate_json(r["payload_json"]) for r in rows]


class ChampionActivationRepository(EvolutionRepository):
    async def upsert(self, record: ChampionActivationRecord) -> ChampionActivationRecord:
        await self._ensure()
        now = record.updated_at.isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO champion_activations (
                    activation_id, promotion_decision_id, experiment_id,
                    challenger_version_id, previous_champion_version_id,
                    registry_applied, history_verified, configuration_status_applied,
                    completed, failure_codes_json, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(promotion_decision_id) DO UPDATE SET
                    registry_applied=excluded.registry_applied,
                    history_verified=excluded.history_verified,
                    configuration_status_applied=excluded.configuration_status_applied,
                    completed=excluded.completed,
                    failure_codes_json=excluded.failure_codes_json,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(record.activation_id),
                    str(record.promotion_decision_id),
                    str(record.experiment_id),
                    str(record.challenger_version_id),
                    str(record.previous_champion_version_id),
                    1 if record.registry_applied else 0,
                    1 if record.history_verified else 0,
                    1 if record.configuration_status_applied else 0,
                    1 if record.completed else 0,
                    json.dumps(list(record.failure_codes)),
                    record.model_dump_json(),
                    record.created_at.isoformat(),
                    now,
                ),
            )
            await db.commit()
        return record

    async def get_by_decision_id(
        self, promotion_decision_id: UUID | str
    ) -> ChampionActivationRecord | None:
        row = await self._fetchone(
            """
            SELECT payload_json FROM champion_activations
            WHERE promotion_decision_id = ?
            """,
            (str(promotion_decision_id),),
        )
        return (
            ChampionActivationRecord.model_validate_json(row["payload_json"])
            if row
            else None
        )


class ShadowAssignmentRepository(EvolutionRepository):
    async def append(self, assignment: ShadowAssignment) -> bool:
        return await self._insert_ignore(
            """
            INSERT INTO shadow_assignments (
                assignment_id, challenger_version_id, champion_version_id, status,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(assignment.assignment_id),
                str(assignment.challenger_version_id),
                str(assignment.champion_version_id),
                assignment.status,
                assignment.model_dump_json(),
                assignment.created_at.isoformat(),
            ),
        )

    async def list_active(self) -> list[ShadowAssignment]:
        rows = await self._fetchall(
            "SELECT payload_json FROM shadow_assignments WHERE status = 'active'"
        )
        return [ShadowAssignment.model_validate_json(r["payload_json"]) for r in rows]

    async def mark_status(self, assignment_id: UUID | str, status: str) -> None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                "SELECT payload_json FROM shadow_assignments WHERE assignment_id = ?",
                (str(assignment_id),),
            )
            row = await cur.fetchone()
            if row is None:
                return
            current = ShadowAssignment.model_validate_json(row[0])
            updated = current.model_copy(update={"status": status})
            await db.execute(
                """
                UPDATE shadow_assignments SET status = ?, payload_json = ?
                WHERE assignment_id = ?
                """,
                (status, updated.model_dump_json(), str(assignment_id)),
            )
            await db.commit()

    async def append_hypothetical_command(
        self,
        *,
        command_id: str,
        assignment_id: UUID | str,
        challenger_version_id: UUID | str,
        payload: dict[str, Any],
        snapshot_id: str | None = None,
        cycle_id: str | None = None,
        created_at: str,
    ) -> bool:
        assert_no_chain_of_thought(payload)
        return await self._insert_ignore(
            """
            INSERT INTO shadow_hypothetical_commands (
                command_id, assignment_id, challenger_version_id, snapshot_id,
                cycle_id, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                command_id,
                str(assignment_id),
                str(challenger_version_id),
                snapshot_id,
                cycle_id,
                stable_json_dumps(payload),
                created_at,
            ),
        )


class MemoryLessonRepository(EvolutionRepository):
    async def append(self, lesson: MemoryLessonEntry) -> bool:
        return await self._insert_ignore(
            """
            INSERT INTO memory_lesson_entries (
                lesson_id, lesson_type, content_hash, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(lesson.lesson_id),
                lesson.lesson_type,
                lesson.content_hash or hash_model(lesson, exclude={"created_at"}),
                lesson.model_dump_json(),
                lesson.created_at.isoformat(),
            ),
        )

    async def list_recent(self, *, limit: int = 50) -> list[MemoryLessonEntry]:
        rows = await self._fetchall(
            """
            SELECT payload_json FROM memory_lesson_entries
            ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        )
        return [MemoryLessonEntry.model_validate_json(r["payload_json"]) for r in rows]


class EvolutionCycleRepository(EvolutionRepository):
    async def upsert(self, record: EvolutionCycleRecord) -> None:
        assert_no_chain_of_thought(record.payload)
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO evolution_cycles (
                    cycle_id, session_id, status, stage, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, cycle_id) DO UPDATE SET
                    status=excluded.status,
                    stage=excluded.stage,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.cycle_id,
                    record.session_id,
                    record.status,
                    record.stage,
                    stable_json_dumps(record.payload),
                    record.updated_at.isoformat(),
                ),
            )
            await db.commit()

    async def list_resumable(self, session_id: str) -> list[EvolutionCycleRecord]:
        rows = await self._fetchall(
            """
            SELECT cycle_id, session_id, status, stage, payload_json, updated_at
            FROM evolution_cycles
            WHERE session_id = ? AND status IN ('pending', 'running')
            ORDER BY updated_at
            """,
            (session_id,),
        )
        out: list[EvolutionCycleRecord] = []
        for row in rows:
            out.append(
                EvolutionCycleRecord(
                    cycle_id=row["cycle_id"],
                    session_id=row["session_id"],
                    status=row["status"],
                    stage=row["stage"],
                    payload=json.loads(row["payload_json"] or "{}"),
                    updated_at=row["updated_at"],
                )
            )
        return out

    async def list_by_session(self, session_id: str) -> list[EvolutionCycleRecord]:
        rows = await self._fetchall(
            """
            SELECT cycle_id, session_id, status, stage, payload_json, updated_at
            FROM evolution_cycles
            WHERE session_id = ?
            ORDER BY updated_at
            """,
            (session_id,),
        )
        out: list[EvolutionCycleRecord] = []
        for row in rows:
            out.append(
                EvolutionCycleRecord(
                    cycle_id=row["cycle_id"],
                    session_id=row["session_id"],
                    status=row["status"],
                    stage=row["stage"],
                    payload=json.loads(row["payload_json"] or "{}"),
                    updated_at=row["updated_at"],
                )
            )
        return out

    async def get(self, session_id: str, cycle_id: str) -> EvolutionCycleRecord | None:
        rows = await self._fetchall(
            """
            SELECT cycle_id, session_id, status, stage, payload_json, updated_at
            FROM evolution_cycles
            WHERE session_id = ? AND cycle_id = ?
            LIMIT 1
            """,
            (session_id, cycle_id),
        )
        if not rows:
            return None
        row = rows[0]
        return EvolutionCycleRecord(
            cycle_id=row["cycle_id"],
            session_id=row["session_id"],
            status=row["status"],
            stage=row["stage"],
            payload=json.loads(row["payload_json"] or "{}"),
            updated_at=row["updated_at"],
        )


def build_evolution_repositories(db_path: str | Path) -> dict[str, EvolutionRepository]:
    """Construct the Task 3 repository suite sharing one DB path."""
    path = Path(db_path)
    return {
        "episodes": TradingEpisodeRepository(path),
        "evaluations": EpisodeEvaluationRepository(path),
        "traces": DecisionTraceRepository(path),
        "configurations": ConfigurationVersionRepository(path),
        "datasets": DatasetRepository(path),
        "proposals": ImprovementProposalRepository(path),
        "experiments": ExperimentRepository(path),
        "promotions": PromotionDecisionRepository(path),
        "activations": ChampionActivationRepository(path),
        "drift": DriftRepository(path),
        "rollbacks": RollbackRepository(path),
        "shadow": ShadowAssignmentRepository(path),
        "memory": MemoryLessonRepository(path),
        "evolution_cycles": EvolutionCycleRepository(path),
    }
