"""Authoritative champion registry with atomic compare-and-swap transitions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import aiosqlite

from joker.evolution.hashing import content_hash, hash_model, stable_json_dumps
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.repositories import ConfigurationVersionRepository
from joker.evolution.schemas import (
    ChampionTransition,
    CognitiveConfigurationVersion,
    ContextPolicyVersion,
    DebatePolicyVersion,
    EscalationPolicyVersion,
    MemoryPolicyVersion,
    RoutingPolicyVersion,
)


class ChampionRegistryError(RuntimeError):
    """Raised when a champion transition violates integrity rules."""


class ChampionRegistry:
    """Exactly one active champion per scope_key; append-only history."""

    def __init__(self, db_path: str | Path, *, scope_key: str = "default") -> None:
        self._db_path = Path(db_path)
        self._scope_key = scope_key
        self._configs = ConfigurationVersionRepository(self._db_path)
        self._initialized = False

    async def initialize(self) -> None:
        apply_task3_migrations(self._db_path)
        await self._configs.initialize()
        self._initialized = True

    async def close(self) -> None:
        await self._configs.close()
        self._initialized = False

    async def bootstrap_champion(
        self,
        *,
        role_model_profiles: dict[str, str] | None = None,
        prompt_versions: dict[str, UUID] | None = None,
    ) -> CognitiveConfigurationVersion:
        """Create baseline policies + champion if none exists."""
        await self.initialize()
        existing = await self.get_current_champion()
        if existing is not None:
            return existing

        now = datetime.now(timezone.utc)
        context = ContextPolicyVersion(
            content={"max_context_characters": 60000, "preserve_data_quality": True},
            content_hash="",
            created_at=now,
        )
        memory = MemoryPolicyVersion(
            content={"max_memories": 8, "include_contradictions": True},
            content_hash="",
            created_at=now,
        )
        debate = DebatePolicyVersion(
            content={"minimum_reviews": 1, "maximum_rounds": 2, "dissent_required": False},
            content_hash="",
            created_at=now,
        )
        routing = RoutingPolicyVersion(
            content={"default_profile": "general_reasoning"},
            content_hash="",
            created_at=now,
        )
        escalation = EscalationPolicyVersion(
            content={"escalate_on_unresolved_conflict": True},
            content_hash="",
            created_at=now,
        )
        for policy in (context, memory, debate, routing, escalation):
            object.__setattr__(
                policy,
                "content_hash",
                content_hash(stable_json_dumps(policy.content)),
            )

        version = CognitiveConfigurationVersion(
            parent_version_id=None,
            status="champion",
            prompt_versions=prompt_versions or {},
            role_model_profiles=role_model_profiles
            or {
                "perception": "fast_structured",
                "world_model": "general_reasoning",
                "strategy": "general_reasoning",
                "critic": "independent_critic",
                "meta_decision": "general_reasoning",
            },
            context_policy_version_id=context.version_id,
            memory_policy_version_id=memory.version_id,
            debate_policy_version_id=debate.version_id,
            routing_policy_version_id=routing.version_id,
            escalation_policy_version_id=escalation.version_id,
            content_hash="",
            created_by="bootstrap",
            created_at=now,
            scope_key=self._scope_key,
        )
        version = version.model_copy(
            update={
                "content_hash": hash_model(
                    version, exclude={"created_at", "status", "configuration_version_id"}
                )
            }
        )
        await self._configs.append(version)
        await self._activate(
            previous_version_id=None,
            new_version=version,
            reason="bootstrap",
            expected_current=None,
        )
        return version

    async def get_current_champion(self) -> CognitiveConfigurationVersion | None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT configuration_version_id FROM champion_current WHERE scope_key = ?",
                (self._scope_key,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return await self._configs.get_by_id(row["configuration_version_id"])

    async def get_previous_champion(self) -> CognitiveConfigurationVersion | None:
        history = await self.compare_champion_history(limit=2)
        if len(history) < 2:
            return None
        prev_id = history[1].previous_version_id or history[1].new_version_id
        # history[0] is latest; previous champion is history[0].previous_version_id
        latest = history[0]
        if latest.previous_version_id is None:
            return None
        return await self._configs.get_by_id(latest.previous_version_id)

    async def compare_champion_history(
        self, *, limit: int = 20
    ) -> list[ChampionTransition]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT payload_json FROM champion_history
                WHERE scope_key = ? ORDER BY activated_at DESC LIMIT ?
                """,
                (self._scope_key, limit),
            )
            rows = await cur.fetchall()
        return [ChampionTransition.model_validate_json(r["payload_json"]) for r in rows]

    async def promote(
        self,
        *,
        challenger: CognitiveConfigurationVersion,
        expected_champion_id: UUID,
        reason: str,
        experiment_id: UUID | None = None,
        promotion_decision_id: UUID | None = None,
    ) -> ChampionTransition:
        """Atomic CAS promotion. Fails if current champion != expected."""
        await self.initialize()
        if challenger.content_hash != hash_model(
            challenger, exclude={"created_at", "status", "configuration_version_id"}
        ) and challenger.content_hash != challenger.content_hash:
            # Always re-verify stored challenger hash matches DB row.
            pass
        stored = await self._configs.get_by_id(challenger.configuration_version_id)
        if stored is None:
            raise ChampionRegistryError("challenger configuration not found")
        if stored.content_hash != challenger.content_hash:
            raise ChampionRegistryError("challenger configuration-hash mismatch")
        return await self._activate(
            previous_version_id=expected_champion_id,
            new_version=stored.model_copy(update={"status": "champion"}),
            reason=reason,
            expected_current=expected_champion_id,
            experiment_id=experiment_id,
            promotion_decision_id=promotion_decision_id,
        )

    async def rollback(
        self,
        *,
        restore_version_id: UUID,
        expected_champion_id: UUID,
        reason: str,
    ) -> ChampionTransition:
        await self.initialize()
        restore = await self._configs.get_by_id(restore_version_id)
        if restore is None:
            raise ChampionRegistryError("restore target not found")
        return await self._activate(
            previous_version_id=expected_champion_id,
            new_version=restore.model_copy(update={"status": "champion"}),
            reason=reason,
            expected_current=expected_champion_id,
        )

    async def _activate(
        self,
        *,
        previous_version_id: UUID | None,
        new_version: CognitiveConfigurationVersion,
        reason: str,
        expected_current: UUID | None,
        experiment_id: UUID | None = None,
        promotion_decision_id: UUID | None = None,
    ) -> ChampionTransition:
        now = datetime.now(timezone.utc)
        transition = ChampionTransition(
            transition_id=uuid4(),
            scope_key=self._scope_key,
            previous_version_id=previous_version_id,
            new_version_id=new_version.configuration_version_id,
            reason=reason,
            experiment_id=experiment_id,
            promotion_decision_id=promotion_decision_id,
            activated_at=now,
            content_hash="",
        )
        transition = transition.model_copy(
            update={"content_hash": hash_model(transition, exclude={"activated_at"})}
        )

        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                db.row_factory = aiosqlite.Row
                cur = await db.execute(
                    "SELECT configuration_version_id FROM champion_current WHERE scope_key = ?",
                    (self._scope_key,),
                )
                row = await cur.fetchone()
                current_id = row["configuration_version_id"] if row else None
                if expected_current is None:
                    if current_id is not None:
                        raise ChampionRegistryError(
                            "champion already exists; refusing bootstrap overwrite"
                        )
                else:
                    if current_id != str(expected_current):
                        raise ChampionRegistryError(
                            "champion changed during promotion (CAS failure)"
                        )

                if previous_version_id is not None:
                    await db.execute(
                        """
                        UPDATE cognitive_configuration_versions
                        SET status = 'retired'
                        WHERE configuration_version_id = ? AND status = 'champion'
                        """,
                        (str(previous_version_id),),
                    )
                    # Mark rolled_back when reason indicates rollback
                    if reason.startswith("rollback"):
                        await db.execute(
                            """
                            UPDATE cognitive_configuration_versions
                            SET status = 'rolled_back'
                            WHERE configuration_version_id = ?
                            """,
                            (str(previous_version_id),),
                        )

                await db.execute(
                    """
                    UPDATE cognitive_configuration_versions
                    SET status = 'champion', payload_json = ?
                    WHERE configuration_version_id = ?
                    """,
                    (
                        new_version.model_copy(update={"status": "champion"}).model_dump_json(),
                        str(new_version.configuration_version_id),
                    ),
                )
                await db.execute(
                    """
                    INSERT INTO champion_history (
                        transition_id, scope_key, previous_version_id, new_version_id,
                        activated_at, content_hash, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(transition.transition_id),
                        self._scope_key,
                        str(previous_version_id) if previous_version_id else None,
                        str(new_version.configuration_version_id),
                        now.isoformat(),
                        transition.content_hash,
                        transition.model_dump_json(),
                    ),
                )
                await db.execute(
                    """
                    INSERT INTO champion_current (
                        scope_key, configuration_version_id, transition_id, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(scope_key) DO UPDATE SET
                        configuration_version_id=excluded.configuration_version_id,
                        transition_id=excluded.transition_id,
                        updated_at=excluded.updated_at
                    """,
                    (
                        self._scope_key,
                        str(new_version.configuration_version_id),
                        str(transition.transition_id),
                        now.isoformat(),
                    ),
                )
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        return transition
