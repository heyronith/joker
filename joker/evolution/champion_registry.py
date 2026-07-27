"""Authoritative champion registry with atomic compare-and-swap transitions."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import aiosqlite

from joker.evolution.hashing import hash_model
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.repositories import ConfigurationVersionRepository
from joker.evolution.schemas import ChampionTransition, CognitiveConfigurationVersion


class ChampionRegistryError(RuntimeError):
    """Raised when a champion transition violates integrity rules."""


class ChampionRegistry:
    """Exactly one active champion per scope_key; append-only history."""

    def __init__(self, db_path: str | Path, *, scope_key: str = "default") -> None:
        self._db_path = Path(db_path)
        self._scope_key = scope_key
        self._configs = ConfigurationVersionRepository(self._db_path)
        self._policies = PolicyVersionStore(self._db_path)
        self._initialized = False

    @property
    def policy_store(self) -> PolicyVersionStore:
        return self._policies

    @property
    def configs(self) -> ConfigurationVersionRepository:
        return self._configs

    async def initialize(self) -> None:
        apply_task3_migrations(self._db_path)
        await self._configs.initialize()
        await self._policies.initialize()
        self._initialized = True

    async def close(self) -> None:
        await self._configs.close()
        await self._policies.close()
        self._initialized = False

    async def bootstrap_champion(
        self,
        *,
        role_model_profiles: dict[str, str] | None = None,
        prompt_versions: dict[str, UUID] | None = None,
    ) -> CognitiveConfigurationVersion:
        """Create and persist baseline prompt/policy versions + champion."""
        await self.initialize()
        existing = await self.get_current_champion()
        if existing is not None:
            ok, problems = await self._policies.verify_configuration_resolvable(existing)
            if not ok:
                raise ChampionRegistryError(
                    "existing champion references missing artefacts: "
                    + ", ".join(problems)
                )
            return existing

        version = await self._policies.bootstrap_defaults()
        if role_model_profiles:
            version = version.model_copy(update={"role_model_profiles": role_model_profiles})
        if prompt_versions:
            version = version.model_copy(update={"prompt_versions": prompt_versions})
        version = version.model_copy(update={"scope_key": self._scope_key})
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
        history = await self.compare_champion_history(limit=1)
        if not history:
            return None
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

    async def repair_promotion_history_if_missing(
        self,
        *,
        previous_version_id: UUID,
        new_version_id: UUID,
        reason: str,
        experiment_id: UUID | None = None,
        promotion_decision_id: UUID | None = None,
    ) -> bool:
        """Idempotently append champion_history when registry already matches challenger."""
        history = await self.compare_champion_history(limit=50)
        if any(
            t.new_version_id == new_version_id
            and t.previous_version_id == previous_version_id
            and (
                promotion_decision_id is None
                or t.promotion_decision_id is None
                or t.promotion_decision_id == promotion_decision_id
            )
            for t in history
        ):
            return True
        current = await self.get_current_champion()
        if current is None or current.configuration_version_id != new_version_id:
            return False
        now = datetime.now(timezone.utc)
        transition = ChampionTransition(
            transition_id=uuid4(),
            scope_key=self._scope_key,
            previous_version_id=previous_version_id,
            new_version_id=new_version_id,
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
                    str(previous_version_id),
                    str(new_version_id),
                    now.isoformat(),
                    transition.content_hash,
                    transition.model_dump_json(),
                ),
            )
            await db.commit()
        return True

    async def promote(
        self,
        *,
        challenger: CognitiveConfigurationVersion,
        expected_champion_id: UUID,
        reason: str,
        experiment_id: UUID | None = None,
        promotion_decision_id: UUID | None = None,
    ) -> ChampionTransition:
        await self.initialize()
        stored = await self._configs.get_by_id(challenger.configuration_version_id)
        if stored is None:
            raise ChampionRegistryError("challenger configuration not found")
        if stored.content_hash != challenger.content_hash:
            raise ChampionRegistryError("challenger configuration-hash mismatch")
        ok, problems = await self._policies.verify_configuration_resolvable(stored)
        if not ok:
            raise ChampionRegistryError(
                "challenger not materialisable: " + ", ".join(problems)
            )
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
        ok, problems = await self._policies.verify_configuration_resolvable(restore)
        if not ok:
            raise ChampionRegistryError(
                "rollback target not materialisable: " + ", ".join(problems)
            )
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
                    new_status = (
                        "rolled_back" if reason.startswith("rollback") else "retired"
                    )
                    await db.execute(
                        """
                        UPDATE cognitive_configuration_versions
                        SET status = ?
                        WHERE configuration_version_id = ?
                        """,
                        (new_status, str(previous_version_id)),
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
