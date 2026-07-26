"""Persist and resolve prompt/policy versions referenced by cognitive configurations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import aiosqlite

from joker.cognition.prompts import all_prompts, get_prompt
from joker.cognition.schemas import AgentRole, PromptSpec
from joker.evolution.hashing import content_hash, hash_model, stable_json_dumps
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.schemas import (
    CognitiveConfigurationVersion,
    ContextPolicyVersion,
    DebatePolicyVersion,
    EscalationPolicyVersion,
    MemoryPolicyVersion,
    PromptVersionRecord,
    RoutingPolicyVersion,
    assert_no_chain_of_thought,
)


class PolicyVersionStore:
    """Owns prompt + policy version tables required to materialise a champion."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        apply_task3_migrations(self._db_path)
        self._initialized = True

    async def close(self) -> None:
        self._initialized = False

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def append_prompt(self, record: PromptVersionRecord) -> bool:
        await self._ensure()
        assert_no_chain_of_thought({"template": record.template})
        async with aiosqlite.connect(self._db_path) as db:
            try:
                await db.execute(
                    """
                    INSERT INTO prompt_versions (
                        prompt_version_id, role, content_hash, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.prompt_version_id),
                        record.role,
                        record.content_hash,
                        record.model_dump_json(),
                        record.created_at.isoformat(),
                    ),
                )
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def get_prompt(self, prompt_version_id: UUID | str) -> PromptVersionRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT payload_json FROM prompt_versions WHERE prompt_version_id = ?",
                (str(prompt_version_id),),
            )
            row = await cur.fetchone()
        return PromptVersionRecord.model_validate_json(row["payload_json"]) if row else None

    async def append_policy(
        self,
        table: str,
        version_id: UUID,
        content_hash_value: str,
        payload_json: str,
        created_at: datetime,
    ) -> bool:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            try:
                await db.execute(
                    f"""
                    INSERT INTO {table} (
                        version_id, content_hash, payload_json, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        str(version_id),
                        content_hash_value,
                        payload_json,
                        created_at.isoformat(),
                    ),
                )
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def get_policy(self, table: str, version_id: UUID | str) -> dict[str, Any] | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                f"SELECT payload_json FROM {table} WHERE version_id = ?",
                (str(version_id),),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        import json

        return json.loads(row["payload_json"])

    async def bootstrap_defaults(self) -> CognitiveConfigurationVersion:
        """Create and persist baseline prompt/policy versions + config object."""
        now = datetime.now(timezone.utc)
        prompt_versions: dict[str, UUID] = {}
        for spec in all_prompts():
            record = PromptVersionRecord(
                prompt_version_id=uuid4(),
                role=spec.agent_role.value,
                template=spec.system_template,
                parent_prompt_version_id=None,
                content_hash=spec.content_hash,
                created_by="bootstrap",
                created_at=now,
            )
            await self.append_prompt(record)
            prompt_versions[spec.agent_role.value] = record.prompt_version_id

        context = ContextPolicyVersion(
            content={
                "max_context_characters": 60000,
                "preserve_data_quality": True,
                "preserve_positions": True,
                "preserve_working_orders": True,
                "preserve_snapshot_identity": True,
            },
            created_at=now,
        )
        memory = MemoryPolicyVersion(
            content={"max_memories": 8, "include_contradictions": True},
            created_at=now,
        )
        debate = DebatePolicyVersion(
            content={
                "minimum_reviews": 1,
                "maximum_rounds": 2,
                "dissent_required": False,
            },
            created_at=now,
        )
        routing = RoutingPolicyVersion(
            content={"default_profile": "general_reasoning"},
            created_at=now,
        )
        escalation = EscalationPolicyVersion(
            content={"escalate_on_unresolved_conflict": True},
            created_at=now,
        )
        for policy, table in (
            (context, "context_policy_versions"),
            (memory, "memory_policy_versions"),
            (debate, "debate_policy_versions"),
            (routing, "routing_policy_versions"),
            (escalation, "escalation_policy_versions"),
        ):
            ch = content_hash(stable_json_dumps(policy.content))
            object.__setattr__(policy, "content_hash", ch)
            await self.append_policy(
                table,
                policy.version_id,
                ch,
                policy.model_dump_json(),
                now,
            )

        version = CognitiveConfigurationVersion(
            parent_version_id=None,
            status="champion",
            prompt_versions=prompt_versions,
            role_model_profiles={
                "market_structure": "fast_structured",
                "volatility": "fast_structured",
                "options_microstructure": "fast_structured",
                "temporal_context": "fast_structured",
                "anomaly": "fast_structured",
                "world_model_synthesiser": "general_reasoning",
                "bullish_inventor": "general_reasoning",
                "bearish_inventor": "general_reasoning",
                "strategy_advocate": "general_reasoning",
                "falsifier": "independent_critic",
                "execution_critic": "independent_critic",
                "meta_decision": "general_reasoning",
                "entry_tactician": "general_reasoning",
                "order_manager": "general_reasoning",
                "position_thesis": "general_reasoning",
                "position_decision": "general_reasoning",
            },
            context_policy_version_id=context.version_id,
            memory_policy_version_id=memory.version_id,
            debate_policy_version_id=debate.version_id,
            routing_policy_version_id=routing.version_id,
            escalation_policy_version_id=escalation.version_id,
            content_hash="",
            created_by="bootstrap",
            created_at=now,
            scope_key="default",
        )
        return version.model_copy(
            update={
                "content_hash": hash_model(
                    version, exclude={"created_at", "status", "configuration_version_id"}
                )
            }
        )

    async def materialise_prompt_override(
        self,
        *,
        role: str,
        template: str,
        parent_prompt_version_id: UUID,
        change_rationale: str,
    ) -> PromptVersionRecord:
        record = PromptVersionRecord(
            prompt_version_id=uuid4(),
            role=role,
            template=template,
            parent_prompt_version_id=parent_prompt_version_id,
            content_hash=content_hash(template, role, change_rationale),
            created_by="agent",
            created_at=datetime.now(timezone.utc),
        )
        await self.append_prompt(record)
        return record

    async def resolve_prompt_spec(
        self, configuration: CognitiveConfigurationVersion, role: AgentRole | str
    ) -> PromptSpec:
        role_key = role.value if isinstance(role, AgentRole) else str(role)
        agent_role = role if isinstance(role, AgentRole) else AgentRole(role_key)
        version_id = configuration.prompt_versions.get(role_key)
        if version_id is None:
            return get_prompt(agent_role)
        record = await self.get_prompt(version_id)
        if record is None:
            raise RuntimeError(
                f"configuration {configuration.configuration_version_id} references "
                f"missing prompt version {version_id} for role {role_key}"
            )
        base = get_prompt(agent_role)
        return PromptSpec(
            prompt_id=base.prompt_id,
            version=str(record.prompt_version_id),
            agent_role=agent_role,
            system_template=record.template,
            output_schema_name=base.output_schema_name,
            required_context_schema=base.required_context_schema,
            created_at=record.created_at,
            content_hash=record.content_hash,
        )

    async def resolve_role_profile(
        self, configuration: CognitiveConfigurationVersion, role: str
    ) -> str | None:
        return configuration.role_model_profiles.get(role)

    async def verify_configuration_resolvable(
        self, configuration: CognitiveConfigurationVersion
    ) -> tuple[bool, tuple[str, ...]]:
        problems: list[str] = []
        for role, pid in configuration.prompt_versions.items():
            if await self.get_prompt(pid) is None:
                problems.append(f"missing_prompt:{role}:{pid}")
        for table, vid in (
            ("context_policy_versions", configuration.context_policy_version_id),
            ("memory_policy_versions", configuration.memory_policy_version_id),
            ("debate_policy_versions", configuration.debate_policy_version_id),
            ("routing_policy_versions", configuration.routing_policy_version_id),
            ("escalation_policy_versions", configuration.escalation_policy_version_id),
        ):
            if await self.get_policy(table, vid) is None:
                problems.append(f"missing_policy:{table}:{vid}")
        return (not problems, tuple(problems))
