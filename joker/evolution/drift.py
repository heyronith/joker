"""Drift detection and atomic rollback."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID, uuid4

from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.idempotency import rollback_idempotency_key
from joker.evolution.repositories import DriftRepository, RollbackRepository
from joker.evolution.schemas import DriftObservation, RollbackRecord


class RollbackReviewAgent:
    """Strategic rollback judgement for non-safety degradation."""

    def recommend(
        self,
        observations: list[DriftObservation],
    ) -> Literal["rollback", "observe", "gather_evidence"]:
        if any(o.severity == "critical" for o in observations):
            return "rollback"
        if any(o.severity == "warning" for o in observations):
            return "gather_evidence"
        return "observe"


class DriftMonitor:
    def __init__(
        self,
        drift_repo: DriftRepository,
        rollback_repo: RollbackRepository,
        champion_registry: ChampionRegistry,
        *,
        safety_rollback_immediate: bool = True,
        strategic_requires_agent: bool = True,
        review_agent: RollbackReviewAgent | None = None,
    ) -> None:
        self._drift = drift_repo
        self._rollbacks = rollback_repo
        self._champions = champion_registry
        self._safety_immediate = safety_rollback_immediate
        self._strategic_requires_agent = strategic_requires_agent
        self._review = review_agent or RollbackReviewAgent()

    async def observe(
        self,
        *,
        configuration_version_id: UUID,
        dimension: str,
        baseline_value: Any,
        observed_value: Any,
        severity: Literal["info", "warning", "critical"] = "warning",
        evidence: dict[str, Any] | None = None,
    ) -> DriftObservation:
        obs = DriftObservation(
            observation_id=uuid4(),
            configuration_version_id=configuration_version_id,
            dimension=dimension,
            baseline_value=baseline_value,
            observed_value=observed_value,
            severity=severity,
            evidence=evidence or {},
        )
        await self._drift.append(obs)
        return obs

    async def evaluate_and_maybe_rollback(
        self,
        *,
        current_champion_id: UUID,
        previous_champion_id: UUID,
        observations: list[DriftObservation],
        safety_violation: bool = False,
    ) -> RollbackRecord | None:
        if safety_violation and self._safety_immediate:
            return await self.rollback(
                rolled_back_version_id=current_champion_id,
                restored_version_id=previous_champion_id,
                trigger="safety_violation",
                initiator="deterministic",
                trigger_metrics={"safety_violation": True},
            )

        if self._strategic_requires_agent:
            recommendation = self._review.recommend(observations)
            if recommendation != "rollback":
                return None
            initiator: Literal["deterministic", "agent", "human"] = "agent"
            trigger = "strategic_degradation_agent_review"
        else:
            initiator = "deterministic"
            trigger = "strategic_degradation"

        return await self.rollback(
            rolled_back_version_id=current_champion_id,
            restored_version_id=previous_champion_id,
            trigger=trigger,
            initiator=initiator,
            trigger_metrics={
                "observation_count": len(observations),
                "critical": sum(1 for o in observations if o.severity == "critical"),
            },
        )

    async def rollback(
        self,
        *,
        rolled_back_version_id: UUID,
        restored_version_id: UUID,
        trigger: str,
        initiator: Literal["deterministic", "agent", "human"],
        trigger_metrics: dict[str, Any] | None = None,
        affected_episode_ids: tuple[UUID, ...] = (),
    ) -> RollbackRecord:
        key = rollback_idempotency_key(
            trigger, rolled_back_version_id, restored_version_id
        )
        record = RollbackRecord(
            rollback_id=uuid4(),
            rolled_back_version_id=rolled_back_version_id,
            restored_version_id=restored_version_id,
            trigger=trigger,
            trigger_metrics=trigger_metrics or {},
            initiator=initiator,
            affected_episode_ids=affected_episode_ids,
            detection_timestamp=datetime.now(timezone.utc),
            completion_timestamp=None,
            active_cycles_retained_original_config=True,
            recovery_status="pending",
            idempotency_key=key,
        )
        inserted = await self._rollbacks.append(record)
        if not inserted:
            # Idempotent: return existing pending/completed with same key via scan.
            pending = await self._rollbacks.list_pending()
            for item in pending:
                if item.idempotency_key == key:
                    return item

        await self._champions.rollback(
            restore_version_id=restored_version_id,
            expected_champion_id=rolled_back_version_id,
            reason=f"rollback:{trigger}",
        )
        completed = record.model_copy(
            update={
                "completion_timestamp": datetime.now(timezone.utc),
                "recovery_status": "completed",
            }
        )
        # Append-only: store completion as updated payload via new insert ignored;
        # for tests, return completed view.
        return completed
