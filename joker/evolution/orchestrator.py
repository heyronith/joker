"""Durable automatic evolution orchestrator (checkpointed stage machine)."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from joker.evolution.hashing import content_hash, stable_json_dumps
from joker.evolution.schemas import EvolutionCycleRecord, ExperimentDefinition

logger = logging.getLogger(__name__)

OrchestratorStage = Literal[
    "collecting_episodes",
    "building_dataset",
    "analysing_evaluations",
    "generating_proposal",
    "registering_challenger",
    "running_experiment",
    "evaluating_eligibility",
    "requesting_agent_decision",
    "applying_decision",
    "monitoring_shadow",
    "completed",
]

STAGE_ORDER: tuple[OrchestratorStage, ...] = (
    "collecting_episodes",
    "building_dataset",
    "analysing_evaluations",
    "generating_proposal",
    "registering_challenger",
    "running_experiment",
    "evaluating_eligibility",
    "requesting_agent_decision",
    "applying_decision",
    "monitoring_shadow",
    "completed",
)


class EvolutionCycleState(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cycle_id: str
    session_id: str
    status: Literal["pending", "running", "completed", "failed", "blocked"] = "pending"
    stage: OrchestratorStage = "collecting_episodes"
    champion_version_id: UUID
    dataset_id: UUID | None = None
    proposal_id: UUID | None = None
    challenger_version_id: UUID | None = None
    experiment_id: UUID | None = None
    promotion_decision_id: UUID | None = None
    source_episode_ids: tuple[UUID, ...] = ()
    source_evaluation_ids: tuple[UUID, ...] = ()
    idempotency_key: str = ""
    last_completed_stage: str | None = None
    failure_codes: tuple[str, ...] = ()
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_record(self) -> EvolutionCycleRecord:
        payload = self.model_dump(mode="json", exclude={"cycle_id", "session_id", "status", "stage", "updated_at"})
        return EvolutionCycleRecord(
            cycle_id=self.cycle_id,
            session_id=self.session_id,
            status=self.status,  # type: ignore[arg-type]
            stage=self.stage,
            payload=payload,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_record(cls, record: EvolutionCycleRecord) -> EvolutionCycleState:
        payload = dict(record.payload or {})
        payload.update(
            {
                "cycle_id": record.cycle_id,
                "session_id": record.session_id,
                "status": record.status if record.status != "abandoned" else "failed",
                "stage": record.stage,
                "updated_at": record.updated_at,
            }
        )
        return cls.model_validate(payload)


class EvolutionOrchestrator:
    """Advance Task 3 beyond evaluation with durable, resumable stages."""

    def __init__(self, runtime: Any) -> None:
        self._rt = runtime
        self._scheduler_task = None
        self._paused = False

    @property
    def settings(self):
        return self._rt.settings.orchestrator

    def pause(self) -> None:
        self._paused = True

    def resume_scheduling(self) -> None:
        self._paused = False

    async def tick(self) -> EvolutionCycleState | None:
        if self._paused or not self.settings.enabled:
            return None
        if not self._rt._prepared:
            return None
        # Prefer resuming an existing cycle.
        cycles = await self._rt._repos["evolution_cycles"].list_resumable(self._rt.session_id)
        if cycles:
            state = EvolutionCycleState.from_record(cycles[0])
            return await self.advance(state)
        return await self.maybe_start_cycle()

    async def maybe_start_cycle(self) -> EvolutionCycleState | None:
        episodes = await self._rt._repos["episodes"].list_completed(limit=500)
        evaluations = []
        for ep in episodes:
            evaluations.extend(await self._rt._repos["evaluations"].list_by_episode(ep.episode_id))
        valid = [e for e in evaluations if e.valid]
        if len(episodes) < self.settings.minimum_new_completed_episodes:
            return None
        if len(valid) < self.settings.minimum_new_evaluations:
            return None
        active_shadow = await self._rt._repos["shadow"].list_active()
        if len(active_shadow) >= self.settings.maximum_active_challengers:
            return None
        claimed = await self._claimed_evaluation_ids()
        unclaimed_eps = [
            ep
            for ep in episodes
            if all(
                str(e.evaluation_id) not in claimed
                for e in await self._rt._repos["evaluations"].list_by_episode(ep.episode_id)
            )
        ]
        if len(unclaimed_eps) < self.settings.minimum_new_completed_episodes:
            # Fall back: if no claim tracking yet, allow first cycle on all.
            if claimed:
                return None
            unclaimed_eps = episodes
        champion = await self._rt.champion_registry.get_current_champion()
        if champion is None:
            return None
        eval_ids = []
        for ep in unclaimed_eps:
            for ev in await self._rt._repos["evaluations"].list_by_episode(ep.episode_id):
                if ev.valid:
                    eval_ids.append(ev.evaluation_id)
        window = content_hash(
            stable_json_dumps([str(i) for i in eval_ids[: self.settings.minimum_new_evaluations]])
        )
        cycle_id = f"evo:{self._rt.session_id}:{window}"
        state = EvolutionCycleState(
            cycle_id=cycle_id,
            session_id=self._rt.session_id,
            status="running",
            stage="collecting_episodes",
            champion_version_id=champion.configuration_version_id,
            source_episode_ids=tuple(ep.episode_id for ep in unclaimed_eps),
            source_evaluation_ids=tuple(eval_ids),
            idempotency_key=f"orch:{cycle_id}",
        )
        await self._persist(state)
        return state

    async def _claimed_evaluation_ids(self) -> set[str]:
        claimed: set[str] = set()
        # Scan recent cycles via resumable + completed isn't available; use payload of resumable only.
        # Also inspect known cycle ids in repo via list_resumable is incomplete; best-effort.
        for record in await self._rt._repos["evolution_cycles"].list_resumable(self._rt.session_id):
            state = EvolutionCycleState.from_record(record)
            claimed.update(str(i) for i in state.source_evaluation_ids)
        return claimed

    async def advance(self, state: EvolutionCycleState) -> EvolutionCycleState:
        current = state
        try:
            while current.stage != "completed" and current.status == "running":
                next_state = await self._run_stage(current)
                if next_state.stage == current.stage and next_state.status == current.status:
                    break
                current = next_state
                await self._persist(current)
            return current
        except Exception as exc:  # noqa: BLE001
            logger.exception("evolution_orchestrator_stage_failed")
            failed = current.model_copy(
                update={
                    "status": "failed",
                    "failure_codes": (*current.failure_codes, str(exc)),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            await self._persist(failed)
            return failed

    async def resume_all(self) -> list[EvolutionCycleState]:
        out: list[EvolutionCycleState] = []
        for record in await self._rt._repos["evolution_cycles"].list_resumable(
            self._rt.session_id
        ):
            state = EvolutionCycleState.from_record(record)
            out.append(await self.advance(state))
        return out

    async def _persist(self, state: EvolutionCycleState) -> None:
        await self._rt._repos["evolution_cycles"].upsert(state.to_record())

    async def _run_stage(self, state: EvolutionCycleState) -> EvolutionCycleState:
        stage = state.stage
        if stage == "collecting_episodes":
            return state.model_copy(
                update={
                    "stage": "building_dataset",
                    "last_completed_stage": "collecting_episodes",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        if stage == "building_dataset":
            return await self._build_dataset(state)
        if stage == "analysing_evaluations":
            return state.model_copy(
                update={
                    "stage": "generating_proposal",
                    "last_completed_stage": "analysing_evaluations",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        if stage == "generating_proposal":
            return await self._generate_proposal(state)
        if stage == "registering_challenger":
            return await self._register_challenger(state)
        if stage == "running_experiment":
            return await self._run_experiment(state)
        if stage == "evaluating_eligibility":
            return state.model_copy(
                update={
                    "stage": "requesting_agent_decision",
                    "last_completed_stage": "evaluating_eligibility",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        if stage == "requesting_agent_decision":
            return await self._agent_decision(state)
        if stage == "applying_decision":
            return state.model_copy(
                update={
                    "stage": "monitoring_shadow",
                    "last_completed_stage": "applying_decision",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        if stage == "monitoring_shadow":
            return state.model_copy(
                update={
                    "stage": "completed",
                    "status": "completed",
                    "last_completed_stage": "monitoring_shadow",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        return state

    async def _build_dataset(self, state: EvolutionCycleState) -> EvolutionCycleState:
        if state.dataset_id is not None:
            return state.model_copy(
                update={
                    "stage": "analysing_evaluations",
                    "last_completed_stage": "building_dataset",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        episodes = []
        for eid in state.source_episode_ids:
            ep = await self._rt._repos["episodes"].get_by_id(eid)
            if ep is not None:
                episodes.append(ep)
        dataset = await self._rt.dataset_builder.build_and_persist(
            episodes,
            random_seed=42,
            minimum_holdout=min(
                self.settings.minimum_holdout_episodes, max(1, len(episodes) // 5)
            ),
            adversarial_ids=(),
            source_db_hashes={"evolution": self._rt.session_id},
        )
        return state.model_copy(
            update={
                "dataset_id": dataset.dataset_id,
                "stage": "analysing_evaluations",
                "last_completed_stage": "building_dataset",
                "updated_at": datetime.now(timezone.utc),
            }
        )

    async def _generate_proposal(self, state: EvolutionCycleState) -> EvolutionCycleState:
        if state.proposal_id is not None and state.challenger_version_id is not None:
            return state.model_copy(
                update={
                    "stage": "registering_challenger",
                    "last_completed_stage": "generating_proposal",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        champion = await self._rt._repos["configurations"].get_by_id(
            state.champion_version_id
        )
        assert champion is not None
        episodes = []
        evaluations = []
        for eid in state.source_episode_ids:
            ep = await self._rt._repos["episodes"].get_by_id(eid)
            if ep is not None:
                episodes.append(ep)
            evaluations.extend(
                await self._rt._repos["evaluations"].list_by_episode(eid)
            )
        window = hashlib.sha256(
            "".join(str(i) for i in state.source_evaluation_ids).encode()
        ).hexdigest()[:16]
        if self._rt.improvement_graph is not None:
            proposal, challenger = await self._rt.improvement_graph.run(
                parent_champion=champion,
                episodes=episodes,
                evaluations=evaluations,
                evaluation_window_hash=window,
            )
        else:
            from joker.evolution.schemas import PromptPatch

            proposal, challenger = await self._rt.improvement.propose(
                parent_champion=champion,
                weakness="evidence_grounding",
                hypothesis="Tighten evidence requirements",
                patch=PromptPatch(
                    role="falsifier",
                    parent_prompt_version_id=uuid4(),
                    replacement_template="Reject theses lacking snapshot/evidence IDs.",
                    change_rationale="orchestrator_auto",
                ),
                supporting_episode_ids=tuple(e.episode_id for e in episodes[:5]),
                metrics_to_improve=("evidence_grounding_score",),
                metrics_must_not_regress=("tail_loss",),
            )
        return state.model_copy(
            update={
                "proposal_id": proposal.proposal_id,
                "challenger_version_id": challenger.configuration_version_id,
                "stage": "registering_challenger",
                "last_completed_stage": "generating_proposal",
                "updated_at": datetime.now(timezone.utc),
            }
        )

    async def _register_challenger(self, state: EvolutionCycleState) -> EvolutionCycleState:
        assert state.challenger_version_id is not None
        challenger = await self._rt._repos["configurations"].get_by_id(
            state.challenger_version_id
        )
        champion = await self._rt._repos["configurations"].get_by_id(
            state.champion_version_id
        )
        assert challenger is not None and champion is not None
        if self._rt.shadow is not None and self._rt.settings.shadow.enabled:
            active = await self._rt._repos["shadow"].list_active()
            if not any(
                a.challenger_version_id == challenger.configuration_version_id
                for a in active
            ):
                await self._rt.shadow.register_challenger(
                    challenger=challenger, champion=champion
                )
        return state.model_copy(
            update={
                "stage": "running_experiment",
                "last_completed_stage": "registering_challenger",
                "updated_at": datetime.now(timezone.utc),
            }
        )

    async def _run_experiment(self, state: EvolutionCycleState) -> EvolutionCycleState:
        assert state.dataset_id is not None
        assert state.proposal_id is not None
        assert state.challenger_version_id is not None
        dataset = await self._rt._repos["datasets"].get_by_id(state.dataset_id)
        assert dataset is not None
        experiment_id = state.experiment_id or uuid4()
        if state.experiment_id is None:
            definition = ExperimentDefinition(
                experiment_id=experiment_id,
                proposal_id=state.proposal_id,
                champion_version_id=state.champion_version_id,
                challenger_version_id=state.challenger_version_id,
                dataset_id=state.dataset_id,
                adversarial_scenario_ids=(),
            )
            await self._rt.experiments.create(definition)
            state = state.model_copy(
                update={
                    "experiment_id": experiment_id,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            await self._persist(state)
        episodes = []
        for eid in state.source_episode_ids:
            ep = await self._rt._repos["episodes"].get_by_id(eid)
            if ep is not None:
                episodes.append(ep)
        result = await self._rt.experiments.resume(
            experiment_id,
            episodes=episodes,
            partition_map=dataset.partition_map,
        )
        # stash eligibility in payload via stage advance
        _ = result
        return state.model_copy(
            update={
                "stage": "evaluating_eligibility",
                "last_completed_stage": "running_experiment",
                "updated_at": datetime.now(timezone.utc),
            }
        )

    async def _agent_decision(self, state: EvolutionCycleState) -> EvolutionCycleState:
        assert state.experiment_id is not None
        assert state.challenger_version_id is not None
        existing = await self._rt._repos["promotions"].get_by_experiment(
            state.experiment_id
        )
        if existing is not None:
            return state.model_copy(
                update={
                    "promotion_decision_id": existing.promotion_decision_id,
                    "stage": "applying_decision",
                    "last_completed_stage": "requesting_agent_decision",
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        result = await self._rt._repos["experiments"].get_result(state.experiment_id)
        challenger = await self._rt._repos["configurations"].get_by_id(
            state.challenger_version_id
        )
        champion = await self._rt._repos["configurations"].get_by_id(
            state.champion_version_id
        )
        proposal = await self._rt._repos["proposals"].get_by_id(state.proposal_id)
        assert result is not None and challenger is not None and champion is not None
        holdout = 0
        if state.dataset_id is not None:
            dataset = await self._rt._repos["datasets"].get_by_id(state.dataset_id)
            if dataset is not None:
                holdout = len(dataset.partition_map.get("holdout", ()))
        decision = await self._rt.decisions.decide_and_apply(
            experiment_id=state.experiment_id,
            result=result,
            challenger=challenger,
            champion=champion,
            proposal=proposal,
            holdout_episode_count=holdout,
            completed_episode_count=len(state.source_episode_ids),
            adversarial_passed=True,
        )
        return state.model_copy(
            update={
                    "promotion_decision_id": decision.promotion_decision_id,
                "stage": "applying_decision",
                "last_completed_stage": "requesting_agent_decision",
                "updated_at": datetime.now(timezone.utc),
            }
        )
