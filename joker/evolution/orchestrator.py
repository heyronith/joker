"""LangGraph-checkpointed automatic evolution orchestrator."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, TypedDict
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from joker.evolution.adversarial import required_scenario_ids
from joker.evolution.crash_injector import EvolutionCrashInjector, NoopCrashInjector
from joker.evolution.evidence_claims import EvidenceClaim, EvidenceClaimStore
from joker.evolution.hashing import content_hash, stable_json_dumps
from joker.evolution.schemas import EvolutionCycleRecord, ExperimentDefinition
from joker.evolution.shadow_ledger import ShadowEvidenceSummary

logger = logging.getLogger(__name__)

OrchestratorStage = Literal[
    "load_or_create_cycle",
    "claim_evidence",
    "building_dataset",
    "analysing_evaluations",
    "generating_proposal",
    "registering_challenger",
    "create_experiment",
    "running_experiment",
    "run_adversarial_suite",
    "calculating_eligibility",
    "collect_shadow_evidence",
    "requesting_agent_decision",
    "applying_decision",
    "finalise_cycle",
    "completed",
    # Legacy aliases retained for persisted cycle records:
    "collecting_episodes",
    "evaluating_eligibility",
    "monitoring_shadow",
]


class EvolutionCycleState(BaseModel):
    """Audit/index record for an evolution cycle (not the LangGraph checkpoint)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cycle_id: str
    session_id: str
    status: Literal["pending", "running", "completed", "failed", "blocked"] = "pending"
    stage: str = "claim_evidence"
    champion_version_id: UUID
    dataset_id: UUID | None = None
    proposal_id: UUID | None = None
    challenger_version_id: UUID | None = None
    experiment_id: UUID | None = None
    promotion_decision_id: UUID | None = None
    shadow_evidence_id: UUID | None = None
    source_episode_ids: tuple[UUID, ...] = ()
    source_evaluation_ids: tuple[UUID, ...] = ()
    deterministic_eligible: bool | None = None
    deterministic_gate_codes: tuple[str, ...] = ()
    adversarial_passed: bool | None = None
    idempotency_key: str = ""
    last_completed_stage: str | None = None
    failure_codes: tuple[str, ...] = ()
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_record(self) -> EvolutionCycleRecord:
        payload = self.model_dump(
            mode="json",
            exclude={"cycle_id", "session_id", "status", "stage", "updated_at"},
        )
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


class EvolutionOrchestratorState(TypedDict, total=False):
    cycle_id: str
    session_id: str
    champion_version_id: str
    source_episode_ids: list[str]
    source_evaluation_ids: list[str]
    dataset_id: str | None
    proposal_id: str | None
    challenger_version_id: str | None
    experiment_id: str | None
    promotion_decision_id: str | None
    shadow_evidence_id: str | None
    deterministic_eligible: bool | None
    deterministic_gate_codes: list[str]
    adversarial_passed: bool | None
    stage: str
    status: str
    failure_codes: list[str]
    pending_evidence: bool


def _uuid(value: str | None) -> UUID | None:
    return UUID(value) if value else None


class EvolutionOrchestrator:
    """Checkpointed Task 3 evolution workflow compiled with the owned saver."""

    def __init__(
        self,
        runtime: Any,
        *,
        checkpointer_saver: AsyncSqliteSaver | None = None,
        crash_injector: EvolutionCrashInjector | None = None,
        evidence_claims: EvidenceClaimStore | None = None,
    ) -> None:
        self._rt = runtime
        self._checkpointer = checkpointer_saver
        self._crash = crash_injector or NoopCrashInjector()
        self._claims = evidence_claims
        self._scheduler_task = None
        self._paused = False
        self._compiled = None

    @property
    def settings(self):
        return self._rt.settings.orchestrator

    def pause(self) -> None:
        self._paused = True

    def resume_scheduling(self) -> None:
        self._paused = False

    def thread_id(self, *, session_id: str, cycle_id: str, champion_version_id: UUID | str) -> str:
        return (
            f"evolution-orchestrator:{session_id}:{cycle_id}:{champion_version_id}"
        )

    def _graph(self):
        if self._compiled is not None:
            return self._compiled
        if self._checkpointer is None:
            raise RuntimeError(
                "EvolutionOrchestrator requires durable orchestrator checkpointer "
                "when evolution is enabled"
            )
        builder = self._build_graph()
        self._compiled = builder.compile(checkpointer=self._checkpointer)
        return self._compiled

    def _build_graph(self):
        g = StateGraph(EvolutionOrchestratorState)
        nodes = {
            "load_or_create_cycle": self._node_load_or_create,
            "claim_evidence": self._node_claim_evidence,
            "build_dataset": self._node_build_dataset,
            "analyse_evaluations": self._node_analyse,
            "generate_improvement": self._node_generate_improvement,
            "register_challenger": self._node_register_challenger,
            "create_experiment": self._node_create_experiment,
            "run_experiment": self._node_run_experiment,
            "run_adversarial_suite": self._node_adversarial,
            "calculate_eligibility": self._node_eligibility,
            "collect_shadow_evidence": self._node_shadow_evidence,
            "run_promotion_decision": self._node_promotion_decision,
            "apply_promotion_decision": self._node_apply_decision,
            "finalise_cycle": self._node_finalise,
        }
        for name, fn in nodes.items():
            g.add_node(name, _make_wrapped_node(self, name, fn))

        g.add_edge(START, "load_or_create_cycle")
        g.add_edge("load_or_create_cycle", "claim_evidence")
        g.add_edge("claim_evidence", "build_dataset")
        g.add_edge("build_dataset", "analyse_evaluations")
        g.add_edge("analyse_evaluations", "generate_improvement")
        g.add_edge("generate_improvement", "register_challenger")
        g.add_edge("register_challenger", "create_experiment")
        g.add_edge("create_experiment", "run_experiment")
        g.add_edge("run_experiment", "run_adversarial_suite")
        g.add_edge("run_adversarial_suite", "calculate_eligibility")
        g.add_edge("calculate_eligibility", "collect_shadow_evidence")
        g.add_conditional_edges(
            "collect_shadow_evidence",
            _shadow_route,
            {
                "pending": END,
                "ready": "run_promotion_decision",
                "blocked": "finalise_cycle",
            },
        )
        g.add_edge("run_promotion_decision", "apply_promotion_decision")
        g.add_edge("apply_promotion_decision", "finalise_cycle")
        g.add_edge("finalise_cycle", END)
        return g

    async def _sync_audit(self, state: Mapping[str, Any]) -> None:
        cycle = EvolutionCycleState(
            cycle_id=str(state["cycle_id"]),
            session_id=str(state["session_id"]),
            status=state.get("status") or "running",  # type: ignore[arg-type]
            stage=str(state.get("stage") or "running"),
            champion_version_id=UUID(str(state["champion_version_id"])),
            dataset_id=_uuid(state.get("dataset_id")),
            proposal_id=_uuid(state.get("proposal_id")),
            challenger_version_id=_uuid(state.get("challenger_version_id")),
            experiment_id=_uuid(state.get("experiment_id")),
            promotion_decision_id=_uuid(state.get("promotion_decision_id")),
            shadow_evidence_id=_uuid(state.get("shadow_evidence_id")),
            source_episode_ids=tuple(UUID(x) for x in state.get("source_episode_ids") or []),
            source_evaluation_ids=tuple(
                UUID(x) for x in state.get("source_evaluation_ids") or []
            ),
            deterministic_eligible=state.get("deterministic_eligible"),
            deterministic_gate_codes=tuple(state.get("deterministic_gate_codes") or ()),
            adversarial_passed=state.get("adversarial_passed"),
            failure_codes=tuple(state.get("failure_codes") or ()),
            idempotency_key=f"orch:{state['cycle_id']}",
            last_completed_stage=str(state.get("stage") or ""),
        )
        await self._rt._repos["evolution_cycles"].upsert(cycle.to_record())

    async def tick(self) -> EvolutionCycleState | None:
        if self._paused or not self.settings.enabled:
            return None
        if not self._rt._prepared:
            return None
        cycles = await self._rt._repos["evolution_cycles"].list_resumable(self._rt.session_id)
        if cycles:
            state = EvolutionCycleState.from_record(cycles[0])
            return await self.advance(state)
        started = await self.maybe_start_cycle()
        if started is None:
            return None
        return await self.advance(started)

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
        owned: set[str] = set()
        if self._claims is not None:
            owned = await self._claims.list_unclaimed_evaluation_ids()
        unclaimed_evals = [e for e in valid if str(e.evaluation_id) not in owned]
        if len(unclaimed_evals) < self.settings.minimum_new_evaluations:
            return None
        # Map episodes that still have unclaimed evals.
        unclaimed_eps = []
        for ep in episodes:
            ep_evals = await self._rt._repos["evaluations"].list_by_episode(ep.episode_id)
            if any(str(e.evaluation_id) not in owned and e.valid for e in ep_evals):
                unclaimed_eps.append(ep)
        if len(unclaimed_eps) < self.settings.minimum_new_completed_episodes:
            return None
        champion = await self._rt.champion_registry.get_current_champion()
        if champion is None:
            return None
        eval_ids = [e.evaluation_id for e in unclaimed_evals]
        window = content_hash(
            stable_json_dumps([str(i) for i in eval_ids[: self.settings.minimum_new_evaluations]])
        )
        cycle_id = f"evo:{self._rt.session_id}:{window}"
        state = EvolutionCycleState(
            cycle_id=cycle_id,
            session_id=self._rt.session_id,
            status="running",
            stage="load_or_create_cycle",
            champion_version_id=champion.configuration_version_id,
            source_episode_ids=tuple(ep.episode_id for ep in unclaimed_eps),
            source_evaluation_ids=tuple(eval_ids),
            idempotency_key=f"orch:{cycle_id}",
        )
        await self._rt._repos["evolution_cycles"].upsert(state.to_record())
        return state

    def _cycle_to_graph_state(
        self, state: EvolutionCycleState
    ) -> EvolutionOrchestratorState:
        return {
            "cycle_id": state.cycle_id,
            "session_id": state.session_id,
            "champion_version_id": str(state.champion_version_id),
            "source_episode_ids": [str(i) for i in state.source_episode_ids],
            "source_evaluation_ids": [str(i) for i in state.source_evaluation_ids],
            "dataset_id": str(state.dataset_id) if state.dataset_id else None,
            "proposal_id": str(state.proposal_id) if state.proposal_id else None,
            "challenger_version_id": (
                str(state.challenger_version_id) if state.challenger_version_id else None
            ),
            "experiment_id": str(state.experiment_id) if state.experiment_id else None,
            "promotion_decision_id": (
                str(state.promotion_decision_id) if state.promotion_decision_id else None
            ),
            "shadow_evidence_id": (
                str(state.shadow_evidence_id) if state.shadow_evidence_id else None
            ),
            "deterministic_eligible": state.deterministic_eligible,
            "deterministic_gate_codes": list(state.deterministic_gate_codes),
            "adversarial_passed": state.adversarial_passed,
            "stage": state.stage,
            "status": state.status,
            "failure_codes": list(state.failure_codes),
            "pending_evidence": state.stage == "collect_shadow_evidence",
        }

    async def _state_after_graph(
        self,
        state: EvolutionCycleState,
        result: dict[str, Any],
    ) -> EvolutionCycleState:
        record = await self._rt._repos["evolution_cycles"].get(
            state.session_id, state.cycle_id
        )
        if record is not None:
            return EvolutionCycleState.from_record(record)
        return EvolutionCycleState(
            cycle_id=result["cycle_id"],
            session_id=result["session_id"],
            status=result.get("status") or "running",  # type: ignore[arg-type]
            stage=result.get("stage") or "completed",
            champion_version_id=UUID(result["champion_version_id"]),
            dataset_id=_uuid(result.get("dataset_id")),
            proposal_id=_uuid(result.get("proposal_id")),
            challenger_version_id=_uuid(result.get("challenger_version_id")),
            experiment_id=_uuid(result.get("experiment_id")),
            promotion_decision_id=_uuid(result.get("promotion_decision_id")),
            shadow_evidence_id=_uuid(result.get("shadow_evidence_id")),
            source_episode_ids=tuple(
                UUID(x) for x in result.get("source_episode_ids") or []
            ),
            source_evaluation_ids=tuple(
                UUID(x) for x in result.get("source_evaluation_ids") or []
            ),
            deterministic_eligible=result.get("deterministic_eligible"),
            deterministic_gate_codes=tuple(
                result.get("deterministic_gate_codes") or ()
            ),
            adversarial_passed=result.get("adversarial_passed"),
            failure_codes=tuple(result.get("failure_codes") or ()),
        )

    async def _run_nodes(
        self,
        merged: EvolutionOrchestratorState,
        nodes: list[tuple[str, Any]],
    ) -> EvolutionOrchestratorState:
        for name, fn in nodes:
            updates = await fn(merged)
            merged = {**merged, **updates}
            await self._sync_audit(merged)
            await self._crash.after_node(name, merged)
            if merged.get("pending_evidence"):
                break
            if merged.get("status") in {"failed", "blocked"} and name == "finalise_cycle":
                break
        return merged

    async def _advance_from_shadow_pause(
        self, state: EvolutionCycleState
    ) -> EvolutionCycleState:
        """Resume after soft-pause at shadow evidence without replaying experiment/adversarial."""
        merged = self._cycle_to_graph_state(state)
        merged = await self._run_nodes(
            merged,
            [
                ("collect_shadow_evidence", self._node_shadow_evidence),
            ],
        )
        if merged.get("pending_evidence"):
            return await self._state_after_graph(state, merged)
        merged = await self._run_nodes(
            merged,
            [
                ("run_promotion_decision", self._node_promotion_decision),
                ("apply_promotion_decision", self._node_apply_decision),
                ("finalise_cycle", self._node_finalise),
            ],
        )
        return await self._state_after_graph(state, merged)

    async def _advance_from_activation_pause(
        self, state: EvolutionCycleState
    ) -> EvolutionCycleState:
        """Retry activation apply without replaying earlier orchestrator nodes."""
        merged = self._cycle_to_graph_state(state)
        merged = await self._run_nodes(
            merged,
            [
                ("apply_promotion_decision", self._node_apply_decision),
                ("finalise_cycle", self._node_finalise),
            ],
        )
        return await self._state_after_graph(state, merged)

    async def advance(self, state: EvolutionCycleState) -> EvolutionCycleState:
        if state.status in {"completed", "failed"} and state.stage == "completed":
            return state
        # Soft-pauses end the LangGraph run at END. Re-ainvoke from START would
        # replay experiment + adversarial on every tick — hang CI under load.
        if state.stage == "collect_shadow_evidence" and state.status == "running":
            return await self._advance_from_shadow_pause(state)
        if state.stage == "applying_decision" and state.status == "running":
            return await self._advance_from_activation_pause(state)
        graph_state = self._cycle_to_graph_state(state)
        graph_state["pending_evidence"] = False
        thread = self.thread_id(
            session_id=state.session_id,
            cycle_id=state.cycle_id,
            champion_version_id=state.champion_version_id,
        )
        try:
            result = await self._graph().ainvoke(
                graph_state, config={"configurable": {"thread_id": thread}}
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("evolution_orchestrator_graph_failed")
            failed = state.model_copy(
                update={
                    "status": "failed" if "injected_crash" not in str(exc) else "running",
                    "failure_codes": (*state.failure_codes, str(exc)),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            # Keep running status for crash-injector recovery tests.
            if "injected_crash" in str(exc):
                # Audit record already synced by last successful node.
                record = await self._rt._repos["evolution_cycles"].get(
                    state.session_id, state.cycle_id
                )
                if record is not None:
                    return EvolutionCycleState.from_record(record)
            await self._rt._repos["evolution_cycles"].upsert(failed.to_record())
            return failed
        return await self._state_after_graph(state, result)

    async def resume_all(self) -> list[EvolutionCycleState]:
        out: list[EvolutionCycleState] = []
        for record in await self._rt._repos["evolution_cycles"].list_resumable(
            self._rt.session_id
        ):
            state = EvolutionCycleState.from_record(record)
            out.append(await self.advance(state))
        return out

    # --- Nodes -----------------------------------------------------------------

    async def _audit_payload(self, state: EvolutionOrchestratorState) -> dict[str, Any]:
        record = await self._rt._repos["evolution_cycles"].get(
            state["session_id"], state["cycle_id"]
        )
        return dict(record.payload) if record is not None else {}

    async def _node_load_or_create(self, state: EvolutionOrchestratorState) -> dict[str, Any]:
        return {"stage": "claim_evidence", "status": "running"}

    async def _node_claim_evidence(self, state: EvolutionOrchestratorState) -> dict[str, Any]:
        if self._claims is None:
            return {"stage": "building_dataset"}
        existing = await self._claims.list_by_cycle(state["cycle_id"])
        if existing:
            return {
                "stage": "building_dataset",
                "source_evaluation_ids": [str(c.evaluation_id) for c in existing],
            }
        fixed: list[EvidenceClaim] = []
        for eid in state.get("source_evaluation_ids") or []:
            episode_id = None
            for ep_id in state.get("source_episode_ids") or []:
                evals = await self._rt._repos["evaluations"].list_by_episode(UUID(ep_id))
                if any(str(e.evaluation_id) == eid for e in evals):
                    episode_id = UUID(ep_id)
                    break
            if episode_id is None:
                continue
            fixed.append(
                EvidenceClaim(
                    evaluation_id=UUID(eid),
                    episode_id=episode_id,
                    evolution_cycle_id=state["cycle_id"],
                    claim_reason="automatic_cycle",
                )
            )
        ok, inserted = await self._claims.claim_batch(
            evolution_cycle_id=state["cycle_id"],
            claims=fixed,
            minimum_count=self.settings.minimum_new_evaluations,
        )
        if not ok:
            return {
                "status": "failed",
                "stage": "claim_evidence",
                "failure_codes": list(state.get("failure_codes") or [])
                + ["evidence_claim_threshold_not_met"],
            }
        return {
            "stage": "building_dataset",
            "source_evaluation_ids": [str(c.evaluation_id) for c in inserted],
        }

    async def _node_build_dataset(self, state: EvolutionOrchestratorState) -> dict[str, Any]:
        audit = await self._audit_payload(state)
        dataset_id = state.get("dataset_id") or audit.get("dataset_id")
        if dataset_id:
            return {"dataset_id": str(dataset_id), "stage": "analysing_evaluations"}
        episodes = []
        for eid in state.get("source_episode_ids") or []:
            ep = await self._rt._repos["episodes"].get_by_id(UUID(eid))
            if ep is None:
                continue
            if getattr(ep, "snapshot_identity_status", "verified") == "missing":
                continue
            if "legacy_lifecycle_inference" in (ep.completeness_findings or ()):
                continue
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
        if self._claims is not None:
            await self._claims.attach_dataset(state["cycle_id"], dataset.dataset_id)
        return {
            "dataset_id": str(dataset.dataset_id),
            "stage": "analysing_evaluations",
        }

    async def _node_analyse(self, state: EvolutionOrchestratorState) -> dict[str, Any]:
        return {"stage": "generating_proposal"}

    async def _node_generate_improvement(
        self, state: EvolutionOrchestratorState
    ) -> dict[str, Any]:
        audit = await self._audit_payload(state)
        proposal_id = state.get("proposal_id") or audit.get("proposal_id")
        challenger_id = state.get("challenger_version_id") or audit.get(
            "challenger_version_id"
        )
        if proposal_id and challenger_id:
            return {
                "proposal_id": str(proposal_id),
                "challenger_version_id": str(challenger_id),
                "stage": "registering_challenger",
            }
        champion = await self._rt._repos["configurations"].get_by_id(
            UUID(state["champion_version_id"])
        )
        assert champion is not None
        episodes = []
        evaluations = []
        for eid in state.get("source_episode_ids") or []:
            ep = await self._rt._repos["episodes"].get_by_id(UUID(eid))
            if ep is not None:
                episodes.append(ep)
            evaluations.extend(
                await self._rt._repos["evaluations"].list_by_episode(UUID(eid))
            )
        window = hashlib.sha256(
            "".join(state.get("source_evaluation_ids") or []).encode()
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
        return {
            "proposal_id": str(proposal.proposal_id),
            "challenger_version_id": str(challenger.configuration_version_id),
            "stage": "registering_challenger",
        }

    async def _node_register_challenger(
        self, state: EvolutionOrchestratorState
    ) -> dict[str, Any]:
        challenger = await self._rt._repos["configurations"].get_by_id(
            UUID(state["challenger_version_id"])
        )
        champion = await self._rt._repos["configurations"].get_by_id(
            UUID(state["champion_version_id"])
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
        return {"stage": "create_experiment"}

    async def _node_create_experiment(
        self, state: EvolutionOrchestratorState
    ) -> dict[str, Any]:
        audit = await self._audit_payload(state)
        experiment_id = state.get("experiment_id") or audit.get("experiment_id")
        if experiment_id:
            return {"experiment_id": str(experiment_id), "stage": "running_experiment"}
        experiment_id = uuid4()
        definition = ExperimentDefinition(
            experiment_id=experiment_id,
            proposal_id=UUID(str(state.get("proposal_id") or audit.get("proposal_id"))),
            champion_version_id=UUID(state["champion_version_id"]),
            challenger_version_id=UUID(
                str(state.get("challenger_version_id") or audit.get("challenger_version_id"))
            ),
            dataset_id=UUID(str(state.get("dataset_id") or audit.get("dataset_id"))),
            adversarial_scenario_ids=required_scenario_ids(),
        )
        await self._rt.experiments.create(definition)
        return {
            "experiment_id": str(experiment_id),
            "stage": "running_experiment",
        }

    async def _node_run_experiment(
        self, state: EvolutionOrchestratorState
    ) -> dict[str, Any]:
        audit = await self._audit_payload(state)
        dataset_id = UUID(str(state.get("dataset_id") or audit.get("dataset_id")))
        experiment_id = UUID(str(state.get("experiment_id") or audit.get("experiment_id")))
        dataset = await self._rt._repos["datasets"].get_by_id(dataset_id)
        assert dataset is not None
        episodes = []
        for eid in state.get("source_episode_ids") or []:
            ep = await self._rt._repos["episodes"].get_by_id(UUID(eid))
            if ep is not None:
                episodes.append(ep)
        await self._rt.experiments.resume(
            experiment_id,
            episodes=episodes,
            partition_map=dataset.partition_map,
        )
        return {"experiment_id": str(experiment_id), "stage": "run_adversarial_suite"}

    async def _node_adversarial(self, state: EvolutionOrchestratorState) -> dict[str, Any]:
        suite = getattr(self._rt, "adversarial_suite", None)
        if suite is None:
            return {
                "adversarial_passed": False,
                "stage": "calculating_eligibility",
                "failure_codes": list(state.get("failure_codes") or [])
                + ["adversarial_suite_missing"],
            }
        passed, _results = await suite.run_for_experiment(
            experiment_id=UUID(state["experiment_id"]),
            champion_version_id=UUID(state["champion_version_id"]),
            challenger_version_id=UUID(state["challenger_version_id"]),
        )
        return {
            "adversarial_passed": passed,
            "stage": "calculating_eligibility",
        }

    async def _node_eligibility(self, state: EvolutionOrchestratorState) -> dict[str, Any]:
        result = await self._rt._repos["experiments"].get_result(
            UUID(state["experiment_id"])
        )
        eligible = bool(result and result.eligibility_outcome and state.get("adversarial_passed"))
        codes: list[str] = []
        if result is not None:
            codes.extend(result.gate_rejection_codes)
        if not state.get("adversarial_passed"):
            codes.append("adversarial_scenario_failure")
        return {
            "deterministic_eligible": eligible,
            "deterministic_gate_codes": codes,
            "stage": "collect_shadow_evidence",
        }

    async def _node_shadow_evidence(
        self, state: EvolutionOrchestratorState
    ) -> dict[str, Any]:
        shadow_settings = self._rt.settings.shadow
        if not getattr(shadow_settings, "allow_promotion_before_shadow", False):
            # Policy A: require shadow evidence before promotion.
            pass
        else:
            return {"pending_evidence": False, "stage": "requesting_agent_decision"}

        ledger = getattr(self._rt, "shadow_ledger", None)
        assignments = await self._rt._repos["shadow"].list_active()
        assignment = None
        for a in assignments:
            if str(a.challenger_version_id) == state.get("challenger_version_id"):
                assignment = a
                break
        if assignment is None and assignments:
            assignment = assignments[0]
        if assignment is None or ledger is None:
            # No shadow infrastructure — fail closed unless allow_promotion_before_shadow.
            if getattr(shadow_settings, "allow_promotion_before_shadow", False):
                return {"pending_evidence": False, "stage": "requesting_agent_decision"}
            # For tests with zero minimums, synthesise empty met summary.
            min_cycles = int(getattr(shadow_settings, "minimum_completed_cycles", 20))
            if min_cycles <= 0:
                summary = ShadowEvidenceSummary(
                    assignment_id=uuid4(),
                    challenger_version_id=UUID(state["challenger_version_id"]),
                    champion_version_id=UUID(state["champion_version_id"]),
                    minimum_requirements_met=True,
                )
                return {
                    "shadow_evidence_id": str(summary.shadow_evidence_id),
                    "pending_evidence": False,
                    "stage": "requesting_agent_decision",
                }
            return {
                "pending_evidence": True,
                "status": "running",
                "stage": "collect_shadow_evidence",
            }

        observed = await ledger.count_cycles(assignment.assignment_id)
        traded = await ledger.count_traded_cycles(assignment.assignment_id)
        open_positions = await ledger.list_open_positions(assignment.assignment_id)
        rejection: list[str] = []
        min_cycles = int(getattr(shadow_settings, "minimum_completed_cycles", 20))
        min_traded = int(getattr(shadow_settings, "minimum_traded_cycles", 5))
        if observed < min_cycles:
            rejection.append("insufficient_shadow_cycles")
        if traded < min_traded:
            rejection.append("insufficient_traded_cycles")
        met = not rejection
        summary = ShadowEvidenceSummary(
            assignment_id=assignment.assignment_id,
            challenger_version_id=assignment.challenger_version_id,
            champion_version_id=assignment.champion_version_id,
            observed_cycle_count=observed,
            traded_cycle_count=traded,
            open_position_count=len(open_positions),
            minimum_requirements_met=met,
            rejection_codes=tuple(rejection),
        )
        await ledger.save_evidence_summary(summary)
        if not met:
            return {
                "shadow_evidence_id": str(summary.shadow_evidence_id),
                "pending_evidence": True,
                "status": "running",
                "stage": "collect_shadow_evidence",
            }
        return {
            "shadow_evidence_id": str(summary.shadow_evidence_id),
            "pending_evidence": False,
            "stage": "requesting_agent_decision",
        }

    async def _node_promotion_decision(
        self, state: EvolutionOrchestratorState
    ) -> dict[str, Any]:
        existing = await self._rt._repos["promotions"].get_by_experiment(
            UUID(state["experiment_id"])
        )
        if existing is not None:
            return {
                "promotion_decision_id": str(existing.promotion_decision_id),
                "stage": "applying_decision",
            }
        result = await self._rt._repos["experiments"].get_result(
            UUID(state["experiment_id"])
        )
        challenger = await self._rt._repos["configurations"].get_by_id(
            UUID(state["challenger_version_id"])
        )
        champion = await self._rt._repos["configurations"].get_by_id(
            UUID(state["champion_version_id"])
        )
        proposal = await self._rt._repos["proposals"].get_by_id(UUID(state["proposal_id"]))
        if result is None or challenger is None or champion is None:
            return {
                "status": "blocked",
                "stage": "finalise_cycle",
                "failure_codes": list(state.get("failure_codes") or [])
                + list(state.get("deterministic_gate_codes") or [])
                + ["missing_promotion_inputs"],
            }
        holdout = 0
        if state.get("dataset_id"):
            dataset = await self._rt._repos["datasets"].get_by_id(UUID(state["dataset_id"]))
            if dataset is not None:
                holdout = len(dataset.partition_map.get("holdout", ()))
        decision = await self._rt.decisions.decide(
            experiment_id=UUID(state["experiment_id"]),
            result=result,
            challenger=challenger,
            champion=champion,
            proposal=proposal,
            holdout_episode_count=holdout,
            completed_episode_count=len(state.get("source_episode_ids") or []),
            adversarial_passed=bool(state.get("adversarial_passed")),
        )
        return {
            "promotion_decision_id": str(decision.promotion_decision_id),
            "stage": "applying_decision",
        }

    async def _node_apply_decision(
        self, state: EvolutionOrchestratorState
    ) -> dict[str, Any]:
        decision_id = state.get("promotion_decision_id")
        if not decision_id:
            return {
                "status": "blocked",
                "stage": "finalise_cycle",
                "failure_codes": list(state.get("failure_codes") or [])
                + ["missing_promotion_decision"],
            }
        await self._rt.decisions.apply_persisted_decision(
            promotion_decision_id=UUID(str(decision_id))
        )
        activation = None
        activations = self._rt._repos.get("activations")
        if activations is not None:
            activation = await activations.get_by_decision_id(UUID(str(decision_id)))
        if activation is not None and not activation.completed:
            # Fail closed: never leave the cycle running forever waiting on activation.
            return {
                "status": "blocked",
                "stage": "finalise_cycle",
                "failure_codes": list(state.get("failure_codes") or [])
                + list(activation.failure_codes)
                + ["activation_incomplete"],
            }
        return {"stage": "finalise_cycle"}

    async def _node_finalise(self, state: EvolutionOrchestratorState) -> dict[str, Any]:
        if self._claims is not None and state.get("status") != "failed":
            if state.get("experiment_id"):
                await self._claims.mark_consumed(state["cycle_id"])
            elif state.get("dataset_id") is None:
                await self._claims.release_cycle(
                    state["cycle_id"], reason="cycle_failed_pre_dataset"
                )
        status = state.get("status") or "completed"
        if status == "running":
            promotion_id = state.get("promotion_decision_id")
            if promotion_id:
                activations = self._rt._repos.get("activations")
                if activations is not None:
                    activation = await activations.get_by_decision_id(
                        UUID(str(promotion_id))
                    )
                    if activation is not None and not activation.completed:
                        return {
                            "stage": "completed",
                            "status": "blocked",
                            "failure_codes": list(state.get("failure_codes") or [])
                            + list(activation.failure_codes)
                            + ["activation_incomplete"],
                            "pending_evidence": False,
                        }
        if status == "running":
            status = "completed"
        return {"stage": "completed", "status": status, "pending_evidence": False}


def _make_wrapped_node(orch: EvolutionOrchestrator, name: str, fn):
    async def node(state: EvolutionOrchestratorState) -> dict[str, Any]:
        # Idempotent short-circuit when artefact already present for later stages.
        updates = await fn(state)
        merged = {**state, **updates}
        await orch._sync_audit(merged)
        await orch._crash.after_node(name, merged)
        return updates

    return node


def _shadow_route(state: EvolutionOrchestratorState) -> str:
    if state.get("status") in {"failed", "blocked"}:
        return "blocked"
    if state.get("pending_evidence"):
        return "pending"
    return "ready"
