"""Async Task 3 evolution runtime wired to Task 1/2 events and champion pinning."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import UUID

from joker.evaluation.agentic_graph import AgenticEvaluationGraphRunner
from joker.evaluation.dataset_builder import DatasetBuilder
from joker.evaluation.graph import EvaluationGraphRunner
from joker.events.bus import InProcessAsyncEventBus
from joker.events.schemas import DomainEvent, EventType
from joker.evolution.adversarial_suite import AdversarialResultStore, AdversarialSuiteRunner
from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.checkpointers import EvolutionCheckpointerOwner
from joker.evolution.config import EvolutionSettings
from joker.evolution.configuration_applicator import (
    AppliedConfiguration,
    ConfigurationApplicator,
)
from joker.evolution.decision import EvolutionDecisionService
from joker.evolution.drift import DriftMonitor
from joker.evolution.episode_compiler import EpisodeCompiler
from joker.evolution.evidence_claims import EvidenceClaimStore
from joker.evolution.experiment_runner import ExperimentRunner
from joker.evolution.improvement import ImprovementProposalService
from joker.evolution.improvement_graph import ImprovementGraphRunner
from joker.evolution.lifecycle import PositionLifecycleResolver
from joker.evolution.orchestrator import EvolutionOrchestrator
from joker.evolution.replay import CognitiveReplayService
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import CognitiveConfigurationVersion
from joker.evolution.shadow import ShadowRuntime
from joker.evolution.shadow_ledger import ShadowLedger
from joker.models.router import ModelRouter

logger = logging.getLogger(__name__)

ProjectionLoader = Callable[[], Awaitable[Any]]


@dataclass
class EvolutionRuntimeHealth:
    enabled: bool
    champion_version_id: str | None
    episode_queue_depth: int
    evaluation_queue_depth: int
    shadow_backlog: int
    degraded: bool = False
    last_error: str | None = None


@dataclass
class EvolutionRuntime:
    """Owns Task 3 workers; subscribes to Task 1/2 events; pins champion into Task 2."""

    db_path: Path
    settings: EvolutionSettings
    scope_key: str = "default"
    session_id: str = ""
    run_id: str = ""
    event_bus: InProcessAsyncEventBus | None = None
    execution_runtime: Any | None = None
    model_router: ModelRouter | None = None
    cognitive_graph_deps: Any | None = None
    _repos: dict[str, Any] = field(default_factory=dict)
    champion_registry: ChampionRegistry | None = None
    applicator: ConfigurationApplicator | None = None
    episode_compiler: EpisodeCompiler | None = None
    evaluation_runner: AgenticEvaluationGraphRunner | EvaluationGraphRunner | None = None
    dataset_builder: DatasetBuilder | None = None
    improvement: ImprovementProposalService | None = None
    improvement_graph: ImprovementGraphRunner | None = None
    experiments: ExperimentRunner | None = None
    decisions: EvolutionDecisionService | None = None
    replay: CognitiveReplayService | None = None
    shadow: ShadowRuntime | None = None
    shadow_ledger: ShadowLedger | None = None
    evidence_claims: EvidenceClaimStore | None = None
    adversarial_suite: AdversarialSuiteRunner | None = None
    drift: DriftMonitor | None = None
    orchestrator: EvolutionOrchestrator | None = None
    checkpointer_owner: EvolutionCheckpointerOwner | None = None
    lifecycle_resolver: PositionLifecycleResolver | None = None
    _episode_queue: asyncio.Queue[dict[str, Any]] | None = None
    _eval_queue: asyncio.Queue[UUID] | None = None
    _workers: list[asyncio.Task[None]] = field(default_factory=list)
    _prepared: bool = False
    _workers_started: bool = False
    _subscribed: bool = False
    _pinned_cycle_configs: dict[str, UUID] = field(default_factory=dict)
    _applied_by_cycle: dict[str, AppliedConfiguration] = field(default_factory=dict)
    _cycle_snapshot: dict[str, UUID] = field(default_factory=dict)
    _contract_entry_snapshot: dict[str, UUID] = field(default_factory=dict)
    _position_origin_config: dict[str, UUID] = field(default_factory=dict)
    _latest_snapshot_id: UUID | None = None
    last_error: str | None = None
    _unsubscribe: list[Callable[[], None]] = field(default_factory=list)

    async def prepare(self) -> None:
        """Open stores, champion registry, and durable Task 3 checkpointers."""
        if not self.settings.enabled or self._prepared:
            return
        self._repos = build_evolution_repositories(self.db_path)
        for repo in self._repos.values():
            await repo.initialize()
        self.champion_registry = ChampionRegistry(self.db_path, scope_key=self.scope_key)
        await self.champion_registry.bootstrap_champion()
        self.applicator = ConfigurationApplicator(self.champion_registry.policy_store)
        provenance = None
        if self.cognitive_graph_deps is not None:
            provenance = getattr(self.cognitive_graph_deps, "provenance_registry", None)
        self.lifecycle_resolver = PositionLifecycleResolver(provenance=provenance)
        self.episode_compiler = EpisodeCompiler(
            self._repos["episodes"],
            self._repos["traces"],
            lifecycle_resolver=self.lifecycle_resolver,
            provenance=provenance,
            cycle_registry=(
                getattr(self.cognitive_graph_deps, "cycle_registry", None)
                if self.cognitive_graph_deps is not None
                else None
            ),
        )
        self.checkpointer_owner = EvolutionCheckpointerOwner(self.db_path)
        savers = await self.checkpointer_owner.open_all()
        if self.model_router is not None:
            self.evaluation_runner = AgenticEvaluationGraphRunner(
                self._repos["evaluations"],
                self._repos["traces"],
                router=self.model_router,
                evaluator_version=self.settings.evaluation.evaluator_version,
                checkpointer_saver=savers.evaluation,
            )
        else:
            self.evaluation_runner = EvaluationGraphRunner(
                self._repos["evaluations"],
                self._repos["traces"],
                evaluator_version=self.settings.evaluation.evaluator_version,
            )
        self.dataset_builder = DatasetBuilder(self._repos["datasets"])
        self.improvement = ImprovementProposalService(
            self._repos["proposals"],
            self._repos["configurations"],
            self.champion_registry.policy_store,
        )
        if self.model_router is not None:
            self.improvement_graph = ImprovementGraphRunner(
                router=self.model_router,
                service=self.improvement,
                checkpointer_saver=savers.improvement,
                session_id=self.session_id or "evolution",
            )
        if self.cognitive_graph_deps is not None:
            from joker.evolution.replay_store import ReplayExecutionStore

            exec_store = ReplayExecutionStore(self.db_path)
            await exec_store.initialize()
            session_cash = None
            ledger_store = None
            if self.execution_runtime is not None:
                ledger_store = getattr(self.execution_runtime, "_ledger", None)
                broker = getattr(self.execution_runtime, "_broker", None)
                if broker is not None and hasattr(broker, "initial_balance"):
                    from decimal import Decimal as _D

                    session_cash = _D(str(broker.initial_balance))
            self.replay = CognitiveReplayService(
                template_deps=self.cognitive_graph_deps,
                config_repo=self._repos["configurations"],
                policy_store=self.champion_registry.policy_store,
                checkpointer_saver=savers.replay,
                execution_store=exec_store,
                ledger_store=ledger_store,
                session_starting_cash=session_cash,
                allow_synthetic_starting_cash=session_cash is None,
            )
        self.experiments = ExperimentRunner(
            self._repos["experiments"],
            db_path=self.db_path,
            repeated_samples=self.settings.experiments.repeated_samples,
            replay_service=self.replay,
        )
        self.decisions = EvolutionDecisionService(
            self._repos["promotions"],
            self._repos["configurations"],
            self.champion_registry,
            router=self.model_router,
            checkpointer_saver=savers.decision,
            session_id=self.session_id or "evolution",
        )
        challenger_runner = None
        if self.replay is not None:
            challenger_runner = self.replay.run_challenger_shadow
        self.shadow_ledger = ShadowLedger(self.db_path)
        await self.shadow_ledger.initialize()
        self.evidence_claims = EvidenceClaimStore(self.db_path)
        await self.evidence_claims.initialize()
        self.adversarial_suite = AdversarialSuiteRunner(
            AdversarialResultStore(str(self.db_path))
        )
        self.shadow = ShadowRuntime(
            self._repos["shadow"],
            policy_store=self.champion_registry.policy_store,
            queue_size=self.settings.shadow.queue_size,
            challenger_runner=challenger_runner,
            ledger=self.shadow_ledger,
            config_repo=self._repos["configurations"],
            replay_service=self.replay,
        )
        self.drift = DriftMonitor(
            self._repos["drift"],
            self._repos["rollbacks"],
            self.champion_registry,
            safety_rollback_immediate=self.settings.drift.safety_rollback_immediate,
            strategic_requires_agent=self.settings.drift.strategic_rollback_requires_agent,
        )
        self.orchestrator = EvolutionOrchestrator(
            self,
            checkpointer_saver=savers.orchestrator,
            evidence_claims=self.evidence_claims,
        )
        self._episode_queue = asyncio.Queue(maxsize=256)
        self._eval_queue = asyncio.Queue(maxsize=256)
        self._prepared = True
        self._started = True  # compatibility alias for prepare-complete

    def subscribe_events(self) -> None:
        if self.event_bus is None or self._subscribed or not self.settings.enabled:
            return
        for event_type, handler in (
            (EventType.POSITION_CLOSED, self._on_position_closed),
            (EventType.POSITION_OPENED, self._on_position_opened),
            (EventType.ORDER_REJECTED, self._on_order_rejected),
            (EventType.ORDER_CANCELLED, self._on_order_cancelled),
            (EventType.COGNITIVE_CYCLE_COMPLETED, self._on_cognitive_cycle_completed),
            (EventType.COGNITIVE_CYCLE_STARTED, self._on_cognitive_cycle_started),
        ):
            self.event_bus.subscribe(event_type, handler)
        self._subscribed = True

    async def start_workers(self) -> None:
        if not self.settings.enabled or not self._prepared or self._workers_started:
            return
        if self.shadow is not None and self.settings.shadow.enabled:
            await self.shadow.start()
        self._workers = [
            asyncio.create_task(self._episode_worker(), name="evolution-episode"),
            asyncio.create_task(self._evaluation_worker(), name="evolution-eval"),
            asyncio.create_task(self._orchestrator_worker(), name="evolution-orchestrator"),
        ]
        self._workers_started = True

    async def resume(self) -> None:
        if not self.settings.enabled or not self._prepared:
            return
        # Restore durable shadow state before orchestrator resume.
        if self.shadow is not None:
            await self.shadow.restore_from_ledger()
        # Resume unfinished experiments first, then orchestrator cycles.
        if self.experiments is not None:
            for definition in await self._repos["experiments"].list_resumable():
                dataset = await self._repos["datasets"].get_by_id(definition.dataset_id)
                if dataset is None:
                    continue
                episodes = []
                for eid in dataset.episode_ids:
                    ep = await self._repos["episodes"].get_by_id(eid)
                    if ep is not None:
                        episodes.append(ep)
                try:
                    await self.experiments.resume(
                        definition.experiment_id,
                        episodes=episodes,
                        partition_map=dataset.partition_map,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.last_error = str(exc)
                    logger.exception("evolution_experiment_resume_failed")
        if self.orchestrator is not None:
            await self.orchestrator.resume_all()

    async def start(self) -> None:
        """Compatibility: prepare → subscribe → workers → resume."""
        await self.prepare()
        self.subscribe_events()
        await self.start_workers()
        await self.resume()

    async def shutdown(self) -> None:
        if self.orchestrator is not None:
            self.orchestrator.pause()
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self._workers.clear()
        self._workers_started = False
        if self.shadow is not None:
            await self.shadow.stop()
        if self.checkpointer_owner is not None:
            await self.checkpointer_owner.close_all()
            self.checkpointer_owner = None
        if self.champion_registry is not None:
            await self.champion_registry.close()
        for repo in self._repos.values():
            await repo.close()
        self._prepared = False
        self._started = False

    async def configuration_for_new_cycle(self) -> CognitiveConfigurationVersion | None:
        if self.champion_registry is None:
            return None
        return await self.champion_registry.get_current_champion()

    async def pin_and_apply_for_cycle(self, cycle_id: str) -> AppliedConfiguration | None:
        """Pin the current champion into a new Task 2 cycle."""
        if not self._prepared or self.champion_registry is None or self.applicator is None:
            return None
        if cycle_id in self._applied_by_cycle:
            return self._applied_by_cycle[cycle_id]
        champion = await self.champion_registry.get_current_champion()
        if champion is None:
            return None
        applied = await self.applicator.apply(champion)
        self._pinned_cycle_configs[cycle_id] = champion.configuration_version_id
        self._applied_by_cycle[cycle_id] = applied
        return applied

    async def apply_configuration_version(
        self, cycle_id: str, configuration_version_id: UUID
    ) -> AppliedConfiguration | None:
        """Apply a specific configuration (for recovered Task 2 cycles)."""
        if not self._prepared or self.applicator is None:
            return None
        if cycle_id in self._applied_by_cycle:
            existing = self._applied_by_cycle[cycle_id]
            if existing.configuration_version_id == configuration_version_id:
                return existing
        cfg = await self._repos["configurations"].get_by_id(configuration_version_id)
        if cfg is None:
            return None
        applied = await self.applicator.apply(cfg)
        self._pinned_cycle_configs[cycle_id] = configuration_version_id
        self._applied_by_cycle[cycle_id] = applied
        return applied

    def get_pinned(self, cycle_id: str) -> UUID | None:
        return self._pinned_cycle_configs.get(cycle_id)

    def get_applied(self, cycle_id: str) -> AppliedConfiguration | None:
        return self._applied_by_cycle.get(cycle_id)

    def remember_position_configuration(
        self, contract_id: str, configuration_version_id: UUID
    ) -> None:
        self._position_origin_config[contract_id] = configuration_version_id

    def originating_configuration_for_contract(self, contract_id: str) -> UUID | None:
        return self._position_origin_config.get(contract_id)

    async def current_champion_id(self) -> UUID | None:
        champ = await self.configuration_for_new_cycle()
        return None if champ is None else champ.configuration_version_id

    async def enqueue_episode_job(self, job: dict[str, Any]) -> bool:
        if not self._prepared or self._episode_queue is None:
            return False
        if self._episode_queue.full():
            return False
        await self._episode_queue.put(job)
        return True

    async def health(self) -> EvolutionRuntimeHealth:
        champ_id = await self.current_champion_id()
        return EvolutionRuntimeHealth(
            enabled=self.settings.enabled and self._prepared,
            champion_version_id=str(champ_id) if champ_id else None,
            episode_queue_depth=(
                self._episode_queue.qsize() if self._episode_queue else 0
            ),
            evaluation_queue_depth=self._eval_queue.qsize() if self._eval_queue else 0,
            shadow_backlog=self.shadow.backlog if self.shadow else 0,
            degraded=self.last_error is not None,
            last_error=self.last_error,
        )

    async def _on_cognitive_cycle_started(self, event: DomainEvent) -> None:
        cycle_id = str(event.payload.get("cycle_id") or "")
        snapshot_id = str(event.payload.get("snapshot_id") or "")
        if cycle_id and snapshot_id:
            try:
                snap = UUID(snapshot_id)
                self._cycle_snapshot[cycle_id] = snap
                self._latest_snapshot_id = snap
            except Exception:
                pass
        # New cycles pin champion unless already pinned (recovery path).
        if cycle_id and cycle_id not in self._applied_by_cycle:
            await self.pin_and_apply_for_cycle(cycle_id)

    async def _on_position_opened(self, event: DomainEvent) -> None:
        contract_id = str(event.payload.get("contract_id") or "")
        cycle_id = str(event.payload.get("cycle_id") or "")
        if contract_id and self._latest_snapshot_id is not None:
            self._contract_entry_snapshot[contract_id] = self._latest_snapshot_id
        pinned = self.get_pinned(cycle_id) if cycle_id else None
        if contract_id and pinned is not None:
            self.remember_position_configuration(contract_id, pinned)

    async def _on_position_closed(self, event: DomainEvent) -> None:
        await self.enqueue_episode_job(
            {
                "kind": "position_closed",
                "event_id": str(event.event_id),
                "payload": dict(event.payload),
                "exchange_timestamp": event.exchange_timestamp,
            }
        )

    async def _on_order_rejected(self, event: DomainEvent) -> None:
        await self.enqueue_episode_job(
            {
                "kind": "entry_rejected",
                "event_id": str(event.event_id),
                "payload": dict(event.payload),
                "exchange_timestamp": event.exchange_timestamp,
            }
        )

    async def _on_order_cancelled(self, event: DomainEvent) -> None:
        await self.enqueue_episode_job(
            {
                "kind": "entry_cancelled",
                "event_id": str(event.event_id),
                "payload": dict(event.payload),
                "exchange_timestamp": event.exchange_timestamp,
            }
        )

    async def _on_cognitive_cycle_completed(self, event: DomainEvent) -> None:
        outcome = str(event.payload.get("outcome") or "")
        if outcome in {"no_trade", "hold", "reject", "rejected", ""}:
            await self.enqueue_episode_job(
                {
                    "kind": "no_trade",
                    "event_id": str(event.event_id),
                    "payload": dict(event.payload),
                    "exchange_timestamp": event.exchange_timestamp,
                }
            )
        if self.shadow is not None and self.settings.shadow.enabled:
            snapshot_id = str(event.payload.get("snapshot_id") or "")
            for assignment in await self._repos["shadow"].list_active():
                await self.shadow.enqueue_snapshot(
                    assignment_id=assignment.assignment_id,
                    challenger_version_id=assignment.challenger_version_id,
                    snapshot_id=snapshot_id,
                    payload={"source_event": str(event.event_id)},
                    coalesce=self.settings.shadow.snapshot_coalescing,
                )

    def _trading_date_from_job(self, job: dict[str, Any]) -> date | None:
        ts = job.get("exchange_timestamp")
        if isinstance(ts, datetime):
            return ts.date()
        return None

    async def _episode_worker(self) -> None:
        assert self._episode_queue is not None
        assert self.episode_compiler is not None
        assert self._eval_queue is not None
        while True:
            job = await self._episode_queue.get()
            try:
                episode = await self._compile_job(job)
                if episode is not None:
                    await self._eval_queue.put(episode.episode_id)
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.exception("evolution_episode_worker_failed")
            finally:
                self._episode_queue.task_done()

    async def _compile_job(self, job: dict[str, Any]):
        assert self.episode_compiler is not None
        champ_id = await self.current_champion_id()
        if champ_id is None or self.execution_runtime is None:
            return None
        kind = job.get("kind")
        trading_day = self._trading_date_from_job(job)
        if trading_day is None:
            self.last_error = "missing_exchange_trading_date"
            logger.error("evolution_missing_exchange_trading_date", extra={"kind": kind})
            return None
        if kind == "position_closed":
            payload = job["payload"]
            cycle_id = str(payload.get("cycle_id") or "")
            contract_id = str(payload.get("contract_id") or "")
            snap = self._cycle_snapshot.get(cycle_id)
            if snap is None and contract_id:
                snap = self._contract_entry_snapshot.get(contract_id)
            if snap is None:
                snap = self._latest_snapshot_id
            if snap is None and self.cognitive_graph_deps is not None:
                provenance = getattr(
                    self.cognitive_graph_deps, "provenance_registry", None
                )
                if provenance is not None and contract_id:
                    record = await provenance.get_latest_by_contract_id(contract_id)
                    if record is not None and record.snapshot_id:
                        try:
                            snap = UUID(str(record.snapshot_id))
                        except Exception:
                            snap = None
                        if not cycle_id and record.cycle_id:
                            cycle_id = str(record.cycle_id)
            config_id = (
                self.get_pinned(cycle_id)
                or self.originating_configuration_for_contract(contract_id)
                or champ_id
            )
            entry_snap = snap
            terminal_snap = self._latest_snapshot_id
            return await self.episode_compiler.compile_from_position_closed(
                session_id=self.session_id,
                run_id=self.run_id,
                trading_date=trading_day,
                configuration_version_id=config_id,
                event_payload=payload,
                event_id=job["event_id"],
                execution=self.execution_runtime,
                initial_snapshot_id=entry_snap,
                terminal_snapshot_id=terminal_snap,
                entry_cycle_id=cycle_id or None,
            )
        if kind in {"entry_rejected", "entry_cancelled"}:
            return await self.episode_compiler.compile_from_order_rejected_or_cancelled(
                session_id=self.session_id,
                run_id=self.run_id,
                trading_date=trading_day,
                configuration_version_id=champ_id,
                event_payload=job["payload"],
                event_id=job["event_id"],
                action_class=kind,
                execution=self.execution_runtime,
            )
        if kind == "no_trade":
            payload = job["payload"]
            cycle_id = str(payload.get("cycle_id") or "")
            snapshot_raw = str(payload.get("snapshot_id") or "")
            try:
                snapshot_id: UUID | None = UUID(snapshot_raw)
            except Exception:
                snapshot_id = self._cycle_snapshot.get(cycle_id)
            return await self.episode_compiler.compile_from_no_trade_cycle(
                session_id=self.session_id,
                run_id=self.run_id,
                trading_date=trading_day,
                configuration_version_id=self.get_pinned(cycle_id) or champ_id,
                cycle_id=cycle_id,
                snapshot_id=snapshot_id,
                event_id=job["event_id"],
                outcome=str(payload.get("outcome") or "no_trade"),
            )
        return None

    async def _evaluation_worker(self) -> None:
        assert self._eval_queue is not None
        assert self.evaluation_runner is not None
        while True:
            episode_id = await self._eval_queue.get()
            try:
                episode = await self._repos["episodes"].get_by_id(episode_id)
                if episode is not None:
                    await self.evaluation_runner.evaluate(episode)
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.exception("evolution_evaluation_worker_failed")
            finally:
                self._eval_queue.task_done()

    async def _orchestrator_worker(self) -> None:
        assert self.orchestrator is not None
        interval = max(1, int(self.settings.orchestrator.automatic_cycle_interval_minutes) * 60)
        # Fast tick in tests when interval is very small via monkeypatched settings.
        if self.settings.orchestrator.automatic_cycle_interval_minutes <= 0:
            interval = 1
        while True:
            try:
                await self.orchestrator.tick()
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.exception("evolution_orchestrator_tick_failed")
            await asyncio.sleep(interval)


async def build_status_report(runtime: EvolutionRuntime) -> dict[str, Any]:
    health = await runtime.health()
    history = []
    if runtime.champion_registry is not None:
        history = [
            t.model_dump(mode="json")
            for t in await runtime.champion_registry.compare_champion_history(limit=10)
        ]
    return {
        "health": health.__dict__,
        "champion_history": history,
        "paper_only": True,
        "live_trading_enabled": False,
    }
