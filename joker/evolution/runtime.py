"""Async Task 3 evolution runtime wired to Task 1/2 events and champion pinning."""

from __future__ import annotations

import asyncio
import json
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
from joker.evolution.event_horizon import Task1EventHorizonLoader
from joker.evolution.evidence_claims import EvidenceClaimStore
from joker.evolution.experiment_runner import ExperimentRunner
from joker.evolution.improvement import ImprovementProposalService
from joker.evolution.improvement_graph import ImprovementGraphRunner
from joker.evolution.lifecycle import PositionLifecycleResolver
from joker.evolution.orchestrator import EvolutionOrchestrator
from joker.evolution.promotion_gate import PromotionEligibilityGate
from joker.evolution.replay import CognitiveReplayService
from joker.evolution.replay_truth import ReplayTruthLoader
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import CognitiveConfigurationVersion
from joker.evolution.session_event_index import (
    SessionEventIndexRecord,
    SessionEventIndexRepository,
)
from joker.evolution.shadow import ShadowRuntime
from joker.evolution.shadow_ledger import ShadowLedger
from joker.models.router import ModelRouter

logger = logging.getLogger(__name__)

_INDEX_EVENT_TYPES = frozenset(
    {
        EventType.MARKET_SNAPSHOT_CREATED,
        EventType.ORDER_SUBMITTED,
        EventType.ORDER_ACCEPTED,
        EventType.ORDER_PARTIALLY_FILLED,
        EventType.ORDER_FILLED,
        EventType.ORDER_CANCELLED,
        EventType.ORDER_REJECTED,
        EventType.POSITION_OPENED,
        EventType.POSITION_CHANGED,
        EventType.POSITION_CLOSED,
        EventType.COGNITIVE_CYCLE_STARTED,
        EventType.COGNITIVE_CYCLE_COMPLETED,
    }
)

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
    session_event_index: SessionEventIndexRepository | None = None
    event_horizon_loader: Task1EventHorizonLoader | None = None
    _episode_queue: asyncio.Queue[dict[str, Any]] | None = None
    _eval_queue: asyncio.Queue[UUID] | None = None
    _episode_in_flight: int = 0
    _eval_in_flight: int = 0
    _workers: list[asyncio.Task[None]] = field(default_factory=list)
    _index_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    _prepared: bool = False
    _workers_started: bool = False
    _subscribed: bool = False
    _pinned_cycle_configs: dict[str, UUID] = field(default_factory=dict)
    _applied_by_cycle: dict[str, AppliedConfiguration] = field(default_factory=dict)
    _cycle_snapshot: dict[str, UUID] = field(default_factory=dict)
    _contract_entry_snapshot: dict[str, UUID] = field(default_factory=dict)
    _cycle_exchange_timestamp: dict[str, datetime] = field(default_factory=dict)
    _contract_entry_timestamp: dict[str, datetime] = field(default_factory=dict)
    _position_origin_config: dict[str, UUID] = field(default_factory=dict)
    _latest_snapshot_id: UUID | None = None
    last_error: str | None = None
    _unsubscribe: list[Callable[[], None]] = field(default_factory=list)
    _orchestrator_wake: asyncio.Event | None = None

    @property
    def repositories(self) -> dict[str, Any]:
        """Public access to initialized Task-3 repositories (empty before prepare)."""
        return dict(self._repos)

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
        cycle_registry = None
        snapshot_repo = None
        if self.cognitive_graph_deps is not None:
            provenance = getattr(self.cognitive_graph_deps, "provenance_registry", None)
            cycle_registry = getattr(self.cognitive_graph_deps, "cycle_registry", None)
            snapshot_repo = getattr(self.cognitive_graph_deps, "snapshot_repo", None)
            if snapshot_repo is not None and hasattr(snapshot_repo, "inner"):
                snapshot_repo = snapshot_repo.inner
        self.session_event_index = SessionEventIndexRepository(str(self.db_path))
        await self.session_event_index.initialize()
        self.event_horizon_loader = Task1EventHorizonLoader(
            index_repo=self.session_event_index,
            snapshot_repo=snapshot_repo,
            data_quality_repo=(
                getattr(self.cognitive_graph_deps, "data_quality_repo", None)
                if self.cognitive_graph_deps is not None
                else None
            ),
            option_surface_repo=(
                getattr(self.cognitive_graph_deps, "option_surface_repo", None)
                if self.cognitive_graph_deps is not None
                else None
            ),
        )
        self.lifecycle_resolver = PositionLifecycleResolver(
            provenance=provenance,
            cycle_registry=cycle_registry,
            event_index=self.session_event_index,
        )
        strategy_repo = (
            getattr(self.cognitive_graph_deps, "strategy_repo", None)
            if self.cognitive_graph_deps is not None
            else None
        )
        world_model_repo = (
            getattr(self.cognitive_graph_deps, "world_model_repo", None)
            if self.cognitive_graph_deps is not None
            else None
        )
        self.episode_compiler = EpisodeCompiler(
            self._repos["episodes"],
            self._repos["traces"],
            lifecycle_resolver=self.lifecycle_resolver,
            provenance=provenance,
            cycle_registry=cycle_registry,
            event_horizon_loader=self.event_horizon_loader,
            strategy_repo=strategy_repo,
            world_model_repo=world_model_repo,
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

            if (
                self.model_router is not None
                and self.cognitive_graph_deps.model_call_repo is not None
            ):
                self.model_router.set_model_call_repo(
                    self.cognitive_graph_deps.model_call_repo
                )
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
                truth_loader=ReplayTruthLoader(
                    snapshot_repo=self.cognitive_graph_deps.snapshot_repo,
                    option_surface_repo=self.cognitive_graph_deps.option_surface_repo,
                    data_quality_repo=self.cognitive_graph_deps.data_quality_repo,
                    ledger_store=ledger_store,
                    session_starting_cash=session_cash,
                    allow_synthetic_starting_cash=session_cash is None,
                    event_horizon_loader=self.event_horizon_loader,
                ),
            )
        self.experiments = ExperimentRunner(
            self._repos["experiments"],
            db_path=self.db_path,
            repeated_samples=self.settings.experiments.repeated_samples,
            replay_service=self.replay,
            gate=PromotionEligibilityGate(self.settings.promotion),
        )
        self.decisions = EvolutionDecisionService(
            self._repos["promotions"],
            self._repos["configurations"],
            self.champion_registry,
            router=self.model_router,
            checkpointer_saver=savers.decision,
            session_id=self.session_id or "evolution",
            activation_repo=self._repos.get("activations"),
            gate=PromotionEligibilityGate(self.settings.promotion),
        )
        challenger_runner = None
        if self.replay is not None:
            challenger_runner = self.replay.run_challenger_shadow
        self.shadow_ledger = ShadowLedger(self.db_path)
        await self.shadow_ledger.initialize()
        self.evidence_claims = EvidenceClaimStore(self.db_path)
        await self.evidence_claims.initialize()
        self.adversarial_suite = AdversarialSuiteRunner(
            AdversarialResultStore(str(self.db_path)),
            template_deps=self.cognitive_graph_deps,
            policy_store=self.champion_registry.policy_store,
            config_repo=self._repos["configurations"],
            checkpointer_saver=savers.replay,
            replay_service=self.replay,
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
        self._orchestrator_wake = asyncio.Event()
        self._prepared = True
        self._started = True  # compatibility alias for prepare-complete

    def subscribe_events(self) -> None:
        if self.event_bus is None or self._subscribed or not self.settings.enabled:
            return
        for event_type in _INDEX_EVENT_TYPES:
            self.event_bus.subscribe(event_type, self._index_domain_event)
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

    async def _index_domain_event(self, event: DomainEvent) -> None:
        """Enqueue session_event_index persistence (fail-soft, non-blocking on bus)."""
        if self.session_event_index is None or getattr(self, "_quiesced", False):
            return
        self._spawn_background(
            self._persist_session_event_index(event),
            name=f"evolution-index-{event.event_id}",
        )

    async def _persist_session_event_index(self, event: DomainEvent) -> None:
        if self.session_event_index is None:
            return
        payload = dict(event.payload or {})
        try:
            record = SessionEventIndexRecord(
                event_id=str(event.event_id),
                session_id=event.session_id,
                event_type=str(event.event_type.value),
                exchange_timestamp=event.exchange_timestamp,
                sequence=getattr(event, "sequence", None),
                correlation_id=str(event.correlation_id),
                cycle_id=str(payload.get("cycle_id") or "") or None,
                snapshot_id=str(payload.get("snapshot_id") or "") or None,
                data_quality_id=str(payload.get("data_quality_id") or "") or None,
                option_surface_id=str(payload.get("option_surface_id") or "") or None,
                client_order_id=str(payload.get("client_order_id") or "") or None,
                contract_id=str(payload.get("contract_id") or "") or None,
                position_lifecycle_id=str(payload.get("position_lifecycle_id") or "")
                or None,
                payload_json=json.dumps(payload),
            )
            await self.session_event_index.record(record)
        except Exception:  # noqa: BLE001
            logger.warning(
                "session_event_index_record_failed",
                exc_info=True,
                extra={"event_id": str(event.event_id), "event_type": str(event.event_type)},
            )

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

    async def _cancel_worker_tasks(
        self, *, names: frozenset[str] | None = None
    ) -> None:
        """Cancel worker tasks.

        When ``names`` is set, only tasks whose asyncio name is in that set are
        cancelled (and removed from ``_workers``). Index tasks are always
        drained when cancelling the full worker set (``names is None``).
        """
        if names is None:
            pending_index = list(self._index_tasks)
            if pending_index:
                await asyncio.wait(pending_index, timeout=5.0)
                still = [t for t in pending_index if not t.done()]
                for task in still:
                    task.cancel()
                if still:
                    await asyncio.gather(*still, return_exceptions=True)
            self._index_tasks.clear()
            targets = list(self._workers)
        else:
            targets = [
                w
                for w in self._workers
                if (w.get_name() if hasattr(w, "get_name") else "") in names
            ]
        for worker in targets:
            worker.cancel()
        for worker in targets:
            try:
                await asyncio.wait_for(worker, timeout=5.0)
            except asyncio.CancelledError:
                continue
            except TimeoutError:
                logger.warning(
                    "evolution_worker_shutdown_timeout",
                    extra={"worker": repr(worker)},
                )
                continue
            except Exception:
                logger.exception(
                    "evolution_worker_shutdown_failed",
                    extra={"worker": repr(worker)},
                )
                raise
        if names is None:
            self._workers.clear()
            self._workers_started = False
        else:
            surviving = [w for w in self._workers if w not in targets]
            self._workers = surviving
            # Keep _workers_started True if any worker remains (e.g. orchestrator).
            self._workers_started = bool(surviving)

    def workers_idle(self) -> bool:
        """True when episode/eval queues are empty and no job is in-flight."""
        ep_q = self._episode_queue
        ev_q = self._eval_queue
        ep_idle = ep_q is None or ep_q.empty()
        ev_idle = ev_q is None or ev_q.empty()
        index_idle = not self._index_tasks
        return (
            ep_idle
            and ev_idle
            and index_idle
            and self._episode_in_flight == 0
            and self._eval_in_flight == 0
        )

    async def pause_workers(self) -> None:
        """Temporarily stop episode/eval workers without quiescing ingestion.

        Used by paper harnesses so cognitive entry is not raced by episode/eval
        workers; POSITION_CLOSED jobs can still enqueue for later drain.

        Does not pause the orchestrator or start/stop its worker — crash-recovery
        seeds that disable the orchestrator worker must remain disabled.

        Callers must wait until :meth:`workers_idle` before pausing; cancelling
        mid-job drops the dequeued item and loses evaluation/compile work.
        """
        await self._cancel_worker_tasks(
            names=frozenset({"evolution-episode", "evolution-eval"})
        )

    async def resume_workers(self) -> None:
        """Restart episode/eval workers after :meth:`pause_workers`."""
        self._quiesced = False
        if not self.settings.enabled or not self._prepared:
            return
        existing = {
            (w.get_name() if hasattr(w, "get_name") else "") for w in self._workers
        }
        if "evolution-episode" not in existing:
            self._workers.append(
                asyncio.create_task(self._episode_worker(), name="evolution-episode")
            )
        if "evolution-eval" not in existing:
            self._workers.append(
                asyncio.create_task(self._evaluation_worker(), name="evolution-eval")
            )
        self._workers_started = bool(self._workers)

    async def stop_workers(self) -> None:
        """Stop episode/eval/orchestrator workers and drain index tasks.

        Leaves repositories and checkpointers open for controlled graph invokes.
        """
        self._quiesced = True
        if self.orchestrator is not None:
            self.orchestrator.pause()
        await self._cancel_worker_tasks()

    async def shutdown(self) -> None:
        await self.stop_workers()
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
        if (
            not self._prepared
            or self._episode_queue is None
            or getattr(self, "_quiesced", False)
        ):
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
        if cycle_id and event.exchange_timestamp is not None:
            self._cycle_exchange_timestamp[cycle_id] = event.exchange_timestamp
        # New cycles pin champion unless already pinned (recovery path).
        # Offload DB work — bus handlers must stay under handler_timeout.
        if cycle_id and cycle_id not in self._applied_by_cycle:
            self._spawn_background(
                self.pin_and_apply_for_cycle(cycle_id),
                name=f"evolution-pin-{cycle_id}",
            )

    async def _on_position_opened(self, event: DomainEvent) -> None:
        # Capture immutable event fields then finish on a background task so the
        # bus handler cannot block snapshot fan-out under SQLite contention.
        self._spawn_background(
            self._handle_position_opened(event),
            name=f"evolution-position-opened-{event.event_id}",
        )

    async def _handle_position_opened(self, event: DomainEvent) -> None:
        contract_id = str(event.payload.get("contract_id") or "")
        cycle_id = str(event.payload.get("cycle_id") or "")
        client_order_id = str(event.payload.get("client_order_id") or "")
        # POSITION_OPENED from execution may omit cycle_id; resolve via provenance.
        if not cycle_id and client_order_id and self.cognitive_graph_deps is not None:
            provenance = getattr(
                self.cognitive_graph_deps, "provenance_registry", None
            )
            if provenance is not None:
                try:
                    record = await provenance.get_by_client_order_id(client_order_id)
                except Exception:
                    record = None
                if record is not None and getattr(record, "cycle_id", None):
                    cycle_id = str(record.cycle_id)
        if contract_id and self._latest_snapshot_id is not None:
            self._contract_entry_snapshot[contract_id] = self._latest_snapshot_id
        if contract_id and event.exchange_timestamp is not None:
            self._contract_entry_timestamp[contract_id] = event.exchange_timestamp
        pinned = self.get_pinned(cycle_id) if cycle_id else None
        if pinned is None and cycle_id:
            applied = await self.pin_and_apply_for_cycle(cycle_id)
            if applied is not None:
                pinned = applied.configuration_version_id
        if pinned is None:
            # No cycle pin (missing cycle_id / provenance): attribute to current champion.
            champ = await self.configuration_for_new_cycle()
            if champ is not None:
                pinned = champ.configuration_version_id
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
        self._spawn_background(
            self._handle_cognitive_cycle_completed(event),
            name=f"evolution-cycle-completed-{event.event_id}",
        )

    async def _handle_cognitive_cycle_completed(self, event: DomainEvent) -> None:
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
                    exchange_timestamp=event.exchange_timestamp,
                    event_sequence=getattr(event, "sequence", None),
                )
                self.wake_orchestrator(reason="shadow_snapshot_enqueued")

    def _spawn_background(self, coro: Awaitable[Any], *, name: str) -> None:
        """Schedule durable evolution work outside the event-bus timeout window."""
        task = asyncio.create_task(coro, name=name)
        self._index_tasks.add(task)
        task.add_done_callback(self._index_tasks.discard)

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
            self._episode_in_flight += 1
            try:
                episode = await self._compile_job(job)
                if episode is not None:
                    await self._eval_queue.put(episode.episode_id)
                    self.wake_orchestrator(reason="episode_compiled")
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.exception("evolution_episode_worker_failed")
            finally:
                self._episode_in_flight = max(0, self._episode_in_flight - 1)
                self._episode_queue.task_done()

    async def _flush_session_event_index(self, *, timeout: float = 5.0) -> None:
        """Wait for in-flight session_event_index writes before horizon load."""
        pending = []
        for task in list(self._index_tasks):
            if task.done():
                continue
            name = task.get_name() if hasattr(task, "get_name") else ""
            if str(name).startswith("evolution-index-"):
                pending.append(task)
        if not pending:
            return
        _done, still = await asyncio.wait(pending, timeout=timeout)
        if still:
            logger.warning(
                "session_event_index_flush_timeout",
                extra={"pending": len(still)},
            )

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
        # Horizon load must see the same events that triggered compilation.
        await self._flush_session_event_index()
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
            entry_decision_ts = (
                payload.get("entry_decision_timestamp")
                or self._cycle_exchange_timestamp.get(cycle_id)
                or self._contract_entry_timestamp.get(contract_id)
            )
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
                terminal_event_timestamp=job.get("exchange_timestamp")
                or payload.get("exchange_timestamp"),
                entry_decision_timestamp=entry_decision_ts,
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
            self._eval_in_flight += 1
            try:
                episode = await self._repos["episodes"].get_by_id(episode_id)
                if episode is not None:
                    await self.evaluation_runner.evaluate(episode)
                    self.wake_orchestrator(reason="evaluation_persisted")
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.exception("evolution_evaluation_worker_failed")
            finally:
                self._eval_in_flight = max(0, self._eval_in_flight - 1)
                self._eval_queue.task_done()

    def wake_orchestrator(self, *, reason: str = "progress") -> None:
        """Event-driven wake for the orchestrator worker (non-blocking)."""
        if self._orchestrator_wake is not None:
            self._orchestrator_wake.set()
            logger.debug("evolution_orchestrator_wake reason=%s", reason)

    async def wait_for_evolution_cycle_terminal(
        self,
        *,
        session_id: str | None = None,
        timeout: float = 120.0,
        poll_interval: float = 0.25,
    ) -> Any:
        """Poll persisted cycle status until a terminal outcome or timeout."""
        from joker.evolution.orchestrator import EvolutionCycleState

        sid = session_id or self.session_id
        deadline = asyncio.get_event_loop().time() + timeout
        terminal = {"completed", "failed", "blocked"}
        while asyncio.get_event_loop().time() < deadline:
            records = await self._repos["evolution_cycles"].list_by_session(sid)
            for record in records:
                if record.status in terminal:
                    return EvolutionCycleState.from_record(record)
            await asyncio.sleep(poll_interval)
        raise TimeoutError("wait_for_evolution_cycle_terminal timed out")

    async def _orchestrator_worker(self) -> None:
        assert self.orchestrator is not None
        # Bounded recovery interval (seconds). Primary trigger is _orchestrator_wake.
        recovery = max(
            1, int(self.settings.orchestrator.automatic_cycle_interval_minutes) * 60
        )
        if self.settings.orchestrator.automatic_cycle_interval_minutes <= 0:
            recovery = 2
        assert self._orchestrator_wake is not None
        # Kick once on startup for resumable cycles.
        self._orchestrator_wake.set()
        while True:
            try:
                try:
                    await asyncio.wait_for(
                        self._orchestrator_wake.wait(), timeout=recovery
                    )
                except asyncio.TimeoutError:
                    pass
                self._orchestrator_wake.clear()
                await self.orchestrator.tick()
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                logger.exception("evolution_orchestrator_tick_failed")


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
