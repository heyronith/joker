"""Async Task 3 evolution runtime — lower priority than Task 1/2 trading path."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.config import EvolutionSettings
from joker.evolution.decision import EvolutionDecisionService
from joker.evolution.drift import DriftMonitor
from joker.evolution.episode_compiler import EpisodeCompiler
from joker.evolution.experiment_runner import ExperimentRunner
from joker.evolution.improvement import ImprovementProposalService
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.shadow import ShadowRuntime
from joker.evaluation.dataset_builder import DatasetBuilder
from joker.evaluation.graph import EvaluationGraphRunner


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
    """Owns Task 3 workers without blocking Task 1/2."""

    db_path: Path
    settings: EvolutionSettings
    scope_key: str = "default"
    _repos: dict[str, Any] = field(default_factory=dict)
    champion_registry: ChampionRegistry | None = None
    episode_compiler: EpisodeCompiler | None = None
    evaluation_runner: EvaluationGraphRunner | None = None
    dataset_builder: DatasetBuilder | None = None
    improvement: ImprovementProposalService | None = None
    experiments: ExperimentRunner | None = None
    decisions: EvolutionDecisionService | None = None
    shadow: ShadowRuntime | None = None
    drift: DriftMonitor | None = None
    _episode_queue: asyncio.Queue[dict[str, Any]] | None = None
    _eval_queue: asyncio.Queue[UUID] | None = None
    _workers: list[asyncio.Task[None]] = field(default_factory=list)
    _started: bool = False
    _pinned_cycle_configs: dict[str, UUID] = field(default_factory=dict)
    last_error: str | None = None

    async def start(self) -> None:
        if not self.settings.enabled:
            return
        self._repos = build_evolution_repositories(self.db_path)
        for repo in self._repos.values():
            await repo.initialize()
        self.champion_registry = ChampionRegistry(self.db_path, scope_key=self.scope_key)
        await self.champion_registry.bootstrap_champion()
        self.episode_compiler = EpisodeCompiler(
            self._repos["episodes"], self._repos["traces"]
        )
        self.evaluation_runner = EvaluationGraphRunner(
            self._repos["evaluations"],
            self._repos["traces"],
            evaluator_version=self.settings.evaluation.evaluator_version,
        )
        self.dataset_builder = DatasetBuilder(self._repos["datasets"])
        self.improvement = ImprovementProposalService(
            self._repos["proposals"], self._repos["configurations"]
        )
        self.experiments = ExperimentRunner(self._repos["experiments"])
        self.decisions = EvolutionDecisionService(
            self._repos["promotions"],
            self._repos["configurations"],
            self.champion_registry,
        )
        self.shadow = ShadowRuntime(self._repos["shadow"])
        if self.settings.shadow.enabled:
            await self.shadow.start()
        self.drift = DriftMonitor(
            self._repos["drift"],
            self._repos["rollbacks"],
            self.champion_registry,
            safety_rollback_immediate=self.settings.drift.safety_rollback_immediate,
            strategic_requires_agent=self.settings.drift.strategic_rollback_requires_agent,
        )
        self._episode_queue = asyncio.Queue(maxsize=256)
        self._eval_queue = asyncio.Queue(maxsize=256)
        self._workers = [
            asyncio.create_task(self._episode_worker(), name="evolution-episode"),
            asyncio.create_task(self._evaluation_worker(), name="evolution-eval"),
        ]
        self._started = True

    async def shutdown(self) -> None:
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            try:
                await worker
            except asyncio.CancelledError:
                pass
        self._workers.clear()
        if self.shadow is not None:
            await self.shadow.stop()
        if self.champion_registry is not None:
            await self.champion_registry.close()
        for repo in self._repos.values():
            await repo.close()
        self._started = False

    def pin_cycle_configuration(self, cycle_id: str, configuration_version_id: UUID) -> None:
        self._pinned_cycle_configs[cycle_id] = configuration_version_id

    def configuration_for_new_cycle(self) -> UUID | None:
        """New cycles use current champion; active cycles keep pinned config."""
        return None  # caller should fetch champion; pinning is explicit

    def get_pinned(self, cycle_id: str) -> UUID | None:
        return self._pinned_cycle_configs.get(cycle_id)

    async def current_champion_id(self) -> UUID | None:
        if self.champion_registry is None:
            return None
        champ = await self.champion_registry.get_current_champion()
        return None if champ is None else champ.configuration_version_id

    async def enqueue_episode_job(self, job: dict[str, Any]) -> bool:
        if not self._started or self._episode_queue is None:
            return False
        if self._episode_queue.full():
            return False
        await self._episode_queue.put(job)
        return True

    async def health(self) -> EvolutionRuntimeHealth:
        champ_id = await self.current_champion_id()
        return EvolutionRuntimeHealth(
            enabled=self.settings.enabled and self._started,
            champion_version_id=str(champ_id) if champ_id else None,
            episode_queue_depth=(
                self._episode_queue.qsize() if self._episode_queue else 0
            ),
            evaluation_queue_depth=self._eval_queue.qsize() if self._eval_queue else 0,
            shadow_backlog=self.shadow.backlog if self.shadow else 0,
            degraded=self.last_error is not None,
            last_error=self.last_error,
        )

    async def _episode_worker(self) -> None:
        assert self._episode_queue is not None
        assert self.episode_compiler is not None
        assert self._eval_queue is not None
        while True:
            job = await self._episode_queue.get()
            try:
                kind = job.get("kind")
                if kind == "closed_trade":
                    episode = await self.episode_compiler.compile_closed_trade(**job["kwargs"])
                elif kind == "no_trade":
                    episode = await self.episode_compiler.compile_no_trade(**job["kwargs"])
                else:
                    continue
                await self._eval_queue.put(episode.episode_id)
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
            finally:
                self._episode_queue.task_done()

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
            finally:
                self._eval_queue.task_done()


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
