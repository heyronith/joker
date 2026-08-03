"""Public factory for paper cognitive + Task-3 evolution sessions.

Used by LivePaperRunner and acceptance tests. Callers must not hand-build
CognitiveGraphDeps / OrderActionGateway / HistoricalOutcomeService.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from joker.broker.interface import BrokerClient, PaperBroker
from joker.cli.session_confirm import build_objective_engines
from joker.config.settings import AppSettings
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.runtime import EvolutionRuntime
from joker.graph.context_hydrate import context_assembler_from_settings
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.data_quality_store import DataQualityRepository
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.models.fake_provider import FakeModelProvider
from joker.objectives.service import SessionObjectiveService
from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
)
from joker.runtime.cognitive_agent_runtime import (
    CognitiveAgentRuntime,
    build_default_repositories,
)
from joker.runtime.cognitive_binding import bind_cognitive_graph_to_task1
from joker.runtime.cognitive_startup import validate_cognitive_providers
from joker.runtime.compatibility import CompatibilityLivePaperBridge
from joker.runtime.entry_permission import EntryPermissionState
from joker.runtime.objective_recovery import recover_session_objective


@dataclass
class PreparedTradingSession:
    """Public handles for a prepared Task-1/2/3 agentic trading session."""

    app_settings: AppSettings
    broker: BrokerClient
    bridge: CompatibilityLivePaperBridge
    evolution_runtime: EvolutionRuntime
    objective_service: SessionObjectiveService
    agent_runtime: CognitiveAgentRuntime
    session_id: str
    run_id: str
    db_path: Path
    broker_kind: str = "local_paper"
    safety_mode: str = "PAPER"

    @property
    def supervisor(self):
        return self.bridge.supervisor

    @property
    def graph_deps(self) -> CognitiveGraphDeps:
        deps = getattr(self.agent_runtime, "deps", None) or getattr(
            self.agent_runtime, "_deps", None
        )
        assert deps is not None
        return deps

    @property
    def historical_outcome_service(self):
        return self.graph_deps.historical_outcome_service

    @property
    def order_action_gateway(self):
        return self.graph_deps.order_action_gateway

    @property
    def episode_compiler(self):
        return self.evolution_runtime.episode_compiler

    async def shutdown(self) -> None:
        from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers

        try:
            await self.agent_runtime.shutdown()
        except Exception:
            pass
        try:
            await self.evolution_runtime.shutdown()
        except Exception:
            pass
        try:
            await self.bridge.supervisor.shutdown()
        except Exception:
            pass
        try:
            await drain_aiosqlite_workers(timeout=1.0)
        except Exception:
            pass
        loop = getattr(self.bridge, "_loop", None)
        if loop is not None and not loop.is_closed():
            loop.close()


# Backward-compatible alias — paper and live share the same session type.
PreparedCognitivePaperSession = PreparedTradingSession


async def prepare_agentic_trading_session(
    *,
    app_settings: AppSettings,
    objective_service: SessionObjectiveService,
    broker: BrokerClient,
    broker_kind: str,
    safety_mode: str,
    db_path: Path | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    fake_model_provider: FakeModelProvider | None = None,
    clock: Any | None = None,
    start_cognitive_agent: bool = True,
    start_evolution_workers: bool = True,
    broker_account_id: str = "local_paper",
    entry_permission: Any | None = None,
) -> PreparedTradingSession:
    """Shared session construction for paper and live — identical graph/EV/sizing."""
    if not bool(getattr(app_settings.evolution, "enabled", False)):
        raise ValueError("prepare_agentic_trading_session requires evolution.enabled")
    if not bool(getattr(app_settings.objective, "enabled", False)):
        raise ValueError("prepare_agentic_trading_session requires objective.enabled")

    paper = broker
    task1_db = Path(db_path or app_settings.db_path)
    task1_db.parent.mkdir(parents=True, exist_ok=True)
    sid = session_id or f"sess-{uuid4().hex[:12]}"
    rid = run_id or f"run-{uuid4().hex[:12]}"

    fake = fake_model_provider or FakeModelProvider(available=True)
    profiles = {
        name: profile.model_copy(update={"provider": "fake", "model": "fake-model"})
        for name, profile in default_model_profiles().items()
    }
    cfg = ModelsConfig(profiles=profiles)
    cfg = cfg.model_copy(
        update={
            "ollama": cfg.ollama.model_copy(update={"enabled": False}),
            "openai": cfg.openai.model_copy(update={"enabled": False}),
        }
    )
    registry = ModelRegistry(cfg, providers={"fake": fake})
    startup = await validate_cognitive_providers(
        cfg, mock_agents=True, registry=registry
    )
    registry = startup.registry
    # Keep the caller-supplied FakeModelProvider so canned role bindings survive
    # validate_cognitive_providers' remap (which otherwise constructs a new fake).
    registry.register_provider("fake", fake)
    router = ModelRouter(registry, session_id=sid)
    repos = build_default_repositories(task1_db)
    for repo in repos.values():
        await repo.initialize()
    router.set_model_call_repo(repos["model_call_repo"])

    evo_repos = build_evolution_repositories(task1_db)
    obj_repo = getattr(objective_service, "repository", None)
    engines = build_objective_engines(
        app_settings,
        episode_repository=evo_repos["episodes"],
        evaluation_repository=evo_repos["evaluations"],
        dataset_repository=evo_repos["datasets"],
        objective_repository=obj_repo,
    )

    async def _objective_state_loader():
        return await objective_service.get_state()

    max_quote_age = int(
        getattr(app_settings.data_quality, "option_stale_seconds", 30) or 30
    )
    max_spread = float(
        getattr(app_settings.data_quality, "maximum_relative_spread", 0.25) or 0.25
    )
    deps = CognitiveGraphDeps(
        router=router,
        config=app_settings.cognitive_graph,
        session_id=sid,
        run_id=rid,
        context_assembler=context_assembler_from_settings(app_settings.cognitive_graph),
        snapshot_repo=SnapshotRepository(task1_db),
        option_surface_repo=OptionSurfaceRepository(task1_db),
        data_quality_repo=DataQualityRepository(task1_db),
        db_path=task1_db,
        objective_service=objective_service,
        objective_state_loader=_objective_state_loader,
        feasibility_engine=engines.feasibility_engine,
        objective_strategy_scorer=engines.objective_strategy_scorer,
        capital_sizer=engines.capital_sizer,
        historical_outcome_service=engines.historical_outcome_service,
        historical_outcome_settings=engines.historical_outcome_settings,
        objective_execution_settings=getattr(app_settings.objective, "execution", None),
        kill_switch=bool(app_settings.risk.kill_switch),
        entry_permission=entry_permission or EntryPermissionState(),
        max_quote_age_seconds=max_quote_age,
        max_relative_spread=max_spread,
        configuration_repo=evo_repos.get("configurations"),
        **repos,
    )
    deps.require_objective_dependencies()

    agent_runtime = CognitiveAgentRuntime(
        session_id=sid,
        run_id=rid,
        router=router,
        config=app_settings.cognitive_graph,
        graph_deps=deps,
        registry=registry,
        checkpointer_path=task1_db.with_name(task1_db.stem + "_cognitive_ckpt.db"),
    )
    from joker.runtime.market_runtime import MarketRuntimeConfig

    option_stale = int(
        getattr(app_settings.data_quality, "option_stale_seconds", 30) or 30
    )
    underlying_stale = int(
        getattr(app_settings.data_quality, "underlying_stale_seconds", 30) or 30
    )
    bridge = CompatibilityLivePaperBridge(
        broker=paper,
        db_path=task1_db,
        session_id=sid,
        run_id=rid,
        clock=clock,
        broker_account_id=broker_account_id,
        agent_runtime=agent_runtime,
        market_config=MarketRuntimeConfig(
            min_option_contracts=1,
            underlying_stale_seconds=max(underlying_stale, 3600),
            option_stale_seconds=max(option_stale, 3600),
        ),
    )
    await bridge.astart(start_agent=False)

    provenance = CognitiveExecutionProvenanceRegistry(
        task1_db.with_name(task1_db.stem + "_cognitive_provenance.db")
    )
    await provenance.initialize()
    bind_cognitive_graph_to_task1(
        deps,
        bridge,
        data_quality_repo=bridge.supervisor.data_quality_repository,
        provenance_registry=provenance,
    )
    bridge.supervisor.bind_objective_service(objective_service)
    await recover_session_objective(
        objective_service,
        session_id=sid,
        execution_runtime=bridge.execution_runtime,
        unresolved_reconciliation=(
            bridge.supervisor.unresolved_reconciliation is not None
        ),
    )

    evolution_runtime = EvolutionRuntime(
        db_path=task1_db,
        settings=app_settings.evolution,
        session_id=sid,
        run_id=rid,
        event_bus=bridge.supervisor.event_bus,
        execution_runtime=bridge.execution_runtime,
        model_router=router,
        cognitive_graph_deps=deps,
    )
    await evolution_runtime.prepare()
    # Rebind historical service to exact EvolutionRuntime repositories.
    evo_owned = evolution_runtime.repositories
    rebound = build_objective_engines(
        app_settings,
        episode_repository=evo_owned.get("episodes"),
        evaluation_repository=evo_owned.get("evaluations"),
        dataset_repository=evo_owned.get("datasets"),
        objective_repository=obj_repo,
    )
    deps.historical_outcome_service = rebound.historical_outcome_service
    deps.historical_outcome_settings = rebound.historical_outcome_settings
    deps.feasibility_engine = rebound.feasibility_engine
    deps.objective_strategy_scorer = rebound.objective_strategy_scorer
    deps.capital_sizer = rebound.capital_sizer
    deps.evolution_runtime = evolution_runtime
    deps.configuration_repo = evo_owned.get("configurations")

    evolution_runtime.subscribe_events()
    agent_runtime.bind_evolution_runtime(evolution_runtime)
    if start_cognitive_agent:
        await bridge.astart_agent()
    if start_evolution_workers:
        await evolution_runtime.start_workers()
        await evolution_runtime.resume()

    return PreparedTradingSession(
        app_settings=app_settings,
        broker=paper,
        bridge=bridge,
        evolution_runtime=evolution_runtime,
        objective_service=objective_service,
        agent_runtime=agent_runtime,
        session_id=sid,
        run_id=rid,
        db_path=task1_db,
        broker_kind=broker_kind,
        safety_mode=safety_mode,
    )


async def prepare_cognitive_paper_session(
    *,
    app_settings: AppSettings,
    objective_service: SessionObjectiveService,
    broker: BrokerClient | None = None,
    db_path: Path | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    fake_model_provider: FakeModelProvider | None = None,
    clock: Any | None = None,
    start_cognitive_agent: bool = True,
    start_evolution_workers: bool = True,
    entry_permission: Any | None = None,
) -> PreparedTradingSession:
    """Paper wrapper around the shared agentic session factory."""
    from joker.broker.webull_live import WebullLiveClient

    paper = broker or PaperBroker(slippage_pct=0)
    if isinstance(paper, WebullLiveClient):
        raise ValueError("prepare_cognitive_paper_session rejects webull_live broker")
    kind = "webull_paper" if paper.__class__.__name__ == "WebullClient" else "local_paper"
    return await prepare_agentic_trading_session(
        app_settings=app_settings,
        objective_service=objective_service,
        broker=paper,
        broker_kind=kind,
        safety_mode="PAPER",
        db_path=db_path,
        session_id=session_id,
        run_id=run_id,
        fake_model_provider=fake_model_provider,
        clock=clock,
        start_cognitive_agent=start_cognitive_agent,
        start_evolution_workers=start_evolution_workers,
        entry_permission=entry_permission,
        broker_account_id="local_paper" if kind == "local_paper" else "webull_paper",
    )


async def prepare_cognitive_live_session(
    *,
    app_settings: AppSettings,
    objective_service: SessionObjectiveService,
    broker: BrokerClient,
    db_path: Path | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    fake_model_provider: FakeModelProvider | None = None,
    clock: Any | None = None,
    start_cognitive_agent: bool = True,
    start_evolution_workers: bool = True,
    entry_permission: Any | None = None,
) -> PreparedTradingSession:
    """Live wrapper — same graph/EV/sizing as paper; live broker only."""
    from joker.app.safety import SafetyMode
    from joker.broker.webull_live import WebullLiveClient

    if app_settings.mode is not SafetyMode.LIVE_GATED:
        raise ValueError("prepare_cognitive_live_session requires mode LIVE_GATED")
    if not app_settings.live_trading_enabled:
        raise ValueError(
            "prepare_cognitive_live_session requires live_trading_enabled=true"
        )
    if not isinstance(broker, WebullLiveClient):
        raise ValueError(
            "prepare_cognitive_live_session requires WebullLiveClient "
            "(refusing paper broker)"
        )
    return await prepare_agentic_trading_session(
        app_settings=app_settings,
        objective_service=objective_service,
        broker=broker,
        broker_kind="webull_live",
        safety_mode="LIVE_GATED",
        db_path=db_path,
        session_id=session_id,
        run_id=run_id,
        fake_model_provider=fake_model_provider,
        clock=clock,
        start_cognitive_agent=start_cognitive_agent,
        start_evolution_workers=start_evolution_workers,
        entry_permission=entry_permission,
        broker_account_id=broker.account_id_hash,
    )
