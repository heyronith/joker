"""Live paper session — real Webull market data + auto paper execution.

Compatibility façade (Task 1):
- Prefer MarketRuntime for observation/bar/snapshot truth.
- Prefer ExecutionRuntime + ledger for order/fill/position accounting.
- This module remains the CLI entry for `joker paper run` and gradually delegates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
import time as wall_time
from typing import Any, Callable

from joker.agents.council import create_agent_council
from joker.agents.council_analysis import CouncilAnalysis, analyze_council
from joker.agents.llm_client import LLMClientError
from joker.agents.openai_agents import AgentError
from joker.agents.session_memory import SessionMicroMemory
from joker.app.safety import SafetyMode
from joker.broker.factory import BrokerFactoryError, resolve_live_paper_broker
from joker.broker.interface import BrokerClient
from joker.config.settings import AppSettings, EnvSettings
from joker.data.provider_factory import ProviderKind
from joker.data.webull_capability import capability_usable_for_shadow
from joker.runtime.live_market_data_loop import LiveMarketDataError, LiveMarketDataLoop
from joker.runtime.portfolio_recovery import PortfolioRecoveryCoordinator
from joker.runtime.recovery_mode import RecoveryMode, is_recovery_only_mode, recovery_mode_value
from joker.execution.exit_manager import ExitManager
from joker.execution.option_selector import OptionSelector, OptionSelectorConfig
from joker.features.engine import FeatureEngine
from joker.logging.event_log import EventLogWriter
from joker.reporting.metrics import StrategyQualityMetrics, compute_quality_metrics
from joker.reporting.replay_report import ReplayReportContext, ReplayReportGenerator
from joker.risk.capital import CapitalBudget, CapitalPlan
from joker.risk.governor import RiskGovernor
from joker.runtime.live_cli import should_stream_event
from joker.runtime.market_handler import MarketEventHandler
from joker.runtime.compatibility import (
    CompatibilityLivePaperBridge,
    ExecutionDelegatingBroker,
)
from joker.runtime.premarket import PremarketWorkflow
from joker.runtime.reactive_engine import ReactiveEngine, StateMachineError
from joker.runtime.run_manager import RunManager
from joker.schemas.domain import DailyState, Playbook, RiskConfig
from joker.schemas.replay import ReplaySummary
from joker.storage.database import Database, ensure_database
from joker.storage.models import (
    AgentDecisionRecord,
    RiskDecisionRecord,
    SystemEventRecord,
    TradeCandidateRecord,
)
from joker.strategy.playbook_quality import PlaybookQualityValidator, PlaybookValidationResult, trim_playbook_enabled_setups

class LivePaperError(RuntimeError):
    """Fail-closed error for live paper session setup."""


@dataclass
class LivePaperRunConfig:
    symbol: str = "SPY"
    duration_seconds: float = 1800.0
    mock_agents: bool = False
    llm_client: Any | None = None
    webull_api: Any | None = None
    webull_options_api: Any | None = None
    # Test-only: inject a pre-approved playbook (never used for production CLI).
    approved_playbook: Playbook | None = None
    require_options: bool = True
    # Test/DI: inject broker or trade API for Webull paper path.
    broker: BrokerClient | None = None
    trade_api: Any | None = None
    # Session capital budget (required for agent sizing); tests may inject
    capital_budget: CapitalBudget | None = None
    # Task-1 durable objective service (required when objective.enabled)
    objective_service: Any | None = None
    cognitive_session_id_override: str | None = None
    recovery_mode: RecoveryMode | str = RecoveryMode.NORMAL
    # Exchange-aware objective deadline (blocks new entries via objective service).
    objective_deadline_exchange: datetime | None = None
    reconciliation_only_recovery: bool = False
    # Extra wall-clock seconds after duration to finish agent-managed exits only.
    # Default 0 so existing short paper tests are not extended; CLI goal-test sets 120.
    shutdown_grace_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.recovery_mode = RecoveryMode(str(self.recovery_mode).strip().lower())
        if self.reconciliation_only_recovery:
            if self.recovery_mode is RecoveryMode.NORMAL:
                self.recovery_mode = RecoveryMode.RECONCILIATION_ONLY
        self.reconciliation_only_recovery = self.recovery_mode in {
            RecoveryMode.RECONCILIATION_ONLY,
            RecoveryMode.BROKER_ONLY,
        }


@dataclass
class LivePaperRunResult:
    run_id: str
    summary: ReplaySummary | None = None
    report_path: Path | None = None
    failures: list[str] = field(default_factory=list)
    playbook: Playbook | None = None
    council_analysis: CouncilAnalysis | None = None
    playbook_validation: PlaybookValidationResult | None = None
    events_processed: int = 0
    feed_health: str = "OK"
    options_available: bool = False
    paper_pnl_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    broker_kind: str = "local_paper"
    broker_label: str = "local PaperBroker"
    open_positions_remaining: int = 0
    working_orders_remaining: int = 0
    reconciliation_clean: bool | None = None
    objective_deadline_reached: bool = False


def _risk_config_from_settings(
    settings: AppSettings, capital: CapitalBudget | None = None
) -> RiskConfig:
    r = settings.risk
    policy = (r.policy or "strict").strip().lower()
    if policy not in {"strict", "agent_led"}:
        policy = "strict"
    max_open = r.max_open_positions
    authorized = 0.0
    reserved = 0.0
    if capital is not None:
        max_open = capital.plan.max_concurrent_positions
        authorized = capital.authorized_usd
        reserved = capital.reserved_usd
    return RiskConfig(
        max_daily_loss_usd=r.max_daily_loss_usd,
        max_trades_per_day=r.max_trades_per_day,
        max_open_positions=max_open,
        max_premium_usd=r.max_premium_usd,
        max_spread_pct=r.max_spread_pct,
        quote_max_age_seconds=r.quote_max_age_seconds,
        allowed_symbol=r.allowed_symbol,
        kill_switch=r.kill_switch,
        allow_delayed_quotes=r.allow_delayed_quotes,
        feed_max_silence_seconds=r.feed_max_silence_seconds,
        delayed_quote_max_age_seconds=r.delayed_quote_max_age_seconds,
        policy=policy,  # type: ignore[arg-type]
        authorized_capital_usd=authorized,
        reserved_capital_usd=reserved,
    )


def _default_capital_budget(settings: AppSettings) -> CapitalBudget:
    c = settings.capital
    return CapitalBudget(
        plan=CapitalPlan(
            authorized_usd=float(c.authorized_usd),
            target_profit_pct=float(c.target_profit_pct),
            max_concurrent_positions=int(c.max_concurrent_positions),
            max_contracts_per_trade=int(c.max_contracts_per_trade),
            min_contracts_per_trade=int(c.min_contracts_per_trade),
            aggression_mode=str(c.aggression_mode),
            max_kelly_fraction=float(c.max_kelly_fraction),
            min_win_probability=float(c.min_win_probability),
            behind_goal_boost=float(c.behind_goal_boost),
            ahead_goal_dampen=float(c.ahead_goal_dampen),
        )
    )


class LivePaperRunner:
    """
    Real Webull SPY + 0DTE options → agents → risk → auto paper orders.

    Execution is either local PaperBroker or Webull paper account.
    Real-money live orders are never submitted.
    """

    def __init__(
        self,
        app_settings: AppSettings,
        env_settings: EnvSettings,
        db: Database | None = None,
        event_log: EventLogWriter | None = None,
    ) -> None:
        self.app_settings = app_settings
        self.env_settings = env_settings
        self.db = db or ensure_database(app_settings.db_path)
        self.event_log = event_log or EventLogWriter(
            app_settings.event_log_dir,
            redact_keys=app_settings.logging.redact_env_keys,
        )
        self._task1_bridge: CompatibilityLivePaperBridge | None = None
        self._evolution_runtime = None

    @property
    def task1_bridge(self) -> CompatibilityLivePaperBridge | None:
        """Active Task 1 SessionSupervisor bridge when a paper session is running."""
        return self._task1_bridge

    def _log(
        self,
        run_id: str,
        event_type: str,
        payload: dict,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.event_log.append(
            run_id=run_id,
            mode=SafetyMode.PAPER.value,
            source="live_paper",
            event_type=event_type,
            payload=payload,
        )
        if on_event is not None and should_stream_event(event_type):
            on_event(event_type, payload)

    def _assert_safe_mode(self, *, require_market_data: bool = True) -> None:
        if self.app_settings.live_trading_enabled:
            raise LivePaperError(
                "live_trading_enabled must be false for paper sessions"
            )
        if self.env_settings.webull_live_trading_enabled:
            raise LivePaperError("WEBULL_LIVE_TRADING_ENABLED must remain false")
        if self.app_settings.mode is SafetyMode.LIVE_GATED:
            raise LivePaperError(
                "Live paper requires mode PAPER (not LIVE_GATED). "
                "Use config/paper.yaml."
            )
        if require_market_data and not self.env_settings.webull_market_data_enabled:
            raise LivePaperError(
                "WEBULL_MARKET_DATA_ENABLED must be true for live paper"
            )

    @staticmethod
    def _working_client_order_ids(projection: Any | None) -> list[str]:
        orders = getattr(projection, "orders", None) or {}
        values = orders.values() if isinstance(orders, dict) else list(orders or [])
        ids: list[str] = []
        for order in values:
            status = str(getattr(order, "status", "") or "").lower()
            if status not in {
                "submitted",
                "accepted",
                "partially_filled",
                "open",
                "pending",
                "working",
            }:
                continue
            client_order_id = str(getattr(order, "client_order_id", "") or "")
            if client_order_id:
                ids.append(client_order_id)
        return list(dict.fromkeys(ids))

    def _run_reconciliation_only_recovery(
        self,
        *,
        run_id: str,
        selection: Any,
        config: LivePaperRunConfig,
        result: LivePaperRunResult,
        task1_bridge: CompatibilityLivePaperBridge,
        capital_budget: CapitalBudget,
        failures: list[str],
        log: Callable[[str, dict[str, Any]], None],
        on_state: Callable[[dict[str, Any]], None] | None,
        run_manager: RunManager,
        shutdown_task1: Callable[[], None],
        recovery_mode: RecoveryMode,
    ) -> LivePaperRunResult:
        """Broker-only recovery path: poll broker/order truth without market warmup."""
        from joker.persistence.cognitive_execution_provenance import (
            CognitiveExecutionProvenanceRegistry,
            PortfolioExecutionOwner,
        )

        task1_db = Path(self.app_settings.db_path).parent / "joker_task1.db"
        provenance = CognitiveExecutionProvenanceRegistry(
            task1_db.with_name(task1_db.stem + "_cognitive_provenance.db")
        )
        task1_bridge.run_coro(provenance.initialize())
        stable_trading_date = task1_bridge.supervisor.clock.trading_date().isoformat()
        owner = PortfolioExecutionOwner(
            session_id=task1_bridge.session_id,
            broker_account_identity=task1_bridge.execution_runtime.broker_account_identity,
            trading_date=stable_trading_date,
        )
        coordinator = PortfolioRecoveryCoordinator(
            execution_runtime=task1_bridge.execution_runtime,
            provenance_registry=provenance,
            stable_owner=owner,
            clock=task1_bridge.supervisor.clock,
            objective_service=config.objective_service,
            recovery_mode=recovery_mode,
        )
        log(
            "live_paper.started",
            {
                "symbol": config.symbol,
                "duration_seconds": config.duration_seconds,
                "mock_agents": config.mock_agents,
                "broker": selection.kind,
                "broker_label": selection.label,
                "auto_orders": selection.auto_orders,
                "live_money_orders": False,
                "is_synthetic": False,
                "capital": capital_budget.prompt_dict(),
                "task1_session_supervisor": True,
                "task1_session_id": task1_bridge.session_id,
                "reconciliation_only_recovery": True,
                "broker_only_recovery": True,
            },
        )
        log(
            "objective.reconciliation_only_started",
            {
                "new_entries_blocked": True,
                "runtime_seconds": float(config.duration_seconds),
                "original_objective_deadline": (
                    config.objective_deadline_exchange.isoformat()
                    if config.objective_deadline_exchange is not None
                    else None
                ),
                "market_warmup_skipped": True,
                "option_surface_optional": True,
            },
        )
        poll = max(0.5, self.app_settings.data.quote_poll_interval_seconds)
        deadline = wall_time.monotonic() + max(float(config.duration_seconds), poll)
        errors: list[str] = []
        projection = None
        def _poll_working_orders(current_projection: Any | None) -> None:
            for client_order_id in coordinator.working_client_order_ids(current_projection):
                task1_bridge.poll_order_status(client_order_id)

        while wall_time.monotonic() < deadline:
            try:
                projection = task1_bridge.project_session()
                _poll_working_orders(projection)
                latest_snapshot_id = None
                snapshot_repo = task1_bridge.supervisor.snapshot_repository
                if snapshot_repo is not None:
                    latest_snapshot = task1_bridge.run_coro(
                        snapshot_repo.get_latest(task1_bridge.session_id)
                    )
                    if latest_snapshot is not None:
                        latest_snapshot_id = str(latest_snapshot.snapshot_id)
                objective_status = None
                if config.objective_service is not None:
                    objective_state = task1_bridge.run_coro(config.objective_service.get_state())
                    objective_status = str(getattr(objective_state, "status", "unknown") or "unknown")
                task1_bridge.run_coro(
                    coordinator.reconcile_owner_components(
                        projection=projection,
                        latest_snapshot_id=latest_snapshot_id,
                        terminal_recovery_reason=(
                            "reconciliation_only_resume_no_new_entries"
                            if recovery_mode is RecoveryMode.RECONCILIATION_ONLY
                            else None
                        ),
                        objective_status=objective_status,
                        origin_run_id=run_id,
                        state={
                            "run_id": run_id,
                            "cycle_id": None,
                            "snapshot_id": latest_snapshot_id,
                        },
                    )
                )
                if on_state is not None:
                    orders = getattr(projection, "orders", None) or {}
                    positions = getattr(projection, "positions", None) or {}
                    on_state(
                        {
                            "run_id": run_id,
                            "provider": selection.kind,
                            "market_price": None,
                            "feed_health": "RECOVERY_ONLY",
                            "delayed": None,
                            "options_available": False,
                            "signals": 0,
                            "trades_entered": 0,
                            "trades_exited": 0,
                            "paper_pnl": selection.client.get_daily_pnl(),
                            "engine_state": "recovery_only",
                            "intraday_calls": 0,
                            "decision_calls": 0,
                            "proposals_acted": 0,
                            "execution_mode": "recovery_only",
                            "capital_available": capital_budget.available_usd,
                            "capital_goal_pct": capital_budget.progress_to_goal_pct,
                            "broker": selection.kind,
                            "broker_label": selection.label,
                            "pending_order": any(
                                str(getattr(order, "status", "") or "").lower()
                                in {"submitted", "accepted", "partially_filled", "open", "working"}
                                for order in (orders.values() if isinstance(orders, dict) else orders)
                            ),
                            "open_trade": bool(positions),
                            "auto_orders": selection.auto_orders,
                            "live_money_orders": False,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                log("broker_only_recovery_failed", {"reason": str(exc)})
            wall_time.sleep(min(poll, max(0.0, deadline - wall_time.monotonic())))
        report = task1_bridge.run_coro(
            task1_bridge.supervisor.execution_runtime.run_reconciliation()
        )
        projection = task1_bridge.project_session()
        latest_snapshot_id = None
        snapshot_repo = task1_bridge.supervisor.snapshot_repository
        if snapshot_repo is not None:
            latest_snapshot = task1_bridge.run_coro(
                snapshot_repo.get_latest(task1_bridge.session_id)
            )
            if latest_snapshot is not None:
                latest_snapshot_id = str(latest_snapshot.snapshot_id)
        objective_status = None
        if config.objective_service is not None:
            objective_state = task1_bridge.run_coro(config.objective_service.get_state())
            objective_status = str(getattr(objective_state, "status", "unknown") or "unknown")
        task1_bridge.run_coro(
            coordinator.reconcile_owner_components(
                projection=projection,
                latest_snapshot_id=latest_snapshot_id,
                terminal_recovery_reason=(
                    "reconciliation_only_resume_no_new_entries"
                    if recovery_mode is RecoveryMode.RECONCILIATION_ONLY
                    else None
                ),
                objective_status=objective_status,
                origin_run_id=run_id,
                state={
                    "run_id": run_id,
                    "cycle_id": None,
                    "snapshot_id": latest_snapshot_id,
                },
            )
        )
        if recovery_mode is RecoveryMode.BROKER_ONLY:
            unresolved_components = task1_bridge.run_coro(
                provenance.portfolio_executions.has_unresolved(
                    session_id=task1_bridge.session_id,
                    broker_account_identity=task1_bridge.execution_runtime.broker_account_identity,
                    trading_date=stable_trading_date,
                )
            )
            unresolved_requests = task1_bridge.run_coro(
                provenance.portfolio_reoptimizations.has_unresolved(
                    session_id=task1_bridge.session_id,
                    broker_account_identity=task1_bridge.execution_runtime.broker_account_identity,
                    trading_date=stable_trading_date,
                )
            )
            if unresolved_components or unresolved_requests:
                log(
                    "broker_only.operator_resolution_required",
                    {
                        "session_id": task1_bridge.session_id,
                        "broker_account_identity": task1_bridge.execution_runtime.broker_account_identity,
                        "trading_date": stable_trading_date,
                        "unresolved_components": unresolved_components,
                        "unresolved_requests": unresolved_requests,
                    },
                )
        positions = getattr(projection, "positions", None) or {}
        orders = getattr(projection, "orders", None) or {}
        result.feed_health = "RECOVERY_ONLY"
        result.options_available = False
        result.errors.extend(errors)
        result.failures.extend(failures)
        result.working_orders_remaining = len(self._working_client_order_ids(projection))
        result.open_positions_remaining = len(positions)
        result.reconciliation_clean = bool(getattr(report, "is_consistent", False))
        from joker.schemas.replay import ReplaySummary

        result.summary = ReplaySummary(
            run_id=run_id,
            session_name="live_paper",
            is_synthetic=False,
            mock_agents=config.mock_agents,
            events_processed=0,
            signals_detected=0,
            trades_entered=0,
            trades_exited=0,
            final_pnl_usd=selection.client.get_daily_pnl(),
            risk_rejections=0,
            failures=list(result.failures) + list(result.errors),
        )
        run_manager.end_run(run_id)
        shutdown_task1()
        return result

    def run(
        self,
        config: LivePaperRunConfig,
        *,
        on_state: Callable[[dict[str, Any]], None] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> LivePaperRunResult:
        if config.symbol.upper() != "SPY":
            raise LivePaperError("Only SPY is supported")

        recovery_mode = recovery_mode_value(config)
        recovery_only_mode = is_recovery_only_mode(config)
        self._assert_safe_mode(require_market_data=not recovery_only_mode)

        # Force PAPER mode for this session regardless of display toggles.
        mode = SafetyMode.PAPER
        trading_day = date.today()
        run_manager = RunManager(self.db, self.event_log, self.app_settings)
        run_id = run_manager.start_run(trading_day=trading_day)
        result = LivePaperRunResult(run_id=run_id)
        failures: list[str] = []

        try:
            selection = resolve_live_paper_broker(
                self.app_settings,
                self.env_settings,
                trade_api=config.trade_api,
                broker=config.broker,
            )
        except BrokerFactoryError as exc:
            raise LivePaperError(str(exc)) from exc

        broker = selection.client
        from joker.runtime.cognitive_session import paper_account_identity

        account_identity = paper_account_identity(
            broker_kind=selection.kind,
            env=self.env_settings,
        )
        result.broker_kind = selection.kind
        result.broker_label = selection.label
        if (self.app_settings.broker.provider or "").strip().lower() in {
            "webull_paper",
            "webull",
        }:
            from joker.broker.interface import PaperBroker

            if selection.kind != "webull_paper" or isinstance(broker, PaperBroker):
                raise LivePaperError(
                    "broker.provider=webull_paper resolved to a non-Webull broker; "
                    "refusing PaperBroker fallback"
                )

        # Task 1 cutover: SessionSupervisor owns market/execution truth.
        task1_db = Path(self.app_settings.db_path).parent / "joker_task1.db"
        agent_runtime_mode = (self.app_settings.agents.runtime or "legacy").strip().lower()
        cognitive_mode = agent_runtime_mode == "cognitive_graph"
        null_agent_mode = agent_runtime_mode == "null"

        injected_agent_runtime = None
        cognitive_graph_deps = None
        _cognitive_startup_payload: dict[str, Any] | None = None
        objective_service = config.objective_service
        bridge_session_id = config.cognitive_session_id_override or run_id
        if recovery_mode is RecoveryMode.NORMAL and cognitive_mode:
            import asyncio as _asyncio

            from joker.cognition.exceptions import CognitiveRuntimeConfigurationError
            from joker.graph.context_hydrate import context_assembler_from_settings
            from joker.graph.graph_deps import CognitiveGraphDeps
            from joker.market.data_quality_store import DataQualityRepository
            from joker.market.option_surface import OptionSurfaceRepository
            from joker.market.snapshots import SnapshotRepository
            from joker.models.router import ModelRouter
            from joker.runtime.cognitive_agent_runtime import (
                CognitiveAgentRuntime,
                build_default_repositories,
            )
            from joker.runtime.cognitive_startup import validate_cognitive_providers

            try:
                startup = _asyncio.run(
                    validate_cognitive_providers(
                        self.app_settings.models,
                        mock_agents=bool(
                            self.app_settings.agents.mock_agents or config.mock_agents
                        ),
                    )
                )
            except CognitiveRuntimeConfigurationError as exc:
                raise LivePaperError(f"cognitive-runtime configuration error: {exc}") from exc

            registry = startup.registry
            # Stable cognitive session survives process restart; run_id remains audit-only.
            from joker.runtime.cognitive_session import live_paper_cognitive_session_id

            cognitive_session_id = (
                config.cognitive_session_id_override
                or live_paper_cognitive_session_id(
                    broker_kind=selection.kind,
                    env=self.env_settings,
                )
            )
            bridge_session_id = cognitive_session_id
            model_router = ModelRouter(
                registry,
                session_id=cognitive_session_id,
                model_call_repo=None,  # wired after repos below
            )
            repos = build_default_repositories(task1_db)
            model_router.set_model_call_repo(repos["model_call_repo"])
            obj_settings = getattr(self.app_settings, "objective", None)
            if (
                obj_settings is not None
                and bool(getattr(obj_settings, "enabled", False))
                and objective_service is None
            ):
                raise LivePaperError(
                    "objective.enabled requires a confirmed SessionObjectiveService "
                    "before starting the cognitive graph"
                )
            objective_state_loader = None
            objective_engine_kwargs: dict[str, Any] = {}
            if objective_service is not None:
                from joker.cli.session_confirm import build_objective_engines
                from joker.evolution.repositories import build_evolution_repositories

                # When evolution is enabled, Task-3 repos live beside Task-1 DB.
                # Pass those repositories explicitly — never guess settings paths.
                episode_repo = None
                evaluation_repo = None
                dataset_repo = None
                if bool(getattr(self.app_settings.evolution, "enabled", False)):
                    evo_repos = build_evolution_repositories(task1_db)
                    episode_repo = evo_repos["episodes"]
                    evaluation_repo = evo_repos["evaluations"]
                    dataset_repo = evo_repos["datasets"]
                obj_repo = getattr(objective_service, "repository", None)
                if obj_repo is None:
                    obj_repo = getattr(objective_service, "_repo", None)
                engines = build_objective_engines(
                    self.app_settings,
                    episode_repository=episode_repo,
                    evaluation_repository=evaluation_repo,
                    dataset_repository=dataset_repo,
                    objective_repository=obj_repo,
                )
                objective_engine_kwargs = engines.as_deps_kwargs()
                async def _objective_state_loader():
                    return await objective_service.get_state()

                objective_state_loader = _objective_state_loader
            max_quote_age = int(
                getattr(self.app_settings.data_quality, "option_stale_seconds", 30)
                or 30
            )
            max_spread = float(
                getattr(self.app_settings.data_quality, "maximum_relative_spread", 0.25)
                or 0.25
            )
            cognitive_graph_deps = CognitiveGraphDeps(
                router=model_router,
                config=self.app_settings.cognitive_graph,
                session_id=cognitive_session_id,
                run_id=run_id,
                broker_account_identity=account_identity,
                context_assembler=context_assembler_from_settings(
                    self.app_settings.cognitive_graph
                ),
                snapshot_repo=SnapshotRepository(task1_db),
                option_surface_repo=OptionSurfaceRepository(task1_db),
                data_quality_repo=DataQualityRepository(task1_db),
                db_path=task1_db,
                objective_service=objective_service,
                objective_state_loader=objective_state_loader,
                target_attainment_settings=getattr(
                    self.app_settings.objective, "target_attainment", None
                ),
                full_chain_optimizer_settings=getattr(
                    self.app_settings, "full_chain_optimizer", None
                ),
                kill_switch=bool(self.app_settings.risk.kill_switch),
                max_quote_age_seconds=max_quote_age,
                max_relative_spread=max_spread,
                recovery_mode=recovery_mode,
                reconciliation_only_recovery=recovery_only_mode,
                **objective_engine_kwargs,
                **repos,
            )
            if (
                obj_settings is not None
                and bool(getattr(obj_settings, "enabled", False))
            ):
                cognitive_graph_deps.require_objective_dependencies()
            injected_agent_runtime = CognitiveAgentRuntime(
                session_id=cognitive_session_id,
                run_id=run_id,
                router=model_router,
                config=self.app_settings.cognitive_graph,
                graph_deps=cognitive_graph_deps,
                registry=registry,
                checkpointer_path=task1_db.with_name(task1_db.stem + "_cognitive_ckpt.db"),
            )
            if recovery_only_mode:
                injected_agent_runtime.enable_reconciliation_only_recovery(True)
            # Startup details are logged after the session log() helper is defined.
            _cognitive_startup_payload = {
                "mock_session": startup.mock_session,
                "remapped_to_fake": startup.remapped_to_fake,
                "ollama_enabled": startup.availability.ollama_enabled,
                "ollama_healthy": startup.availability.ollama_healthy,
                "openai_enabled": startup.availability.openai_enabled,
                "openai_healthy": startup.availability.openai_healthy,
                "healthy_mandatory": list(
                    startup.availability.healthy_mandatory_profiles
                ),
                "notes": list(startup.availability.notes),
            }
        elif recovery_mode is RecoveryMode.NORMAL and null_agent_mode:
            from joker.runtime.compatibility import NullAgentRuntime

            injected_agent_runtime = NullAgentRuntime()

        task1_bridge = CompatibilityLivePaperBridge(
            broker=broker,
            db_path=task1_db,
            session_id=bridge_session_id,
            run_id=run_id,
            broker_account_id=account_identity,
            broker_account_identity=account_identity,
            agent_runtime=injected_agent_runtime,
        )
        # Two-phase startup for cognitive mode:
        # Create Task 1 stores/ExecutionRuntime → bind gateway → start agent → resume.
        task1_bridge.start(start_agent=not (recovery_only_mode or cognitive_mode))
        if recovery_mode is not RecoveryMode.NORMAL:
            if (
                recovery_mode is RecoveryMode.RECONCILIATION_ONLY
                and objective_service is not None
            ):
                from joker.runtime.objective_recovery import recover_session_objective

                task1_bridge.supervisor.bind_objective_service(objective_service)
                task1_bridge.run_coro(
                    recover_session_objective(
                        objective_service,
                        session_id=bridge_session_id,
                        execution_runtime=task1_bridge.execution_runtime,
                        unresolved_reconciliation=(
                            task1_bridge.supervisor.unresolved_reconciliation is not None
                        ),
                    )
                )
        if recovery_mode is RecoveryMode.NORMAL and cognitive_mode and cognitive_graph_deps is not None:
            from joker.persistence.cognitive_execution_provenance import (
                CognitiveExecutionProvenanceRegistry,
            )
            from joker.runtime.cognitive_binding import bind_cognitive_graph_to_task1

            provenance = CognitiveExecutionProvenanceRegistry(
                task1_db.with_name(task1_db.stem + "_cognitive_provenance.db")
            )
            import asyncio as _asyncio_bind

            _asyncio_bind.run(provenance.initialize())
            bind_cognitive_graph_to_task1(
                cognitive_graph_deps,
                task1_bridge,
                data_quality_repo=task1_bridge.supervisor.data_quality_repository,
                provenance_registry=provenance,
            )
            assert cognitive_graph_deps.execution_runtime is not None
            assert cognitive_graph_deps.order_action_gateway is not None
            graph_event_values = {
                "graph.cycle.started",
                "strategy.thesis.generated",
                "chain.universe.built",
                "contract.outcome.estimated",
                "contract.grid.scored",
                "portfolio.grid.scored",
                "debate.review.completed",
                "target.portfolio.selected",
                "target.wait.selected",
                "execution.revalidation",
                "execution.reoptimization_required",
                "graph.cycle.completed",
            }

            async def _stream_graph_evidence(event) -> None:
                event_name = str(event.event_type.value)
                if event_name in graph_event_values:
                    self._log(
                        run_id,
                        event_name,
                        dict(event.payload),
                        on_event=on_event,
                    )

            assert cognitive_graph_deps.event_bus is not None
            cognitive_graph_deps.event_bus.subscribe(None, _stream_graph_evidence)
            if bool(getattr(self.app_settings.evolution, "enabled", False)):
                from joker.evolution.runtime import EvolutionRuntime

                evolution_runtime = EvolutionRuntime(
                    db_path=task1_db,
                    settings=self.app_settings.evolution,
                    session_id=bridge_session_id,
                    run_id=run_id,
                    event_bus=task1_bridge.supervisor.event_bus,
                    execution_runtime=task1_bridge.execution_runtime,
                    model_router=model_router,
                    cognitive_graph_deps=cognitive_graph_deps,
                )
                # Prepare Task 3 before Task 2 recovery so champion/config/applicator
                # and event subscriptions exist before unfinished cycles resume.
                task1_bridge.run_coro(evolution_runtime.prepare())
                # Prefer the exact repositories owned by EvolutionRuntime.
                if (
                    cognitive_graph_deps.historical_outcome_service is not None
                    and objective_service is not None
                ):
                    from joker.cli.session_confirm import build_objective_engines

                    evo_repos = evolution_runtime.repositories
                    obj_repo = getattr(objective_service, "repository", None)
                    if obj_repo is None:
                        obj_repo = getattr(objective_service, "_repo", None)
                    engines = build_objective_engines(
                        self.app_settings,
                        episode_repository=evo_repos.get("episodes"),
                        evaluation_repository=evo_repos.get("evaluations"),
                        dataset_repository=evo_repos.get("datasets"),
                        objective_repository=obj_repo,
                    )
                    cognitive_graph_deps.historical_outcome_service = (
                        engines.historical_outcome_service
                    )
                    cognitive_graph_deps.historical_outcome_settings = (
                        engines.historical_outcome_settings
                    )
                    cognitive_graph_deps.feasibility_engine = engines.feasibility_engine
                    cognitive_graph_deps.objective_strategy_scorer = (
                        engines.objective_strategy_scorer
                    )
                    cognitive_graph_deps.capital_sizer = engines.capital_sizer
                    cognitive_graph_deps.target_attainment_policy = (
                        engines.target_attainment_policy
                    )
                    cognitive_graph_deps.objective_policy = engines.objective_policy
                    cognitive_graph_deps.shadow_baseline_enabled = (
                        engines.shadow_baseline_enabled
                    )
                cognitive_graph_deps.evolution_runtime = evolution_runtime
                cognitive_graph_deps.configuration_repo = evo_repos.get(
                    "configurations"
                )
                obj_exec = getattr(self.app_settings.objective, "execution", None)
                cognitive_graph_deps.objective_execution_settings = obj_exec
                evolution_runtime.subscribe_events()
                injected_agent_runtime.bind_evolution_runtime(evolution_runtime)
                task1_bridge.start_agent()
                task1_bridge.run_coro(evolution_runtime.start_workers())
                task1_bridge.run_coro(evolution_runtime.resume())
                self._evolution_runtime = evolution_runtime
            else:
                task1_bridge.start_agent()
        self._task1_bridge = task1_bridge
        execution_broker = ExecutionDelegatingBroker(
            inner=broker,
            bridge=task1_bridge,
            broker_account_id=account_identity,
        )
        _http_clients: list[Any] = [broker]

        def shutdown_task1() -> None:
            evolution = getattr(self, "_evolution_runtime", None)
            if evolution is not None and self._task1_bridge is not None:
                try:
                    self._task1_bridge.run_coro(evolution.shutdown())
                except Exception:
                    pass
                self._evolution_runtime = None
            if self._task1_bridge is not None:
                try:
                    self._task1_bridge.shutdown()
                except Exception:
                    pass
                self._task1_bridge = None
            for client in list(_http_clients):
                closer = getattr(client, "close", None)
                if not callable(closer):
                    continue
                try:
                    closer()
                except Exception:
                    pass
            _http_clients.clear()

        def log(event_type: str, payload: dict) -> None:
            self._log(run_id, event_type, payload, on_event=on_event)
            if event_type == "risk.decision" and not payload.get("approved"):
                codes = payload.get("reason_codes") or []
                session_memory.record_risk_note(
                    ",".join(codes) if codes else str(payload.get("message", "rejected"))
                )
            if event_type == "order.filled":
                session_memory.note_entry(
                    direction=session_memory.last_entry_direction,
                    entry_price=payload.get("entry_price"),
                )

        session_memory = SessionMicroMemory()
        capital_budget = config.capital_budget or _default_capital_budget(self.app_settings)
        if _cognitive_startup_payload is not None:
            log("cognitive.startup", _cognitive_startup_payload)

        def on_trade_outcome(payload: dict) -> None:
            rec = session_memory.record_outcome(
                exit_reason=str(payload.get("exit_reason") or "unknown"),
                exit_price=payload.get("exit_price"),
                mae=payload.get("mae"),
                mfe=payload.get("mfe"),
                duration_minutes=payload.get("duration_minutes"),
                realized_pnl_usd=payload.get("realized_pnl_usd"),
            )
            log(
                "agent.outcome",
                {
                    "quality_note": rec.quality_note,
                    "exit_reason": rec.exit_reason,
                    "duration_minutes": rec.duration_minutes,
                },
            )

        log(
            "live_paper.started",
            {
                "symbol": config.symbol,
                "duration_seconds": config.duration_seconds,
                "mock_agents": config.mock_agents,
                "broker": selection.kind,
                "broker_label": selection.label,
                "auto_orders": selection.auto_orders,
                "live_money_orders": False,
                "is_synthetic": False,
                "capital": capital_budget.prompt_dict(),
                "task1_session_supervisor": True,
                "task1_session_id": task1_bridge.session_id,
            },
        )

        if recovery_mode is not RecoveryMode.NORMAL:
            return self._run_reconciliation_only_recovery(
                run_id=run_id,
                selection=selection,
                config=config,
                result=result,
                task1_bridge=task1_bridge,
                capital_budget=capital_budget,
                failures=failures,
                log=log,
                on_state=on_state,
                run_manager=run_manager,
                shutdown_task1=shutdown_task1,
                recovery_mode=recovery_mode,
            )

        market_loop = LiveMarketDataLoop(
            app_settings=self.app_settings,
            env=self.env_settings,
            stock_api=config.webull_api,
            options_api=config.webull_options_api,
            require_options=config.require_options,
            source_label="live_paper",
            log=log,
        )
        try:
            market_loop.authenticate()
        except LiveMarketDataError as exc:
            msg = str(exc)
            result.errors.append(msg)
            result.feed_health = "ERROR"
            log("provider.error", {"error": msg})
            result.summary = ReplaySummary(
                run_id=run_id,
                session_name="live_paper",
                is_synthetic=False,
                mock_agents=config.mock_agents,
                events_processed=0,
                signals_detected=0,
                trades_entered=0,
                trades_exited=0,
                final_pnl_usd=0.0,
                risk_rejections=0,
                failures=[msg],
            )
            run_manager.end_run(run_id)
            shutdown_task1()
            return result

        provider = market_loop.provider
        options_provider = market_loop.options_provider
        if options_provider is not None and (
            getattr(options_provider, "verified", False)
            or capability_usable_for_shadow()
        ):
            result.options_available = True
        for client in market_loop._http_clients:
            if client not in _http_clients:
                _http_clients.append(client)

        # Warm snapshot from real Webull. Feature candles remain for FeatureEngine;
        # Task 1 market truth is owned by MarketRuntime (see poll loop ingest).
        try:
            snapshot = market_loop.warmup(task1_bridge)
        except LiveMarketDataError as exc:
            msg = str(exc)
            result.errors.append(msg)
            result.failures.append(msg)
            log("provider.error", {"error": msg})
            result.summary = ReplaySummary(
                run_id=run_id,
                session_name="live_paper",
                is_synthetic=False,
                mock_agents=config.mock_agents,
                events_processed=0,
                signals_detected=0,
                trades_entered=0,
                trades_exited=0,
                final_pnl_usd=0.0,
                risk_rejections=0,
                failures=list(result.failures),
            )
            run_manager.end_run(run_id)
            shutdown_task1()
            return result

        risk_config = _risk_config_from_settings(self.app_settings, capital_budget)
        validator = PlaybookQualityValidator(risk_config)
        playbook: Playbook | None = None
        council_analysis: CouncilAnalysis | None = None
        playbook_validation: PlaybookValidationResult | None = None
        armed = False

        features = FeatureEngine(
            max_age_seconds=self.app_settings.risk.feed_max_silence_seconds
        ).compute(snapshot, reference_time=provider.current_time)

        from joker.memory import build_day_memory, save_session_lesson

        day_memory = build_day_memory(
            data_dir=self.app_settings.data_dir,
            db=self.db,
            as_of=trading_day,
            lookback_days=self.app_settings.agents.memory_lookback_days,
        )
        log("memory.loaded",
            {
                "available": day_memory.memory_available,
                "lessons": len(day_memory.prior_lessons),
                "recent_pnl_usd": day_memory.recent_pnl_usd,
            },
        )

        if config.approved_playbook is not None:
            # Test/injection path only — production CLI never sets this.
            playbook = config.approved_playbook.model_copy(update={"approved": True})
            playbook = trim_playbook_enabled_setups(playbook, risk_config)
            playbook_validation = validator.validate(playbook)
            if not playbook_validation.approved:
                failures.append(
                    f"injected_playbook_invalid: {playbook_validation.reason_codes}"
                )
        else:
            council_settings = self.app_settings.model_copy(
                update={
                    "agents": self.app_settings.agents.model_copy(
                        update={"mock_agents": config.mock_agents}
                    ),
                    "mode": mode,
                }
            )
            council = create_agent_council(
                council_settings,
                self.env_settings,
                llm_client=config.llm_client,
            )
            premarket = PremarketWorkflow(
                self.db, self.event_log, council_settings, council=council
            )
            try:
                pb = premarket.run(
                    run_id,
                    trading_day,
                    features,
                    env_settings=self.env_settings,
                    memory=day_memory,
                )
                records = self.db.list_by_run(AgentDecisionRecord, run_id)
                if records:
                    from joker.schemas.domain import AgentCouncilDecision

                    council_decision = AgentCouncilDecision.model_validate(
                        records[0].payload
                    )
                    council_analysis = analyze_council(council_decision)
                playbook = trim_playbook_enabled_setups(pb, risk_config)
                if playbook != pb:
                    log("playbook.trimmed",
                        {
                            "enabled_before": sum(1 for s in pb.setups if s.enabled),
                            "enabled_after": sum(1 for s in playbook.setups if s.enabled),
                        },
                    )
                playbook_validation = validator.validate(
                    playbook,
                    critic_blocked=(
                        council_analysis.council_blocked if council_analysis else False
                    ),
                )
                log("playbook.validation",
                    playbook_validation.model_dump(mode="json"),
                )
                if not playbook_validation.approved:
                    failures.append(
                        f"playbook_validation_failed: {playbook_validation.reason_codes}"
                    )
                elif council_analysis and council_analysis.council_blocked:
                    failures.append("council_blocked: critic flagged weak plan")
                else:
                    playbook = premarket.approve_playbook(run_id, playbook)
            except (AgentError, LLMClientError) as exc:
                failures.append(f"openai_council_failed: {exc}")
                log("live_paper.failure", {"error": failures[-1]})
            finally:
                if config.llm_client is None:
                    council_llm = getattr(council, "llm", None)
                    closer = getattr(council_llm, "close", None)
                    if callable(closer):
                        closer()

        reactive = ReactiveEngine(
            RiskGovernor(risk_config, mode, live_enabled=False),
            execution_broker,
        )

        if playbook and playbook.approved and (
            playbook_validation is None or playbook_validation.approved
        ):
            try:
                reactive.arm_playbook(playbook)
                armed = True
                log(
                    "playbook.approved",
                    {
                        "playbook_id": playbook.playbook_id,
                        "enabled_setups": sum(1 for s in playbook.setups if s.enabled),
                        "broker": selection.kind,
                    },
                )
            except StateMachineError as exc:
                failures.append(f"playbook_arm_failed: {exc}")
        else:
            if cognitive_mode:
                # Cognitive mode does not require a legacy playbook to observe/poll.
                log(
                    "cognitive.playbook_optional",
                    {
                        "playbook_present": playbook is not None,
                        "approved": bool(playbook and playbook.approved),
                    },
                )
            else:
                if not failures:
                    failures.append("no_active_playbook")
                log("live_paper.failure", {"error": "no_active_playbook"})

        daily_state = DailyState(
            trading_day=trading_day,
            run_id=run_id,
            mode=mode.value,
            playbook_approved=armed or cognitive_mode,
        )

        def on_log(event_type: str, payload: dict) -> None:
            log(event_type, payload)
            if event_type == "risk.decision":
                self.db.save(
                    RiskDecisionRecord(
                        run_id=run_id,
                        candidate_id=payload.get("candidate_id", "unknown"),
                        approved=payload.get("approved", False),
                        reason_codes=payload.get("reason_codes", []),
                        payload=payload,
                    )
                )
            if event_type == "signal.detected":
                self.db.save(
                    TradeCandidateRecord(
                        run_id=run_id,
                        candidate_id=payload.get("candidate_id", "unknown"),
                        payload=payload,
                    )
                )

        agent_cfg = self.app_settings.agents
        execution_mode = (agent_cfg.execution_mode or "rules_hybrid").strip().lower()
        # Legacy agent_led loop is disabled under cognitive_graph authority.
        agent_led = execution_mode == "agent_led" and not cognitive_mode

        handler = MarketEventHandler(
            provider=provider,
            reactive_engine=reactive,
            risk_governor=reactive.risk_governor,
            broker=execution_broker,
            feature_engine=FeatureEngine(
                max_age_seconds=self.app_settings.risk.feed_max_silence_seconds
            ),
            option_selector=OptionSelector(
                OptionSelectorConfig(
                    max_spread_pct=self.app_settings.risk.max_spread_pct,
                    max_premium_usd=self.app_settings.risk.max_premium_usd,
                    quote_max_age_seconds=self.app_settings.risk.quote_max_age_seconds,
                    allow_delayed_quotes=self.app_settings.risk.allow_delayed_quotes,
                    feed_max_silence_seconds=self.app_settings.risk.feed_max_silence_seconds,
                    delayed_quote_max_age_seconds=(
                        self.app_settings.risk.delayed_quote_max_age_seconds
                    ),
                    soft_liquidity_advisory=(
                        (self.app_settings.agents.execution_mode or "").strip().lower()
                        == "agent_led"
                        or (self.app_settings.risk.policy or "").strip().lower()
                        == "agent_led"
                    ),
                )
            ),
            exit_manager=ExitManager(),
            mode=mode,
            run_id=run_id,
            daily_state=daily_state,
            on_log=on_log,
            options_provider=options_provider,
            rules_auto_entry=(
                not cognitive_mode
                and not agent_led
            ),
            on_trade_outcome=on_trade_outcome,
            capital_budget=capital_budget,
            task1_bridge=task1_bridge,
        )
        handler._pause_when_goal_met = bool(
            self.app_settings.capital.pause_entries_when_goal_met
        )
        # Prior / premarket levels from warmed candle buffer when available
        try:
            from joker.features.engine import split_session_candles

            snap_levels = provider.get_latest_snapshot()
            if snap_levels is not None and snap_levels.candles:
                prior, pm, _rth = split_session_candles(snap_levels.candles)
                handler._prior_day_candles = prior
                handler._premarket_candles = pm
        except Exception:
            pass
        if armed and playbook:
            handler.state.setups_armed = len([s for s in playbook.setups if s.enabled])

        events_processed = 0
        intraday_calls = 0
        decision_calls = 0
        proposals_acted = 0
        log(
            "execution.mode",
            {
                "execution_mode": execution_mode,
                "agent_runtime": agent_runtime_mode,
                "risk_policy": risk_config.policy,
                "rules_auto_entry": not cognitive_mode and not agent_led,
                "cognitive_mode": cognitive_mode,
            },
        )

        if armed or cognitive_mode:
            try:
                import time as _time

                from joker.agents.decision import (
                    DecisionAgentError,
                    mock_decision,
                    run_decision_agent,
                )
                from joker.agents.intraday import (
                    IntradayAgentError,
                    mock_intraday_result,
                    run_intraday_council,
                )
                from joker.agents.llm_client import OpenAILLMClient
                from joker.strategy.playbook_patch import PatchError, apply_patch

                poll = max(0.5, self.app_settings.data.quote_poll_interval_seconds)
                objective_deadline_mono = _time.monotonic() + max(
                    config.duration_seconds, poll
                )
                # Grace window only for finishing agent-managed exits — does not
                # extend the objective deadline itself.
                hard_stop_mono = objective_deadline_mono + max(
                    0.0, float(config.shutdown_grace_seconds or 0.0)
                )
                deadline = hard_stop_mono
                last_intraday_at = 0.0
                last_decision_at = 0.0
                decision_interval = float(
                    getattr(agent_cfg, "decision_interval_seconds", 45.0) or 45.0
                )
                max_decision_calls = int(
                    getattr(agent_cfg, "max_decision_calls_per_session", 40) or 40
                )
                objective_entries_blocked = bool(recovery_only_mode)
                if recovery_only_mode:
                    log(
                        "objective.reconciliation_only_started",
                        {
                            "new_entries_blocked": True,
                            "runtime_seconds": float(config.duration_seconds),
                            "original_objective_deadline": (
                                config.objective_deadline_exchange.isoformat()
                                if config.objective_deadline_exchange is not None
                                else None
                            ),
                        },
                    )

                while _time.monotonic() < deadline:
                    now_mono = _time.monotonic()
                    past_objective = now_mono >= objective_deadline_mono
                    if past_objective and not objective_entries_blocked:
                        objective_entries_blocked = True
                        result.objective_deadline_reached = True
                        log(
                            "objective.deadline_reached",
                            {
                                "new_entries_blocked": True,
                                "grace_seconds": float(
                                    config.shutdown_grace_seconds or 0.0
                                ),
                            },
                        )
                        if config.objective_service is not None:
                            try:
                                task1_bridge.run_coro(
                                    config.objective_service.recompute_from_truth()
                                )
                            except Exception as exc:
                                log(
                                    "objective.deadline_recompute_failed",
                                    {"reason": str(exc)},
                                )
                    # After objective deadline with no open/pending work, finish.
                    if past_objective and (
                        handler.state.open_trade is None
                        and handler.state.pending_entry is None
                    ):
                        log(
                            "objective.session_complete_flat",
                            {"past_objective_deadline": True},
                        )
                        break
                    event = market_loop.poll_once(task1_bridge)
                    if event is None:
                        _time.sleep(poll)
                        continue

                    events_processed += 1

                    if task1_bridge.health.degraded:
                        log(
                            "task1.truth_degraded",
                            {
                                "last_error": task1_bridge.health.last_error,
                                "consecutive_failures": (
                                    task1_bridge.health.consecutive_failures
                                ),
                                "new_entries_blocked": True,
                            },
                        )
                        # Still reconcile pending exits/entries; block new agent entries.
                        handler.handle_event(event)
                        _time.sleep(min(poll, max(0.0, deadline - _time.monotonic())))
                        continue

                    handler.handle_event(event)

                    latest = provider.get_latest_snapshot()
                    if (
                        handler.state.open_trade is not None
                        and options_provider is not None
                        and latest is not None
                    ):
                        try:
                            call_snap, put_snap = options_provider.fetch_atm_snapshots(
                                latest.price
                            )
                            snaps = [s for s in (call_snap, put_snap) if s is not None]
                            if snaps:
                                opt_events = options_provider.to_quote_events(
                                    snaps,
                                    reference_time=provider.current_time,
                                    allow_delayed_quotes=(
                                        self.app_settings.risk.allow_delayed_quotes
                                    ),
                                    feed_max_silence_seconds=(
                                        self.app_settings.risk.feed_max_silence_seconds
                                    ),
                                    delayed_quote_max_age_seconds=(
                                        self.app_settings.risk.delayed_quote_max_age_seconds
                                    ),
                                )
                                for oe in opt_events:
                                    handler.handle_event(oe)
                        except Exception as exc:
                            log(
                                "options.exit_refresh_failed",
                                {"reason": str(exc)},
                            )

                    now_mono = _time.monotonic()

                    # --- agent_led: primary AI decision loop (propose → confirm) ---
                    has_pending = session_memory.pending is not None
                    fast_confirm_min = float(
                        getattr(agent_cfg, "fast_confirm_min_seconds", 8.0) or 8.0
                    )
                    use_prefilter = bool(getattr(agent_cfg, "use_edge_prefilter", True))
                    interval_due = (now_mono - last_decision_at) >= (
                        fast_confirm_min if has_pending else decision_interval
                    )
                    if (
                        agent_led
                        and not objective_entries_blocked
                        and agent_cfg.intraday_enabled
                        and decision_calls < max_decision_calls
                        and proposals_acted < agent_cfg.max_proposals_per_session
                        and interval_due
                        and handler.state.open_trade is None
                        and handler.state.pending_entry is None
                    ):
                        snap_now = provider.get_latest_snapshot()
                        if snap_now is not None:
                            from joker.features.engine import split_session_candles

                            prior_c, pm_c, _ = split_session_candles(snap_now.candles or [])
                            if prior_c:
                                handler._prior_day_candles = prior_c
                            if pm_c:
                                handler._premarket_candles = pm_c
                            feat_now = FeatureEngine(
                                max_age_seconds=self.app_settings.risk.feed_max_silence_seconds
                            ).compute(
                                snap_now,
                                prior_day_candles=getattr(handler, "_prior_day_candles", None),
                                premarket_candles=getattr(handler, "_premarket_candles", None),
                                reference_time=provider.current_time,
                            )

                            skip_llm = False
                            if (
                                use_prefilter
                                and not has_pending
                                and not config.mock_agents
                            ):
                                from joker.strategy.edge_prefilter import edge_prefilter

                                pre = edge_prefilter(
                                    feat_now, goal_met=capital_budget.goal_met
                                )
                                if not pre.candidate:
                                    log(
                                        "agent.prefilter_skip",
                                        {"reason": pre.reason},
                                    )
                                    last_decision_at = now_mono
                                    skip_llm = True

                            if not skip_llm:
                                last_decision_at = now_mono
                                option_context: dict = {}
                                if (
                                    options_provider is not None
                                    and options_provider.is_available()
                                ):
                                    try:
                                        call_snap, put_snap = (
                                            options_provider.fetch_atm_snapshots(
                                                snap_now.price
                                            )
                                        )
                                        if call_snap is not None:
                                            option_context["atm_call"] = {
                                                "strike": call_snap.contract.strike,
                                                "bid": call_snap.bid,
                                                "ask": call_snap.ask,
                                                "mid": call_snap.mid,
                                                "spread_pct": call_snap.spread_pct,
                                            }
                                        if put_snap is not None:
                                            option_context["atm_put"] = {
                                                "strike": put_snap.contract.strike,
                                                "bid": put_snap.bid,
                                                "ask": put_snap.ask,
                                                "mid": put_snap.mid,
                                                "spread_pct": put_snap.spread_pct,
                                            }
                                    except Exception as exc:
                                        option_context["error"] = str(exc)
                                try:
                                    from joker.agents.decision import (
                                        confirm_gate,
                                        decision_from_pending,
                                        ev_entry_allowed,
                                        pending_from_decision,
                                    )

                                    require_propose = bool(
                                        getattr(
                                            agent_cfg,
                                            "require_propose_before_enter",
                                            True,
                                        )
                                    )
                                    # Expire stale proposals before asking the agent
                                    if session_memory.pending is not None:
                                        ok, reason = confirm_gate(
                                            session_memory.pending,
                                            spy_price=float(snap_now.price),
                                            option_context=option_context,
                                            ttl_seconds=float(
                                                getattr(
                                                    agent_cfg, "confirm_ttl_seconds", 120.0
                                                )
                                            ),
                                            max_spy_drift_pct=float(
                                                getattr(
                                                    agent_cfg,
                                                    "max_confirm_spy_drift_pct",
                                                    0.20,
                                                )
                                            ),
                                            max_option_mid_worsen_pct=float(
                                                getattr(
                                                    agent_cfg,
                                                    "max_confirm_option_mid_worsen_pct",
                                                    15.0,
                                                )
                                            ),
                                        )
                                        if not ok and reason.startswith("proposal_expired"):
                                            log(
                                                "agent.propose_expired",
                                                {"reason": reason},
                                            )
                                            session_memory.clear_pending()

                                    if config.mock_agents:
                                        decision = mock_decision(
                                            feat_now,
                                            playbook,
                                            session_memory=session_memory,
                                        )
                                    else:
                                        llm = config.llm_client or OpenAILLMClient(
                                            api_key=self.env_settings.openai_api_key,
                                            model=self.env_settings.openai_model,
                                            max_retries=agent_cfg.max_retries,
                                            default_timeout_seconds=float(
                                                agent_cfg.council_timeout_seconds
                                            ),
                                        )
                                        decision = run_decision_agent(
                                            llm,
                                            agent_cfg,
                                            run_id=run_id,
                                            playbook=playbook,
                                            features=feat_now,
                                            risk=risk_config,
                                            memory=day_memory,
                                            session_memory=session_memory,
                                            capital_budget=capital_budget,
                                            open_position=False,
                                            trades_entered=handler.state.trades_entered,
                                            spy_price=snap_now.price,
                                            option_context=option_context,
                                        )
                                    decision_calls += 1
                                    session_memory.record_decision(
                                        action=decision.action,
                                        direction=decision.direction,
                                        confidence=decision.confidence,
                                        summary=decision.summary or decision.rationale,
                                        spy_price=snap_now.price,
                                    )
                                    session_memory.update_option_mids(option_context)
                                    log(
                                        "agent.decision",
                                        {
                                            "action": decision.action,
                                            "direction": decision.direction,
                                            "confidence": decision.confidence,
                                            "win_probability": decision.win_probability,
                                            "expected_r": decision.expected_r,
                                            "expected_value_usd": decision.expected_value_usd,
                                            "summary": (
                                                decision.summary or decision.rationale
                                            )[:160],
                                            "pending": bool(session_memory.pending),
                                            "goal_gap_pct": capital_budget.goal_gap_pct,
                                            "aggression_cap": capital_budget.aggression_cap(
                                                minutes_to_close=feat_now.minutes_to_close
                                            ),
                                        },
                                    )
                                    if decision.patch is not None and playbook is not None:
                                        try:
                                            playbook = apply_patch(playbook, decision.patch)
                                            reactive.active_playbook = playbook
                                            log(
                                                "playbook.patched",
                                                {
                                                    "disable": decision.patch.disable_setup_ids,
                                                    "enable": decision.patch.enable_setup_ids,
                                                },
                                            )
                                        except PatchError as exc:
                                            log(
                                                "playbook.patch_rejected",
                                                {"reason": str(exc)},
                                            )

                                    action = decision.action
                                    if action == "abandon":
                                        session_memory.clear_pending()
                                        log("agent.propose_abandoned", {})
                                    elif action == "propose":
                                        if decision.direction in ("long_call", "long_put"):
                                            session_memory.set_pending(
                                                pending_from_decision(
                                                    decision,
                                                    spy_price=float(snap_now.price),
                                                    option_context=option_context,
                                                )
                                            )
                                            log(
                                                "agent.propose",
                                                {
                                                    "direction": decision.direction,
                                                    "confidence": decision.confidence,
                                                    "win_probability": decision.win_probability,
                                                    "expected_value_usd": decision.expected_value_usd,
                                                    "goal_gap_pct": capital_budget.goal_gap_pct,
                                                },
                                            )
                                    elif action in ("confirm", "enter"):
                                        pending = session_memory.pending
                                        pending_decision = None
                                        if require_propose and pending is None:
                                            # Treat bare enter as propose when two-step required
                                            if action == "enter" and decision.direction in (
                                                "long_call",
                                                "long_put",
                                            ):
                                                session_memory.set_pending(
                                                    pending_from_decision(
                                                        decision,
                                                        spy_price=float(snap_now.price),
                                                        option_context=option_context,
                                                    )
                                                )
                                                log(
                                                    "agent.propose",
                                                    {
                                                        "direction": decision.direction,
                                                        "confidence": decision.confidence,
                                                        "via": "enter_downgraded_to_propose",
                                                        "win_probability": decision.win_probability,
                                                        "expected_value_usd": decision.expected_value_usd,
                                                    },
                                                )
                                            else:
                                                log(
                                                    "agent.confirm_rejected",
                                                    {"reason": "no_pending_proposal"},
                                                )
                                        else:
                                            if pending is None:
                                                pending_decision = decision
                                            else:
                                                ok, reason = confirm_gate(
                                                    pending,
                                                    spy_price=float(snap_now.price),
                                                    option_context=option_context,
                                                    ttl_seconds=float(
                                                        getattr(
                                                            agent_cfg,
                                                            "confirm_ttl_seconds",
                                                            120.0,
                                                        )
                                                    ),
                                                    max_spy_drift_pct=float(
                                                        getattr(
                                                            agent_cfg,
                                                            "max_confirm_spy_drift_pct",
                                                            0.20,
                                                        )
                                                    ),
                                                    max_option_mid_worsen_pct=float(
                                                        getattr(
                                                            agent_cfg,
                                                            "max_confirm_option_mid_worsen_pct",
                                                            15.0,
                                                        )
                                                    ),
                                                )
                                                if not ok:
                                                    log(
                                                        "agent.confirm_rejected",
                                                        {"reason": reason},
                                                    )
                                                    if reason.startswith("proposal_expired"):
                                                        session_memory.clear_pending()
                                                    pending_decision = None
                                                else:
                                                    pending_decision = decision_from_pending(
                                                        pending,
                                                        confidence=decision.confidence
                                                        or pending.confidence,
                                                        capital_fraction=decision.capital_fraction,
                                                        target_contracts=decision.target_contracts,
                                                        allocation_style=decision.allocation_style,
                                                        win_probability=decision.win_probability,
                                                        expected_r=decision.expected_r,
                                                        expected_value_usd=decision.expected_value_usd,
                                                    )
                                                    if decision.rationale:
                                                        pending_decision = (
                                                            pending_decision.model_copy(
                                                                update={
                                                                    "rationale": decision.rationale,
                                                                    "summary": decision.summary
                                                                    or pending.summary,
                                                                }
                                                            )
                                                        )
                                        if pending_decision is not None:
                                            ok_ev, ev_reason = ev_entry_allowed(
                                                pending_decision,
                                                min_win_probability=float(
                                                    capital_budget.plan.min_win_probability
                                                ),
                                            )
                                            if not ok_ev:
                                                log(
                                                    "agent.confirm_rejected",
                                                    {"reason": ev_reason},
                                                )
                                            else:
                                                acted = handler.try_enter_from_decision(
                                                    pending_decision,
                                                    min_confidence=agent_cfg.min_proposal_confidence,
                                                )
                                                if acted:
                                                    proposals_acted += 1
                                                    session_memory.note_entry(
                                                        direction=pending_decision.direction,
                                                        entry_price=None,
                                                    )
                                                    session_memory.clear_pending()
                                                    log(
                                                        "agent.confirm_executed",
                                                        {
                                                            "direction": pending_decision.direction,
                                                            "confidence": pending_decision.confidence,
                                                            "win_probability": pending_decision.win_probability,
                                                            "expected_value_usd": pending_decision.expected_value_usd,
                                                            "goal_gap_pct": capital_budget.goal_gap_pct,
                                                        },
                                                    )
                                except (DecisionAgentError, Exception) as exc:
                                    log("agent.decision_failed", {"reason": str(exc)})

                    # --- rules_hybrid: legacy intraday council ---
                    elif (
                        not agent_led
                        and agent_cfg.intraday_enabled
                        and playbook is not None
                        and intraday_calls < agent_cfg.max_intraday_calls_per_session
                        and proposals_acted < agent_cfg.max_proposals_per_session
                        and (now_mono - last_intraday_at)
                        >= agent_cfg.intraday_interval_seconds
                        and handler.state.open_trade is None
                    ):
                        last_intraday_at = now_mono
                        snap_now = provider.get_latest_snapshot()
                        if snap_now is not None:
                            feat_now = FeatureEngine(
                                max_age_seconds=self.app_settings.risk.feed_max_silence_seconds
                            ).compute(snap_now, reference_time=provider.current_time)
                            try:
                                if config.mock_agents:
                                    intra = mock_intraday_result(
                                        playbook, feat_now, run_id=run_id
                                    )
                                else:
                                    llm = config.llm_client or OpenAILLMClient(
                                        api_key=self.env_settings.openai_api_key,
                                        model=self.env_settings.openai_model,
                                        max_retries=agent_cfg.max_retries,
                                        default_timeout_seconds=float(
                                            agent_cfg.council_timeout_seconds
                                        ),
                                    )
                                    intra = run_intraday_council(
                                        llm,
                                        agent_cfg,
                                        run_id=run_id,
                                        playbook=playbook,
                                        features=feat_now,
                                        memory=day_memory,
                                        open_position=False,
                                        trades_entered=handler.state.trades_entered,
                                        max_trades=self.app_settings.risk.max_trades_per_day,
                                    )
                                intraday_calls += 1
                                log(
                                    "intraday.council",
                                    {
                                        "summary": intra.summary,
                                        "has_patch": intra.patch is not None,
                                        "propose_entry": bool(
                                            intra.proposal and intra.proposal.propose_entry
                                        ),
                                    },
                                )
                                if intra.patch is not None:
                                    try:
                                        playbook = apply_patch(playbook, intra.patch)
                                        reactive.active_playbook = playbook
                                        log(
                                            "playbook.patched",
                                            {
                                                "disable": intra.patch.disable_setup_ids,
                                                "enable": intra.patch.enable_setup_ids,
                                            },
                                        )
                                    except PatchError as exc:
                                        log(
                                            "playbook.patch_rejected",
                                            {"reason": str(exc)},
                                        )
                                if intra.proposal is not None:
                                    acted = handler.try_enter_from_proposal(
                                        intra.proposal,
                                        min_confidence=agent_cfg.min_proposal_confidence,
                                    )
                                    if acted:
                                        proposals_acted += 1
                            except (IntradayAgentError, Exception) as exc:
                                log("intraday.failed", {"reason": str(exc)})

                    result.feed_health = provider.feed_health
                    result.options_available = bool(
                        options_provider and options_provider.is_available()
                    )
                    if on_state:
                        snap = provider.get_latest_snapshot()
                        on_state(
                            {
                                "run_id": run_id,
                                "provider": ProviderKind.WEBULL.value,
                                "market_price": snap.price if snap else None,
                                "feed_health": provider.feed_health,
                                "delayed": provider.last_quote_delayed,
                                "options_available": result.options_available,
                                "signals": handler.state.signals_detected,
                                "trades_entered": handler.state.trades_entered,
                                "trades_exited": handler.state.trades_exited,
                                "paper_pnl": broker.get_daily_pnl(),
                                "engine_state": reactive.state.value,
                                "intraday_calls": intraday_calls,
                                "decision_calls": decision_calls,
                                "proposals_acted": proposals_acted,
                                "execution_mode": execution_mode,
                                "capital_available": capital_budget.available_usd,
                                "capital_goal_pct": capital_budget.progress_to_goal_pct,
                                "broker": selection.kind,
                                "broker_label": selection.label,
                                "pending_order": bool(handler.state.pending_entry),
                                "open_trade": handler.state.open_trade is not None,
                                "auto_orders": selection.auto_orders,
                                "live_money_orders": False,
                            }
                        )
                    _time.sleep(min(poll, max(0.0, deadline - _time.monotonic())))
            except Exception as exc:
                msg = str(exc)
                result.errors.append(msg)
                result.feed_health = "ERROR"
                failures.append(msg)
                log("provider.error", {"error": msg})

        # Postmarket learner → day memory
        if armed and self.app_settings.agents.postmarket_learner_enabled:
            try:
                from joker.agents.intraday import (
                    mock_session_lesson,
                    run_postmarket_learner,
                )
                from joker.agents.llm_client import OpenAILLMClient

                session_stats = {
                    "events_processed": events_processed,
                    "signals_detected": handler.state.signals_detected,
                    "trades_entered": handler.state.trades_entered,
                    "trades_exited": handler.state.trades_exited,
                    "final_pnl_usd": broker.get_daily_pnl(),
                    "risk_rejections": handler.state.risk_rejections,
                    "intraday_calls": intraday_calls,
                    "decision_calls": decision_calls,
                    "proposals_acted": proposals_acted,
                    "execution_mode": execution_mode,
                    "risk_reject_codes": [],
                }
                if config.mock_agents:
                    lesson = mock_session_lesson(trading_day, session_stats)
                else:
                    llm = config.llm_client or OpenAILLMClient(
                        api_key=self.env_settings.openai_api_key,
                        model=self.env_settings.openai_model,
                        max_retries=agent_cfg.max_retries,
                        default_timeout_seconds=float(agent_cfg.council_timeout_seconds),
                    )
                    try:
                        lesson = run_postmarket_learner(
                            llm,
                            agent_cfg,
                            trading_day=trading_day,
                            session_stats=session_stats,
                            memory=day_memory,
                        )
                    finally:
                        if config.llm_client is None:
                            llm.close()
                save_session_lesson(self.app_settings.data_dir, lesson)
                log("memory.lesson_saved",
                    {
                        "trading_day": lesson.trading_day.isoformat(),
                        "summary": lesson.summary[:120],
                    },
                )
            except Exception as exc:
                log("memory.lesson_failed", {"reason": str(exc)})

        quality_metrics = (
            compute_quality_metrics(handler.state) if armed else StrategyQualityMetrics()
        )
        summary = (
            handler.build_summary(mock_agents=config.mock_agents, is_synthetic=False)
            if armed
            else ReplaySummary(
                run_id=run_id,
                session_name="live_paper",
                is_synthetic=False,
                mock_agents=config.mock_agents,
                events_processed=0,
                signals_detected=0,
                trades_entered=0,
                trades_exited=0,
                final_pnl_usd=0.0,
                risk_rejections=0,
                failures=failures,
            )
        )
        summary = summary.model_copy(
            update={
                "run_id": run_id,
                "events_processed": events_processed,
                "playbook_validation_approved": (
                    playbook_validation.approved if playbook_validation else False
                ),
                "council_blocked": (
                    council_analysis.council_blocked if council_analysis else False
                ),
                "failures": failures,
                "final_pnl_usd": broker.get_daily_pnl(),
            }
        )

        report_ctx = ReplayReportContext(
            run_id=run_id,
            trading_day=trading_day,
            is_synthetic=False,
            mock_agents=config.mock_agents,
            playbook=playbook,
            playbook_validation=playbook_validation,
            council_analysis=council_analysis,
            quality_metrics=quality_metrics,
            exit_decisions=handler.state.exit_decisions if armed else [],
            failures=failures,
            replay_summary=summary.model_dump(mode="json"),
        )
        report_path = ReplayReportGenerator(
            self.db, self.app_settings.reports_dir
        ).generate_replay_postmarket(report_ctx)

        self.db.save(
            SystemEventRecord(
                run_id=run_id,
                event_type="live_paper.summary",
                source="live_paper",
                payload={
                    **summary.model_dump(mode="json"),
                    "quality_metrics": quality_metrics.to_dict(),
                    "feed_health": result.feed_health,
                },
            )
        )
        log("live_paper.completed", summary.model_dump(mode="json"))
        run_manager.end_run(run_id)

        result.summary = summary
        result.report_path = report_path
        result.failures = failures
        result.playbook = playbook
        result.council_analysis = council_analysis
        result.playbook_validation = playbook_validation
        result.events_processed = events_processed
        result.paper_pnl_usd = broker.get_daily_pnl()
        try:
            result.open_positions_remaining = len(broker.list_positions() or [])
            result.working_orders_remaining = len(broker.list_open_orders() or [])
        except Exception as exc:
            result.errors.append(f"broker_flat_check_failed: {exc}")
            result.open_positions_remaining = 1 if handler.state.open_trade else 0
            result.working_orders_remaining = 1 if handler.state.pending_entry else 0
        if task1_bridge is not None and task1_bridge.supervisor.execution_runtime is not None:
            try:
                recon = task1_bridge.run_coro(
                    task1_bridge.supervisor.execution_runtime.run_reconciliation()
                )
                result.reconciliation_clean = bool(
                    getattr(recon, "is_consistent", False)
                )
            except Exception as exc:
                result.reconciliation_clean = False
                result.errors.append(f"reconciliation_failed: {exc}")
        shutdown_task1()
        return result
