"""Shared paper-session harness for Task 3 production acceptance tests."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from dataclasses import dataclass

from joker.broker.interface import PaperBroker
from joker.cognition.schemas import (
    OrderManagementDecision,
    PositionAction,
    PositionThesisVersion,
)
from joker.config.settings import CognitiveGraphSettings
from joker.evolution.agent_schemas import (
    EvolutionDecisionAgentOutput,
    EvaluatorAgentScores,
    ImprovementAgentProposal,
)
from joker.evolution.config import (
    DatasetSettings,
    EvolutionSettings,
    ExperimentSettings,
    OrchestratorSettings,
    PromotionSettings,
    ShadowSettings,
)
from joker.evolution.orchestrator import EvolutionCycleState
from joker.evolution.runtime import EvolutionRuntime
from joker.evaluation.agentic_graph import EVALUATOR_ROLES
from joker.graph.context_hydrate import context_assembler_from_settings
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.data_quality_store import DataQualityRepository
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelRequest, ModelsConfig, default_model_profiles
from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
)
from joker.runtime.cognitive_agent_runtime import (
    CognitiveAgentRuntime,
    build_default_repositories,
)
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.order_action_gateway import ensure_order_action_gateway
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned

ET = ZoneInfo("America/New_York")
FAR_CONTRACT_ID = "SPY:2026-07-01:580.0:call"
EXPECTED_REALIZED_PNL = Decimal("10")


def acceptance_settings() -> EvolutionSettings:
    return EvolutionSettings(
        enabled=True,
        datasets=DatasetSettings(
            minimum_episode_count=2,
            minimum_holdout_count=1,
            minimum_regime_count=1,
        ),
        experiments=ExperimentSettings(repeated_samples=1, maximum_model_calls=5000),
        promotion=PromotionSettings(
            require_known_cost=True,
            minimum_calibration_samples=2,
            require_brier_score=True,
            require_expected_calibration_error=True,
            minimum_completed_episodes=2,
            minimum_holdout_episodes=1,
            maximum_tail_loss_regression_pct=Decimal("100"),
            maximum_calibration_regression_pct=Decimal("100"),
            maximum_latency_regression_pct=Decimal("100"),
            maximum_cost_regression_pct=Decimal("100"),
        ),
        shadow=ShadowSettings(
            enabled=True,
            minimum_completed_cycles=2,
            minimum_traded_cycles=1,
            minimum_regime_coverage=1,
            minimum_observation_minutes=0,
            allow_promotion_before_shadow=False,
        ),
        orchestrator=OrchestratorSettings(
            enabled=True,
            minimum_new_completed_episodes=2,
            minimum_new_evaluations=2,
            minimum_holdout_episodes=1,
            automatic_cycle_interval_minutes=0,
        ),
    )


def _fake_registry(fake: FakeModelProvider) -> ModelRegistry:
    profiles = {
        name: profile.model_copy(update={"provider": "fake", "model": "fake-model"})
        for name, profile in default_model_profiles().items()
    }
    models_config = ModelsConfig(profiles=profiles)
    models_config = models_config.model_copy(
        update={
            "ollama": models_config.ollama.model_copy(update={"enabled": False}),
            "openai": models_config.openai.model_copy(update={"enabled": False}),
        }
    )
    return ModelRegistry(models_config, providers={"fake": fake})


def make_order_manager_decision(request: ModelRequest) -> OrderManagementDecision:
    """Mint a fresh OrderManagementDecision for every distinct model invocation."""
    client_order_id = str(
        request.context_payload.get("client_order_id") or "unknown-order"
    )
    return OrderManagementDecision(
        decision_id=uuid4(),
        session_id="placeholder",  # enriched by CognitiveAgent
        snapshot_id=request.snapshot_id,
        prompt_version=request.prompt_version,
        model_call_id=request.request_id,
        cycle_id=request.cycle_id,
        client_order_id=client_order_id,
        action="continue_waiting",
        rationale_summary="continue waiting in paper-path test",
    )


@dataclass
class PaperPathCannedState:
    """Mutable HOLD/EXIT switch for per-invocation position factories."""

    position_action: PositionAction = PositionAction.HOLD
    contract_id: str = CONTRACT_ID
    session_id: str = "placeholder"


def _position_thesis_from_state(
    state: PaperPathCannedState, request: ModelRequest
) -> PositionThesisVersion:
    action = state.position_action
    return PositionThesisVersion(
        thesis_version_id=uuid4(),
        position_id=state.contract_id,
        contract_id=state.contract_id,
        session_id=state.session_id,
        snapshot_id=request.snapshot_id,
        original_strategy_id=uuid4(),
        current_thesis="exit now" if action == PositionAction.EXIT else "thesis holds",
        recommended_action=action,
        recommended_quantity=1,
        recommended_limit_price=Decimal("1.20"),
        confidence=0.7,
        prompt_version=request.prompt_version or "2.0.0",
        model_call_id=request.request_id,
    )


def install_order_manager_factory(fake: FakeModelProvider) -> None:
    """Ensure order-manager responses never reuse a static decision_id."""
    fake.set_role_factory("order_manager", make_order_manager_decision)


def install_paper_path_factories(
    fake: FakeModelProvider,
    *,
    session_id: str | None = None,
    state: PaperPathCannedState | None = None,
) -> PaperPathCannedState:
    """Install OM + position factories that mint fresh immutable artifact ids."""
    ctl = state or getattr(fake, "_paper_path_state", None)
    if not isinstance(ctl, PaperPathCannedState):
        ctl = PaperPathCannedState()
    if session_id is not None:
        ctl.session_id = session_id
    fake._paper_path_state = ctl  # type: ignore[attr-defined]

    def make_thesis(request: ModelRequest) -> PositionThesisVersion:
        return _position_thesis_from_state(ctl, request)

    def make_decision(request: ModelRequest) -> PositionThesisVersion:
        return _position_thesis_from_state(ctl, request)

    fake.set_role_factory("position_thesis", make_thesis)
    fake.set_role_factory("position_decision", make_decision)
    install_order_manager_factory(fake)
    return ctl


def set_position_action(fake: FakeModelProvider, action: PositionAction) -> None:
    """Flip HOLD/EXIT for subsequent position-factory invocations."""
    ctl = install_paper_path_factories(fake)
    ctl.position_action = action


def _refresh_trade_canned(fake: FakeModelProvider, trade_index: int) -> None:
    """Ensure repeated paper round-trips do not reuse immutable artifact ids."""
    del trade_index  # retained for call-site clarity; factories mint per invocation
    ctl = install_paper_path_factories(fake)
    ctl.position_action = PositionAction.HOLD


async def wire_replay_canned_for_episodes(
    evolution: EvolutionRuntime, fake: FakeModelProvider
) -> None:
    """Register Task 2 canned outputs for episode snapshot ids used by replay."""
    episodes = await evolution._repos["episodes"].list_completed(limit=20)
    for i, episode in enumerate(episodes):
        for snap in (episode.initial_snapshot_id, episode.terminal_snapshot_id):
            if snap is None:
                continue
            register_full_path_canned(
                fake,
                snap,
                f"replay-{i}-{snap}",
                session=evolution.session_id,
                position_action=PositionAction.HOLD,
            )
    register_evolution_router_canned(fake)
    install_paper_path_factories(fake, session_id=evolution.session_id)


def register_evolution_router_canned(fake: FakeModelProvider) -> None:
    scores = EvaluatorAgentScores(
        thesis_quality=Decimal("0.7"),
        evidence_grounding_score=Decimal("0.7"),
        calibration_score=Decimal("0.7"),
        execution_quality=Decimal("0.7"),
        efficiency_score=Decimal("0.5"),
    )
    for role in EVALUATOR_ROLES:
        fake.set_canned_for_role(role, scores)
    proposal = ImprovementAgentProposal(
        weakness="evidence_grounding",
        hypothesis="Require explicit evidence IDs",
        patch_type="prompt",
        role="falsifier",
        replacement_template="Reject theses lacking snapshot/evidence IDs.",
        change_rationale="grounding",
        metrics_to_improve=("evidence_grounding_score",),
        metrics_must_not_regress=("tail_loss",),
        critic_accepted=True,
    )
    fake.set_canned_for_role("improvement_proposer", proposal)
    fake.set_canned_for_role("improvement_critic", proposal)
    fake.set_canned_for_role(
        "evolution_decision",
        EvolutionDecisionAgentOutput(
            action="promote",
            rationale_codes=("improved_grounding",),
            summary="promote challenger",
        ),
    )


def build_acceptance_router(session_id: str) -> tuple[ModelRouter, FakeModelProvider]:
    fake = FakeModelProvider(available=True)
    register_evolution_router_canned(fake)
    install_paper_path_factories(fake, session_id=session_id)
    return ModelRouter(_fake_registry(fake), session_id=session_id), fake


def build_router_from_fake(fake: FakeModelProvider, session_id: str) -> ModelRouter:
    return ModelRouter(_fake_registry(fake), session_id=session_id)


async def build_restart_evolution_runtime(
    db,
    *,
    session_id: str,
    settings: EvolutionSettings | None = None,
    router: ModelRouter | None = None,
    fake: FakeModelProvider | None = None,
) -> tuple[EvolutionRuntime, FakeModelProvider]:
    """Minimal evolution runtime for restart/recovery tests with model-call persistence."""
    if fake is None:
        fake = FakeModelProvider(available=True)
        register_evolution_router_canned(fake)
        install_paper_path_factories(fake, session_id=session_id)
    if router is None:
        router = build_router_from_fake(fake, session_id)
    repos = build_default_repositories(db)
    for repo in repos.values():
        await repo.initialize()
    router.set_model_call_repo(repos["model_call_repo"])
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=30),
        session_id=session_id,
        run_id=session_id,
        context_assembler=context_assembler_from_settings(CognitiveGraphSettings()),
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=DataQualityRepository(db),
        db_path=db,
        **repos,
    )
    runtime = EvolutionRuntime(
        db_path=db,
        settings=settings or restart_settings(),
        session_id=session_id,
        run_id=session_id,
        model_router=router,
        cognitive_graph_deps=deps,
    )
    return runtime, fake


async def seed_market_surface(market, start: datetime, clock: FrozenExchangeClock):
    for i in range(3):
        ts = start + timedelta(minutes=i, seconds=5)
        clock.set_now(ts)
        await market.ingest_underlying_quote(
            symbol="SPY",
            bid=Decimal("499.90"),
            ask=Decimal("500.10"),
            last=Decimal("500") + Decimal(i),
            source_timestamp=ts,
            received_timestamp=ts,
        )
    rows = [
        {
            "contract_id": CONTRACT_ID,
            "symbol": "SPY",
            "expiry": date(2026, 7, 1),
            "strike": "500",
            "option_type": "call",
            "bid": "1.00",
            "ask": "1.20",
            "last": "1.10",
            "quote_timestamp": start + timedelta(minutes=3),
            "is_0dte": True,
        },
        {
            "contract_id": FAR_CONTRACT_ID,
            "symbol": "SPY",
            "expiry": date(2026, 7, 1),
            "strike": "580",
            "option_type": "call",
            "bid": "0.05",
            "ask": "0.15",
            "last": "0.10",
            "quote_timestamp": start + timedelta(minutes=3),
            "is_0dte": True,
        },
    ]
    await market.ingest_option_quotes(rows)
    now = start + timedelta(minutes=3, seconds=3)
    clock.set_now(now)
    tick = await market.tick(now=now)
    assert tick.snapshot is not None
    return tick.snapshot


def restart_settings() -> EvolutionSettings:
    """Restart/recovery tests skip shadow gating so later orchestrator nodes are reachable."""
    settings = acceptance_settings()
    return settings.model_copy(
        update={
            "shadow": settings.shadow.model_copy(
                update={
                    "minimum_completed_cycles": 0,
                    "minimum_traded_cycles": 0,
                    "minimum_regime_coverage": 0,
                    "minimum_observation_minutes": 0,
                }
            )
        }
    )


async def build_paper_evolution_stack(
    tmp_path,
    *,
    session_id: str,
    settings: EvolutionSettings | None = None,
    start_orchestrator_worker: bool = True,
    start_workers: bool = True,
    db_path=None,
):
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = db_path if db_path is not None else tmp_path / f"{session_id}.db"
    broker = PaperBroker(slippage_pct=0)
    gateway_entry_ids: list[str] = []
    gateway_exit_ids: list[str] = []

    router, fake = build_acceptance_router(session_id)
    repos = build_default_repositories(db)
    for repo in repos.values():
        await repo.initialize()
    router.set_model_call_repo(repos["model_call_repo"])
    provenance = CognitiveExecutionProvenanceRegistry(
        db.with_name(f"{session_id}_prov.db")
    )
    await provenance.initialize()

    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=60),
        session_id=session_id,
        run_id=session_id,
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=DataQualityRepository(db),
        db_path=db,
        provenance_registry=provenance,
        **repos,
    )
    agent = CognitiveAgentRuntime(
        session_id=session_id,
        run_id=session_id,
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=60),
        graph_deps=deps,
        registry=_fake_registry(fake),
        checkpointer_path=tmp_path / f"{session_id}_ckpt.db",
    )
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id=session_id,
            broker_account_id="paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
        agent_runtime=agent,
    )
    await supervisor.start()
    assert supervisor.execution_runtime is not None
    deps.execution_runtime = supervisor.execution_runtime
    deps.data_quality_repo = supervisor.data_quality_repository
    deps.option_surface_repo = supervisor.option_surface_repository
    deps.snapshot_repo = supervisor.snapshot_repository
    deps.event_bus = supervisor.event_bus

    async def _submit(provenanced):
        return await supervisor.execution_runtime.submit_execution_command(
            provenanced.command
        )

    async def _projection():
        return await supervisor.execution_runtime.project_session()

    deps.submit_callback = _submit
    deps.projection_loader = _projection
    ensure_order_action_gateway(deps)
    assert deps.order_action_gateway is not None
    original_submit = deps.order_action_gateway.submit

    gateway_blocks: list[str] = []

    async def _tracking_submit(request):
        result = await original_submit(request)
        if result.submitted:
            if request.action.value in {"entry", "probe"}:
                gateway_entry_ids.append(result.client_order_id)
            if request.action.value == "exit":
                gateway_exit_ids.append(result.client_order_id)
        elif result.blocked_reason:
            gateway_blocks.append(
                f"{request.action.value}:{result.blocked_reason}"
            )
        return result

    deps.order_action_gateway.submit = _tracking_submit  # type: ignore[method-assign]

    evolution = EvolutionRuntime(
        db_path=db,
        settings=settings or acceptance_settings(),
        session_id=session_id,
        run_id=session_id,
        event_bus=supervisor.event_bus,
        execution_runtime=supervisor.execution_runtime,
        model_router=router,
        cognitive_graph_deps=deps,
    )
    await evolution.prepare()
    evolution.subscribe_events()
    agent.bind_evolution_runtime(evolution)
    if start_workers:
        await evolution.start_workers()
    if not start_orchestrator_worker and evolution.orchestrator is not None:
        evolution.orchestrator.pause()
    await evolution.resume()

    return {
        "db": db,
        "broker": broker,
        "clock": clock,
        "start": start,
        "supervisor": supervisor,
        "agent": agent,
        "evolution": evolution,
        "fake": fake,
        "router": router,
        "deps": deps,
        "gateway_entry_ids": gateway_entry_ids,
        "gateway_exit_ids": gateway_exit_ids,
        "gateway_blocks": gateway_blocks,
    }


async def _wait_for_position(supervisor, *, want_open: bool, attempts: int = 50) -> None:
    for _ in range(attempts):
        projection = await supervisor.execution_runtime.project_session()
        pos = projection.positions.get(CONTRACT_ID)
        if want_open:
            if pos is not None and pos.quantity != 0:
                return
        elif pos is not None and pos.quantity == 0:
            return
        await asyncio.sleep(0.15)
    state = "open" if want_open else "closed"
    raise AssertionError(f"position never became {state}")


async def _wait_for_gateway_progress(
    stack: dict,
    *,
    key: str,
    before: int,
    attempts: int = 120,
) -> None:
    """Wait until the tracked gateway id list grows (entry or exit)."""
    for _ in range(attempts):
        if len(stack[key]) > before:
            return
        await asyncio.sleep(0.15)
    raise AssertionError(
        f"{key} did not progress through OrderActionGateway "
        f"(before={before}, after={len(stack[key])})"
    )


async def _wait_for_closed_exit(
    stack: dict,
    *,
    exits_before: int,
    trade_index: int,
    attempts: int = 120,
) -> None:
    """Require a new gateway EXIT and a flat position; re-stimulate if stalled."""
    supervisor = stack["supervisor"]
    fake = stack["fake"]
    clock: FrozenExchangeClock = stack["clock"]
    session_id = stack["evolution"].session_id
    for i in range(attempts):
        projection = await supervisor.execution_runtime.project_session()
        pos = projection.positions.get(CONTRACT_ID)
        flat = pos is not None and pos.quantity == 0
        exited = len(stack["gateway_exit_ids"]) > exits_before
        if exited and flat:
            return
        # Re-mint EXIT canned + tick periodically so a stalled position cycle recovers.
        if i > 0 and i % 15 == 0:
            now = clock.now()
            tick = await supervisor.market_runtime.tick(now=now + timedelta(seconds=1))
            clock.set_now(now + timedelta(seconds=1))
            if tick.snapshot is not None:
                _mint_position_exit_canned(
                    fake,
                    session_id=session_id,
                    snapshot_id=tick.snapshot.snapshot_id,
                    trade_index=trade_index,
                )
                register_evolution_router_canned(fake)
        await asyncio.sleep(0.15)
    raise AssertionError(
        "EXIT must pass through OrderActionGateway "
        f"(before={exits_before}, after={len(stack['gateway_exit_ids'])})"
    )


def _mint_position_exit_canned(
    fake: FakeModelProvider,
    *,
    session_id: str,
    snapshot_id,
    trade_index: int,
) -> None:
    """Switch position factories to EXIT with fresh per-invocation artifact ids."""
    del snapshot_id, trade_index  # snapshot/model ids come from each ModelRequest
    ctl = install_paper_path_factories(fake, session_id=session_id)
    ctl.position_action = PositionAction.EXIT


async def ensure_flat_position(stack: dict, *, trade_index: int = 99) -> None:
    """Exit any open contract position so a subsequent entry proof starts flat."""
    supervisor = stack["supervisor"]
    fake = stack["fake"]
    clock: FrozenExchangeClock = stack["clock"]
    start: datetime = stack["start"]
    session_id = stack["evolution"].session_id
    projection = await supervisor.execution_runtime.project_session()
    pos = projection.positions.get(CONTRACT_ID)
    if pos is None or pos.quantity == 0:
        return
    exits_before = len(stack["gateway_exit_ids"])
    exit_start = clock.now() + timedelta(minutes=1)
    clock.set_now(exit_start)
    await supervisor.market_runtime.ingest_underlying_quote(
        symbol="SPY",
        bid=Decimal("499.50"),
        ask=Decimal("499.70"),
        last=Decimal("499.60"),
        source_timestamp=exit_start,
        received_timestamp=exit_start,
    )
    await supervisor.market_runtime.ingest_option_quotes(
        [
            {
                "contract_id": CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "0.80",
                "ask": "1.00",
                "last": "0.90",
                "quote_timestamp": exit_start,
                "is_0dte": True,
            },
            {
                "contract_id": FAR_CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "580",
                "option_type": "call",
                "bid": "0.03",
                "ask": "0.12",
                "last": "0.08",
                "quote_timestamp": exit_start,
                "is_0dte": True,
            },
        ]
    )
    register_evolution_router_canned(fake)
    exit_tick = await supervisor.market_runtime.tick(
        now=exit_start + timedelta(seconds=3)
    )
    assert exit_tick.snapshot is not None
    _mint_position_exit_canned(
        fake,
        session_id=session_id,
        snapshot_id=exit_tick.snapshot.snapshot_id,
        trade_index=trade_index,
    )
    register_evolution_router_canned(fake)
    await _wait_for_closed_exit(
        stack,
        exits_before=exits_before,
        trade_index=trade_index,
        attempts=120,
    )


def _evolution_queues_idle(evolution: EvolutionRuntime) -> bool:
    """True when episode/eval queues are empty and no compile/eval is in-flight."""
    idle = getattr(evolution, "workers_idle", None)
    if callable(idle):
        return bool(idle())
    ep_q = getattr(evolution, "_episode_queue", None)
    ev_q = getattr(evolution, "_eval_queue", None)
    ep_idle = ep_q is None or ep_q.empty()
    ev_idle = ev_q is None or ev_q.empty()
    index_idle = not bool(getattr(evolution, "_index_tasks", None))
    return ep_idle and ev_idle and index_idle


async def settle_after_closed_trade(
    stack: dict, *, min_closed_episodes: int, timeout: float = 60.0
) -> None:
    """Wait for compile/eval of closed trades before starting another entry.

    Without this barrier the next paper entry may pause/cancel evolution workers
    mid-evaluation (queue.empty() is not sufficient while a job is in-flight).
    """
    evolution: EvolutionRuntime = stack["evolution"]
    session_id = evolution.session_id
    await stack["supervisor"].event_bus.drain(timeout=5.0)
    closed = await wait_for_closed_episodes(
        evolution, session_id, min_closed_episodes
    )
    await wait_for_evaluations(evolution, closed)
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if _evolution_queues_idle(evolution):
            break
        await asyncio.sleep(0.1)
    await wait_ready_for_new_entry(stack, trade_index=min_closed_episodes + 90)


async def wait_ready_for_new_entry(
    stack: dict, *, trade_index: int = 98, timeout: float = 45.0
) -> None:
    """Ensure flat book, no working entry, and agent idle enough for a new entry."""
    from joker.runtime.order_action_gateway import (
        has_working_entry_order,
        working_orders_from_projection,
    )

    await ensure_flat_position(stack, trade_index=trade_index)
    agent: CognitiveAgentRuntime = stack["agent"]
    supervisor = stack["supervisor"]
    evolution: EvolutionRuntime = stack["evolution"]
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await supervisor.event_bus.drain(timeout=2.0)
        projection = await supervisor.execution_runtime.project_session()
        pos = projection.positions.get(CONTRACT_ID)
        open_pos = pos is not None and pos.quantity != 0
        working = has_working_entry_order(working_orders_from_projection(projection))
        in_flight = bool(getattr(agent, "_new_entry_in_flight", False))
        evo_idle = _evolution_queues_idle(evolution)
        if not open_pos and not working and not in_flight and evo_idle:
            return
        if open_pos:
            await ensure_flat_position(stack, trade_index=trade_index)
        await asyncio.sleep(0.1)
    raise AssertionError("session never became ready for a new paper entry")


async def run_open_trade_entry_only(
    stack: dict,
    *,
    trade_index: int,
    minute_offset: int,
) -> None:
    """Open a paper position via gateway and stop after entry (no exit leg)."""
    supervisor = stack["supervisor"]
    fake = stack["fake"]
    clock: FrozenExchangeClock = stack["clock"]
    start: datetime = stack["start"]
    session_id = stack["evolution"].session_id
    await wait_ready_for_new_entry(stack, trade_index=trade_index)
    install_paper_path_factories(fake, session_id=session_id)
    set_position_action(fake, PositionAction.HOLD)
    entries_before = len(stack["gateway_entry_ids"])

    entry_start = start + timedelta(minutes=minute_offset)
    clock.set_now(entry_start)
    snapshot = await seed_market_surface(supervisor.market_runtime, entry_start, clock)
    register_full_path_canned(
        fake,
        snapshot.snapshot_id,
        f"cycle-entry-{trade_index}-{uuid4().hex[:8]}",
        session=session_id,
        position_action=PositionAction.HOLD,
    )
    _refresh_trade_canned(fake, trade_index)
    register_evolution_router_canned(fake)
    await stack["supervisor"].event_bus.drain(timeout=5.0)
    for i in range(120):
        if len(stack["gateway_entry_ids"]) > entries_before:
            break
        if i > 0 and i % 15 == 0:
            # Stale open position routes snapshots to the position graph, not entry.
            projection = await supervisor.execution_runtime.project_session()
            pos = projection.positions.get(CONTRACT_ID)
            if pos is not None and pos.quantity != 0:
                await ensure_flat_position(stack, trade_index=trade_index)
                set_position_action(fake, PositionAction.HOLD)
            now = clock.now()
            tick = await supervisor.market_runtime.tick(now=now + timedelta(seconds=1))
            clock.set_now(now + timedelta(seconds=1))
            if tick.snapshot is not None:
                register_full_path_canned(
                    fake,
                    tick.snapshot.snapshot_id,
                    f"cycle-entry-retry-{trade_index}-{i}",
                    session=session_id,
                    position_action=PositionAction.HOLD,
                )
                register_evolution_router_canned(fake)
        await asyncio.sleep(0.15)
    else:
        raise AssertionError(
            "gateway_entry_ids did not progress through OrderActionGateway "
            f"(before={entries_before}, after={len(stack['gateway_entry_ids'])})"
        )
    await _wait_for_position(supervisor, want_open=True, attempts=100)
    projection = await supervisor.execution_runtime.project_session()
    pos = projection.positions.get(CONTRACT_ID)
    assert pos is not None and pos.quantity != 0, "open entry must leave a live position"


async def run_closed_trade_round_trip(
    stack: dict,
    *,
    trade_index: int,
    minute_offset: int,
) -> None:
    supervisor = stack["supervisor"]
    fake = stack["fake"]
    clock: FrozenExchangeClock = stack["clock"]
    start: datetime = stack["start"]
    evolution: EvolutionRuntime = stack["evolution"]
    session_id = evolution.session_id
    # Require a fresh gateway ENTRY — do not treat a leftover open position as success.
    await wait_ready_for_new_entry(stack, trade_index=trade_index)
    # Pause evolution workers during entry so cognitive cycles own the fake
    # provider / SQLite file; keep ingestion so POSITION_CLOSED can enqueue.
    # Never pause while compile/eval is in-flight — cancel drops dequeued jobs.
    workers_were_running = bool(getattr(evolution, "_workers_started", False))
    if workers_were_running:
        deadline = asyncio.get_event_loop().time() + 45.0
        while asyncio.get_event_loop().time() < deadline:
            if _evolution_queues_idle(evolution):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError(
                "evolution workers not idle before pause; refusing to cancel in-flight jobs"
            )
        await evolution.pause_workers()
    install_paper_path_factories(fake, session_id=session_id)
    set_position_action(fake, PositionAction.HOLD)
    entries_before = len(stack["gateway_entry_ids"])
    exits_before = len(stack["gateway_exit_ids"])
    blocks_before = len(stack.get("gateway_blocks") or [])

    entry_start = start + timedelta(minutes=minute_offset)
    clock.set_now(entry_start)
    snapshot = await seed_market_surface(supervisor.market_runtime, entry_start, clock)
    register_full_path_canned(
        fake,
        snapshot.snapshot_id,
        f"cycle-entry-{trade_index}-{uuid4().hex[:8]}",
        session=session_id,
        position_action=PositionAction.HOLD,
    )
    _refresh_trade_canned(fake, trade_index)
    register_evolution_router_canned(fake)
    # Immediate tick so the agent sees the seeded surface without waiting.
    tick0 = await supervisor.market_runtime.tick(
        now=entry_start + timedelta(seconds=3)
    )
    if tick0.snapshot is not None:
        register_full_path_canned(
            fake,
            tick0.snapshot.snapshot_id,
            f"cycle-entry-tick0-{trade_index}",
            session=session_id,
            position_action=PositionAction.HOLD,
        )
    await stack["supervisor"].event_bus.drain(timeout=5.0)
    try:
        for i in range(320):
            if len(stack["gateway_entry_ids"]) > entries_before:
                break
            if i % 4 == 0:
                projection = await supervisor.execution_runtime.project_session()
                pos = projection.positions.get(CONTRACT_ID)
                if pos is not None and pos.quantity != 0:
                    await ensure_flat_position(stack, trade_index=trade_index)
                    set_position_action(fake, PositionAction.HOLD)
                now = clock.now()
                tick = await supervisor.market_runtime.tick(
                    now=now + timedelta(seconds=1)
                )
                clock.set_now(now + timedelta(seconds=1))
                if tick.snapshot is not None:
                    register_full_path_canned(
                        fake,
                        tick.snapshot.snapshot_id,
                        f"cycle-entry-retry-{trade_index}-{i}",
                        session=session_id,
                        position_action=PositionAction.HOLD,
                    )
                    register_evolution_router_canned(fake)
            await asyncio.sleep(0.15)
        else:
            blocks = (stack.get("gateway_blocks") or [])[blocks_before:]
            raise AssertionError(
                "gateway_entry_ids did not progress through OrderActionGateway "
                f"(before={entries_before}, after={len(stack['gateway_entry_ids'])}, "
                f"blocks={blocks!r})"
            )
    finally:
        if workers_were_running:
            await evolution.resume_workers()

    await _wait_for_position(supervisor, want_open=True, attempts=100)

    exit_start = entry_start + timedelta(minutes=8)
    clock.set_now(exit_start)
    await supervisor.market_runtime.ingest_underlying_quote(
        symbol="SPY",
        bid=Decimal("499.50"),
        ask=Decimal("499.70"),
        last=Decimal("499.60"),
        source_timestamp=exit_start,
        received_timestamp=exit_start,
    )
    await supervisor.market_runtime.ingest_option_quotes(
        [
            {
                "contract_id": CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "0.80",
                "ask": "1.00",
                "last": "0.90",
                "quote_timestamp": exit_start,
                "is_0dte": True,
            },
            {
                "contract_id": FAR_CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "580",
                "option_type": "call",
                "bid": "0.03",
                "ask": "0.12",
                "last": "0.08",
                "quote_timestamp": exit_start,
                "is_0dte": True,
            },
        ]
    )
    _mint_position_exit_canned(
        fake,
        session_id=session_id,
        snapshot_id=snapshot.snapshot_id,
        trade_index=trade_index,
    )
    register_evolution_router_canned(fake)
    exit_tick = await supervisor.market_runtime.tick(
        now=exit_start + timedelta(seconds=3)
    )
    assert exit_tick.snapshot is not None
    _mint_position_exit_canned(
        fake,
        session_id=session_id,
        snapshot_id=exit_tick.snapshot.snapshot_id,
        trade_index=trade_index,
    )
    await _wait_for_closed_exit(
        stack, exits_before=exits_before, trade_index=trade_index, attempts=120
    )
    # Barrier so the next round-trip does not race episode compile / evaluation.
    await settle_after_closed_trade(
        stack, min_closed_episodes=trade_index + 1, timeout=60.0
    )


async def wait_for_closed_episodes(evolution: EvolutionRuntime, session_id: str, count: int):
    for _ in range(160):
        episodes = await evolution._repos["episodes"].list_by_session(session_id)
        closed = [e for e in episodes if e.action_class == "closed_trade" and e.completed]
        if len(closed) >= count:
            return closed[:count]
        await asyncio.sleep(0.15)
    raise AssertionError(f"expected {count} closed_trade episodes")


async def wait_for_evaluations(evolution: EvolutionRuntime, episodes) -> list:
    out = []
    for episode in episodes:
        for _ in range(300):
            evaluations = await evolution._repos["evaluations"].list_by_episode(
                episode.episode_id
            )
            if evaluations:
                out.extend(evaluations)
                break
            await asyncio.sleep(0.1)
        else:
            raise AssertionError(
                f"automatic evaluation never persisted for {episode.episode_id}"
            )
    return out


async def feed_shadow_snapshots_via_market(stack: dict, *, cycles: int = 3) -> None:
    """Publish Task 1 snapshots and run cognitive cycles for active shadow assignments."""
    evolution: EvolutionRuntime = stack["evolution"]
    assignments = await evolution._repos["shadow"].list_active()
    if not assignments:
        return
    supervisor = stack["supervisor"]
    agent: CognitiveAgentRuntime = stack["agent"]
    fake = stack["fake"]
    clock: FrozenExchangeClock = stack["clock"]
    start: datetime = stack["start"]
    session_id = evolution.session_id
    base = start + timedelta(minutes=40)

    async def _wait_shadow_idle(*, timeout: float = 30.0) -> None:
        if evolution.shadow is None:
            return
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if evolution.shadow.backlog == 0:
                return
            await asyncio.sleep(0.05)

    # Market ticks create real Task 1 snapshots for shadow, but must not open live
    # paper entries (that pollutes gateway counts and blocks later entry proofs).
    agent.suppress_new_entry_snapshots(True)
    try:
        for i in range(cycles):
            # Do not re-register canned outputs while a prior shadow cycle is mid-graph.
            await _wait_shadow_idle()
            ts = base + timedelta(minutes=i * 4)
            clock.set_now(ts)
            await supervisor.market_runtime.ingest_underlying_quote(
                symbol="SPY",
                bid=Decimal("499.80"),
                ask=Decimal("500.00"),
                last=Decimal("499.90"),
                source_timestamp=ts,
                received_timestamp=ts,
            )
            await supervisor.market_runtime.ingest_option_quotes(
                [
                    {
                        "contract_id": CONTRACT_ID,
                        "symbol": "SPY",
                        "expiry": date(2026, 7, 1),
                        "strike": "500",
                        "option_type": "call",
                        "bid": "1.00",
                        "ask": "1.20",
                        "last": "1.10",
                        "quote_timestamp": ts,
                        "is_0dte": True,
                    }
                ]
            )
            tick = await supervisor.market_runtime.tick(now=ts + timedelta(seconds=3))
            assert tick.snapshot is not None
            register_full_path_canned(
                fake,
                tick.snapshot.snapshot_id,
                f"shadow-feed-{i}",
                session=session_id,
                position_action=PositionAction.HOLD,
            )
            register_evolution_router_canned(fake)
            before = len(evolution.shadow.results) if evolution.shadow is not None else 0
            for assignment in assignments:
                await evolution.shadow.enqueue_snapshot(
                    assignment_id=assignment.assignment_id,
                    challenger_version_id=assignment.challenger_version_id,
                    snapshot_id=str(tick.snapshot.snapshot_id),
                    payload={"snapshot_id": str(tick.snapshot.snapshot_id)},
                    coalesce=False,
                )
            await supervisor.event_bus.drain(timeout=5.0)
            deadline = asyncio.get_event_loop().time() + 30.0
            while asyncio.get_event_loop().time() < deadline:
                if evolution.shadow is None:
                    break
                if (
                    evolution.shadow.backlog == 0
                    and len(evolution.shadow.results) >= before + len(assignments)
                ):
                    break
                await asyncio.sleep(0.05)
            else:
                await _wait_shadow_idle()
    finally:
        agent.suppress_new_entry_snapshots(False)


async def _wait_shadow_threshold(
    evolution: EvolutionRuntime,
    *,
    min_cycles: int | None = None,
    min_traded: int | None = None,
    timeout: float = 20.0,
) -> None:
    settings = evolution.settings
    need_cycles = (
        min_cycles
        if min_cycles is not None
        else int(settings.shadow.minimum_completed_cycles)
    )
    need_traded = (
        min_traded
        if min_traded is not None
        else int(settings.shadow.minimum_traded_cycles)
    )
    assignments = await evolution._repos["shadow"].list_active()
    if not assignments:
        return
    assignment = assignments[0]
    for _ in range(int(timeout * 10)):
        observed = await evolution.shadow_ledger.count_cycles(assignment.assignment_id)
        traded = await evolution.shadow_ledger.count_traded_cycles(assignment.assignment_id)
        if observed >= need_cycles and traded >= need_traded:
            return
        await asyncio.sleep(0.1)


async def wait_for_automatic_evolution(
    stack: dict, *, timeout: float = 180.0, poll_interval: float = 0.25
):
    """Drive the orchestrator to a terminal cycle without concurrent worker races.

    The background orchestrator worker is paused while this helper feeds shadow
    evidence and calls ``advance`` so two coroutines cannot advance the same
    cycle concurrently (observed as champion/challenger identity mismatches).
    """
    evolution: EvolutionRuntime = stack["evolution"]
    assert evolution.orchestrator is not None
    evolution.orchestrator.pause()
    deadline = asyncio.get_event_loop().time() + timeout
    shadow_feeds = 0
    last_state: EvolutionCycleState | None = None
    # Kick once under pause: start/resume the cycle from this helper only.
    await evolution.orchestrator.tick()
    while asyncio.get_event_loop().time() < deadline:
        records = await evolution._repos["evolution_cycles"].list_by_session(
            evolution.session_id
        )
        for record in records:
            state = EvolutionCycleState.from_record(record)
            last_state = state
            if state.status in {"completed", "failed", "blocked"}:
                return state
            if state.stage == "collect_shadow_evidence" and state.status == "running":
                if shadow_feeds < 12:
                    await feed_shadow_snapshots_via_market(stack, cycles=3)
                    await _wait_shadow_threshold(evolution, timeout=30.0)
                    shadow_feeds += 1
                state = await evolution.orchestrator.advance(state)
                last_state = state
                if state.status in {"completed", "failed", "blocked"}:
                    return state
                continue
            state = await evolution.orchestrator.advance(state)
            last_state = state
            if state.status in {"completed", "failed", "blocked"}:
                return state
        started = await evolution.orchestrator.maybe_start_cycle()
        if started is not None:
            last_state = await evolution.orchestrator.advance(started)
            if last_state.status in {"completed", "failed", "blocked"}:
                return last_state
        await asyncio.sleep(poll_interval)
    detail = "no cycle"
    if last_state is not None:
        detail = (
            f"stage={last_state.stage} status={last_state.status} "
            f"failures={last_state.failure_codes}"
        )
    raise AssertionError(
        f"automatic orchestrator did not reach a terminal cycle ({detail})"
    )


async def drain_evolution_orchestrator(stack: dict, *, max_rounds: int = 30):
    """Legacy manual drain retained for restart tests; prefer wait_for_automatic_evolution."""
    evolution: EvolutionRuntime = stack["evolution"]
    assert evolution.orchestrator is not None
    evolution.orchestrator.resume_scheduling()

    state = await evolution.orchestrator.maybe_start_cycle()
    if state is None:
        raise AssertionError("orchestrator cycle did not start")

    shadow_feed_batches = 0
    for _ in range(max_rounds):
        rows = await evolution._repos["evolution_cycles"].list_resumable(
            evolution.session_id
        )
        if rows:
            state = EvolutionCycleState.from_record(rows[0])

        if state.stage == "collect_shadow_evidence" and state.status == "running":
            if shadow_feed_batches < 6:
                await feed_shadow_snapshots_via_market(stack, cycles=4)
                await _wait_shadow_threshold(evolution, timeout=30.0)
                shadow_feed_batches += 1
            await asyncio.sleep(0.2)

        state = await evolution.orchestrator.advance(state)

        if state.stage in {"completed", "finalise_cycle"} and state.status in {
            "completed",
            "failed",
            "blocked",
        }:
            return state

        if state.stage == "collect_shadow_evidence" and state.status == "running":
            continue

    raise AssertionError(
        "orchestrator did not reach terminal state "
        f"(stage={state.stage}, status={state.status})"
    )


async def _wait_for_no_aiosqlite_workers(timeout_seconds: float = 5.0) -> None:
    """Drain/join until no aiosqlite workers remain, or fail with their names."""
    from joker.persistence.aiosqlite_lifecycle import wait_for_no_aiosqlite_workers

    await wait_for_no_aiosqlite_workers(timeout_seconds=timeout_seconds)


async def rebuild_paper_evolution_stack(
    tmp_path,
    *,
    db,
    session_id: str,
    settings: EvolutionSettings | None = None,
    start_orchestrator_worker: bool = True,
    start_workers: bool = True,
):
    """Fresh process: rebuild supervisor/agent/evolution on an existing session db."""
    return await build_paper_evolution_stack(
        tmp_path,
        session_id=session_id,
        settings=settings,
        start_orchestrator_worker=start_orchestrator_worker,
        start_workers=start_workers,
        db_path=db,
    )


async def shutdown_stack(stack: dict, *, strict_workers: bool = False) -> None:
    """Shut down stack components.

    Default is best-effort worker drain. Mid-suite tests share a session-scoped
    event loop; asserting zero aiosqlite workers here deadlocks or flakes on CI
    when earlier tests leave checkpointer connections alive.
    """
    await stack["evolution"].shutdown()
    await stack["agent"].shutdown()
    await stack["supervisor"].shutdown()
    if strict_workers:
        await _wait_for_no_aiosqlite_workers(timeout_seconds=5.0)
        from joker.persistence.aiosqlite_lifecycle import iter_aiosqlite_worker_threads

        assert not iter_aiosqlite_worker_threads()
    else:
        from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers

        await drain_aiosqlite_workers(timeout=0.5)
