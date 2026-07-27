"""Shared paper-session harness for Task 3 production acceptance tests."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from joker.broker.interface import PaperBroker
from joker.cognition.schemas import PositionAction, PositionThesisVersion
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
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.data_quality_store import DataQualityRepository
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
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
            require_known_cost=False,
            minimum_calibration_samples=0,
            require_brier_score=False,
            require_expected_calibration_error=False,
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


def _refresh_trade_canned(fake: FakeModelProvider, trade_index: int) -> None:
    """Ensure repeated paper round-trips do not reuse immutable artifact ids."""
    for role in ("position_thesis", "position_decision"):
        val = fake._by_role.get(role)
        if val is not None and hasattr(val, "model_copy"):
            fake.set_canned_for_role(
                role,
                val.model_copy(
                    update={
                        "thesis_version_id": uuid4(),
                        "model_call_id": uuid4(),
                    }
                ),
            )
    order_mgr = fake._by_role.get("order_manager")
    if order_mgr is not None and hasattr(order_mgr, "model_copy"):
        fake.set_canned_for_role(
            "order_manager",
            order_mgr.model_copy(
                update={
                    "client_order_id": f"order-{trade_index}",
                    "model_call_id": uuid4(),
                }
            ),
        )


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
    return ModelRouter(_fake_registry(fake), session_id=session_id), fake


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


async def build_paper_evolution_stack(
    tmp_path,
    *,
    session_id: str,
    settings: EvolutionSettings | None = None,
    start_orchestrator_worker: bool = True,
):
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / f"{session_id}.db"
    broker = PaperBroker(slippage_pct=0)
    gateway_entry_ids: list[str] = []
    gateway_exit_ids: list[str] = []

    router, fake = build_acceptance_router(session_id)
    repos = build_default_repositories(db)
    for repo in repos.values():
        await repo.initialize()
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

    async def _tracking_submit(request):
        result = await original_submit(request)
        if result.submitted:
            if request.action.value in {"entry", "probe"}:
                gateway_entry_ids.append(result.client_order_id)
            if request.action.value == "exit":
                gateway_exit_ids.append(result.client_order_id)
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


def _mint_position_exit_canned(
    fake: FakeModelProvider,
    *,
    session_id: str,
    snapshot_id,
    trade_index: int,
) -> None:
    """Always mint fresh artifact ids so concurrent/repeated position cycles never collide."""
    exit_thesis = PositionThesisVersion(
        position_id=CONTRACT_ID,
        contract_id=CONTRACT_ID,
        session_id=session_id,
        snapshot_id=snapshot_id,
        original_strategy_id=uuid4(),
        current_thesis=f"exit trade {trade_index}",
        recommended_action=PositionAction.EXIT,
        recommended_quantity=1,
        recommended_limit_price=Decimal("1.20"),
        confidence=0.7,
        prompt_version="2.0.0",
        model_call_id=uuid4(),
        thesis_version_id=uuid4(),
    )
    fake.set_canned_for_role("position_thesis", exit_thesis)
    fake.set_canned_for_role(
        "position_decision",
        exit_thesis.model_copy(
            update={
                "thesis_version_id": uuid4(),
                "model_call_id": uuid4(),
                "recommended_action": PositionAction.EXIT,
            }
        ),
    )


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
    session_id = stack["evolution"].session_id
    exits_before = len(stack["gateway_exit_ids"])

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
    await _wait_for_position(supervisor, want_open=True, attempts=80)
    assert stack["gateway_entry_ids"], "entry must pass through OrderActionGateway"

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
    await _wait_for_position(supervisor, want_open=False, attempts=80)
    assert len(stack["gateway_exit_ids"]) > exits_before, (
        "EXIT must pass through OrderActionGateway"
    )


async def wait_for_closed_episodes(evolution: EvolutionRuntime, session_id: str, count: int):
    for _ in range(80):
        episodes = await evolution._repos["episodes"].list_by_session(session_id)
        closed = [e for e in episodes if e.action_class == "closed_trade" and e.completed]
        if len(closed) >= count:
            return closed[:count]
        await asyncio.sleep(0.15)
    raise AssertionError(f"expected {count} closed_trade episodes")


async def wait_for_evaluations(evolution: EvolutionRuntime, episodes) -> list:
    out = []
    for episode in episodes:
        for _ in range(60):
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
    fake = stack["fake"]
    clock: FrozenExchangeClock = stack["clock"]
    start: datetime = stack["start"]
    session_id = evolution.session_id
    base = start + timedelta(minutes=40)
    for i in range(cycles):
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
        for assignment in assignments:
            await evolution.shadow.enqueue_snapshot(
                assignment_id=assignment.assignment_id,
                challenger_version_id=assignment.challenger_version_id,
                snapshot_id=str(tick.snapshot.snapshot_id),
                payload={"snapshot_id": str(tick.snapshot.snapshot_id)},
                coalesce=False,
            )
        await supervisor.event_bus.drain(timeout=5.0)
        await asyncio.sleep(0.5)


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


async def drain_evolution_orchestrator(stack: dict, *, max_rounds: int = 30):
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
            # Keep feeding while the node waits for shadow thresholds.
            continue

    assignments = await evolution._repos["shadow"].list_active()
    observed = traded = 0
    if assignments and evolution.shadow_ledger is not None:
        observed = await evolution.shadow_ledger.count_cycles(assignments[0].assignment_id)
        traded = await evolution.shadow_ledger.count_traded_cycles(
            assignments[0].assignment_id
        )
    raise AssertionError(
        "orchestrator did not reach terminal state "
        f"(stage={state.stage}, status={state.status}, shadow_cycles={observed}, "
        f"shadow_traded={traded})"
    )


async def shutdown_stack(stack: dict, *, strict_workers: bool = True) -> None:
    await stack["evolution"].shutdown()
    await stack["agent"].shutdown()
    await stack["supervisor"].shutdown()
    from joker.persistence.aiosqlite_lifecycle import (
        drain_aiosqlite_workers,
        iter_aiosqlite_worker_threads,
        join_aiosqlite_workers,
    )

    await drain_aiosqlite_workers()
    join_aiosqlite_workers()
    if strict_workers:
        assert not iter_aiosqlite_worker_threads()
