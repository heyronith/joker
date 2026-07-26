"""Tests for Task 2 final remediation: surface, gateway, conflicts, recovery."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.agents.cognitive.execution import (
    build_truth_from_deps,
    validate_and_compile_proposal,
)
from joker.broker.interface import PaperBroker
from joker.cognition.schemas import (
    ExecutionLeg,
    ExecutionProposal,
    MetaDecisionAction,
    PositionAction,
)
from joker.config.settings import CognitiveGraphSettings
from joker.data.webull_options_provider import SurfaceFetchResult
from joker.events.schemas import EventType, make_event
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
from joker.market.data_quality_store import DataQualityRepository
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.persistence.cognitive_cycle_registry import CognitiveCycleRegistry
from joker.runtime.cognitive_agent_runtime import (
    CognitiveAgentRuntime,
    build_default_repositories,
)
from joker.runtime.execution_runtime import ExecutionCommand, contract_id_for
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.order_action_gateway import (
    OrderActionGateway,
    OrderActionKind,
    OrderActionRequest,
    has_working_entry_order,
    working_orders_from_projection,
)
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.schemas.domain import OptionContract, OrderIntent
from joker.schemas.options_data import OptionContractMetadata, OptionSnapshot
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned

ET = ZoneInfo("America/New_York")
FAR_CONTRACT_ID = "SPY:2026-07-01:580.0:call"


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


async def _seed_with_far_contract(market, start, clock):
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
    assert tick.snapshot.option_surface_id is not None
    return tick.snapshot


@pytest.mark.asyncio
async def test_far_from_atm_contract_persisted_and_valid_for_entry(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "far.db"
    broker = PaperBroker(slippage_pct=0)
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id="far",
            broker_account_id="paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
    )
    await supervisor.start()
    snapshot = await _seed_with_far_contract(supervisor.market_runtime, start, clock)
    surface = await supervisor.option_surface_repository.get_by_id(
        snapshot.option_surface_id
    )
    assert surface is not None
    ids = {c.contract_id for c in surface.contracts}
    assert FAR_CONTRACT_ID in ids

    fake = FakeModelProvider(available=True)
    register_full_path_canned(
        fake, snapshot.snapshot_id, "c-far", session="far", contract_id=FAR_CONTRACT_ID
    )
    # Override entry tactician to the far contract explicitly.
    from tests.cognitive.task2_canned import utc_now
    from joker.cognition.schemas import AgentRole
    from uuid import uuid4 as _uuid4

    mc = _uuid4()
    # Reuse canned registration; ensure proposal uses FAR_CONTRACT_ID.
    proposal = ExecutionProposal(
        proposal_id=_uuid4(),
        decision_id=_uuid4(),
        strategy_id=_uuid4(),
        session_id="far",
        cycle_id="c-far",
        snapshot_id=snapshot.snapshot_id,
        action="execute",
        legs=(
            ExecutionLeg(
                contract_id=FAR_CONTRACT_ID,
                side="buy",
                quantity=1,
                limit_price=Decimal("0.10"),
                sequence_order=0,
                max_quote_age_seconds=3600,
                replacement_policy="none",
                partial_fill_policy="wait",
            ),
        ),
        order_type="limit",
        time_in_force="day",
        entry_rationale="far contract",
        prompt_version="2.0.0",
        model_call_id=mc,
    )
    dq = await supervisor.data_quality_repository.get_by_id(snapshot.data_quality_id)
    truth = build_truth_from_deps(
        snapshot=snapshot,
        data_quality=dq,
        option_surface=surface,
        projection=await supervisor.execution_runtime.project_session(),
    )
    provenanced = validate_and_compile_proposal(proposal, truth=truth)
    assert provenanced.command.intent.contract.strike == 580.0
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_working_entry_blocks_second_entry_order(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "conflict.db"
    broker = PaperBroker(slippage_pct=50.0)
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id="conflict",
            broker_account_id="paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
    )
    await supervisor.start()
    snapshot = await _seed_with_far_contract(supervisor.market_runtime, start, clock)
    # Working buy that does not fill.
    intent = OrderIntent(
        intent_id="working-entry-1",
        candidate_id="prop-a",
        contract=OptionContract(
            symbol="SPY",
            expiration=date(2026, 7, 1),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        ),
        side="buy",
        order_type="limit",
        quantity=1,
        limit_price=0.01,
    )
    await supervisor.execution_runtime.submit_execution_command(
        ExecutionCommand(client_order_id="working-entry-1", intent=intent)
    )
    projection = await supervisor.execution_runtime.project_session()
    working = working_orders_from_projection(projection)
    assert has_working_entry_order(working)

    repos = build_default_repositories(db)
    for r in repos.values():
        await r.initialize()
    fake = FakeModelProvider(available=True)
    register_full_path_canned(fake, snapshot.snapshot_id, "c2", session="conflict")
    registry = _fake_registry(fake)
    router = ModelRouter(registry, session_id="conflict")
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="conflict",
        run_id="conflict",
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=supervisor.data_quality_repository,
        execution_runtime=supervisor.execution_runtime,
        projection_loader=supervisor.execution_runtime.project_session,
        db_path=db,
        **repos,
    )
    deps.order_action_gateway = OrderActionGateway(deps)
    result = await deps.order_action_gateway.submit(
        OrderActionRequest(
            action=OrderActionKind.ENTRY,
            snapshot_id=str(snapshot.snapshot_id),
            contract_id=CONTRACT_ID,
            side="buy",
            quantity=1,
            client_order_id="second-entry",
            limit_price=1.10,
            proposal_id=str(uuid4()),
        )
    )
    assert result.submitted is False
    assert "working entry" in (result.blocked_reason or "").lower()
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_proposal_b_rejected_when_working_order_for_proposal_a(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "prop.db"
    broker = PaperBroker(slippage_pct=50.0)
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id="prop",
            broker_account_id="paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
    )
    await supervisor.start()
    snapshot = await _seed_with_far_contract(supervisor.market_runtime, start, clock)
    proposal_a = str(uuid4())
    await supervisor.execution_runtime.submit_execution_command(
        ExecutionCommand(
            client_order_id=f"cog-{proposal_a}-leg",
            intent=OrderIntent(
                intent_id=f"cog-{proposal_a}-leg",
                candidate_id=proposal_a,
                contract=OptionContract(
                    symbol="SPY",
                    expiration=date(2026, 7, 1),
                    strike=500.0,
                    option_type="call",
                    is_0dte=True,
                ),
                side="buy",
                order_type="limit",
                quantity=1,
                limit_price=0.01,
            ),
        )
    )
    surface = await supervisor.option_surface_repository.get_by_id(
        snapshot.option_surface_id
    )
    dq = await supervisor.data_quality_repository.get_by_id(snapshot.data_quality_id)
    projection = await supervisor.execution_runtime.project_session()
    working = working_orders_from_projection(
        projection, proposal_id_by_client_order={f"cog-{proposal_a}-leg": proposal_a}
    )
    truth = build_truth_from_deps(
        snapshot=snapshot,
        data_quality=dq,
        option_surface=surface,
        projection=projection,
    )
    # Inject proposal mapping into truth working orders.
    from dataclasses import replace
    from joker.runtime.order_action_gateway import WorkingOrderTruth

    mapped = tuple(
        WorkingOrderTruth(
            client_order_id=o.client_order_id,
            contract_id=o.contract_id,
            side=o.side,
            requested_quantity=o.requested_quantity,
            filled_quantity=o.filled_quantity,
            remaining_quantity=o.remaining_quantity,
            status=o.status,
            proposal_id=proposal_a if proposal_a in o.client_order_id else None,
        )
        for o in working
    )
    truth = replace(truth, working_orders=mapped)
    proposal_b = ExecutionProposal(
        proposal_id=uuid4(),
        decision_id=uuid4(),
        strategy_id=uuid4(),
        session_id="prop",
        cycle_id="c-b",
        snapshot_id=snapshot.snapshot_id,
        action="execute",
        legs=(
            ExecutionLeg(
                contract_id=CONTRACT_ID,
                side="buy",
                quantity=1,
                limit_price=Decimal("1.10"),
                sequence_order=0,
                max_quote_age_seconds=3600,
                replacement_policy="none",
                partial_fill_policy="wait",
            ),
        ),
        order_type="limit",
        time_in_force="day",
        entry_rationale="b",
        prompt_version="2.0.0",
        model_call_id=uuid4(),
    )
    with pytest.raises(Exception, match="working entry|conflicting"):
        validate_and_compile_proposal(proposal_b, truth=truth)
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_surface_fetch_result_records_partial_batches() -> None:
    result = SurfaceFetchResult(
        snapshots=[],
        discovered_count=10,
        selected_count=10,
        fetched_count=8,
        failed_batches=("batch[0:20]: missing_ids=a,b",),
        complete=False,
        trading_date=date(2026, 7, 1),
    )
    findings = result.to_data_quality_findings()
    assert findings
    assert findings[0].code.value == "partial_option_surface"


@pytest.mark.asyncio
async def test_partial_batch_without_exception_marks_incomplete() -> None:
    """20 requested → 18 returned without exception → complete=False → DQ error."""
    from joker.config.settings import EnvSettings
    from joker.data.webull_options_provider import WebullOptionsDataProvider
    from joker.schemas.options_data import OptionContractMetadata, OptionSnapshot

    exp = date(2026, 7, 1)
    contracts = [
        OptionContractMetadata(
            underlying_symbol="SPY",
            expiration=exp,
            strike=400.0 + i,
            option_type="call",
            contract_id=f"SPY{i}",
        )
        for i in range(20)
    ]
    snaps = {
        c.contract_id: OptionSnapshot(
            contract=c,
            bid=1.0,
            ask=1.1,
            last=1.05,
            quote_timestamp=datetime(2026, 7, 1, 14, 0, tzinfo=ET),
        )
        for c in contracts[:18]
    }

    class _PartialApi:
        CONTRACT_DISCOVERY_VERIFIED = True
        SNAPSHOT_VERIFIED = True

        def find_option_contracts(self, symbol, expiration):
            return contracts

        def get_option_snapshots(self, batch):
            # Return only contracts that exist in snaps — silent shortfall.
            out = []
            for c in batch:
                if c.contract_id in snaps:
                    out.append(snaps[c.contract_id])
            return out

    provider = WebullOptionsDataProvider(
        env=EnvSettings.model_construct(),
        api=_PartialApi(),  # type: ignore[arg-type]
    )
    result = provider.fetch_surface_snapshots(500.0, trading_date=exp, batch_size=20)
    assert result.selected_count == 20
    assert result.fetched_count == 18
    assert result.complete is False
    findings = result.to_data_quality_findings()
    assert findings
    assert findings[0].severity.value == "error"
    assert findings[0].code.value == "partial_option_surface"


@pytest.mark.asyncio
async def test_row_conversion_failure_marks_partial_surface() -> None:
    """Valid snapshots with one conversion failure → partial finding."""
    from joker.runtime.option_surface_ingest import convert_option_snapshots_to_surface_rows
    from joker.schemas.options_data import OptionContractMetadata, OptionSnapshot
    from unittest.mock import patch

    exp = date(2026, 7, 1)
    good = OptionSnapshot(
        contract=OptionContractMetadata(
            underlying_symbol="SPY",
            expiration=exp,
            strike=500.0,
            option_type="call",
            contract_id="good",
        ),
        bid=1.0,
        ask=1.1,
        quote_timestamp=datetime(2026, 7, 1, 14, 0, tzinfo=ET),
    )
    bad = OptionSnapshot(
        contract=OptionContractMetadata(
            underlying_symbol="SPY",
            expiration=exp,
            strike=501.0,
            option_type="call",
            contract_id="bad",
        ),
        bid=1.0,
        ask=1.1,
        quote_timestamp=datetime(2026, 7, 1, 14, 0, tzinfo=ET),
    )
    original = __import__(
        "joker.runtime.option_surface_ingest", fromlist=["option_snapshot_to_surface_row"]
    ).option_snapshot_to_surface_row

    def _maybe_fail(snap, *, trading_date=None):
        if snap.contract.contract_id == "bad":
            raise ValueError("invalid strike payload")
        return original(snap, trading_date=trading_date)

    with patch(
        "joker.runtime.option_surface_ingest.option_snapshot_to_surface_row",
        side_effect=_maybe_fail,
    ):
        conversion = convert_option_snapshots_to_surface_rows(
            [good, bad], trading_date=exp
        )
    assert conversion.converted_count == 1
    assert conversion.complete is False
    findings = conversion.to_data_quality_findings()
    assert findings
    assert findings[0].code.value == "partial_option_surface"
    assert findings[0].severity.value == "error"


@pytest.mark.asyncio
async def test_cycle_registry_resume_after_interrupt(tmp_path) -> None:
    """Unfinished cycle resumes on start() without re-delivering the event."""
    from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
    from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
    from joker.persistence.cognitive_cycle_registry import CognitiveCycleRecord

    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "resume.db"
    broker = PaperBroker(slippage_pct=0)
    session_id = "resume-stable"
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
    )
    await supervisor.start(start_agent=False)
    snapshot = await _seed_with_far_contract(supervisor.market_runtime, start, clock)
    fake = FakeModelProvider(available=True)
    cycle_id = "cycle-resume"
    register_full_path_canned(fake, snapshot.snapshot_id, cycle_id, session=session_id)
    registry = _fake_registry(fake)
    router = ModelRouter(registry, session_id=session_id)
    repos = build_default_repositories(db)
    for r in repos.values():
        await r.initialize()

    submitted: list[str] = []

    async def submit_callback(provenanced):
        submitted.append(provenanced.command.client_order_id)
        return await supervisor.execution_runtime.submit_execution_command(
            provenanced.command
        )

    ckpt_path = tmp_path / "resume_ckpt.db"
    cycle_reg = CognitiveCycleRegistry(tmp_path / "cycles.db")
    await cycle_reg.initialize()
    checkpointer = CognitiveCheckpointer(ckpt_path)
    saver = await checkpointer.open()

    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=60),
        session_id=session_id,
        run_id="run-1",
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=supervisor.data_quality_repository,
        execution_runtime=supervisor.execution_runtime,
        projection_loader=supervisor.execution_runtime.project_session,
        submit_callback=submit_callback,
        db_path=db,
        cycle_registry=cycle_reg,
        checkpointer=saver,
        **repos,
    )
    deps.order_action_gateway = OrderActionGateway(deps)

    # Interrupt after a middle node so a durable checkpoint exists.
    graph = build_cognitive_graph(deps)
    state = initial_cycle_state(
        session_id=session_id,
        run_id="run-1",
        cycle_id=cycle_id,
        trigger_event_id=str(uuid4()),
        trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
        snapshot_id=str(snapshot.snapshot_id),
    )
    config = ainvoke_config(
        session_id=session_id, graph_kind="decision", cycle_id=cycle_id
    )
    config = {
        **config,
        "configurable": {
            **config["configurable"],
            "checkpoint_ns": "",
        },
        "interrupt_before": ["synthesise_world_model"],
    }
    await graph.ainvoke(state, config=config)
    assert len(submitted) == 0
    thread_id = f"{session_id}:decision:{cycle_id}"
    await cycle_reg.upsert(
        CognitiveCycleRecord(
            session_id=session_id,
            graph_kind="decision",
            cycle_id=cycle_id,
            trigger_event_id=str(state["trigger_event_id"]),
            snapshot_id=str(snapshot.snapshot_id),
            status="running",
            checkpoint_thread_id=thread_id,
            last_completed_node="perception",
        )
    )
    await checkpointer.close()

    # New runtime — start() alone resumes; no event redelivery.
    deps2 = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=60),
        session_id=session_id,
        run_id="run-2",
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=supervisor.data_quality_repository,
        execution_runtime=supervisor.execution_runtime,
        projection_loader=supervisor.execution_runtime.project_session,
        submit_callback=submit_callback,
        db_path=db,
        cycle_registry=cycle_reg,
        submitted_proposal_ids=set(deps.submitted_proposal_ids),
        **repos,
    )
    deps2.order_action_gateway = OrderActionGateway(deps2)
    runtime2 = CognitiveAgentRuntime(
        session_id=session_id,
        run_id="run-2",
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=60),
        graph_deps=deps2,
        registry=registry,
        checkpointer_path=ckpt_path,
    )
    await runtime2.start()
    await asyncio.sleep(1.5)
    await runtime2.shutdown()
    projection = await supervisor.execution_runtime.project_session()
    assert projection.orders, "resumed cycle must submit exactly one order"
    assert len(projection.orders) == 1
    remaining = await cycle_reg.list_resumable(session_id)
    assert not any(r.cycle_id == cycle_id for r in remaining)
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_two_phase_supervisor_bind_before_agent_recovery(tmp_path) -> None:
    """Live ordering: Task1 start → bind → agent start resumes unfinished cycle."""
    from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
    from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
    from joker.persistence.cognitive_cycle_registry import CognitiveCycleRecord
    from joker.runtime.cognitive_binding import bind_cognitive_graph_to_task1
    from joker.runtime.cognitive_session import stable_cognitive_session_id

    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "twophase.db"
    broker = PaperBroker(slippage_pct=0)
    session_id = stable_cognitive_session_id(
        trading_date=date(2026, 7, 1),
        account_identity="local_paper",
        mode="paper",
    )
    fake = FakeModelProvider(available=True)
    registry = _fake_registry(fake)
    router = ModelRouter(registry, session_id=session_id)
    repos = build_default_repositories(db)
    for r in repos.values():
        await r.initialize()

    submitted: list[str] = []
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=60),
        session_id=session_id,
        run_id="audit-run-1",
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=DataQualityRepository(db),
        db_path=db,
        **repos,
    )
    runtime = CognitiveAgentRuntime(
        session_id=session_id,
        run_id="audit-run-1",
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=60),
        graph_deps=deps,
        registry=registry,
        checkpointer_path=tmp_path / "twophase_ckpt.db",
    )
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id=session_id,
            run_id="audit-run-1",
            broker_account_id="paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
        agent_runtime=runtime,
    )

    class _BridgeStub:
        def __init__(self, sup: SessionSupervisor) -> None:
            self.supervisor = sup

    # Phase 1 — Task 1 only.
    await supervisor.start(start_agent=False)
    assert runtime._started is False  # noqa: SLF001
    assert deps.execution_runtime is None
    bind_cognitive_graph_to_task1(
        deps,
        _BridgeStub(supervisor),  # type: ignore[arg-type]
        data_quality_repo=supervisor.data_quality_repository,
    )
    assert deps.execution_runtime is not None
    assert deps.order_action_gateway is not None

    async def _track(provenanced):
        submitted.append(provenanced.command.client_order_id)
        return await deps.execution_runtime.submit_execution_command(provenanced.command)

    deps.submit_callback = _track
    snapshot = await _seed_with_far_contract(supervisor.market_runtime, start, clock)
    cycle_id = "phase-cycle"
    register_full_path_canned(fake, snapshot.snapshot_id, cycle_id, session=session_id)

    # Persist an interrupted decision cycle before the agent is started.
    ckpt = CognitiveCheckpointer(tmp_path / "twophase_ckpt.db")
    saver = await ckpt.open()
    deps.checkpointer = saver
    graph = build_cognitive_graph(deps)
    state = initial_cycle_state(
        session_id=session_id,
        run_id="audit-run-1",
        cycle_id=cycle_id,
        trigger_event_id=str(uuid4()),
        trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
        snapshot_id=str(snapshot.snapshot_id),
    )
    config = ainvoke_config(
        session_id=session_id, graph_kind="decision", cycle_id=cycle_id
    )
    config = {**config, "interrupt_before": ["synthesise_world_model"]}
    await graph.ainvoke(state, config=config)
    assert submitted == []
    if deps.cycle_registry is None:
        deps.cycle_registry = CognitiveCycleRegistry(
            db.with_name(db.stem + "_cognitive_cycles.db")
        )
        await deps.cycle_registry.initialize()
    await deps.cycle_registry.upsert(
        CognitiveCycleRecord(
            session_id=session_id,
            graph_kind="decision",
            cycle_id=cycle_id,
            trigger_event_id=str(state["trigger_event_id"]),
            snapshot_id=str(snapshot.snapshot_id),
            status="running",
            checkpoint_thread_id=f"{session_id}:decision:{cycle_id}",
            last_completed_node="perception",
        )
    )
    await ckpt.close()
    deps.checkpointer = None

    # Phase 2 — start agent after bind; recovery runs with gateway present.
    await supervisor.start_agent_runtime()
    await asyncio.sleep(1.5)
    await runtime.shutdown()
    projection = await supervisor.execution_runtime.project_session()
    assert projection.orders, "recovered cycle must submit through bound gateway"
    assert len(projection.orders) == 1
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_position_reduce_then_exit_lifecycle(tmp_path) -> None:
    """Entry → REDUCE → later EXIT; both commands once; quantity reaches zero."""
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "lifecycle.db"
    broker = PaperBroker(slippage_pct=0)
    session_id = "lifecycle"
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
    )
    await supervisor.start()
    snapshot = await _seed_with_far_contract(supervisor.market_runtime, start, clock)

    intent = OrderIntent(
        intent_id="entry-2",
        candidate_id="entry-2",
        contract=OptionContract(
            symbol="SPY",
            expiration=date(2026, 7, 1),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        ),
        side="buy",
        order_type="limit",
        quantity=2,
        limit_price=1.10,
    )
    await supervisor.execution_runtime.submit_execution_command(
        ExecutionCommand(client_order_id="entry-2", intent=intent)
    )
    proj = await supervisor.execution_runtime.project_session()
    assert proj.positions[CONTRACT_ID].quantity == Decimal("2")

    repos = build_default_repositories(db)
    for r in repos.values():
        await r.initialize()
    fake = FakeModelProvider(available=True)
    registry = _fake_registry(fake)
    router = ModelRouter(registry, session_id=session_id)
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id=session_id,
        run_id=session_id,
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=supervisor.data_quality_repository,
        execution_runtime=supervisor.execution_runtime,
        projection_loader=supervisor.execution_runtime.project_session,
        db_path=db,
        **repos,
    )
    gateway = OrderActionGateway(deps)
    deps.order_action_gateway = gateway

    reduce_result = await gateway.submit(
        OrderActionRequest(
            action=OrderActionKind.REDUCE,
            snapshot_id=str(snapshot.snapshot_id),
            contract_id=CONTRACT_ID,
            side="sell",
            quantity=1,
            client_order_id="reduce-1",
            limit_price=1.20,
            max_quote_age_seconds=3600,
        )
    )
    assert reduce_result.submitted is True
    mid = await supervisor.execution_runtime.project_session()
    assert mid.positions[CONTRACT_ID].quantity == Decimal("1")

    later = start + timedelta(minutes=6)
    clock.set_now(later)
    await supervisor.market_runtime.ingest_underlying_quote(
        symbol="SPY",
        bid=Decimal("499.5"),
        ask=Decimal("499.7"),
        last=Decimal("499.6"),
        source_timestamp=later,
        received_timestamp=later,
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
                "quote_timestamp": later,
                "is_0dte": True,
            }
        ]
    )
    exit_tick = await supervisor.market_runtime.tick(now=later + timedelta(seconds=3))
    assert exit_tick.snapshot is not None

    exit_result = await gateway.submit(
        OrderActionRequest(
            action=OrderActionKind.EXIT,
            snapshot_id=str(exit_tick.snapshot.snapshot_id),
            contract_id=CONTRACT_ID,
            side="sell",
            quantity=1,
            client_order_id="exit-1",
            limit_price=1.20,
            max_quote_age_seconds=3600,
        )
    )
    assert exit_result.submitted is True
    # Duplicate EXIT must be rejected.
    dup = await gateway.submit(
        OrderActionRequest(
            action=OrderActionKind.EXIT,
            snapshot_id=str(exit_tick.snapshot.snapshot_id),
            contract_id=CONTRACT_ID,
            side="sell",
            quantity=1,
            client_order_id="exit-1",
            limit_price=1.20,
            max_quote_age_seconds=3600,
        )
    )
    assert dup.submitted is False

    final = await supervisor.execution_runtime.project_session()
    assert final.positions[CONTRACT_ID].quantity == Decimal("0")
    sell_ids = {
        o.client_order_id
        for o in final.orders.values()
        if getattr(o, "side", None) == "sell"
    }
    assert "reduce-1" in sell_ids
    assert "exit-1" in sell_ids
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_order_management_replace_idempotent_across_restart(tmp_path) -> None:
    """Replacement decision persisted → restart → same cycle → no second replace."""
    from joker.persistence.order_management_actions import (
        OrderManagementActionRecord,
        OrderManagementActionRepository,
        make_order_management_action_key,
    )

    db = tmp_path / "om.db"
    repo = OrderManagementActionRepository(tmp_path / "om_actions.db")
    await repo.initialize()
    key = make_order_management_action_key(
        source_order_id="ord-1",
        source_order_state="submitted",
        trigger_event_id="evt-1",
        decision_id="dec-1",
        action="replace",
    )
    first = await repo.record(
        OrderManagementActionRecord(
            action_key=key,
            session_id="om",
            source_order_id="ord-1",
            action="replace",
            source_order_state="submitted",
            trigger_event_id="evt-1",
            decision_id="dec-1",
            replacement_client_order_id="ord-1:replace:dec-1",
        )
    )
    assert first is True
    # Simulate process restart with a fresh repository handle on the same durable DB.
    repo2 = OrderManagementActionRepository(tmp_path / "om_actions.db")
    assert await repo2.has_key(key) is True
    second = await repo2.record(
        OrderManagementActionRecord(
            action_key=key,
            session_id="om",
            source_order_id="ord-1",
            action="replace",
            decision_id="dec-1",
        )
    )
    assert second is False
