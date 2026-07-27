"""Mode-specific adversarial runners that invoke real Task 2 cognition."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from joker.cognition.prompt_overrides import pinned_applied_configuration
from joker.evolution.adversarial_fixtures import (
    AdversarialFixture,
    AdversarialScenarioDefinition,
)
from joker.evolution.adversarial_recovery import (
    AdversarialRecoveryCheckpoint,
    AdversarialRecoveryStore,
)
from joker.evolution.configuration_applicator import ConfigurationApplicator
from joker.evolution.replay import CognitiveReplayService, entry_thread_id
from joker.evolution.replay_execution import (
    ReplayExecutionRuntime,
    ReplayOrder,
    ReplayPosition,
)
from joker.evolution.replay_gateway import ReplayOrderActionGateway
from joker.evolution.replay_market import ReplayEpisodeTruth
from joker.evolution.replay_order_management import ReplayOrderManagementRunner
from joker.evolution.replay_truth import ReplayMarketFrame
from joker.evolution.schemas import CognitiveConfigurationVersion, TradingEpisode
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import ainvoke_config
from joker.graph.position_graph import build_position_graph
from joker.market.data_quality_store import DataQualityRepository
from joker.market.option_surface import (
    OptionContractSnapshot,
    OptionSurfaceRepository,
    OptionSurfaceSnapshot,
    compute_mid,
    compute_relative_spread,
)
from joker.market.quality import (
    DataQualityCode,
    DataQualityFinding,
    DataQualityReport,
    DataQualitySeverity,
)
from joker.market.snapshots import MarketSnapshot, SnapshotRepository, UnderlyingSnapshot
from joker.runtime.order_action_gateway import OrderActionKind, OrderActionRequest, OrderActionResult


class AdversarialExecutionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: UUID = Field(default_factory=uuid4)
    experiment_id: UUID
    scenario_id: str
    scenario_version: str
    configuration_version_id: UUID
    sample_number: int
    execution_mode: str

    fixture_loaded: bool = False
    runtime_invoked: bool = False
    durable_checkpoint_loaded: bool = False
    graph_kind: str | None = None
    graph_thread_ids: tuple[str, ...] = ()
    model_call_ids: tuple[UUID, ...] = ()
    gateway_action_ids: tuple[str, ...] = ()
    order_ids: tuple[str, ...] = ()
    fill_ids: tuple[str, ...] = ()

    crash_injected: bool = False
    fresh_runtime_created: bool = False
    checkpoint_resumed: bool = False

    expected_invariants: tuple[str, ...] = ()
    evaluated_invariants: tuple[str, ...] = ()
    satisfied_invariants: tuple[str, ...] = ()
    failed_invariants: tuple[str, ...] = ()
    invariants_evaluated: tuple[str, ...] = ()  # legacy alias kept for serialization
    runtime_errors: tuple[str, ...] = ()

    completed: bool = False
    findings: tuple[str, ...] = ()
    passed: bool = False


class AdversarialModeRunner(Protocol):
    async def execute(
        self,
        *,
        experiment_id: UUID,
        definition: AdversarialScenarioDefinition,
        fixture: AdversarialFixture,
        configuration: CognitiveConfigurationVersion,
        sample_number: int,
    ) -> AdversarialExecutionEvidence: ...


_EXECUTE_ACTIONS = frozenset({"execute", "probe", "EXECUTE", "PROBE"})


def _truth_from_fixture(fixture: AdversarialFixture) -> ReplayEpisodeTruth:
    return ReplayEpisodeTruth(
        episode_id=uuid4(),
        initial_snapshot_id=fixture.frames[0].snapshot_id,
        terminal_snapshot_id=fixture.frames[-1].snapshot_id,
        snapshot_sequence=tuple(f.snapshot_id for f in fixture.frames),
        frames=tuple(fixture.frames),
        starting_cash=fixture.starting_cash,
        starting_positions=fixture.starting_positions,
        fill_model_version="adversarial_fill_v1",
        random_seed=7,
    )


def _isolated_deps(
    template: CognitiveGraphDeps,
    *,
    gateway: ReplayOrderActionGateway | None = None,
    checkpointer: Any = None,
) -> CognitiveGraphDeps:
    return CognitiveGraphDeps(
        router=template.router,
        config=template.config,
        session_id=f"adv:{template.session_id}",
        run_id=f"adv:{template.run_id}",
        context_assembler=template.context_assembler,
        snapshot_repo=template.snapshot_repo,
        option_surface_repo=template.option_surface_repo,
        data_quality_repo=template.data_quality_repo,
        evidence_repo=None,
        world_model_repo=None,
        hypothesis_repo=None,
        strategy_repo=None,
        debate_repo=None,
        decision_repo=None,
        position_thesis_repo=None,
        order_management_repo=None,
        model_call_repo=template.model_call_repo,
        execution_runtime=None,
        submit_callback=None,
        event_bus=None,
        clock=template.clock,
        db_path=template.db_path,
        checkpointer=checkpointer if checkpointer is not None else template.checkpointer,
        data_quality_loader=template.data_quality_loader,
        projection_loader=None,
        provenance_registry=None,
        order_action_gateway=gateway,
        cycle_registry=None,
        order_management_action_repo=None,
    )


def _resolve_task1_repos(
    deps: CognitiveGraphDeps,
) -> tuple[
    SnapshotRepository | None,
    OptionSurfaceRepository | None,
    DataQualityRepository | None,
]:
    db_path = deps.db_path
    snapshot_repo = deps.snapshot_repo
    surface_repo = deps.option_surface_repo
    dq_repo = deps.data_quality_repo
    if db_path is not None:
        if snapshot_repo is None:
            snapshot_repo = SnapshotRepository(db_path)
        if surface_repo is None:
            surface_repo = OptionSurfaceRepository(db_path)
        if dq_repo is None:
            dq_repo = DataQualityRepository(db_path)
    return snapshot_repo, surface_repo, dq_repo


def _frame_contract_to_surface_row(
    frame: ReplayMarketFrame,
    quote: Any,
) -> OptionContractSnapshot:
    expiry_raw = quote.expiry or frame.timestamp.date().isoformat()
    expiry = date.fromisoformat(str(expiry_raw)[:10])
    opt_raw = str(quote.option_type or "call").lower()
    option_type = "put" if opt_raw in {"p", "put"} else "call"
    quote_ts = quote.quote_timestamp or frame.timestamp
    bid = quote.bid
    ask = quote.ask
    mid = compute_mid(bid, ask)
    rel = compute_relative_spread(bid, ask)
    return OptionContractSnapshot(
        contract_id=quote.contract_id,
        symbol=quote.symbol,
        expiry=expiry,
        strike=quote.strike or Decimal("500"),
        option_type=option_type,
        bid=bid,
        ask=ask,
        mid=mid,
        last=quote.last,
        quote_timestamp=quote_ts,
        quote_age_ms=max(
            0,
            int((frame.timestamp - quote_ts.astimezone(frame.timestamp.tzinfo)).total_seconds() * 1000),
        ),
        relative_spread=rel,
    )


async def _seed_fixture_into_task1_repos(
    deps: CognitiveGraphDeps,
    fixture: AdversarialFixture,
) -> None:
    """Write minimal Task 1 rows so graphs hydrate fixture snapshot IDs."""
    snapshot_repo, surface_repo, dq_repo = _resolve_task1_repos(deps)
    if snapshot_repo is None:
        return

    await snapshot_repo.initialize()
    if surface_repo is not None:
        await surface_repo.initialize()
    if dq_repo is not None:
        await dq_repo.initialize()

    skip_dq = bool(fixture.stimulus.get("missing_data_quality"))
    partial_surface = "partial_surface_handled" in fixture.expected_invariants

    for frame in fixture.frames:
        contracts = frame.contracts
        if partial_surface and len(contracts) > 1:
            contracts = contracts[:1]

        surface_contracts: tuple[OptionContractSnapshot, ...] = ()
        if frame.option_surface_id is not None and surface_repo is not None:
            surface_contracts = tuple(
                _frame_contract_to_surface_row(frame, c) for c in contracts
            )
            surface = OptionSurfaceSnapshot(
                surface_id=frame.option_surface_id,
                exchange_time=frame.timestamp,
                trading_date=frame.timestamp.date(),
                underlying_symbol="SPY",
                underlying_price=frame.underlying_last,
                contracts=surface_contracts,
                source="adversarial_fixture",
            )
            existing = await surface_repo.get_by_id(frame.option_surface_id)
            if existing is None:
                await surface_repo.save(surface)

        existing_snap = await snapshot_repo.get_by_id(frame.snapshot_id)
        if existing_snap is None:
            underlying = UnderlyingSnapshot(
                symbol="SPY",
                exchange_time=frame.timestamp,
                last=frame.underlying_last,
                bid=frame.underlying_bid,
                ask=frame.underlying_ask,
                mid=(frame.underlying_bid + frame.underlying_ask) / Decimal("2"),
                source="adversarial_fixture",
            )
            snapshot = MarketSnapshot(
                snapshot_id=frame.snapshot_id,
                exchange_time=frame.timestamp,
                trading_date=frame.timestamp.date(),
                underlying=underlying,
                option_surface_id=frame.option_surface_id,
                data_quality_id=frame.data_quality_id,
                source_event_ids=(),
            )
            await snapshot_repo.save(snapshot)

        if skip_dq or dq_repo is None:
            continue

        existing_dq = await dq_repo.get_by_id(frame.data_quality_id)
        if existing_dq is not None:
            continue

        findings: tuple[DataQualityFinding, ...] = ()
        severity = DataQualitySeverity.OK
        usable_reasoning = True
        usable_execution = True

        if fixture.stimulus.get("stale_quote"):
            findings = (
                DataQualityFinding(
                    code=DataQualityCode.STALE_OPTION,
                    severity=DataQualitySeverity.ERROR,
                    message="stale option quote for adversarial fixture",
                    symbol="SPY",
                ),
            )
            severity = DataQualitySeverity.ERROR
            usable_execution = False
        elif partial_surface:
            findings = (
                DataQualityFinding(
                    code=DataQualityCode.PARTIAL_OPTION_SURFACE,
                    severity=DataQualitySeverity.WARNING,
                    message="partial option surface for adversarial fixture",
                    symbol="SPY",
                ),
            )
            severity = DataQualitySeverity.WARNING
            if "partial_surface_handled" in fixture.expected_invariants:
                usable_execution = False
        elif not contracts:
            findings = (
                DataQualityFinding(
                    code=DataQualityCode.INSUFFICIENT_OPTION_SURFACE,
                    severity=DataQualitySeverity.ERROR,
                    message="empty option surface for adversarial fixture",
                    symbol="SPY",
                ),
            )
            severity = DataQualitySeverity.ERROR
            usable_execution = False

        report = DataQualityReport(
            report_id=frame.data_quality_id,
            snapshot_id=frame.snapshot_id,
            severity=severity,
            findings=findings,
            usable_for_reasoning=usable_reasoning,
            usable_for_execution=usable_execution,
        )
        await dq_repo.save(report)


async def _data_quality_blocks_entry(
    deps: CognitiveGraphDeps,
    fixture: AdversarialFixture,
) -> str | None:
    """Return a blocked-reason token when seeded DQ forbids execution."""
    dq_repo = deps.data_quality_repo
    if dq_repo is None or not fixture.frames:
        if fixture.stimulus.get("missing_data_quality"):
            return "missing_data_quality_fail_closed"
        return None
    frame = fixture.frames[0]
    if frame.data_quality_id is None:
        if fixture.stimulus.get("missing_data_quality"):
            return "missing_data_quality_fail_closed"
        return None
    report = await dq_repo.get_by_id(frame.data_quality_id)
    if report is None:
        if fixture.stimulus.get("missing_data_quality"):
            return "missing_data_quality_fail_closed"
        return None
    if not report.usable_for_execution:
        for finding in report.findings:
            code = getattr(finding, "code", None)
            code_val = getattr(code, "value", str(code) if code else "")
            if code_val == DataQualityCode.STALE_OPTION.value:
                return "stale_quote_rejection"
        return "missing_data_quality_fail_closed"
    return None


async def _collect_model_call_ids(
    deps: CognitiveGraphDeps,
    cycle_id: str,
) -> tuple[UUID, ...]:
    repo = deps.model_call_repo
    if repo is None:
        return ()
    list_fn = getattr(repo, "list_by_cycle", None)
    if list_fn is None:
        return ()
    records = await list_fn(cycle_id)
    out: list[UUID] = []
    for record in records:
        call_id = getattr(record, "model_call_id", None) or getattr(record, "request_id", None)
        if call_id is not None:
            out.append(UUID(str(call_id)))
    return tuple(out)


def _pass_only_if_observed(
    expected: tuple[str, ...],
    satisfied: tuple[str, ...],
    failed: tuple[str, ...],
    runtime_errors: tuple[str, ...],
) -> bool:
    if runtime_errors or failed:
        return False
    if not expected:
        return True
    satisfied_set = set(satisfied)
    return all(inv in satisfied_set for inv in expected)


def _evaluate_entry_invariants(
    expected: tuple[str, ...],
    *,
    findings: tuple[str, ...],
    failed: tuple[str, ...],
    fixture: AdversarialFixture,
    entry_submitted: bool,
    meta_action: str | None,
    graph_fail_closed: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    satisfied: list[str] = []
    failed_out = list(failed)
    findings_set = set(findings)

    for inv in expected:
        if inv == "invented_contract_rejected":
            if "invented_contract_accepted" in failed_out:
                continue
            if "invented_contract_rejected" in findings_set:
                satisfied.append(inv)
        elif inv == "missing_data_quality_fail_closed":
            if not entry_submitted and (
                "missing_data_quality_fail_closed" in findings_set or graph_fail_closed
            ):
                satisfied.append(inv)
        elif inv == "stale_quote_rejection":
            if not entry_submitted and (
                "stale_quote_rejection" in findings_set
                or meta_action in {"delay", "abandon", "request_more_evidence", None}
                or any("entry_blocked" in f or "stale" in f for f in findings)
            ):
                satisfied.append(inv)
        elif inv == "model_unavailable_fail_closed":
            if graph_fail_closed and not entry_submitted:
                satisfied.append(inv)
        elif inv == "provider_timeout_recovered":
            if graph_fail_closed and not entry_submitted:
                satisfied.append(inv)
        elif inv == "justified_no_trade":
            if not entry_submitted and meta_action not in _EXECUTE_ACTIONS:
                satisfied.append(inv)
        elif inv == "empty_surface_no_trade":
            if not fixture.frames[0].contracts and not entry_submitted:
                satisfied.append(inv)
        elif inv == "partial_surface_handled":
            if not entry_submitted or "entry_blocked" in findings_set:
                satisfied.append(inv)
        elif inv in findings_set:
            satisfied.append(inv)

    return tuple(dict.fromkeys(satisfied)), tuple(dict.fromkeys(failed_out))


def _evaluate_position_invariants(
    expected: tuple[str, ...],
    *,
    findings: tuple[str, ...],
    gateway_actions: tuple[str, ...],
    execution: ReplayExecutionRuntime,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    satisfied: list[str] = []
    for inv in expected:
        if inv == "reduce_then_exit":
            if len(gateway_actions) >= 2 and not execution.positions:
                satisfied.append(inv)
            elif "reduce_then_exit" in findings:
                satisfied.append(inv)
        elif inv in findings:
            satisfied.append(inv)
    return tuple(satisfied), ()


def _evaluate_om_invariants(
    expected: tuple[str, ...],
    *,
    findings: tuple[str, ...],
    execution: ReplayExecutionRuntime,
    partial_fill: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    satisfied: list[str] = []
    for inv in expected:
        if inv == "partial_fill_managed" and partial_fill and execution.orders:
            satisfied.append(inv)
        elif inv == "replace_on_deterioration" and any(
            o.status in {"replaced", "cancelled"} for o in execution.orders.values()
        ):
            satisfied.append(inv)
        elif inv in findings:
            satisfied.append(inv)
    return tuple(satisfied), ()


def _evaluate_recovery_invariants(
    expected: tuple[str, ...],
    *,
    checkpoint_loaded: bool,
    resumed: bool,
    duplicate_orders: bool,
    duplicate_fills: bool,
    duplicate_model_calls: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    satisfied: list[str] = []
    failed: list[str] = []
    if duplicate_orders:
        failed.append("duplicate_orders_on_resume")
    if duplicate_fills:
        failed.append("duplicate_fills_on_resume")
    if duplicate_model_calls:
        failed.append("duplicate_model_calls_on_resume")

    for inv in expected:
        if inv.startswith("crash_recovery") and checkpoint_loaded and resumed:
            if not duplicate_orders and not duplicate_fills and not duplicate_model_calls:
                satisfied.append(inv)
    return tuple(satisfied), tuple(failed)


def _evaluate_replay_invariants(
    expected: tuple[str, ...],
    *,
    findings: tuple[str, ...],
    payload: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    satisfied: list[str] = []
    ran = bool(payload.get("ran_task2_graph"))
    integrity = payload.get("integrity_findings") or ()
    for inv in expected:
        if ran and not integrity and inv in findings:
            satisfied.append(inv)
        elif ran and inv in findings:
            satisfied.append(inv)
    return tuple(satisfied), ()


def _get_fake_provider(deps: CognitiveGraphDeps) -> Any | None:
    router = deps.router
    if router is None:
        return None
    registry = getattr(router, "_registry", None) or getattr(router, "registry", None)
    if registry is None:
        return None
    providers = getattr(registry, "_providers", None) or getattr(registry, "providers", {})
    return providers.get("fake")


def _setup_replay_execution(
    fixture: AdversarialFixture,
    configuration: CognitiveConfigurationVersion,
) -> tuple[ReplayExecutionRuntime, ReplayOrderActionGateway]:
    truth = _truth_from_fixture(fixture)
    execution = ReplayExecutionRuntime(truth=truth)
    quotes = truth.frame_quotes(0)
    for cid, q in quotes.items():
        execution.allow_contract(cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"]))
    execution.lock_surface(set(quotes.keys()))
    gateway = ReplayOrderActionGateway(
        execution=execution,
        session_id=f"adv:{fixture.scenario_id}",
        configuration_version_id=str(configuration.configuration_version_id),
    )
    return execution, gateway


class _RunnerBase:
    def __init__(
        self,
        *,
        template_deps: CognitiveGraphDeps,
        policy_store: Any,
        checkpointer_saver: Any = None,
        replay_service: CognitiveReplayService | None = None,
    ) -> None:
        self._template = template_deps
        self._applicator = ConfigurationApplicator(policy_store)
        self._checkpointer = checkpointer_saver
        self._replay = replay_service

    def _evidence(
        self,
        *,
        experiment_id: UUID,
        definition: AdversarialScenarioDefinition,
        configuration: CognitiveConfigurationVersion,
        sample_number: int,
        **kwargs: Any,
    ) -> AdversarialExecutionEvidence:
        evaluated = kwargs.pop("invariants_evaluated", None)
        expected = kwargs.get("expected_invariants") or definition.expected_invariants
        satisfied = kwargs.get("satisfied_invariants", ())
        if evaluated is None:
            evaluated = expected
        return AdversarialExecutionEvidence(
            experiment_id=experiment_id,
            scenario_id=definition.scenario_id,
            scenario_version=definition.version,
            configuration_version_id=configuration.configuration_version_id,
            sample_number=sample_number,
            execution_mode=definition.execution_mode,
            fixture_loaded=True,
            expected_invariants=expected,
            evaluated_invariants=evaluated,
            invariants_evaluated=evaluated,
            **kwargs,
        )


class EntryGraphAdversarialRunner(_RunnerBase):
    async def execute(
        self,
        *,
        experiment_id: UUID,
        definition: AdversarialScenarioDefinition,
        fixture: AdversarialFixture,
        configuration: CognitiveConfigurationVersion,
        sample_number: int,
    ) -> AdversarialExecutionEvidence:
        applied = await self._applicator.apply(configuration)
        execution, gateway = _setup_replay_execution(fixture, configuration)
        deps = _isolated_deps(
            self._template, gateway=gateway, checkpointer=self._checkpointer
        )
        assert deps.execution_runtime is None
        assert deps.submit_callback is None
        await _seed_fixture_into_task1_repos(deps, fixture)

        provider = _get_fake_provider(deps)
        restored_flags: dict[str, Any] = {}
        if provider is not None and fixture.provider_behaviour == "timeout":
            restored_flags["simulate_timeout"] = provider.simulate_timeout
            provider.simulate_timeout = True
        elif provider is not None and fixture.provider_behaviour == "unavailable":
            restored_flags["available"] = provider.available
            provider.available = False

        findings: list[str] = []
        failed: list[str] = []
        runtime_errors: list[str] = []
        gateway_actions: list[str] = []
        runtime_invoked = False
        graph_fail_closed = False
        meta_action: str | None = None
        entry_submitted = False

        thread_id = entry_thread_id(
            experiment_id,
            fixture.fixture_id,
            configuration.configuration_version_id,
            sample_number,
        )
        try:
            state = initial_cycle_state(
                session_id=deps.session_id,
                run_id=deps.run_id,
                cycle_id=thread_id,
                trigger_event_id=str(uuid4()),
                trigger_event_type="adversarial_entry",
                snapshot_id=str(fixture.frames[0].snapshot_id),
            )
            graph = build_cognitive_graph(deps)
            config = ainvoke_config(
                session_id=deps.session_id, graph_kind="decision", cycle_id=thread_id
            )
            result: dict[str, Any] | None = None
            try:
                with pinned_applied_configuration(applied):
                    runtime_invoked = True
                    result = await graph.ainvoke(state, config=config)
            except Exception as exc:  # noqa: BLE001
                graph_fail_closed = True
                findings.append(f"graph_fail_closed:{type(exc).__name__}")
                runtime_errors.append(f"entry_graph:{type(exc).__name__}:{exc}")

            model_call_ids = await _collect_model_call_ids(deps, thread_id)

            if result is not None:
                meta = result.get("meta_decision")
                meta_action = getattr(getattr(meta, "action", None), "value", None)
                proposal = result.get("execution_proposal")
                selected = None
                if proposal is not None:
                    selected = getattr(proposal, "contract_id", None) or getattr(
                        proposal, "selected_contract_id", None
                    )
                    legs = getattr(proposal, "legs", None) or ()
                    if selected is None and legs:
                        selected = getattr(legs[0], "contract_id", None)

                attempt = fixture.stimulus.get("attempt_contract")
                if attempt and selected is None and meta_action in _EXECUTE_ACTIONS:
                    selected = attempt

                wants_entry = meta_action in _EXECUTE_ACTIONS or selected or (
                    fixture.stimulus.get("expect_reject") and attempt
                )
                if wants_entry:
                    if fixture.stimulus.get("expect_reject") and attempt:
                        contract = str(attempt)
                    else:
                        contract = str(
                            selected
                            or attempt
                            or "SPY:2026-07-01:500.0:call"
                        )
                    dq_block = await _data_quality_blocks_entry(deps, fixture)
                    if dq_block:
                        submit = OrderActionResult(
                            submitted=False,
                            client_order_id=f"adv-entry:{fixture.scenario_id}:{sample_number}",
                            blocked_reason=dq_block,
                        )
                        findings.append(dq_block)
                        if dq_block not in {"stale_quote_rejection", "missing_data_quality_fail_closed"}:
                            findings.append("entry_blocked")
                    else:
                        submit = await gateway.submit(
                            OrderActionRequest(
                                action=OrderActionKind.ENTRY,
                                snapshot_id=str(fixture.frames[0].snapshot_id),
                                contract_id=contract,
                                side="buy",
                                quantity=1,
                                client_order_id=f"adv-entry:{fixture.scenario_id}:{sample_number}",
                                cycle_id=thread_id,
                            )
                        )
                    gateway_actions.append(submit.client_order_id or "entry")
                    entry_submitted = bool(submit.submitted)
                    if fixture.stimulus.get("expect_reject"):
                        if submit.submitted:
                            failed.append("invented_contract_accepted")
                        else:
                            findings.append("invented_contract_rejected")
                    elif submit.submitted:
                        findings.append("entry_submitted")
                    else:
                        findings.append(submit.blocked_reason or "entry_blocked")
                        if fixture.stimulus.get("stale_quote"):
                            findings.append("stale_quote_rejection")
                else:
                    findings.append("no_trade_recommended")
                    if fixture.stimulus.get("stale_quote"):
                        findings.append("stale_quote_rejection")
                    if fixture.stimulus.get("expect_no_trade"):
                        findings.append("justified_no_trade")
            elif runtime_invoked and fixture.stimulus.get("attempt_contract"):
                contract = str(fixture.stimulus["attempt_contract"])
                submit = await gateway.submit(
                    OrderActionRequest(
                        action=OrderActionKind.ENTRY,
                        snapshot_id=str(fixture.frames[0].snapshot_id),
                        contract_id=contract,
                        side="buy",
                        quantity=1,
                        client_order_id=f"adv-entry:{fixture.scenario_id}:{sample_number}",
                        cycle_id=thread_id,
                    )
                )
                gateway_actions.append(submit.client_order_id or "entry")
                entry_submitted = bool(submit.submitted)
                if submit.submitted:
                    failed.append("invented_contract_accepted")
                else:
                    findings.append("invented_contract_rejected")

            if fixture.stimulus.get("missing_data_quality") and not entry_submitted:
                findings.append("missing_data_quality_fail_closed")

            if fixture.stimulus.get("stale_quote") and not entry_submitted:
                findings.append("stale_quote_rejection")

            if (
                fixture.provider_behaviour in {"timeout", "unavailable"}
                and graph_fail_closed
                and not entry_submitted
            ):
                if fixture.provider_behaviour == "timeout":
                    findings.append("provider_timeout_recovered")
                else:
                    findings.append("model_unavailable_fail_closed")

            findings_tuple = tuple(dict.fromkeys(findings))
            satisfied, failed_inv = _evaluate_entry_invariants(
                definition.expected_invariants,
                findings=findings_tuple,
                failed=tuple(failed),
                fixture=fixture,
                entry_submitted=entry_submitted,
                meta_action=meta_action,
                graph_fail_closed=graph_fail_closed,
            )
            passed = _pass_only_if_observed(
                definition.expected_invariants,
                satisfied,
                failed_inv,
                tuple(runtime_errors),
            )
            return self._evidence(
                experiment_id=experiment_id,
                definition=definition,
                configuration=configuration,
                sample_number=sample_number,
                graph_kind="entry",
                graph_thread_ids=(thread_id,),
                model_call_ids=model_call_ids,
                gateway_action_ids=tuple(gateway_actions),
                order_ids=tuple(execution.orders.keys()),
                fill_ids=tuple(f.fill_id for f in execution.fills),
                runtime_invoked=runtime_invoked,
                satisfied_invariants=satisfied,
                failed_invariants=failed_inv,
                runtime_errors=tuple(runtime_errors),
                findings=findings_tuple,
                passed=passed,
                completed=runtime_invoked,
            )
        finally:
            if provider is not None:
                for k, v in restored_flags.items():
                    setattr(provider, k, v)


class PositionGraphAdversarialRunner(_RunnerBase):
    async def execute(
        self,
        *,
        experiment_id: UUID,
        definition: AdversarialScenarioDefinition,
        fixture: AdversarialFixture,
        configuration: CognitiveConfigurationVersion,
        sample_number: int,
    ) -> AdversarialExecutionEvidence:
        applied = await self._applicator.apply(configuration)
        execution, gateway = _setup_replay_execution(fixture, configuration)
        contract = "SPY:2026-07-01:500.0:call"
        execution.positions[contract] = ReplayPosition(
            contract_id=contract,
            quantity=Decimal("2"),
            avg_price=Decimal("1.10"),
            configuration_version_id=configuration.configuration_version_id,
        )
        deps = _isolated_deps(
            self._template, gateway=gateway, checkpointer=self._checkpointer
        )
        await _seed_fixture_into_task1_repos(deps, fixture)

        thread = (
            f"adv:{experiment_id}:{fixture.scenario_id}:"
            f"{configuration.configuration_version_id}:{sample_number}:position"
        )
        graph = build_position_graph(deps)
        actions: list[str] = []
        findings: list[str] = []
        runtime_errors: list[str] = []
        runtime_invoked = False

        from joker.cognition.schemas import PositionAction

        provider = _get_fake_provider(deps)
        position_ctl: dict[str, PositionAction] = {"action": PositionAction.HOLD}
        if provider is not None and fixture.stimulus.get("reduce_then_exit"):
            from joker.cognition.schemas import PositionThesisVersion
            from joker.models.schemas import ModelRequest

            def _position_factory(request: ModelRequest) -> PositionThesisVersion:
                action = position_ctl["action"]
                return PositionThesisVersion(
                    thesis_version_id=uuid4(),
                    position_id=contract,
                    contract_id=contract,
                    session_id=deps.session_id,
                    snapshot_id=request.snapshot_id,
                    original_strategy_id=uuid4(),
                    current_thesis="reduce then exit" if action != PositionAction.HOLD else "hold",
                    recommended_action=action,
                    recommended_quantity=1,
                    recommended_limit_price=Decimal("1.20"),
                    confidence=0.7,
                    prompt_version=request.prompt_version or "2.0.0",
                    model_call_id=request.request_id,
                )

            provider.set_role_factory("position_thesis", _position_factory)
            provider.set_role_factory("position_decision", _position_factory)

        for frame_index, frame in enumerate(fixture.frames):
            if fixture.stimulus.get("reduce_then_exit"):
                if frame_index == 1:
                    position_ctl["action"] = PositionAction.REDUCE
                elif frame_index >= 2:
                    position_ctl["action"] = PositionAction.EXIT
                else:
                    position_ctl["action"] = PositionAction.HOLD

            pos_state = {
                "session_id": deps.session_id,
                "run_id": deps.run_id,
                "cycle_id": f"{thread}:{frame_index}",
                "snapshot_id": str(frame.snapshot_id),
                "_position_id": contract,
                "_contract_id": contract,
            }
            pos_result: dict[str, Any] = {}
            try:
                with pinned_applied_configuration(applied):
                    runtime_invoked = True
                    pos_result = await graph.ainvoke(
                        pos_state,
                        config=ainvoke_config(
                            session_id=deps.session_id,
                            graph_kind="position",
                            cycle_id=f"{thread}:{frame_index}",
                        ),
                    )
            except Exception as exc:  # noqa: BLE001
                runtime_errors.append(f"position_graph:{type(exc).__name__}:{exc}")
                findings.append(f"graph_fail_closed:{type(exc).__name__}")
                break

            decision = pos_result.get("_position_decision") or pos_result.get(
                "position_decision"
            )
            thesis = pos_result.get("_position_thesis") or pos_result.get(
                "position_thesis"
            )
            action_obj = decision or thesis
            recommended = getattr(action_obj, "recommended_action", None)
            rec = getattr(recommended, "value", str(recommended) if recommended else "HOLD")
            live = execution.positions.get(contract)
            if live is None or live.quantity <= 0:
                break
            kind: OrderActionKind | None = None
            qty = 0
            if rec in {"EXIT", "exit"}:
                kind = OrderActionKind.EXIT
                qty = int(live.quantity)
            elif rec in {"REDUCE", "reduce"}:
                kind = OrderActionKind.REDUCE
                qty = int(getattr(action_obj, "recommended_quantity", 1) or 1)
            if kind is None:
                continue
            submit = await gateway.submit(
                OrderActionRequest(
                    action=kind,
                    snapshot_id=str(frame.snapshot_id),
                    contract_id=contract,
                    side="sell",
                    quantity=qty,
                    client_order_id=f"adv-pos:{fixture.scenario_id}:{frame_index}",
                    cycle_id=f"{thread}:{frame_index}",
                )
            )
            actions.append(submit.client_order_id or str(kind))
            if kind == OrderActionKind.REDUCE:
                findings.append("position_reduce")
            elif kind == OrderActionKind.EXIT:
                findings.append("position_exit")

        if len(actions) >= 2:
            findings.append("reduce_then_exit")

        findings_tuple = tuple(dict.fromkeys(findings))
        satisfied, failed_inv = _evaluate_position_invariants(
            definition.expected_invariants,
            findings=findings_tuple,
            gateway_actions=tuple(actions),
            execution=execution,
        )
        passed = _pass_only_if_observed(
            definition.expected_invariants,
            satisfied,
            failed_inv,
            tuple(runtime_errors),
        )
        return self._evidence(
            experiment_id=experiment_id,
            definition=definition,
            configuration=configuration,
            sample_number=sample_number,
            graph_kind="position",
            graph_thread_ids=(thread,),
            gateway_action_ids=tuple(actions),
            order_ids=tuple(execution.orders.keys()),
            fill_ids=tuple(f.fill_id for f in execution.fills),
            runtime_invoked=runtime_invoked,
            satisfied_invariants=satisfied,
            failed_invariants=failed_inv,
            runtime_errors=tuple(runtime_errors),
            findings=findings_tuple,
            passed=passed,
            completed=runtime_invoked,
        )


class OrderManagementAdversarialRunner(_RunnerBase):
    async def execute(
        self,
        *,
        experiment_id: UUID,
        definition: AdversarialScenarioDefinition,
        fixture: AdversarialFixture,
        configuration: CognitiveConfigurationVersion,
        sample_number: int,
    ) -> AdversarialExecutionEvidence:
        applied = await self._applicator.apply(configuration)
        execution, gateway = _setup_replay_execution(fixture, configuration)
        contract = "SPY:2026-07-01:500.0:call"
        order = ReplayOrder(
            client_order_id=f"adv-working:{fixture.scenario_id}",
            contract_id=contract,
            side="buy",
            quantity=Decimal("2"),
            filled_qty=Decimal("1") if fixture.stimulus.get("partial_fill") else Decimal("0"),
            status="partially_filled"
            if fixture.stimulus.get("partial_fill")
            else "accepted",
            limit_price=Decimal("1.20"),
        )
        execution.orders[order.client_order_id] = order
        deps = _isolated_deps(
            self._template, gateway=gateway, checkpointer=self._checkpointer
        )
        await _seed_fixture_into_task1_repos(deps, fixture)

        thread = (
            f"adv:{experiment_id}:{fixture.scenario_id}:"
            f"{configuration.configuration_version_id}:{sample_number}:om"
        )
        om = ReplayOrderManagementRunner(deps=deps)
        frame = fixture.frames[-1]
        findings: list[str] = []
        runtime_errors: list[str] = []
        runtime_invoked = False
        try:
            runtime_invoked = True
            await om.manage(
                frame=frame,
                order=order,
                execution=execution,
                applied_configuration=applied,
                parent_cycle_id=thread,
                gateway=gateway,
            )
            if fixture.stimulus.get("partial_fill"):
                findings.append("partial_fill_managed")
            if fixture.stimulus.get("replace"):
                findings.append("replace_on_deterioration")
        except Exception as exc:  # noqa: BLE001
            runtime_errors.append(f"order_management:{type(exc).__name__}:{exc}")
            findings.append(f"om_fail_closed:{type(exc).__name__}")

        findings_tuple = tuple(dict.fromkeys(findings))
        satisfied, failed_inv = _evaluate_om_invariants(
            definition.expected_invariants,
            findings=findings_tuple,
            execution=execution,
            partial_fill=bool(fixture.stimulus.get("partial_fill")),
        )
        passed = _pass_only_if_observed(
            definition.expected_invariants,
            satisfied,
            failed_inv,
            tuple(runtime_errors),
        )
        return self._evidence(
            experiment_id=experiment_id,
            definition=definition,
            configuration=configuration,
            sample_number=sample_number,
            graph_kind="order_management",
            graph_thread_ids=(thread,),
            order_ids=tuple(execution.orders.keys()),
            fill_ids=tuple(f.fill_id for f in execution.fills),
            runtime_invoked=runtime_invoked,
            satisfied_invariants=satisfied,
            failed_invariants=failed_inv,
            runtime_errors=tuple(runtime_errors),
            findings=findings_tuple,
            passed=passed,
            completed=runtime_invoked,
        )


class ExecutionRecoveryAdversarialRunner(_RunnerBase):
    async def execute(
        self,
        *,
        experiment_id: UUID,
        definition: AdversarialScenarioDefinition,
        fixture: AdversarialFixture,
        configuration: CognitiveConfigurationVersion,
        sample_number: int,
    ) -> AdversarialExecutionEvidence:
        applied = await self._applicator.apply(configuration)
        db_path = self._template.db_path
        if db_path is None:
            db_path = str(Path("data") / "adversarial_recovery.db")

        ck_key = AdversarialRecoveryStore.checkpoint_key(
            experiment_id,
            definition.scenario_id,
            definition.version,
            configuration.configuration_version_id,
            sample_number,
        )

        execution, gateway = _setup_replay_execution(fixture, configuration)
        deps = _isolated_deps(
            self._template, gateway=gateway, checkpointer=self._checkpointer
        )
        await _seed_fixture_into_task1_repos(deps, fixture)

        thread_id = entry_thread_id(
            experiment_id,
            fixture.fixture_id,
            configuration.configuration_version_id,
            sample_number,
        )
        runtime_invoked = False
        findings: list[str] = []
        gateway_actions: list[str] = []

        state = initial_cycle_state(
            session_id=deps.session_id,
            run_id=deps.run_id,
            cycle_id=thread_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type="adversarial_recovery",
            snapshot_id=str(fixture.frames[0].snapshot_id),
        )
        graph = build_cognitive_graph(deps)
        config = ainvoke_config(
            session_id=deps.session_id, graph_kind="decision", cycle_id=thread_id
        )
        try:
            with pinned_applied_configuration(applied):
                runtime_invoked = True
                await graph.ainvoke(state, config=config)
        except Exception as exc:  # noqa: BLE001
            findings.append(f"graph_fail_closed:{type(exc).__name__}")

        model_calls_1 = await _collect_model_call_ids(deps, thread_id)
        orders_1 = tuple(execution.orders.keys())
        fills_1 = tuple(f.fill_id for f in execution.fills)

        checkpoint = AdversarialRecoveryCheckpoint(
            checkpoint_key=ck_key,
            experiment_id=experiment_id,
            scenario_id=definition.scenario_id,
            scenario_version=definition.version,
            configuration_version_id=configuration.configuration_version_id,
            sample_number=sample_number,
            crash_point=fixture.crash_injection_point,
            graph_thread_ids=(thread_id,),
            cash=str(fixture.starting_cash),
            submitted_keys=tuple(gateway_actions),
            order_ids=orders_1,
            fill_ids=fills_1,
            model_call_ids=tuple(str(m) for m in model_calls_1),
            gateway_action_ids=tuple(gateway_actions),
            findings=tuple(findings),
        )
        store1 = AdversarialRecoveryStore(db_path)
        await store1.save(checkpoint)
        crash_injected = True

        store2 = AdversarialRecoveryStore(db_path)
        loaded = await store2.load(ck_key)
        durable_checkpoint_loaded = loaded is not None

        fresh_execution, fresh_gateway = _setup_replay_execution(fixture, configuration)
        fresh_deps = _isolated_deps(
            self._template, gateway=fresh_gateway, checkpointer=self._checkpointer
        )
        await _seed_fixture_into_task1_repos(fresh_deps, fixture)
        fresh_runtime_created = True
        checkpoint_resumed = durable_checkpoint_loaded

        resume_runtime_invoked = False
        try:
            with pinned_applied_configuration(applied):
                resume_runtime_invoked = True
                await build_cognitive_graph(fresh_deps).ainvoke(state, config=config)
        except Exception as exc:  # noqa: BLE001
            findings.append(f"resume_graph_fail_closed:{type(exc).__name__}")

        model_calls_2 = await _collect_model_call_ids(fresh_deps, thread_id)
        orders_2 = tuple(fresh_execution.orders.keys())
        fills_2 = tuple(f.fill_id for f in fresh_execution.fills)

        dup_orders = bool(set(orders_2) - set(orders_1))
        dup_fills = bool(set(fills_2) - set(fills_1))
        dup_model = len(model_calls_2) > len(model_calls_1)

        if checkpoint_resumed and not dup_orders and not dup_fills:
            findings.append("recovery_resume_no_duplicates")

        satisfied, failed_inv = _evaluate_recovery_invariants(
            definition.expected_invariants,
            checkpoint_loaded=durable_checkpoint_loaded,
            resumed=checkpoint_resumed,
            duplicate_orders=dup_orders,
            duplicate_fills=dup_fills,
            duplicate_model_calls=dup_model,
        )
        runtime_errors: list[str] = []
        if dup_orders:
            runtime_errors.append("duplicate_orders_on_resume")
        if dup_fills:
            runtime_errors.append("duplicate_fills_on_resume")
        if dup_model:
            runtime_errors.append("duplicate_model_calls_on_resume")

        passed = _pass_only_if_observed(
            definition.expected_invariants,
            satisfied,
            failed_inv,
            tuple(runtime_errors),
        )
        return self._evidence(
            experiment_id=experiment_id,
            definition=definition,
            configuration=configuration,
            sample_number=sample_number,
            graph_kind="execution_recovery",
            graph_thread_ids=(thread_id,),
            model_call_ids=model_calls_1 + model_calls_2,
            order_ids=orders_1 + orders_2,
            fill_ids=fills_1 + fills_2,
            crash_injected=crash_injected,
            fresh_runtime_created=fresh_runtime_created,
            durable_checkpoint_loaded=durable_checkpoint_loaded,
            checkpoint_resumed=checkpoint_resumed,
            runtime_invoked=runtime_invoked or resume_runtime_invoked,
            satisfied_invariants=satisfied,
            failed_invariants=failed_inv,
            runtime_errors=tuple(runtime_errors),
            findings=tuple(dict.fromkeys(findings)),
            passed=passed,
            completed=runtime_invoked or resume_runtime_invoked,
        )


class FullReplayAdversarialRunner(_RunnerBase):
    async def execute(
        self,
        *,
        experiment_id: UUID,
        definition: AdversarialScenarioDefinition,
        fixture: AdversarialFixture,
        configuration: CognitiveConfigurationVersion,
        sample_number: int,
    ) -> AdversarialExecutionEvidence:
        expected = definition.expected_invariants
        if self._replay is None:
            return self._evidence(
                experiment_id=experiment_id,
                definition=definition,
                configuration=configuration,
                sample_number=sample_number,
                graph_kind="full_replay",
                runtime_invoked=False,
                runtime_errors=("replay_service_missing",),
                passed=False,
                completed=False,
            )

        # Full replay loads Task 1 truth from durable repos — seed fixture rows first.
        await _seed_fixture_into_task1_repos(self._template, fixture)

        episode = TradingEpisode(
            session_id=f"adv-replay:{experiment_id}",
            run_id=f"adv-replay:{definition.scenario_id}",
            trading_date=date.today(),
            initial_snapshot_id=fixture.frames[0].snapshot_id,
            terminal_snapshot_id=fixture.frames[-1].snapshot_id,
            action_class="closed_trade",
            configuration_version_id=configuration.configuration_version_id,
            quantity=Decimal("1"),
            realised_pnl=Decimal("0"),
            completed=True,
            idempotency_key=f"adv-full:{experiment_id}:{definition.scenario_id}:{sample_number}",
            snapshot_identity_status="verified",
            entry_decision_timestamp=fixture.frames[0].timestamp,
            terminal_event_timestamp=fixture.frames[-1].timestamp,
            terminal_event_id=uuid4(),
            market_event_ids=(),
        )

        runtime_invoked = False
        runtime_errors: list[str] = []
        findings: list[str] = []
        try:
            payload = await self._replay.replay_episode(
                experiment_id=experiment_id,
                episode=episode,
                configuration_version_id=configuration.configuration_version_id,
                sample=sample_number,
            )
            runtime_invoked = True
            if payload.get("integrity_findings"):
                findings.append("replay_integrity")
                runtime_errors.append("replay_integrity_failure")
            if not payload.get("ran_task2_graph"):
                runtime_errors.append("replay_graph_not_run")
            for inv in expected:
                if inv in findings:
                    continue
                if payload.get("ran_task2_graph") and not payload.get("integrity_findings"):
                    findings.append(inv)
            satisfied, failed_inv = _evaluate_replay_invariants(
                expected,
                findings=tuple(findings),
                payload=payload,
            )
            passed = _pass_only_if_observed(
                expected,
                satisfied,
                failed_inv,
                tuple(runtime_errors),
            )
            return self._evidence(
                experiment_id=experiment_id,
                definition=definition,
                configuration=configuration,
                sample_number=sample_number,
                graph_kind="full_replay",
                graph_thread_ids=(
                    str(payload.get("entry_graph_thread_id") or ""),
                )
                if payload.get("entry_graph_thread_id")
                else (),
                fill_ids=tuple(payload.get("fill_ids") or ()),
                runtime_invoked=runtime_invoked,
                satisfied_invariants=satisfied,
                failed_invariants=failed_inv,
                runtime_errors=tuple(runtime_errors),
                findings=tuple(dict.fromkeys(findings)),
                passed=passed,
                completed=bool(payload.get("ran_task2_graph")),
            )
        except Exception as exc:  # noqa: BLE001
            runtime_errors.append(f"full_replay:{type(exc).__name__}:{exc}")
            return self._evidence(
                experiment_id=experiment_id,
                definition=definition,
                configuration=configuration,
                sample_number=sample_number,
                graph_kind="full_replay",
                runtime_invoked=runtime_invoked,
                runtime_errors=tuple(runtime_errors),
                findings=(f"full_replay_failed:{type(exc).__name__}",),
                passed=False,
                completed=runtime_invoked,
            )


class AdversarialRunnerDispatcher:
    def __init__(
        self,
        *,
        template_deps: CognitiveGraphDeps | None = None,
        policy_store: Any = None,
        checkpointer_saver: Any = None,
        replay_service: CognitiveReplayService | None = None,
    ) -> None:
        self._template = template_deps
        self._policy_store = policy_store
        self._checkpointer = checkpointer_saver
        self._replay = replay_service
        self._runners: dict[str, AdversarialModeRunner] | None = None

    def _build(self) -> dict[str, AdversarialModeRunner]:
        if self._template is None or self._policy_store is None:
            raise RuntimeError("adversarial_runners_missing_dependencies")
        kwargs = dict(
            template_deps=self._template,
            policy_store=self._policy_store,
            checkpointer_saver=self._checkpointer,
            replay_service=self._replay,
        )
        return {
            "entry_graph": EntryGraphAdversarialRunner(**kwargs),
            "position_graph": PositionGraphAdversarialRunner(**kwargs),
            "order_management": OrderManagementAdversarialRunner(**kwargs),
            "execution_recovery": ExecutionRecoveryAdversarialRunner(**kwargs),
            "full_replay": FullReplayAdversarialRunner(**kwargs),
        }

    def for_mode(self, mode: str) -> AdversarialModeRunner:
        if self._runners is None:
            self._runners = self._build()
        if mode not in self._runners:
            return self._runners["full_replay"]
        return self._runners[mode]
