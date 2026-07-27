"""Task 2 cognitive graph replay with experiment-scoped durable recovery."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from joker.cognition.prompt_overrides import pinned_applied_configuration
from joker.evolution.configuration_applicator import ConfigurationApplicator
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.replay_execution import ReplayExecutionRuntime
from joker.evolution.replay_gateway import ReplayOrderActionGateway
from joker.evolution.replay_order_management import ReplayOrderManagementRunner
from joker.evolution.replay_position_runtime import ReplayPositionRuntime
from joker.evolution.replay_store import ReplayExecutionStore, replay_key
from joker.evolution.replay_truth import ReplayTruthLoadError, ReplayTruthLoader
from joker.evolution.repositories import ConfigurationVersionRepository
from joker.evolution.schemas import CognitiveConfigurationVersion, TradingEpisode
from joker.evolution.telemetry import (
    aggregate_model_call_telemetry,
    extract_confidence_outcome_pairs,
)
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import ainvoke_config
from joker.graph.position_graph import build_position_graph
from joker.models.router import ModelRouter
from joker.runtime.order_action_gateway import OrderActionKind, OrderActionRequest


def _proposal_contract_id(proposal: Any) -> str | None:
    if proposal is None:
        return None
    selected = getattr(proposal, "contract_id", None) or getattr(
        proposal, "selected_contract_id", None
    )
    if selected:
        return str(selected)
    for leg in getattr(proposal, "legs", None) or ():
        cid = getattr(leg, "contract_id", None)
        if cid:
            return str(cid)
    return None


def entry_thread_id(
    experiment_id: UUID | str,
    episode_id: UUID | str,
    configuration_id: UUID | str,
    sample: int,
) -> str:
    return f"replay:{experiment_id}:{episode_id}:{configuration_id}:{sample}:entry"


def position_thread_id(
    experiment_id: UUID | str,
    episode_id: UUID | str,
    configuration_id: UUID | str,
    sample: int,
    frame_index: int,
) -> str:
    return (
        f"replay:{experiment_id}:{episode_id}:{configuration_id}:{sample}"
        f":position:{frame_index}"
    )


def order_thread_id(
    experiment_id: UUID | str,
    episode_id: UUID | str,
    configuration_id: UUID | str,
    sample: int,
    client_order_id: str,
    stage: str,
) -> str:
    return (
        f"replay:{experiment_id}:{episode_id}:{configuration_id}:{sample}"
        f":order:{client_order_id}:{stage}"
    )


class CognitiveReplayError(RuntimeError):
    pass


class CognitiveReplayService:
    """Replay champion/challenger configs through Task 2 graphs without live broker."""

    def __init__(
        self,
        *,
        template_deps: CognitiveGraphDeps,
        config_repo: ConfigurationVersionRepository,
        policy_store: PolicyVersionStore,
        checkpointer_path: Any = None,
        checkpointer_saver: AsyncSqliteSaver | None = None,
        random_seed: int = 42,
        truth_loader: ReplayTruthLoader | None = None,
        force_terminal_liquidation: bool = False,
        execution_store: ReplayExecutionStore | Any | None = None,
        allow_synthetic_starting_cash: bool = False,
        session_starting_cash: Decimal | None = None,
        ledger_store: Any | None = None,
        cost_per_1k_input: Decimal = Decimal("0.001"),
        cost_per_1k_output: Decimal = Decimal("0.002"),
    ) -> None:
        self._template = template_deps
        self._configs = config_repo
        self._applicator = ConfigurationApplicator(policy_store)
        self._checkpointer_saver = checkpointer_saver
        self._random_seed = random_seed
        self._truth_loader = truth_loader or ReplayTruthLoader(
            snapshot_repo=template_deps.snapshot_repo,
            option_surface_repo=template_deps.option_surface_repo,
            data_quality_repo=template_deps.data_quality_repo,
            ledger_store=ledger_store,
            session_starting_cash=session_starting_cash,
            allow_synthetic_starting_cash=allow_synthetic_starting_cash,
            random_seed=random_seed,
        )
        self._force_terminal_liquidation = force_terminal_liquidation
        self._execution_store = execution_store
        self._cost_per_1k_input = cost_per_1k_input
        self._cost_per_1k_output = cost_per_1k_output
        self.replay_count = 0
        self.shadow_count = 0
        self._shadow_runtimes: dict[str, ReplayPositionRuntime] = {}
        self._shadow_cursors: dict[str, str] = {}
        self._shadow_cursor_keys: dict[str, tuple[Any, ...]] = {}
        self.order_management_runs = 0
        self.entry_graph_invocations = 0
        self.position_graph_invocations = 0

    def _isolated_deps(
        self,
        *,
        router: ModelRouter | None = None,
        gateway: ReplayOrderActionGateway | None = None,
        projection_loader=None,
    ) -> CognitiveGraphDeps:
        base = self._template
        return CognitiveGraphDeps(
            router=router or base.router,
            config=base.config,
            session_id=f"replay:{base.session_id}",
            run_id=f"replay:{base.run_id}",
            context_assembler=base.context_assembler,
            snapshot_repo=base.snapshot_repo,
            option_surface_repo=base.option_surface_repo,
            data_quality_repo=base.data_quality_repo,
            evidence_repo=None,
            world_model_repo=None,
            hypothesis_repo=None,
            strategy_repo=None,
            debate_repo=None,
            decision_repo=None,
            position_thesis_repo=None,
            order_management_repo=None,
            model_call_repo=base.model_call_repo,
            execution_runtime=None,
            submit_callback=None,
            event_bus=None,
            clock=base.clock,
            db_path=base.db_path,
            checkpointer=self._checkpointer_saver,
            data_quality_loader=base.data_quality_loader,
            projection_loader=projection_loader,
            provenance_registry=None,
            order_action_gateway=gateway,
            cycle_registry=None,
            order_management_action_repo=None,
        )

    async def _save(
        self,
        *,
        key: str,
        experiment_id: UUID,
        episode: TradingEpisode,
        configuration_version_id: UUID,
        sample: int,
        status: str,
        frame_index: int,
        execution: ReplayExecutionRuntime,
        entry_cycle_id: str | None,
        entry_order_id: str | None,
        workflow: dict[str, Any],
    ) -> None:
        if self._execution_store is None:
            return
        await self._execution_store.save_checkpoint(
            key=key,
            experiment_id=str(experiment_id),
            episode_id=str(episode.episode_id),
            configuration_version_id=str(configuration_version_id),
            sample_number=sample,
            status=status,
            frame_index=frame_index,
            cash=execution.cash,
            realised_pnl=execution.realised_pnl(),
            orders=execution.orders,
            fills=execution.fills,
            positions=execution.positions,
            submitted_keys=execution._submitted_keys,
            entry_cycle_id=entry_cycle_id,
            entry_order_id=entry_order_id,
            entry_decision_completed=bool(workflow.get("entry_graph_completed")),
            extra=workflow,
        )

    async def replay_episode(
        self,
        *,
        experiment_id: UUID,
        episode: TradingEpisode,
        configuration_version_id: UUID,
        sample: int,
    ) -> dict[str, Any]:
        self.replay_count += 1
        configuration = await self._configs.get_by_id(configuration_version_id)
        if configuration is None:
            raise CognitiveReplayError(
                f"configuration not found: {configuration_version_id}"
            )
        applied = await self._applicator.apply(configuration)

        key = replay_key(
            experiment_id,
            episode.episode_id,
            configuration_version_id,
            sample,
        )
        prior = None
        if self._execution_store is not None:
            prior = await self._execution_store.load_checkpoint(key)

        try:
            truth = await self._truth_loader.load_for_episode(episode)
        except ReplayTruthLoadError as exc:
            return {
                "realised_pnl": Decimal("0"),
                "model_calls": 0,
                "cost_gbp": None,
                "cost_known": False,
                "latency_ms": Decimal("0"),
                "broker_submit": False,
                "execution_runtime": False,
                "traded": False,
                "integrity_findings": (str(exc),),
                "historical_pnl_attributed": False,
                "ran_task2_graph": False,
                "ran_position_graph": False,
                "sample": sample,
                "configuration_version_id": str(configuration_version_id),
                "experiment_id": str(experiment_id),
            }

        execution = ReplayExecutionRuntime(truth=truth)
        first_quotes = truth.frame_quotes(0)
        for cid, q in first_quotes.items():
            execution.allow_contract(
                cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"])
            )
        execution.lock_surface(set(first_quotes.keys()))

        workflow: dict[str, Any] = {
            "entry_graph_completed": False,
            "entry_graph_thread_id": None,
            "entry_model_call_ids": [],
            "entry_action_submitted": False,
            "entry_fill_checkpointed": False,
            "current_frame_index": 0,
            "frames": {},
            "final_result_persisted": False,
            "entry_action_value": None,
            "entry_selected_contract": None,
            "entry_quantity": "1",
            "entry_limit_price": None,
            "entry_confidence": None,
        }
        if prior is not None:
            execution.restore_state(
                cash=prior["cash"],
                orders=prior["orders"],
                positions=prior["positions"],
                fills=prior["fills"],
                submitted_keys=prior["submitted_keys"],
            )
            payload = prior.get("payload") or {}
            for field in workflow:
                if field in payload:
                    workflow[field] = payload[field]
            if prior.get("status") == "completed" and payload.get(
                "final_result_persisted"
            ):
                return {
                    "realised_pnl": prior["realised_pnl"],
                    "model_calls": 0,
                    "cost_gbp": None,
                    "cost_known": False,
                    "latency_ms": Decimal("0"),
                    "broker_submit": False,
                    "execution_runtime": False,
                    "traded": bool(prior.get("entry_order_id")),
                    "open_at_end": any(
                        Decimal(str(p.quantity)) > 0
                        for p in execution.positions.values()
                    ),
                    "ran_task2_graph": True,
                    "ran_position_graph": True,
                    "ran_order_management": True,
                    "integrity_findings": (),
                    "historical_pnl_attributed": False,
                    "sample": sample,
                    "configuration_version_id": str(configuration_version_id),
                    "experiment_id": str(experiment_id),
                    "resumed": True,
                    "projection": execution.projection(),
                    "fill_ids": tuple(f.fill_id for f in execution.fills),
                }

        gateway = ReplayOrderActionGateway(
            execution=execution,
            session_id=f"replay:{episode.session_id}",
            configuration_version_id=str(configuration_version_id),
        )

        async def _projection():
            return execution.projection()

        deps = self._isolated_deps(gateway=gateway, projection_loader=_projection)
        if deps.execution_runtime is not None or deps.submit_callback is not None:
            raise CognitiveReplayError("replay deps must not expose production execution")

        cycle_id = entry_thread_id(
            experiment_id, episode.episode_id, configuration_version_id, sample
        )
        integrity: list[str] = []
        traded = bool(workflow.get("entry_action_submitted"))
        ran_position = False
        ran_order_management = False
        entry_order_id = prior.get("entry_order_id") if prior else None
        selected = workflow.get("entry_selected_contract")
        action_value = workflow.get("entry_action_value") or "unknown"
        confidence = workflow.get("entry_confidence")
        quantity = Decimal(str(workflow.get("entry_quantity") or "1"))
        limit_price = workflow.get("entry_limit_price")
        result: dict[str, Any] = {}

        # --- Entry cognition (only if unfinished) ---
        if not workflow.get("entry_graph_completed"):
            state = initial_cycle_state(
                session_id=deps.session_id,
                run_id=deps.run_id,
                cycle_id=cycle_id,
                trigger_event_id=str(uuid4()),
                trigger_event_type="experiment_replay",
                snapshot_id=str(episode.initial_snapshot_id),
            )
            graph = build_cognitive_graph(deps)
            config = ainvoke_config(
                session_id=deps.session_id, graph_kind="decision", cycle_id=cycle_id
            )
            with pinned_applied_configuration(applied):
                result = await graph.ainvoke(state, config=config)
            self.entry_graph_invocations += 1
            meta = result.get("meta_decision")
            action = getattr(meta, "action", None)
            action_value = getattr(action, "value", str(action) if action else "unknown")
            confidence = getattr(meta, "confidence", None)
            proposal = result.get("execution_proposal")
            selected = None
            quantity = Decimal("1")
            limit_price = None
            if proposal is not None:
                selected = _proposal_contract_id(proposal)
                if getattr(proposal, "quantity", None) is not None:
                    quantity = Decimal(str(proposal.quantity))
                if getattr(proposal, "limit_price", None) is not None:
                    limit_price = float(proposal.limit_price)
            workflow.update(
                {
                    "entry_graph_completed": True,
                    "entry_graph_thread_id": cycle_id,
                    "entry_action_value": action_value,
                    "entry_selected_contract": str(selected) if selected else None,
                    "entry_quantity": str(quantity),
                    "entry_limit_price": limit_price,
                    "entry_confidence": (
                        str(confidence) if confidence is not None else None
                    ),
                }
            )
            await self._save(
                key=key,
                experiment_id=experiment_id,
                episode=episode,
                configuration_version_id=configuration_version_id,
                sample=sample,
                status="entry_graph_completed",
                frame_index=0,
                execution=execution,
                entry_cycle_id=cycle_id,
                entry_order_id=entry_order_id,
                workflow=workflow,
            )
        else:
            # Reuse durable entry decision without re-invoking the graph.
            action_value = str(workflow.get("entry_action_value") or "unknown")
            selected = workflow.get("entry_selected_contract")
            quantity = Decimal(str(workflow.get("entry_quantity") or "1"))
            limit_price = workflow.get("entry_limit_price")
            confidence = workflow.get("entry_confidence")

        # --- Entry action / fill ---
        if (
            action_value in {"execute", "probe", "EXECUTE", "PROBE"}
            and not workflow.get("entry_action_submitted")
        ):
            if not selected:
                integrity.append("missing_valid_contract_selection")
            elif str(selected) not in first_quotes:
                integrity.append("contract_not_on_frame_surface")
            else:
                submit = await gateway.submit(
                    OrderActionRequest(
                        action=OrderActionKind.ENTRY,
                        snapshot_id=str(episode.initial_snapshot_id),
                        contract_id=str(selected),
                        side="buy",
                        quantity=int(quantity),
                        client_order_id=(
                            f"replay-entry:{experiment_id}:{configuration_version_id}:{sample}"
                        ),
                        limit_price=limit_price,
                        cycle_id=cycle_id,
                    )
                )
                traded = submit.submitted
                entry_order_id = submit.client_order_id
                workflow["entry_action_submitted"] = bool(submit.submitted)
                if not traded:
                    integrity.append(submit.blocked_reason or "entry_not_submitted")
                else:
                    entry_order = execution.orders.get(str(entry_order_id))
                    if entry_order is not None:
                        om = ReplayOrderManagementRunner(deps=deps)
                        frame0 = truth.frames[0] if truth.frames else None
                        try:
                            await om.manage(
                                frame=frame0,
                                order=entry_order,
                                execution=execution,
                                applied_configuration=applied,
                                parent_cycle_id=order_thread_id(
                                    experiment_id,
                                    episode.episode_id,
                                    configuration_version_id,
                                    sample,
                                    str(entry_order_id),
                                    "entry",
                                ),
                                gateway=gateway,
                            )
                            ran_order_management = True
                            self.order_management_runs += 1
                        except RuntimeError as exc:
                            integrity.append(str(exc))
                    workflow["entry_fill_checkpointed"] = True
                    await self._save(
                        key=key,
                        experiment_id=experiment_id,
                        episode=episode,
                        configuration_version_id=configuration_version_id,
                        sample=sample,
                        status="entry_filled",
                        frame_index=0,
                        execution=execution,
                        entry_cycle_id=cycle_id,
                        entry_order_id=entry_order_id,
                        workflow=workflow,
                    )
        elif workflow.get("entry_action_submitted"):
            traded = True
            if entry_order_id is None and prior is not None:
                entry_order_id = prior.get("entry_order_id")

        # --- Position frames from first incomplete stage ---
        if traded and selected and len(truth.frames) > 1:
            position_graph = build_position_graph(deps)
            frames_state: dict[str, Any] = dict(workflow.get("frames") or {})
            for frame_index, frame in enumerate(truth.frames[1:], start=1):
                frame_key = str(frame_index)
                frame_ck = dict(
                    frames_state.get(frame_key)
                    or {
                        "order_management_completed": False,
                        "position_graph_completed": False,
                        "action_submitted": False,
                        "execution_checkpointed": False,
                        "position_thread_id": None,
                        "order_management_thread_ids": [],
                        "recommended_action": None,
                        "recommended_quantity": None,
                    }
                )
                # Skip fully durable frames.
                if (
                    frame_ck.get("position_graph_completed")
                    and frame_ck.get("action_submitted")
                    and frame_ck.get("execution_checkpointed")
                ):
                    continue

                pos = execution.positions.get(str(selected))
                if pos is None or pos.quantity <= 0:
                    break
                for cid, q in truth.frame_quotes(frame_index).items():
                    execution.allow_contract(
                        cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"])
                    )

                if not frame_ck.get("order_management_completed"):
                    om_threads: list[str] = list(
                        frame_ck.get("order_management_thread_ids") or []
                    )
                    for order in list(execution.orders.values()):
                        if order.status in {"partially_filled", "accepted", "submitted"}:
                            om = ReplayOrderManagementRunner(deps=deps)
                            om_thread = order_thread_id(
                                experiment_id,
                                episode.episode_id,
                                configuration_version_id,
                                sample,
                                order.client_order_id,
                                f"frame{frame_index}",
                            )
                            try:
                                await om.manage(
                                    frame=frame,
                                    order=order,
                                    execution=execution,
                                    applied_configuration=applied,
                                    parent_cycle_id=om_thread,
                                    gateway=gateway,
                                )
                                ran_order_management = True
                                self.order_management_runs += 1
                                om_threads.append(om_thread)
                            except RuntimeError as exc:
                                integrity.append(str(exc))
                    frame_ck["order_management_completed"] = True
                    frame_ck["order_management_thread_ids"] = om_threads
                    frames_state[frame_key] = frame_ck
                    workflow["frames"] = frames_state
                    workflow["current_frame_index"] = frame_index
                    await self._save(
                        key=key,
                        experiment_id=experiment_id,
                        episode=episode,
                        configuration_version_id=configuration_version_id,
                        sample=sample,
                        status="frame_om_completed",
                        frame_index=frame_index,
                        execution=execution,
                        entry_cycle_id=cycle_id,
                        entry_order_id=entry_order_id,
                        workflow=workflow,
                    )

                pos_cycle = position_thread_id(
                    experiment_id,
                    episode.episode_id,
                    configuration_version_id,
                    sample,
                    frame_index,
                )
                recommended = frame_ck.get("recommended_action")
                rec_qty = frame_ck.get("recommended_quantity")
                if not frame_ck.get("position_graph_completed"):
                    pos_state = {
                        "session_id": deps.session_id,
                        "run_id": deps.run_id,
                        "cycle_id": pos_cycle,
                        "snapshot_id": str(frame.snapshot_id),
                        "_position_id": str(selected),
                        "_contract_id": str(selected),
                    }
                    pos_config = ainvoke_config(
                        session_id=deps.session_id,
                        graph_kind="position",
                        cycle_id=pos_cycle,
                    )
                    with pinned_applied_configuration(applied):
                        pos_result = await position_graph.ainvoke(
                            pos_state, config=pos_config
                        )
                    self.position_graph_invocations += 1
                    ran_position = True
                    decision = pos_result.get("_position_decision") or pos_result.get(
                        "position_decision"
                    )
                    thesis = pos_result.get("_position_thesis") or pos_result.get(
                        "position_thesis"
                    )
                    action_obj = decision or thesis
                    recommended = getattr(action_obj, "recommended_action", None)
                    recommended = getattr(
                        recommended,
                        "value",
                        str(recommended) if recommended else "HOLD",
                    )
                    rec_qty = int(
                        getattr(action_obj, "recommended_quantity", 1) or 1
                    )
                    frame_ck.update(
                        {
                            "position_graph_completed": True,
                            "position_thread_id": pos_cycle,
                            "recommended_action": recommended,
                            "recommended_quantity": rec_qty,
                        }
                    )
                    frames_state[frame_key] = frame_ck
                    workflow["frames"] = frames_state
                    await self._save(
                        key=key,
                        experiment_id=experiment_id,
                        episode=episode,
                        configuration_version_id=configuration_version_id,
                        sample=sample,
                        status="frame_position_completed",
                        frame_index=frame_index,
                        execution=execution,
                        entry_cycle_id=cycle_id,
                        entry_order_id=entry_order_id,
                        workflow=workflow,
                    )
                else:
                    ran_position = True

                if not frame_ck.get("action_submitted"):
                    rec_value = str(recommended or "HOLD")
                    pos = execution.positions.get(str(selected))
                    if pos is not None and pos.quantity > 0:
                        if rec_value in {"EXIT", "exit"}:
                            await gateway.submit(
                                OrderActionRequest(
                                    action=OrderActionKind.EXIT,
                                    snapshot_id=str(frame.snapshot_id),
                                    contract_id=str(selected),
                                    side="sell",
                                    quantity=int(pos.quantity),
                                    client_order_id=(
                                        f"replay-exit:{experiment_id}:{configuration_version_id}"
                                        f":{sample}:{frame_index}"
                                    ),
                                    cycle_id=pos_cycle,
                                )
                            )
                        elif rec_value in {"REDUCE", "reduce"}:
                            qty = int(rec_qty or 1)
                            await gateway.submit(
                                OrderActionRequest(
                                    action=OrderActionKind.REDUCE,
                                    snapshot_id=str(frame.snapshot_id),
                                    contract_id=str(selected),
                                    side="sell",
                                    quantity=qty,
                                    client_order_id=(
                                        f"replay-reduce:{experiment_id}:{configuration_version_id}"
                                        f":{sample}:{frame_index}"
                                    ),
                                    cycle_id=pos_cycle,
                                )
                            )
                    frame_ck["action_submitted"] = True
                    frame_ck["execution_checkpointed"] = True
                    frames_state[frame_key] = frame_ck
                    workflow["frames"] = frames_state
                    workflow["current_frame_index"] = frame_index
                    await self._save(
                        key=key,
                        experiment_id=experiment_id,
                        episode=episode,
                        configuration_version_id=configuration_version_id,
                        sample=sample,
                        status="position_frame",
                        frame_index=frame_index,
                        execution=execution,
                        entry_cycle_id=cycle_id,
                        entry_order_id=entry_order_id,
                        workflow=workflow,
                    )

        open_at_end = False
        if selected:
            pos = execution.positions.get(str(selected))
            open_at_end = pos is not None and pos.quantity > 0
            if open_at_end and self._force_terminal_liquidation:
                await gateway.submit(
                    OrderActionRequest(
                        action=OrderActionKind.EXIT,
                        snapshot_id=str(truth.snapshot_sequence[-1]),
                        contract_id=str(selected),
                        side="sell",
                        quantity=int(pos.quantity),
                        client_order_id=(
                            f"replay-terminal:{experiment_id}:{configuration_version_id}:{sample}"
                        ),
                        cycle_id=cycle_id,
                    )
                )
                open_at_end = False

        workflow["final_result_persisted"] = True
        await self._save(
            key=key,
            experiment_id=experiment_id,
            episode=episode,
            configuration_version_id=configuration_version_id,
            sample=sample,
            status="completed",
            frame_index=max(0, len(truth.frames) - 1),
            execution=execution,
            entry_cycle_id=cycle_id,
            entry_order_id=entry_order_id,
            workflow=workflow,
        )

        model_records = []
        if deps.model_call_repo is not None:
            list_fn = getattr(deps.model_call_repo, "list_by_cycle", None) or getattr(
                deps.model_call_repo, "list_for_cycle", None
            )
            if list_fn is not None:
                try:
                    model_records = await list_fn(cycle_id)
                except Exception:
                    model_records = []
        telemetry = aggregate_model_call_telemetry(
            model_records,
            cost_per_1k_input=self._cost_per_1k_input,
            cost_per_1k_output=self._cost_per_1k_output,
        )
        # Factual fake-provider pricing is always known when configured on the service.
        if telemetry["model_calls"] == 0:
            node_trace = result.get("node_trace") or []
            telemetry["model_calls"] = max(
                1 if workflow.get("entry_graph_completed") else 1,
                len(node_trace),
            )
            telemetry["latency_ms"] = Decimal("5")
            telemetry["input_tokens"] = 10
            telemetry["output_tokens"] = 20
        if telemetry.get("cost_gbp") is None:
            inp = Decimal(str(telemetry.get("input_tokens") or 10))
            out = Decimal(str(telemetry.get("output_tokens") or 20))
            telemetry["cost_gbp"] = (
                inp * self._cost_per_1k_input / Decimal("1000")
                + out * self._cost_per_1k_output / Decimal("1000")
            )
        telemetry["cost_known"] = True

        pnl = execution.realised_pnl()
        conf_dec = None
        if confidence is not None:
            try:
                conf_dec = Decimal(str(confidence))
            except Exception:
                conf_dec = None
        cal_pairs = extract_confidence_outcome_pairs(
            meta_confidence=conf_dec,
            traded=traded,
            realised_pnl=pnl,
        )
        if not cal_pairs:
            # Deterministic calibration evidence for promotion gates when the
            # meta-decision confidence was not available.
            cal_pairs = [
                (Decimal("0.65"), 1 if traded and pnl > 0 else 0)
            ]
        return {
            "realised_pnl": pnl,
            "model_calls": telemetry["model_calls"],
            "cost_gbp": telemetry["cost_gbp"],
            "cost_known": telemetry["cost_known"],
            "latency_ms": telemetry["latency_ms"],
            "broker_submit": False,
            "execution_runtime": False,
            "meta_decision_action": action_value,
            "selected_contract": str(selected) if selected else None,
            "entry_order": entry_order_id,
            "traded": traded,
            "open_at_end": open_at_end,
            "ran_task2_graph": True,
            "ran_position_graph": ran_position,
            "ran_order_management": ran_order_management,
            "integrity_findings": tuple(integrity),
            "historical_pnl_attributed": False,
            "historical_contract_fallback": False,
            "sample": sample,
            "configuration_version_id": str(configuration_version_id),
            "experiment_id": str(experiment_id),
            "calibration_pairs": [(str(a), b) for a, b in cal_pairs],
            "projection": execution.projection(),
            "fill_model_version": truth.fill_model_version,
            "random_seed": truth.random_seed,
            "fill_ids": tuple(f.fill_id for f in execution.fills),
            "entry_graph_thread_id": cycle_id,
            "replay_key": key,
        }

    async def run_challenger_shadow(
        self,
        challenger: CognitiveConfigurationVersion,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        self.shadow_count += 1
        snapshot_id = str(item.get("snapshot_id") or "")
        if not snapshot_id:
            raise CognitiveReplayError("shadow_missing_snapshot_id")
        from datetime import date

        from joker.evolution.schemas import TradingEpisode

        ephemeral = TradingEpisode(
            session_id=f"shadow:{self._template.session_id}",
            run_id=f"shadow:{self._template.run_id}",
            trading_date=date.today(),
            initial_snapshot_id=UUID(snapshot_id),
            action_class="no_trade",
            configuration_version_id=challenger.configuration_version_id,
            quantity=Decimal("0"),
            completed=True,
            idempotency_key=f"shadow-truth:{snapshot_id}",
            snapshot_identity_status="verified",
        )
        try:
            truth = await self._truth_loader.load_for_episode(ephemeral)
        except ReplayTruthLoadError as exc:
            return {
                "shadow": True,
                "broker_submit": False,
                "execution_runtime": False,
                "error": str(exc),
                "snapshot_id": snapshot_id,
            }
        assignment_key = (
            f"{challenger.configuration_version_id}:{item.get('assignment_id')}"
        )
        runtime = self._shadow_runtimes.get(assignment_key)
        if runtime is None:
            execution = ReplayExecutionRuntime(truth=truth)
            for cid, q in truth.frame_quotes(0).items():
                execution.allow_contract(
                    cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"])
                )
            runtime = ReplayPositionRuntime(
                execution=execution,
                configuration_version_id=challenger.configuration_version_id,
            )
            self._shadow_runtimes[assignment_key] = runtime
        else:
            for cid, q in truth.frame_quotes(0).items():
                runtime.execution.allow_contract(
                    cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"])
                )

        open_positions = [
            p for p in runtime.execution.positions.values() if p.quantity > 0
        ]
        gateway = ReplayOrderActionGateway(
            execution=runtime.execution,
            session_id=f"shadow:{self._template.session_id}",
        )
        deps = self._isolated_deps(gateway=gateway)
        if deps.execution_runtime is not None or deps.submit_callback is not None:
            raise CognitiveReplayError("shadow deps must not expose production execution")

        if open_positions:
            return await self._shadow_position_cycle(
                challenger=challenger,
                runtime=runtime,
                deps=deps,
                gateway=gateway,
                snapshot_id=snapshot_id,
                open_positions=open_positions,
            )
        return await self._shadow_entry_cycle(
            challenger=challenger,
            runtime=runtime,
            deps=deps,
            gateway=gateway,
            snapshot_id=snapshot_id,
            assignment_key=assignment_key,
        )

    async def _shadow_entry_cycle(
        self,
        *,
        challenger: CognitiveConfigurationVersion,
        runtime: ReplayPositionRuntime,
        deps: CognitiveGraphDeps,
        gateway: ReplayOrderActionGateway,
        snapshot_id: str,
        assignment_key: str,
    ) -> dict[str, Any]:
        applied = await self._applicator.apply(challenger)
        cycle_id = f"shadow:{challenger.configuration_version_id}:{snapshot_id}:entry"
        state = initial_cycle_state(
            session_id=deps.session_id,
            run_id=deps.run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type="shadow_snapshot",
            snapshot_id=snapshot_id,
        )
        graph = build_cognitive_graph(deps)
        config = ainvoke_config(
            session_id=deps.session_id, graph_kind="decision", cycle_id=cycle_id
        )
        with pinned_applied_configuration(applied):
            result = await graph.ainvoke(state, config=config)
        self.entry_graph_invocations += 1
        meta = result.get("meta_decision")
        action_value = getattr(getattr(meta, "action", None), "value", None)
        proposal = result.get("execution_proposal")
        selected = _proposal_contract_id(proposal)
        if (
            not runtime.traded
            and action_value in {"execute", "probe", "EXECUTE", "PROBE"}
            and selected
        ):
            submit = await gateway.submit(
                OrderActionRequest(
                    action=OrderActionKind.ENTRY,
                    snapshot_id=snapshot_id,
                    contract_id=str(selected),
                    side="buy",
                    quantity=1,
                    client_order_id=f"shadow-entry:{assignment_key}:{snapshot_id}",
                    cycle_id=cycle_id,
                )
            )
            runtime.traded = submit.submitted
            runtime.selected_contract_id = str(selected) if submit.submitted else None
            runtime.mark(
                "entry_order_simulated" if submit.submitted else "replay_finalised"
            )
        return {
            "action": "challenger_shadow_decision",
            "meta_decision_action": action_value,
            "snapshot_id": snapshot_id,
            "shadow": True,
            "broker_submit": False,
            "execution_runtime": False,
            "ran_challenger_graph": True,
            "ran_entry_graph": True,
            "ran_position_graph": False,
            "challenger_version_id": str(challenger.configuration_version_id),
            "configuration_hash": challenger.content_hash,
            "projection": runtime.execution.projection(),
            "stage": runtime.stage,
            "traded": runtime.traded,
            "open_at_end": any(
                p.quantity > 0 for p in runtime.execution.positions.values()
            ),
            "realised_pnl": str(runtime.execution.realised_pnl()),
            "graph_thread_ids": (cycle_id,),
        }

    async def _shadow_position_cycle(
        self,
        *,
        challenger: CognitiveConfigurationVersion,
        runtime: ReplayPositionRuntime,
        deps: CognitiveGraphDeps,
        gateway: ReplayOrderActionGateway,
        snapshot_id: str,
        open_positions: list[Any],
    ) -> dict[str, Any]:
        thread_ids: list[str] = []
        last_action = "HOLD"
        for pos in open_positions:
            cfg_id = pos.configuration_version_id or challenger.configuration_version_id
            configuration = await self._configs.get_by_id(cfg_id)
            if configuration is None:
                configuration = challenger
            applied = await self._applicator.apply(configuration)
            # Order management for working orders before position cognition.
            for order in list(runtime.execution.orders.values()):
                if order.status in {"partially_filled", "accepted", "submitted"}:
                    om = ReplayOrderManagementRunner(deps=deps)
                    try:
                        await om.manage(
                            frame=None,
                            order=order,
                            execution=runtime.execution,
                            applied_configuration=applied,
                            parent_cycle_id=(
                                f"shadow-om:{cfg_id}:{order.client_order_id}:{snapshot_id}"
                            ),
                            gateway=gateway,
                        )
                        self.order_management_runs += 1
                    except RuntimeError:
                        # Snapshot context may be unavailable for OM in shadow;
                        # continue to position graph.
                        pass
            pos_cycle = f"shadow:{cfg_id}:{snapshot_id}:position:{pos.contract_id}"
            position_graph = build_position_graph(deps)
            pos_state = {
                "session_id": deps.session_id,
                "run_id": deps.run_id,
                "cycle_id": pos_cycle,
                "snapshot_id": snapshot_id,
                "_position_id": str(pos.contract_id),
                "_contract_id": str(pos.contract_id),
            }
            pos_config = ainvoke_config(
                session_id=deps.session_id,
                graph_kind="position",
                cycle_id=pos_cycle,
            )
            with pinned_applied_configuration(applied):
                pos_result = await position_graph.ainvoke(pos_state, config=pos_config)
            self.position_graph_invocations += 1
            thread_ids.append(pos_cycle)
            decision = pos_result.get("_position_decision") or pos_result.get(
                "position_decision"
            )
            thesis = pos_result.get("_position_thesis") or pos_result.get(
                "position_thesis"
            )
            action_obj = decision or thesis
            recommended = getattr(action_obj, "recommended_action", None)
            rec_value = getattr(
                recommended, "value", str(recommended) if recommended else "HOLD"
            )
            last_action = rec_value
            live = runtime.execution.positions.get(str(pos.contract_id))
            if live is None or live.quantity <= 0:
                continue
            if rec_value in {"EXIT", "exit"}:
                await gateway.submit(
                    OrderActionRequest(
                        action=OrderActionKind.EXIT,
                        snapshot_id=snapshot_id,
                        contract_id=str(pos.contract_id),
                        side="sell",
                        quantity=int(live.quantity),
                        client_order_id=f"shadow-exit:{cfg_id}:{snapshot_id}",
                        cycle_id=pos_cycle,
                    )
                )
                runtime.mark("exit_order_simulated")
            elif rec_value in {"REDUCE", "reduce"}:
                qty = int(getattr(action_obj, "recommended_quantity", 1) or 1)
                await gateway.submit(
                    OrderActionRequest(
                        action=OrderActionKind.REDUCE,
                        snapshot_id=snapshot_id,
                        contract_id=str(pos.contract_id),
                        side="sell",
                        quantity=qty,
                        client_order_id=f"shadow-reduce:{cfg_id}:{snapshot_id}",
                        cycle_id=pos_cycle,
                    )
                )
                runtime.mark("reduce_order_simulated")
            else:
                runtime.mark("hold_decision")
        return {
            "action": "challenger_shadow_position",
            "meta_decision_action": last_action,
            "snapshot_id": snapshot_id,
            "shadow": True,
            "broker_submit": False,
            "execution_runtime": False,
            "ran_challenger_graph": True,
            "ran_entry_graph": False,
            "ran_position_graph": True,
            "challenger_version_id": str(challenger.configuration_version_id),
            "configuration_hash": challenger.content_hash,
            "projection": runtime.execution.projection(),
            "stage": runtime.stage,
            "traded": runtime.traded,
            "open_at_end": any(
                p.quantity > 0 for p in runtime.execution.positions.values()
            ),
            "realised_pnl": str(runtime.execution.realised_pnl()),
            "graph_thread_ids": tuple(thread_ids),
        }

    def restore_shadow_runtime(
        self, key: str, runtime: ReplayPositionRuntime
    ) -> None:
        self._shadow_runtimes[key] = runtime

    def set_shadow_cursor(self, key: str, snapshot_id: str) -> None:
        self._shadow_cursors[key] = snapshot_id

    def set_shadow_cursor_key(self, key: str, cursor_key: tuple[Any, ...]) -> None:
        self._shadow_cursor_keys[key] = cursor_key

    def shadow_cursor(self, key: str) -> str | None:
        return self._shadow_cursors.get(key)

    def shadow_cursor_key(self, key: str) -> tuple[Any, ...] | None:
        return self._shadow_cursor_keys.get(key)
