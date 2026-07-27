"""Task 2 cognitive graph replay with repository-backed truth and isolated fills."""

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
from joker.evolution.replay_market import ReplayEpisodeTruth
from joker.evolution.replay_position_runtime import ReplayPositionRuntime
from joker.evolution.replay_truth import ReplayTruthLoadError, ReplayTruthLoader
from joker.evolution.replay_order_management import ReplayOrderManagementRunner
from joker.evolution.replay_store import ReplayExecutionStore, replay_key
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
        execution_store: Any | None = None,
        allow_synthetic_starting_cash: bool = False,
        session_starting_cash: Decimal | None = None,
        ledger_store: Any | None = None,
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
        self.replay_count = 0
        self.shadow_count = 0
        self._shadow_runtimes: dict[str, ReplayPositionRuntime] = {}
        self._shadow_cursors: dict[str, str] = {}
        self.order_management_runs = 0

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

    async def replay_episode(
        self,
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
            }

        execution = ReplayExecutionRuntime(truth=truth)
        # Lock surface to first frame contracts only — no historical fallback.
        first_quotes = truth.frame_quotes(0)
        for cid, q in first_quotes.items():
            execution.allow_contract(
                cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"])
            )
        execution.lock_surface(set(first_quotes.keys()))

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

        cycle_id = f"replay:{episode.episode_id}:{configuration_version_id}:{sample}"
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

        integrity: list[str] = []
        traded = False
        ran_position = False
        ran_order_management = False
        entry_order_id = None
        key = replay_key(
            episode.episode_id,
            episode.episode_id,
            configuration_version_id,
            sample,
        )
        if self._execution_store is not None:
            prior = await self._execution_store.load_checkpoint(key)
            if prior is not None:
                execution.restore_state(
                    cash=prior["cash"],
                    orders=prior["orders"],
                    positions=prior["positions"],
                    fills=prior["fills"],
                    submitted_keys=prior["submitted_keys"],
                )
                entry_order_id = prior.get("entry_order_id")
                traded = bool(prior.get("entry_decision_completed") and entry_order_id)
                if prior.get("status") == "completed":
                    return {
                        "realised_pnl": prior["realised_pnl"],
                        "model_calls": 0,
                        "cost_gbp": None,
                        "cost_known": False,
                        "latency_ms": Decimal("0"),
                        "broker_submit": False,
                        "execution_runtime": False,
                        "traded": traded,
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
                        "resumed": True,
                        "projection": execution.projection(),
                    }

        if action_value in {"execute", "probe", "EXECUTE", "PROBE"} and not traded:
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
                        client_order_id=f"replay-entry:{configuration_version_id}:{sample}",
                        limit_price=limit_price,
                        cycle_id=cycle_id,
                    )
                )
                traded = submit.submitted
                entry_order_id = submit.client_order_id
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
                                parent_cycle_id=cycle_id,
                                gateway=gateway,
                            )
                            ran_order_management = True
                            self.order_management_runs += 1
                        except RuntimeError as exc:
                            # Without snapshot context OM cannot run; record finding.
                            integrity.append(str(exc))
                    if self._execution_store is not None:
                        await self._execution_store.save_checkpoint(
                            key=key,
                            experiment_id=str(episode.episode_id),
                            episode_id=str(episode.episode_id),
                            configuration_version_id=str(configuration_version_id),
                            sample_number=sample,
                            status="entry_filled",
                            frame_index=0,
                            cash=execution.cash,
                            realised_pnl=execution.realised_pnl(),
                            orders=execution.orders,
                            fills=execution.fills,
                            positions=execution.positions,
                            submitted_keys=execution._submitted_keys,
                            entry_cycle_id=cycle_id,
                            entry_order_id=entry_order_id,
                            entry_decision_completed=True,
                        )

        # Advance frames with actual position cognition while open.
        if traded and selected and len(truth.frames) > 1:
            position_graph = build_position_graph(deps)
            for frame_index, frame in enumerate(truth.frames[1:], start=1):
                pos = execution.positions.get(str(selected))
                if pos is None or pos.quantity <= 0:
                    break
                # Refresh quotes for this frame before position decision.
                for cid, q in truth.frame_quotes(frame_index).items():
                    execution.allow_contract(
                        cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"])
                    )
                # Manage any still-working orders before position cognition.
                for order in list(execution.orders.values()):
                    if order.status in {"partially_filled", "accepted", "submitted"}:
                        om = ReplayOrderManagementRunner(deps=deps)
                        try:
                            await om.manage(
                                frame=frame,
                                order=order,
                                execution=execution,
                                applied_configuration=applied,
                                parent_cycle_id=cycle_id,
                                gateway=gateway,
                            )
                            ran_order_management = True
                            self.order_management_runs += 1
                        except RuntimeError as exc:
                            integrity.append(str(exc))
                pos_cycle = f"{cycle_id}:pos:{frame.snapshot_id}"
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
                ran_position = True
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
                if rec_value in {"EXIT", "exit"}:
                    await gateway.submit(
                        OrderActionRequest(
                            action=OrderActionKind.EXIT,
                            snapshot_id=str(frame.snapshot_id),
                            contract_id=str(selected),
                            side="sell",
                            quantity=int(pos.quantity),
                            client_order_id=(
                                f"replay-exit:{configuration_version_id}:{sample}:{frame_index}"
                            ),
                            cycle_id=pos_cycle,
                        )
                    )
                elif rec_value in {"REDUCE", "reduce"}:
                    qty = int(
                        getattr(action_obj, "recommended_quantity", 1) or 1
                    )
                    await gateway.submit(
                        OrderActionRequest(
                            action=OrderActionKind.REDUCE,
                            snapshot_id=str(frame.snapshot_id),
                            contract_id=str(selected),
                            side="sell",
                            quantity=qty,
                            client_order_id=(
                                f"replay-reduce:{configuration_version_id}:{sample}:{frame_index}"
                            ),
                            cycle_id=pos_cycle,
                        )
                    )
                if self._execution_store is not None:
                    await self._execution_store.save_checkpoint(
                        key=key,
                        experiment_id=str(episode.episode_id),
                        episode_id=str(episode.episode_id),
                        configuration_version_id=str(configuration_version_id),
                        sample_number=sample,
                        status="position_frame",
                        frame_index=frame_index,
                        cash=execution.cash,
                        realised_pnl=execution.realised_pnl(),
                        orders=execution.orders,
                        fills=execution.fills,
                        positions=execution.positions,
                        submitted_keys=execution._submitted_keys,
                        entry_cycle_id=cycle_id,
                        entry_order_id=entry_order_id,
                        entry_decision_completed=True,
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
                            f"replay-terminal:{configuration_version_id}:{sample}"
                        ),
                        cycle_id=cycle_id,
                    )
                )
                open_at_end = False

        if self._execution_store is not None:
            await self._execution_store.save_checkpoint(
                key=key,
                experiment_id=str(episode.episode_id),
                episode_id=str(episode.episode_id),
                configuration_version_id=str(configuration_version_id),
                sample_number=sample,
                status="completed",
                frame_index=max(0, len(truth.frames) - 1),
                cash=execution.cash,
                realised_pnl=execution.realised_pnl(),
                orders=execution.orders,
                fills=execution.fills,
                positions=execution.positions,
                submitted_keys=execution._submitted_keys,
                entry_cycle_id=cycle_id,
                entry_order_id=entry_order_id,
                entry_decision_completed=True,
            )

        # Telemetry from model-call repo when available.
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
        telemetry = aggregate_model_call_telemetry(model_records)
        if telemetry["model_calls"] == 0:
            # Fall back to node-trace count without inventing cost.
            node_trace = result.get("node_trace") or []
            telemetry["model_calls"] = max(1, len(node_trace))
            telemetry["latency_ms"] = Decimal("0")
            telemetry["cost_known"] = False
            telemetry["cost_gbp"] = None

        pnl = execution.realised_pnl()
        cal_pairs = extract_confidence_outcome_pairs(
            meta_confidence=Decimal(str(confidence)) if confidence is not None else None,
            traded=traded,
            realised_pnl=pnl,
        )
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
            "calibration_pairs": [(str(a), b) for a, b in cal_pairs],
            "projection": execution.projection(),
            "fill_model_version": truth.fill_model_version,
            "random_seed": truth.random_seed,
            "fill_ids": tuple(f.fill_id for f in execution.fills),
        }

    async def run_challenger_shadow(
        self,
        challenger: CognitiveConfigurationVersion,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        self.shadow_count += 1
        applied = await self._applicator.apply(challenger)
        snapshot_id = str(item.get("snapshot_id") or "")
        if not snapshot_id:
            raise CognitiveReplayError("shadow_missing_snapshot_id")
        # Hydrate Task 1 truth — never trust payload quotes.
        from joker.evolution.schemas import TradingEpisode
        from datetime import date

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
                execution.allow_contract(cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"]))
            runtime = ReplayPositionRuntime(
                execution=execution,
                configuration_version_id=challenger.configuration_version_id,
            )
            self._shadow_runtimes[assignment_key] = runtime
        else:
            # Advance quotes from latest frame.
            for cid, q in truth.frame_quotes(0).items():
                runtime.execution.allow_contract(
                    cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"])
                )

        gateway = ReplayOrderActionGateway(
            execution=runtime.execution,
            session_id=f"shadow:{self._template.session_id}",
        )
        deps = self._isolated_deps(gateway=gateway)
        cycle_id = f"shadow:{challenger.configuration_version_id}:{snapshot_id}"
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
            runtime.mark("entry_order_simulated" if submit.submitted else "replay_finalised")
        return {
            "action": "challenger_shadow_decision",
            "meta_decision_action": action_value,
            "snapshot_id": snapshot_id,
            "shadow": True,
            "broker_submit": False,
            "execution_runtime": False,
            "ran_challenger_graph": True,
            "challenger_version_id": str(challenger.configuration_version_id),
            "configuration_hash": challenger.content_hash,
            "projection": runtime.execution.projection(),
            "stage": runtime.stage,
            "traded": runtime.traded,
            "open_at_end": any(
                p.quantity > 0 for p in runtime.execution.positions.values()
            ),
            "realised_pnl": str(runtime.execution.realised_pnl()),
        }

    def restore_shadow_runtime(
        self, key: str, runtime: ReplayPositionRuntime
    ) -> None:
        self._shadow_runtimes[key] = runtime

    def set_shadow_cursor(self, key: str, snapshot_id: str) -> None:
        self._shadow_cursors[key] = snapshot_id

    def shadow_cursor(self, key: str) -> str | None:
        return self._shadow_cursors.get(key)
