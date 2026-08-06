"""Asynchronous cognitive agent runtime with independent decision/position workers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from joker.agents.cognitive.order_management import OrderManagerAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import (
    AgentRole,
    CognitiveError,
    CognitiveRuntimeHealth,
    OrderManagementDecision,
)
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import DomainEvent, EventType, make_event
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.context_hydrate import assemble_role_context, load_snapshot_truth
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import (
    CognitiveCheckpointer,
    ainvoke_config,
)
from joker.graph.position_graph import build_position_graph
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.persistence.cognitive_repositories import (
    DebateRepository,
    DecisionRepository,
    EvidenceRepository,
    HypothesisRepository,
    ModelCallRepository,
    OrderManagementRepository,
    PositionThesisRepository,
    StrategyRepository,
    WorldModelRepository,
)
from joker.cognition.artifacts import CognitiveArtifactStore

logger = logging.getLogger(__name__)


class _Priority(IntEnum):
    CRITICAL = 0
    POSITION_SNAPSHOT = 1
    NEW_ENTRY_SNAPSHOT = 2


@dataclass(order=True)
class _QueuedWork:
    priority: int
    sequence: int
    event: DomainEvent = field(compare=False)
    kind: str = field(compare=False, default="decision")


@dataclass
class _RuntimeCounters:
    active_decision_cycles: int = 0
    active_position_cycles: int = 0
    queued_events: int = 0
    last_success_at: datetime | None = None
    last_error: CognitiveError | None = None


class CognitiveAgentRuntime:
    """Task 2 agent runtime — independent decision and position workers."""

    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        router: ModelRouter,
        config: CognitiveGraphSettings,
        graph_deps: CognitiveGraphDeps | None = None,
        registry: ModelRegistry | None = None,
        checkpointer_path: Path | str | None = None,
        evolution_runtime: Any | None = None,
    ) -> None:
        self._session_id = session_id
        self._run_id = run_id
        self._router = router
        self._config = config
        self._registry = registry
        self._evolution_runtime = evolution_runtime
        self._deps = graph_deps or CognitiveGraphDeps(
            router=router,
            config=config,
            session_id=session_id,
            run_id=run_id,
        )
        self._checkpointer_helper: CognitiveCheckpointer | None = None
        if checkpointer_path is not None:
            self._checkpointer_helper = CognitiveCheckpointer(Path(checkpointer_path))
        elif self._deps.db_path is not None:
            self._checkpointer_helper = CognitiveCheckpointer(
                Path(self._deps.db_path).with_name(
                    Path(self._deps.db_path).stem + "_cognitive_ckpt.db"
                )
            )
        self._decision_graph = None
        self._position_graph = None
        self._decision_queue: asyncio.PriorityQueue[_QueuedWork] = asyncio.PriorityQueue()
        self._position_queue: asyncio.PriorityQueue[_QueuedWork] = asyncio.PriorityQueue()
        self._sequence = 0
        self._shutdown = False
        self._started = False
        self._decision_worker: asyncio.Task[None] | None = None
        self._position_worker: asyncio.Task[None] | None = None
        self._active_decision_tasks: set[asyncio.Task[None]] = set()
        self._active_position_tasks: set[asyncio.Task[None]] = set()
        self._new_entry_lock = asyncio.Lock()
        self._new_entry_in_flight = False
        self._pending_new_entry_snapshot: str | None = None
        # When True, market snapshots still drive open-position management but do
        # not enqueue new-entry decision cycles (shadow evidence collection).
        self._suppress_new_entry_snapshots = False
        self._reconciliation_only_recovery = False
        self._reoptimization_retry_tasks: dict[str, asyncio.Task[None]] = {}
        self._counters = _RuntimeCounters()
        self._status: str = "healthy"
        self._received_events: list[DomainEvent] = []
        self._order_decision_ids: set[str] = set()

    @property
    def received_events(self) -> list[DomainEvent]:
        return list(self._received_events)

    @property
    def deps(self) -> CognitiveGraphDeps:
        return self._deps

    def _broker_account_identity(self) -> str:
        execution = self._deps.execution_runtime
        if execution is not None:
            return execution.broker_account_identity
        return str(self._deps.broker_account_identity or "")

    def _stable_trading_date(self) -> str:
        from joker.runtime.cognitive_session import stable_cognitive_session_trading_date

        parsed = stable_cognitive_session_trading_date(self._session_id)
        if parsed is not None:
            return parsed.isoformat()
        if self._deps.clock is None:
            raise RuntimeError("exchange clock required for stable trading date")
        return self._deps.clock.trading_date().isoformat()

    def suppress_new_entry_snapshots(self, suppressed: bool = True) -> None:
        """Block new-entry decision enqueue while still allowing position cycles."""
        self._suppress_new_entry_snapshots = bool(suppressed)
        if suppressed:
            self._pending_new_entry_snapshot = None

    def enable_reconciliation_only_recovery(self, enabled: bool = True) -> None:
        """Fail closed after terminal objectives: reconcile only, never enter anew."""
        self._reconciliation_only_recovery = bool(enabled)
        if enabled:
            self.suppress_new_entry_snapshots(True)

    def bind_evolution_runtime(self, evolution_runtime: Any) -> None:
        """Inject Task 3 runtime before workers start (supported public API)."""
        if self._started and self._evolution_runtime is not None:
            raise RuntimeError(
                "cannot rebind evolution runtime after CognitiveAgentRuntime has started"
            )
        self._evolution_runtime = evolution_runtime

    async def start(self) -> None:
        if self._started:
            return
        if self._checkpointer_helper is not None:
            saver = await self._checkpointer_helper.open()
            self._deps.checkpointer = saver
        from joker.runtime.order_action_gateway import ensure_order_action_gateway

        ensure_order_action_gateway(self._deps)
        if self._deps.provenance_registry is None and self._deps.db_path is not None:
            from joker.persistence.cognitive_execution_provenance import (
                CognitiveExecutionProvenanceRegistry,
            )

            provenance = CognitiveExecutionProvenanceRegistry(self._deps.db_path)
            await provenance.initialize()
            self._deps.provenance_registry = provenance
        if self._deps.cycle_registry is None and self._deps.db_path is not None:
            from joker.persistence.cognitive_cycle_registry import CognitiveCycleRegistry

            registry = CognitiveCycleRegistry(
                Path(self._deps.db_path).with_name(
                    Path(self._deps.db_path).stem + "_cognitive_cycles.db"
                )
            )
            await registry.initialize()
            self._deps.cycle_registry = registry
        if self._deps.order_management_action_repo is None and self._deps.db_path is not None:
            from joker.persistence.order_management_actions import (
                OrderManagementActionRepository,
            )

            om_repo = OrderManagementActionRepository(
                Path(self._deps.db_path).with_name(Path(self._deps.db_path).stem + "_om_actions.db")
            )
            await om_repo.initialize()
            self._deps.order_management_action_repo = om_repo
        self._decision_graph = build_cognitive_graph(self._deps)
        self._position_graph = build_position_graph(self._deps)
        self._started = True
        self._shutdown = False
        # Resume unfinished cycles before accepting new events.
        try:
            await self._resume_unfinished_cycles()
            await self._resume_pending_portfolio_executions()
            await self._resume_pending_portfolio_reoptimizations()
        except Exception as exc:  # noqa: BLE001
            logger.exception("cognitive_cycle_recovery_failed", exc_info=exc)
            self._status = "degraded"
            self._counters.last_error = CognitiveError(
                error_code="cycle_recovery_failed",
                message=str(exc),
                recoverable=True,
            )
        self._decision_worker = asyncio.create_task(
            self._decision_worker_loop(), name="cognitive-decision-worker"
        )
        self._position_worker = asyncio.create_task(
            self._position_worker_loop(), name="cognitive-position-worker"
        )

    async def _resume_pending_portfolio_executions(self) -> None:
        """Reconcile durable component state and resume only the next eligible leg."""
        registry = self._deps.provenance_registry
        if registry is None or self._decision_graph is None or self._deps.clock is None:
            return
        broker_account_id = self._broker_account_identity()
        if not broker_account_id:
            return
        records = await registry.portfolio_executions.list_resumable(
            session_id=self._session_id,
            broker_account_identity=broker_account_id,
            trading_date=self._stable_trading_date(),
        )
        decision_ids = sorted({record.target_portfolio_decision_id for record in records})
        for decision_id in decision_ids:
            await self._resume_portfolio_decision(decision_id)

    async def _resume_portfolio_decision(self, decision_id: str) -> None:
        registry = self._deps.provenance_registry
        if registry is None or self._decision_graph is None:
            return
        from joker.persistence.cognitive_execution_provenance import (
            PortfolioComponentStatus,
            PortfolioExecutionOwner,
        )

        if self._deps.clock is None:
            return
        owner = PortfolioExecutionOwner(
            session_id=self._session_id,
            broker_account_identity=self._broker_account_identity(),
            trading_date=self._stable_trading_date(),
        )
        records = await registry.portfolio_executions.list_by_decision(decision_id, owner=owner)
        if not records:
            return
        resumed_at = self._deps.clock.now().isoformat()
        records = [
            await registry.portfolio_executions.record_resume(
                record.authorized_position_tuple_id,
                owner=owner,
                current_run_id=self._run_id,
                resumed_at=resumed_at,
            )
            for record in records
        ]

        execution_runtime = self._deps.execution_runtime
        if execution_runtime is not None:
            for record in records:
                if record.status not in {
                    PortfolioComponentStatus.SUBMITTED,
                    PortfolioComponentStatus.WORKING,
                    PortfolioComponentStatus.PARTIALLY_FILLED,
                }:
                    continue
                try:
                    await execution_runtime.poll_order_status(record.client_order_id)
                except Exception as exc:  # noqa: BLE001
                    # Fail closed: without reconciled broker truth, no later
                    # authorized component may advance on restart.
                    logger.warning(
                        "portfolio_resume_reconciliation_failed",
                        extra={
                            "target_portfolio_decision_id": decision_id,
                            "client_order_id": record.client_order_id,
                            "error": str(exc),
                        },
                    )
                    return
        payload = dict(records[0].extra or {})
        proposal_raw = payload.get("execution_proposal")
        portfolio_decision = payload.get("portfolio_decision")
        authorized_positions = payload.get("authorized_positions")
        if (
            not isinstance(proposal_raw, dict)
            or not isinstance(portfolio_decision, dict)
            or not isinstance(authorized_positions, list)
        ):
            logger.warning(
                "portfolio_resume_payload_incomplete",
                extra={"target_portfolio_decision_id": decision_id},
            )
            return
        from joker.cognition.schemas import ExecutionProposal

        proposal = ExecutionProposal.model_validate(proposal_raw)
        resume_state: dict[str, Any] = {
            "session_id": self._session_id,
            "run_id": self._run_id,
            "cycle_id": proposal.cycle_id,
            "snapshot_id": str(proposal.snapshot_id),
            "execution_proposal": proposal,
            "execution_command_id": None,
            "_execution_command_ids": None,
            "_target_portfolio_decision": portfolio_decision,
            "_target_authorized_positions": authorized_positions,
            "evidence": [],
            "errors": [],
            "node_trace": [],
            "_block_new_entries": bool(self._reconciliation_only_recovery),
            "_reconciliation_only_recovery": bool(self._reconciliation_only_recovery),
        }
        submit_node = self._decision_graph.nodes["submit_execution_command"]
        await submit_node.ainvoke(resume_state)

    async def _resume_portfolio_for_order(self, client_order_id: str) -> None:
        registry = self._deps.provenance_registry
        if registry is None or self._deps.clock is None:
            return
        from joker.persistence.cognitive_execution_provenance import (
            PortfolioExecutionOwner,
        )

        broker_account_id = self._broker_account_identity()
        if not broker_account_id:
            return
        owner = PortfolioExecutionOwner(
            session_id=self._session_id,
            broker_account_identity=broker_account_id,
            trading_date=self._stable_trading_date(),
        )
        component = await registry.portfolio_executions.get_by_client_order_id(
            client_order_id, owner=owner
        )
        if component is None:
            return
        await self._resume_portfolio_decision(component.target_portfolio_decision_id)

    async def _resume_pending_portfolio_reoptimizations(self) -> None:
        """Run durable continuation optimization even while positions are open."""
        if self._reconciliation_only_recovery:
            return
        registry = self._deps.provenance_registry
        if registry is None or self._decision_graph is None or self._deps.clock is None:
            return
        broker_account_id = self._broker_account_identity()
        if not broker_account_id:
            return
        requests = await registry.portfolio_reoptimizations.list_pending(
            session_id=self._session_id,
            broker_account_identity=broker_account_id,
            trading_date=self._stable_trading_date(),
        )
        for request in requests:
            event = make_event(
                EventType.MARKET_SNAPSHOT_CREATED,
                session_id=self._session_id,
                source="portfolio_reoptimization",
                exchange_timestamp=self._deps.clock.now(),
                payload={
                    "snapshot_id": request.latest_snapshot_id,
                    "portfolio_reoptimization_request_id": request.request_id,
                    "cycle_id": f"portfolio-reoptimization-{request.request_id}",
                },
            )
            await self._invoke_decision_graph(event)

    async def _schedule_reoptimization_retry(
        self, request: Any, *, lease_expiry: str
    ) -> None:
        """Retry a fenced request after the current owner's lease expires."""
        if self._deps.provenance_registry is None or self._deps.clock is None:
            return
        try:
            expiry = datetime.fromisoformat(lease_expiry)
        except ValueError:
            return
        if expiry.tzinfo is None:
            return
        existing = self._reoptimization_retry_tasks.get(str(request.request_id))
        if existing is not None and not existing.done():
            return
        delay = max(0.0, (expiry - self._deps.clock.now()).total_seconds()) + 0.05
        request_id = str(request.request_id)
        attempt_generation = int(getattr(request, "attempt_generation", 0) or 0)

        async def _retry() -> None:
            try:
                await asyncio.sleep(delay)
                if self._shutdown or self._reconciliation_only_recovery:
                    return
                latest = await self._deps.provenance_registry.portfolio_reoptimizations.get(
                    request_id
                )
                if latest is None:
                    return
                if latest.status.value not in {"PENDING", "RUNNING"}:
                    return
                if (
                    latest.status.value == "RUNNING"
                    and latest.attempt_owner_run_id != self._run_id
                    and latest.attempt_lease_expires_at
                ):
                    try:
                        latest_expiry = datetime.fromisoformat(latest.attempt_lease_expires_at)
                    except ValueError:
                        latest_expiry = None
                    if latest_expiry is not None and latest_expiry.tzinfo is not None:
                        if self._deps.clock.now() < latest_expiry:
                            if latest.attempt_generation != attempt_generation:
                                self._reoptimization_retry_tasks.pop(request_id, None)
                            await self._schedule_reoptimization_retry(
                                latest, lease_expiry=latest.attempt_lease_expires_at
                            )
                            return
                await self._resume_pending_portfolio_reoptimizations()
            finally:
                current = self._reoptimization_retry_tasks.get(request_id)
                if current is asyncio.current_task():
                    self._reoptimization_retry_tasks.pop(request_id, None)

        self._reoptimization_retry_tasks[request_id] = asyncio.create_task(
            _retry(), name=f"portfolio-reoptimization-retry:{request_id}"
        )

    async def _resume_unfinished_cycles(self) -> None:
        registry = self._deps.cycle_registry
        if registry is None or self._deps.checkpointer is None:
            return
        from joker.graph.langgraph_checkpointer import ainvoke_config
        from joker.persistence.cognitive_cycle_registry import CognitiveCycleRecord

        resumable = await registry.list_resumable(self._session_id)
        for record in resumable:
            graph = (
                self._decision_graph if record.graph_kind == "decision" else self._position_graph
            )
            if graph is None:
                continue
            config = ainvoke_config(
                session_id=record.session_id,
                graph_kind=record.graph_kind,
                cycle_id=record.cycle_id,
            )
            await registry.upsert(
                CognitiveCycleRecord(
                    session_id=record.session_id,
                    graph_kind=record.graph_kind,
                    cycle_id=record.cycle_id,
                    trigger_event_id=record.trigger_event_id,
                    snapshot_id=record.snapshot_id,
                    status="running",
                    checkpoint_thread_id=record.checkpoint_thread_id,
                    last_completed_node=record.last_completed_node,
                    parent_entry_cycle_id=record.parent_entry_cycle_id,
                    original_strategy_id=record.original_strategy_id,
                    original_proposal_id=record.original_proposal_id,
                    payload=record.payload,
                )
            )
            try:
                from joker.cognition.prompt_overrides import pinned_applied_configuration

                applied = None
                payload = record.payload or {}
                cfg_raw = payload.get("configuration_version_id")
                if self._evolution_runtime is not None and cfg_raw:
                    from uuid import UUID as _UUID

                    applied = await self._evolution_runtime.apply_configuration_version(
                        record.cycle_id, _UUID(str(cfg_raw))
                    )
                if applied is not None:
                    with pinned_applied_configuration(applied):
                        resumed = await graph.ainvoke(None, config=config)
                else:
                    resumed = await graph.ainvoke(None, config=config)
                terminal_ok = self._cycle_reached_terminal_outcome(
                    resumed, graph_kind=record.graph_kind
                )
                await registry.upsert(
                    CognitiveCycleRecord(
                        session_id=record.session_id,
                        graph_kind=record.graph_kind,
                        cycle_id=record.cycle_id,
                        trigger_event_id=record.trigger_event_id,
                        snapshot_id=record.snapshot_id,
                        status="completed" if terminal_ok else "running",
                        checkpoint_thread_id=record.checkpoint_thread_id,
                        last_completed_node=record.last_completed_node,
                        parent_entry_cycle_id=record.parent_entry_cycle_id,
                        original_strategy_id=record.original_strategy_id,
                        original_proposal_id=record.original_proposal_id,
                        payload={
                            **(record.payload or {}),
                            "recovery_terminal_ok": terminal_ok,
                        },
                    )
                )
                if not terminal_ok:
                    self._status = "degraded"
                    logger.warning(
                        "cycle_resume_incomplete",
                        extra={
                            "cycle_id": record.cycle_id,
                            "errors": [
                                getattr(e, "error_code", None)
                                for e in (resumed or {}).get("errors") or []
                            ],
                        },
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "cycle_resume_failed",
                    extra={"cycle_id": record.cycle_id, "error": str(exc)},
                )
                self._status = "degraded"

    @staticmethod
    def _cycle_reached_terminal_outcome(
        state: dict[str, Any] | None,
        *,
        graph_kind: str,
    ) -> bool:
        """Return True only for a valid terminal graph outcome without blocking errors."""
        if not isinstance(state, dict):
            return False
        errors = state.get("errors") or []
        blocking_codes = {
            "no_submit_callback",
            "no_order_action_gateway",
            "gateway_blocked",
            "submit_validation_failed",
            "validation_failed",
            "cycle_recovery_failed",
        }
        for err in errors:
            code = getattr(err, "error_code", None)
            if code is None and isinstance(err, dict):
                code = err.get("error_code")
            if code in blocking_codes:
                return False
        traces = state.get("node_trace") or []
        terminal_nodes = {
            "persist_cycle",
            "persist_pending_cycle",
            "persist_evidence_request",
            "persist_stale",
            "route_position_action",
        }
        for trace in traces:
            name = getattr(trace, "node_name", None)
            status = getattr(trace, "status", None)
            if name is None and isinstance(trace, dict):
                name = trace.get("node_name")
                status = trace.get("status")
            if name in terminal_nodes and status == "completed":
                return True
        if graph_kind == "decision" and state.get("execution_command_id"):
            return True
        if graph_kind == "position" and state.get("_position_command_id"):
            return True
        # Delayed / evidence / hold without order still count if a persist node ran.
        return False

    async def _validate_reoptimization_result(
        self,
        request: Any,
        result: Any,
    ) -> tuple[bool, str, str | None, str | None]:
        """Validate graph output before durable reoptimization completion."""
        if not isinstance(result, dict):
            return False, "reoptimization_result_not_dictionary", None, None
        blocking_error_codes = {
            "no_submit_callback",
            "no_order_action_gateway",
            "gateway_blocked",
            "submit_validation_failed",
            "validation_failed",
            "cycle_recovery_failed",
            "target_attainment_recalculation_required",
        }
        for error in result.get("errors") or []:
            code = getattr(error, "error_code", None)
            if code is None and isinstance(error, dict):
                code = error.get("error_code")
            if code in blocking_error_codes:
                return False, "reoptimization_graph_has_blocking_error", None, None
        if not self._cycle_reached_terminal_outcome(result, graph_kind="decision"):
            return False, "reoptimization_terminal_outcome_missing", None, None
        if str(result.get("_portfolio_reoptimization_request_id") or "") != request.request_id:
            return False, "reoptimization_request_provenance_mismatch", None, None
        if result.get("_portfolio_execution_owner") != {
            "session_id": request.session_id,
            "broker_account_identity": request.broker_account_identity,
            "trading_date": request.trading_date,
        }:
            return False, "reoptimization_owner_provenance_mismatch", None, None
        replacement_raw = result.get("_target_portfolio_decision")
        if not isinstance(replacement_raw, dict):
            return False, "replacement_decision_missing", None, None
        replacement_id = str(replacement_raw.get("decision_id") or "")
        if not replacement_id or replacement_id == request.original_portfolio_decision_id:
            return False, "replacement_decision_id_invalid", None, None
        action = str(replacement_raw.get("action") or "").upper()
        if action not in {"WAIT", "ENTER"}:
            return False, "replacement_action_invalid", replacement_id, action or None
        positions = replacement_raw.get("authorized_positions")
        state_positions = result.get("_target_authorized_positions")
        if not isinstance(positions, list) or not isinstance(state_positions, list):
            return False, "replacement_authority_invalid", replacement_id, action
        if positions != state_positions:
            return False, "replacement_authority_channels_disagree", replacement_id, action
        proposal = result.get("execution_proposal")
        command_ids = result.get("_execution_command_ids") or []
        if action == "WAIT":
            if positions or proposal is not None or command_ids or result.get("execution_command_id"):
                return False, "wait_reoptimization_retains_authority", replacement_id, action
            return True, "", replacement_id, action

        if not positions or proposal is None:
            return False, "enter_reoptimization_missing_authority", replacement_id, action
        old_tuple_ids = set(request.already_filled_tuple_ids) | set(
            request.remaining_authorized_tuple_ids
        )
        tuple_ids = [str(position.get("position_tuple_id") or "") for position in positions]
        if any(not tuple_id for tuple_id in tuple_ids) or len(tuple_ids) != len(set(tuple_ids)):
            return False, "replacement_tuple_identity_invalid", replacement_id, action
        if old_tuple_ids.intersection(tuple_ids):
            return False, "replacement_reuses_old_tuple", replacement_id, action
        existing_contract_ids = {
            str(position.get("contract_id") or position.get("symbol") or "")
            for position in request.open_positions
        }
        current_open_contract_ids = {
            str(contract_id)
            for contract_id in (result.get("_reoptimization_excluded_contract_ids") or [])
        }
        selected_contract_ids = {
            str(position.get("contract_id") or "") for position in positions
        }
        if (existing_contract_ids | current_open_contract_ids).intersection(
            selected_contract_ids
        ):
            return False, "replacement_selects_existing_contract", replacement_id, action
        current_objective_version = int(
            result.get("_reoptimization_expected_objective_version") or -1
        )
        latest_snapshot_id = str(
            result.get("_reoptimization_expected_snapshot_id") or ""
        )
        if current_objective_version < 0 or not latest_snapshot_id:
            return False, "reoptimization_expected_provenance_missing", replacement_id, action
        evaluated_objective_version = int(
            replacement_raw.get("objective_version") or -1
        )
        if evaluated_objective_version < current_objective_version:
            return False, "replacement_objective_provenance_stale", replacement_id, action
        if str(replacement_raw.get("snapshot_id") or "") != latest_snapshot_id:
            return False, "replacement_snapshot_provenance_stale", replacement_id, action
        for position in positions:
            if (
                int(position.get("objective_version") or -1)
                != evaluated_objective_version
                or str(position.get("snapshot_id") or "") != latest_snapshot_id
                or str(position.get("decision_id") or "") != replacement_id
            ):
                return False, "replacement_tuple_provenance_stale", replacement_id, action
        registry = self._deps.provenance_registry
        if registry is None:
            return False, "replacement_durable_authority_missing", replacement_id, action
        durable_components = await registry.portfolio_executions.list_by_decision(
            replacement_id,
            owner=request.owner,
        )
        durable_by_tuple = {
            component.authorized_position_tuple_id: component
            for component in durable_components
        }
        if set(tuple_ids) != set(durable_by_tuple):
            return False, "replacement_durable_authority_mismatch", replacement_id, action
        from joker.persistence.cognitive_execution_provenance import (
            PortfolioComponentStatus,
            stable_portfolio_client_order_id,
        )

        expected_indexes = list(range(len(durable_components)))
        if (
            [component.component_index for component in durable_components]
            != expected_indexes
            or any(
                component.component_count != len(durable_components)
                or component.target_portfolio_decision_id != replacement_id
                for component in durable_components
            )
        ):
            return False, "replacement_component_order_invalid", replacement_id, action

        phase = "filled_prefix"
        active_count = 0
        unsubmitted = {
            PortfolioComponentStatus.AUTHORIZED,
            PortfolioComponentStatus.READY,
        }
        active = {
            PortfolioComponentStatus.SUBMITTED,
            PortfolioComponentStatus.WORKING,
            PortfolioComponentStatus.PARTIALLY_FILLED,
        }
        for component in durable_components:
            position = next(
                item
                for item in positions
                if str(item.get("position_tuple_id") or "")
                == component.authorized_position_tuple_id
            )
            if (
                component.contract_id != str(position.get("contract_id") or "")
                or component.authorized_quantity != int(position.get("quantity") or 0)
                or component.original_decision_snapshot_id != latest_snapshot_id
                or component.evaluated_objective_version != evaluated_objective_version
            ):
                return False, "replacement_component_authority_invalid", replacement_id, action
            if component.strategy_id != str(position.get("strategy_id") or ""):
                return False, "replacement_strategy_mismatch", replacement_id, action
            if component.selected_portfolio_id != str(
                replacement_raw.get("selected_portfolio_id") or ""
            ):
                return False, "replacement_portfolio_id_mismatch", replacement_id, action
            if component.capital_allocation != Decimal(
                str(position.get("capital_allocation") or "0")
            ):
                return False, "replacement_capital_allocation_mismatch", replacement_id, action
            if component.client_order_id != stable_portfolio_client_order_id(
                replacement_id, component.authorized_position_tuple_id
            ):
                return False, "replacement_component_authority_invalid", replacement_id, action
            if component.status == PortfolioComponentStatus.FILLED:
                if phase != "filled_prefix":
                    return False, "replacement_component_sequence_invalid", replacement_id, action
                if (
                    component.filled_quantity != component.authorized_quantity
                    or not component.broker_order_id
                    or component.submission_objective_version is None
                    or component.submission_objective_version
                    < component.evaluated_objective_version
                    or component.latest_validation_snapshot_id != latest_snapshot_id
                    or not component.continuation_ready
                    or component.post_fill_objective_version is None
                    or not component.post_fill_objective_fingerprint
                    or component.post_fill_snapshot_id != latest_snapshot_id
                    or not component.post_fill_exchange_time
                    or component.reconciled_filled_quantity != component.authorized_quantity
                ):
                    return False, "replacement_submission_provenance_invalid", replacement_id, action
                continue
            if component.status in active:
                if phase == "authorized_suffix" or active_count:
                    return False, "replacement_component_sequence_invalid", replacement_id, action
                phase = "active"
                active_count += 1
                if (
                    not component.broker_order_id
                    or component.submitted_quantity <= 0
                    or component.submission_objective_version is None
                    or component.submission_objective_version < evaluated_objective_version
                    or component.latest_validation_snapshot_id != latest_snapshot_id
                ):
                    return False, "replacement_submission_provenance_invalid", replacement_id, action
                continue
            if component.status in unsubmitted:
                phase = "authorized_suffix"
                if component.submitted_quantity != 0 or component.filled_quantity != 0:
                    return False, "replacement_unsubmitted_quantity_invalid", replacement_id, action
                if component.status == PortfolioComponentStatus.READY and (
                    component.latest_validation_snapshot_id != latest_snapshot_id
                ):
                    return False, "replacement_ready_provenance_invalid", replacement_id, action
                continue
            return False, "replacement_component_sequence_invalid", replacement_id, action
        completed_nodes: set[str] = set()
        for trace in result.get("node_trace") or []:
            name = getattr(trace, "node_name", None)
            status = getattr(trace, "status", None)
            if isinstance(trace, dict):
                name = name or trace.get("node_name")
                status = status or trace.get("status")
            if name and status == "completed":
                completed_nodes.add(str(name))
        if not {"validate_execution_proposal", "submit_execution_command"}.issubset(
            completed_nodes
        ):
            return False, "replacement_execution_validation_incomplete", replacement_id, action
        return True, "", replacement_id, action

    async def pause_event_workers(self) -> None:
        """Stop event-driven decision/position workers without tearing down deps.

        Used when a caller owns a single controlled graph invoke (acceptance
        tests / diagnostics) and must avoid concurrent SQLite writers on the
        shared Task-1/3 database.
        """
        self._shutdown = True
        pending = list(self._active_decision_tasks | self._active_position_tasks)
        if pending:
            _done, still = await asyncio.wait(pending, timeout=5.0)
            for task in still:
                task.cancel()
            for task in still:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        for worker in (self._decision_worker, self._position_worker):
            if worker is not None:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
        retry_tasks = list(self._reoptimization_retry_tasks.values())
        for task in retry_tasks:
            task.cancel()
        for task in retry_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._reoptimization_retry_tasks.clear()
        self._decision_worker = None
        self._position_worker = None
        self._status = "paused"

    async def shutdown(self) -> None:
        """Checkpoint active cycles rather than merely cancelling them."""
        self._status = "shutting_down"
        await self.pause_event_workers()
        if self._checkpointer_helper is not None:
            await self._checkpointer_helper.close()
            self._deps.checkpointer = None
        await self._registry.aclose()
        from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers

        await drain_aiosqlite_workers()
        self._started = False

    async def health(self) -> CognitiveRuntimeHealth:
        local_ok = await self._provider_available(local_only=True)
        remote_ok = await self._provider_available(local_only=False)
        status = self._status
        if not local_ok and not remote_ok:
            status = "unavailable"
        elif self._counters.last_error is not None:
            status = "degraded"
        return CognitiveRuntimeHealth(
            status=status,  # type: ignore[arg-type]
            local_provider_available=local_ok,
            remote_provider_available=remote_ok,
            active_decision_cycles=self._counters.active_decision_cycles,
            active_position_cycles=self._counters.active_position_cycles,
            queued_events=self._decision_queue.qsize() + self._position_queue.qsize(),
            last_success_at=self._counters.last_success_at,
            last_error=self._counters.last_error,
        )

    async def on_event(self, event: DomainEvent) -> None:
        """Enqueue cognitive work and return immediately — never drops position/order events."""
        self._received_events.append(event)
        if not self._started or self._shutdown:
            return

        if event.event_type == EventType.MARKET_SNAPSHOT_CREATED:
            # Never await ledger projection on the bus handler path — that can
            # exceed handler_timeout under SQLite contention and cancel routing.
            self._sequence += 1
            await self._decision_queue.put(
                _QueuedWork(
                    priority=_Priority.NEW_ENTRY_SNAPSHOT,
                    sequence=self._sequence,
                    event=event,
                    kind="snapshot_route",
                )
            )
            self._counters.queued_events = (
                self._decision_queue.qsize() + self._position_queue.qsize()
            )
            return

        priority = self._event_priority(event)
        kind = self._classify(event)
        self._sequence += 1
        work = _QueuedWork(priority=priority, sequence=self._sequence, event=event, kind=kind)
        if kind in {"position", "order"}:
            await self._position_queue.put(work)
        else:
            await self._decision_queue.put(work)
        self._counters.queued_events = self._decision_queue.qsize() + self._position_queue.qsize()

    async def _enqueue_snapshot_work(self, event: DomainEvent) -> None:
        """Route snapshots using authoritative ledger projection, not event metadata."""
        snapshot_id = str(
            event.payload.get("snapshot_id") or event.payload.get("market_snapshot_id") or ""
        )
        open_positions = await self._open_position_contract_ids()
        working_entry = await self._has_working_entry_order()
        if open_positions and self._config.position.enabled:
            for contract_id in open_positions:
                enriched = make_event(
                    EventType.MARKET_SNAPSHOT_CREATED,
                    session_id=event.session_id,
                    source="cognitive_runtime_position_route",
                    exchange_timestamp=event.exchange_timestamp,
                    correlation_id=event.correlation_id,
                    causation_id=event.event_id,
                    payload={
                        **dict(event.payload),
                        "snapshot_id": snapshot_id,
                        "position_id": contract_id,
                        "contract_id": contract_id,
                        "active_position_id": contract_id,
                        "trigger_event_id": str(event.event_id),
                    },
                )
                self._sequence += 1
                await self._position_queue.put(
                    _QueuedWork(
                        priority=_Priority.POSITION_SNAPSHOT,
                        sequence=self._sequence,
                        event=enriched,
                        kind="position",
                    )
                )
        # New-entry only when flat, no working entry, and not suppressed (shadow).
        if not open_positions and not working_entry and not self._suppress_new_entry_snapshots:
            if self._config.market_snapshot_coalescing and self._new_entry_in_flight:
                self._pending_new_entry_snapshot = snapshot_id or self._pending_new_entry_snapshot
            else:
                self._sequence += 1
                await self._decision_queue.put(
                    _QueuedWork(
                        priority=_Priority.NEW_ENTRY_SNAPSHOT,
                        sequence=self._sequence,
                        event=event,
                        kind="decision",
                    )
                )
        self._counters.queued_events = self._decision_queue.qsize() + self._position_queue.qsize()

    async def _has_working_entry_order(self) -> bool:
        if self._deps.projection_loader is None:
            return False
        try:
            projection = await self._deps.projection_loader()
        except Exception:
            return False
        from joker.runtime.order_action_gateway import (
            has_working_entry_order,
            working_orders_from_projection,
        )

        return has_working_entry_order(working_orders_from_projection(projection))

    async def _open_position_contract_ids(self) -> list[str]:
        if self._deps.projection_loader is None:
            return []
        try:
            projection = await self._deps.projection_loader()
        except Exception as exc:  # noqa: BLE001
            logger.warning("projection_loader_failed", extra={"error": str(exc)})
            return []
        if projection is None:
            return []
        positions = getattr(projection, "positions", None) or {}
        open_ids: list[str] = []
        items = (
            positions.items()
            if isinstance(positions, dict)
            else ((getattr(p, "contract_id", None), p) for p in positions)
        )
        for key, pos in items:
            qty = getattr(pos, "quantity", None)
            if qty is None and isinstance(pos, dict):
                qty = pos.get("quantity") or pos.get("net_quantity")
            try:
                from decimal import Decimal

                q = Decimal(str(qty)) if qty is not None else Decimal("0")
            except Exception:
                continue
            if q == 0:
                continue
            cid = getattr(pos, "contract_id", None) or key
            if cid is None and isinstance(pos, dict):
                cid = pos.get("contract_id")
            if cid:
                open_ids.append(str(cid))
        return open_ids

    def _classify(self, event: DomainEvent) -> str:
        if event.event_type in {
            EventType.POSITION_OPENED,
            EventType.POSITION_CHANGED,
            EventType.POSITION_CLOSED,
        }:
            return "position"
        if event.event_type in {
            EventType.ORDER_FILLED,
            EventType.ORDER_PARTIALLY_FILLED,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_ACCEPTED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
        }:
            return "order"
        if event.event_type == EventType.MARKET_SNAPSHOT_CREATED:
            if event.payload.get("active_position_id") or event.payload.get("position_id"):
                return "position"
            return "decision"
        return "decision"

    def _event_priority(self, event: DomainEvent) -> int:
        if event.event_type in {
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_ACCEPTED,
            EventType.ORDER_PARTIALLY_FILLED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
            EventType.POSITION_OPENED,
            EventType.POSITION_CHANGED,
            EventType.POSITION_CLOSED,
        }:
            return _Priority.CRITICAL
        if event.event_type == EventType.MARKET_SNAPSHOT_CREATED:
            if event.payload.get("active_position_id") or event.payload.get("position_id"):
                return _Priority.POSITION_SNAPSHOT
            return _Priority.NEW_ENTRY_SNAPSHOT
        return _Priority.NEW_ENTRY_SNAPSHOT

    async def _decision_worker_loop(self) -> None:
        while not self._shutdown:
            try:
                work = await asyncio.wait_for(self._decision_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if self._pending_new_entry_snapshot and not self._new_entry_in_flight:
                    await self._run_coalesced_new_entry()
                continue
            try:
                if work.kind == "snapshot_route":
                    await self._enqueue_snapshot_work(work.event)
                else:
                    await self._run_decision_work(work.event)
            except Exception as exc:
                logger.exception("cognitive decision worker error", exc_info=exc)
                self._counters.last_error = CognitiveError(
                    error_code="decision_worker_failure",
                    message=str(exc),
                    recoverable=True,
                )
                self._status = "degraded"
            finally:
                self._decision_queue.task_done()
                self._counters.queued_events = (
                    self._decision_queue.qsize() + self._position_queue.qsize()
                )

    async def _position_worker_loop(self) -> None:
        while not self._shutdown:
            try:
                work = await asyncio.wait_for(self._position_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                if work.kind == "order":
                    await self._run_order_work(work.event)
                else:
                    await self._run_position_work(work.event)
            except Exception as exc:
                logger.exception("cognitive position worker error", exc_info=exc)
                self._counters.last_error = CognitiveError(
                    error_code="position_worker_failure",
                    message=str(exc),
                    recoverable=True,
                )
                self._status = "degraded"
            finally:
                self._position_queue.task_done()
                self._counters.queued_events = (
                    self._decision_queue.qsize() + self._position_queue.qsize()
                )

    async def _run_coalesced_new_entry(self) -> None:
        snapshot_id = self._pending_new_entry_snapshot
        self._pending_new_entry_snapshot = None
        if not snapshot_id:
            return
        fake_event = make_event(
            EventType.MARKET_SNAPSHOT_CREATED,
            session_id=self._session_id,
            source="cognitive_runtime_coalesce",
            exchange_timestamp=datetime.now(timezone.utc),
            payload={"snapshot_id": snapshot_id},
        )
        await self._run_decision_work(fake_event)

    async def _run_decision_work(self, event: DomainEvent) -> None:
        if self._new_entry_in_flight:
            snapshot_id = str(
                event.payload.get("snapshot_id") or event.payload.get("market_snapshot_id") or ""
            )
            self._pending_new_entry_snapshot = snapshot_id or self._pending_new_entry_snapshot
            return
        async with self._new_entry_lock:
            if self._new_entry_in_flight:
                return
            self._new_entry_in_flight = True
            self._counters.active_decision_cycles += 1
            task = asyncio.create_task(self._invoke_decision_graph(event))
            self._active_decision_tasks.add(task)
            try:
                await task
            finally:
                self._active_decision_tasks.discard(task)
                self._new_entry_in_flight = False
                self._counters.active_decision_cycles = max(
                    0, self._counters.active_decision_cycles - 1
                )

    async def _invoke_decision_graph(self, event: DomainEvent) -> None:
        assert self._decision_graph is not None
        reoptimization_request = None
        reoptimization_request_id = str(
            event.payload.get("portfolio_reoptimization_request_id") or ""
        )
        if reoptimization_request_id:
            from joker.persistence.cognitive_execution_provenance import (
                PortfolioExecutionOwner,
            )

            registry = self._deps.provenance_registry
            if registry is None:
                return
            reoptimization_request = await registry.portfolio_reoptimizations.get(
                reoptimization_request_id
            )
            if reoptimization_request is None or self._deps.clock is None:
                return
            broker_account_id = self._broker_account_identity()
            if not broker_account_id:
                return
            runtime_owner = PortfolioExecutionOwner(
                session_id=self._session_id,
                broker_account_identity=broker_account_id,
                trading_date=self._stable_trading_date(),
            )
            if reoptimization_request.owner != runtime_owner:
                logger.warning(
                    "portfolio_reoptimization_owner_mismatch",
                    extra={"request_id": reoptimization_request_id},
                )
                return
        if await self._has_working_entry_order():
            logger.info(
                "decision_cycle_skipped_working_entry",
                extra={"event_id": str(event.event_id)},
            )
            return
        snapshot_id = str(
            event.payload.get("snapshot_id") or event.payload.get("market_snapshot_id") or ""
        )
        if not snapshot_id:
            return
        cycle_id = str(event.payload.get("cycle_id") or uuid4())
        if reoptimization_request is not None:
            from joker.persistence.cognitive_execution_provenance import (
                PortfolioAttemptLeaseActive,
                PortfolioReoptimizationStatus,
            )

            preclaim_request = reoptimization_request
            try:
                reoptimization_request = await (
                    self._deps.provenance_registry.portfolio_reoptimizations.begin_attempt(
                        reoptimization_request.request_id,
                        owner=reoptimization_request.owner,
                        current_run_id=self._run_id,
                        attempt_exchange_time=self._deps.clock.now().isoformat(),
                    )
                )
            except PortfolioAttemptLeaseActive:
                if preclaim_request.attempt_lease_expires_at:
                    await self._schedule_reoptimization_retry(
                        preclaim_request,
                        lease_expiry=preclaim_request.attempt_lease_expires_at,
                    )
                return
            if (
                preclaim_request.status == PortfolioReoptimizationStatus.RUNNING
                and reoptimization_request.state_version == preclaim_request.state_version
            ):
                # This run already owns an active attempt. Do not execute the
                # graph twice for a duplicate wake-up from the same process.
                return
        from joker.graph.langgraph_checkpointer import cognitive_thread_id
        from joker.persistence.cognitive_cycle_registry import CognitiveCycleRecord

        thread_id = cognitive_thread_id(
            session_id=self._session_id, graph_kind="decision", cycle_id=cycle_id
        )
        if self._deps.cycle_registry is not None:
            await self._deps.cycle_registry.upsert(
                CognitiveCycleRecord(
                    session_id=self._session_id,
                    graph_kind="decision",
                    cycle_id=cycle_id,
                    trigger_event_id=str(event.event_id),
                    snapshot_id=snapshot_id,
                    status="running",
                    checkpoint_thread_id=thread_id,
                )
            )
        state = initial_cycle_state(
            session_id=self._session_id,
            run_id=self._run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(event.event_id),
            trigger_event_type=event.event_type.value,
            snapshot_id=snapshot_id,
        )
        if reoptimization_request is not None:
            persisted_contract_ids = {
                str(position.get("contract_id") or position.get("symbol") or "")
                for position in reoptimization_request.open_positions
                if (position.get("contract_id") or position.get("symbol"))
            }
            current_open_contract_ids = set(await self._open_position_contract_ids())
            excluded_contract_ids = sorted(
                persisted_contract_ids | current_open_contract_ids
            )
            expected_objective_version = reoptimization_request.latest_objective_version
            if (
                self._deps.objective_service is not None
                and hasattr(self._deps.objective_service, "recompute_from_truth")
            ):
                recomputed = await self._deps.objective_service.recompute_from_truth(
                    now=self._deps.clock.now()
                )
                expected_objective_version = int(
                    getattr(recomputed, "version", expected_objective_version)
                )
            state.update(
                {
                    "_portfolio_reoptimization_request_id": (reoptimization_request.request_id),
                    "_portfolio_execution_owner": {
                        "session_id": reoptimization_request.session_id,
                        "broker_account_identity": (
                            reoptimization_request.broker_account_identity
                        ),
                        "trading_date": reoptimization_request.trading_date,
                    },
                    "_reoptimization_excluded_contract_ids": excluded_contract_ids,
                    "_reoptimization_existing_positions": list(
                        reoptimization_request.open_positions
                    ),
                    "_reoptimization_expected_objective_version": (
                        expected_objective_version
                    ),
                    "_reoptimization_expected_snapshot_id": (
                        reoptimization_request.latest_snapshot_id
                    ),
                }
            )
        if self._deps.event_bus is not None:
            await self._deps.event_bus.publish(
                make_event(
                    EventType.COGNITIVE_CYCLE_STARTED,
                    session_id=self._session_id,
                    source="cognitive_runtime",
                    exchange_timestamp=event.exchange_timestamp,
                    correlation_id=event.correlation_id,
                    causation_id=event.event_id,
                    payload={"cycle_id": cycle_id, "snapshot_id": snapshot_id},
                )
            )
        config = ainvoke_config(
            session_id=self._session_id,
            graph_kind="decision",
            cycle_id=cycle_id,
        )
        try:
            from joker.cognition.prompt_overrides import pinned_applied_configuration

            applied = None
            if self._evolution_runtime is not None:
                applied = await self._evolution_runtime.pin_and_apply_for_cycle(cycle_id)
            if applied is not None:
                with pinned_applied_configuration(applied):
                    result_state = await asyncio.wait_for(
                        self._decision_graph.ainvoke(state, config=config),
                        timeout=float(self._config.max_cycle_seconds),
                    )
            else:
                result_state = await asyncio.wait_for(
                    self._decision_graph.ainvoke(state, config=config),
                    timeout=float(self._config.max_cycle_seconds),
                )
            if reoptimization_request is not None:
                from joker.persistence.cognitive_execution_provenance import (
                    PortfolioReoptimizationStatus,
                )

                valid, reason, replacement_id, replacement_action = (
                    await self._validate_reoptimization_result(
                        reoptimization_request, result_state
                    )
                )
                if not valid:
                    raise RuntimeError(reason)
                await self._deps.provenance_registry.portfolio_reoptimizations.complete_attempt(
                    attempt=reoptimization_request,
                    completed_at=self._deps.clock.now().isoformat(),
                    replacement_decision_id=str(replacement_id),
                    replacement_action=str(replacement_action),
                )
            self._counters.last_success_at = datetime.now(timezone.utc)
            self._status = "healthy"
            if self._deps.cycle_registry is not None:
                terminal_ok = self._cycle_reached_terminal_outcome(
                    result_state, graph_kind="decision"
                )
                await self._deps.cycle_registry.upsert(
                    CognitiveCycleRecord(
                        session_id=self._session_id,
                        graph_kind="decision",
                        cycle_id=cycle_id,
                        trigger_event_id=str(event.event_id),
                        snapshot_id=snapshot_id,
                        status="completed" if terminal_ok else "running",
                        checkpoint_thread_id=thread_id,
                        payload={
                            "configuration_version_id": (
                                str(applied.configuration_version_id)
                                if applied is not None
                                else None
                            )
                        },
                    )
                )
        except asyncio.TimeoutError:
            self._counters.last_error = CognitiveError(
                error_code="cycle_timeout",
                message=f"cycle exceeded {self._config.max_cycle_seconds}s",
                recoverable=True,
            )
            self._status = "degraded"
            if reoptimization_request is not None:
                from joker.persistence.cognitive_execution_provenance import (
                    PortfolioReoptimizationStatus,
                )

                await self._deps.provenance_registry.portfolio_reoptimizations.fail_attempt(
                    attempt=reoptimization_request,
                    failed_at=self._deps.clock.now().isoformat(),
                    failure_reason="reoptimization_cycle_timeout",
                )
        except Exception as exc:
            self._counters.last_error = CognitiveError(
                error_code="cycle_failed",
                message=str(exc),
                recoverable=True,
            )
            self._status = "degraded"
            if reoptimization_request is not None:
                from joker.persistence.cognitive_execution_provenance import (
                    PortfolioReoptimizationStatus,
                )

                latest_request = await self._deps.provenance_registry.portfolio_reoptimizations.get(
                    reoptimization_request.request_id
                )
                if latest_request is not None and latest_request.status in {
                    PortfolioReoptimizationStatus.PENDING,
                    PortfolioReoptimizationStatus.RUNNING,
                }:
                    await self._deps.provenance_registry.portfolio_reoptimizations.fail_attempt(
                        attempt=reoptimization_request,
                        failed_at=self._deps.clock.now().isoformat(),
                        failure_reason=str(exc),
                    )

    async def _resolve_provenance(self, event: DomainEvent) -> dict[str, Any]:
        """Resolve cognitive metadata from registry using Task 1 event fields."""
        payload = dict(event.payload)
        client_order_id = str(payload.get("client_order_id") or "")
        contract_id = str(payload.get("contract_id") or "")
        registry = self._deps.provenance_registry
        record = None
        if registry is not None and client_order_id:
            record = await registry.get_by_client_order_id(client_order_id)
        if record is None and registry is not None and contract_id:
            record = await registry.get_latest_by_contract_id(contract_id)
        if record is not None:
            if not contract_id and record.contract_id:
                contract_id = record.contract_id
            payload.setdefault("snapshot_id", record.snapshot_id)
            payload.setdefault("strategy_id", record.strategy_id)
            payload.setdefault("original_strategy_id", record.strategy_id)
            payload.setdefault("proposal_id", record.proposal_id)
            payload.setdefault("decision_id", record.decision_id)
            payload.setdefault("cycle_id", record.cycle_id)
            if record.contract_id:
                payload.setdefault("contract_id", record.contract_id)
                contract_id = contract_id or record.contract_id
        # Task 1 uses contract_id as the authoritative position identity.
        if contract_id:
            payload.setdefault("position_id", contract_id)
            payload.setdefault("active_position_id", contract_id)
            payload.setdefault("contract_id", contract_id)
        # Fall back to latest market snapshot when provenance lacks snapshot_id.
        if not payload.get("snapshot_id") and self._deps.snapshot_repo is not None:
            try:
                latest = await self._deps.snapshot_repo.get_latest(self._session_id)
                if latest is not None:
                    payload["snapshot_id"] = str(latest.snapshot_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("latest_snapshot_lookup_failed", extra={"error": str(exc)})
        return payload

    async def _sync_objective_reservation(
        self, event: DomainEvent, *, client_order_id: str
    ) -> None:
        """Task 2 observes order events only — financial mutations are Task 1 owned.

        Objective capital projection runs via SessionSupervisor's
        ObjectiveCapitalProjector subscribed to the Task 1 event bus.
        """
        return

    async def _sync_objective_on_position_closed(self, event: DomainEvent) -> None:
        """Task 2 observes closes only — realised PnL / exposure release is Task 1."""
        return

    async def _run_position_work(self, event: DomainEvent) -> None:
        if not self._config.position.enabled:
            return
        assert self._position_graph is not None
        if event.event_type == EventType.POSITION_CLOSED:
            await self._sync_objective_on_position_closed(event)
            return
        resolved = await self._resolve_provenance(event)
        position_id = str(
            resolved.get("position_id")
            or resolved.get("active_position_id")
            or resolved.get("contract_id")
            or ""
        )
        if not position_id:
            logger.info(
                "position_work_skipped_missing_identity",
                extra={"event_type": event.event_type.value},
            )
            return
        snapshot_id = str(resolved.get("snapshot_id") or resolved.get("market_snapshot_id") or "")
        if not snapshot_id:
            logger.info(
                "position_work_skipped_missing_snapshot",
                extra={"position_id": position_id},
            )
            return
        # Position reassessment always gets its own cycle — never reuse entry.
        from joker.persistence.cognitive_cycle_registry import stable_position_cycle_id

        cycle_id = stable_position_cycle_id(
            self._session_id,
            str(event.event_id),
        )
        parent_entry_cycle_id = (
            str(resolved.get("parent_entry_cycle_id") or resolved.get("cycle_id") or "") or None
        )
        from joker.graph.langgraph_checkpointer import cognitive_thread_id
        from joker.persistence.cognitive_cycle_registry import CognitiveCycleRecord

        thread_id = cognitive_thread_id(
            session_id=self._session_id, graph_kind="position", cycle_id=cycle_id
        )
        if self._deps.cycle_registry is not None:
            await self._deps.cycle_registry.upsert(
                CognitiveCycleRecord(
                    session_id=self._session_id,
                    graph_kind="position",
                    cycle_id=cycle_id,
                    trigger_event_id=str(event.event_id),
                    snapshot_id=snapshot_id,
                    status="running",
                    checkpoint_thread_id=thread_id,
                    parent_entry_cycle_id=parent_entry_cycle_id,
                    original_strategy_id=str(
                        resolved.get("original_strategy_id") or resolved.get("strategy_id") or ""
                    )
                    or None,
                    original_proposal_id=str(resolved.get("proposal_id") or "") or None,
                )
            )
        self._counters.active_position_cycles += 1
        state: dict[str, Any] = {
            "session_id": self._session_id,
            "run_id": self._run_id,
            "cycle_id": cycle_id,
            "snapshot_id": snapshot_id,
            "_position_id": position_id,
            "_contract_id": resolved.get("contract_id") or position_id,
            "_original_strategy_id": resolved.get("original_strategy_id")
            or resolved.get("strategy_id"),
            "_parent_entry_cycle_id": parent_entry_cycle_id,
            "_original_proposal_id": resolved.get("proposal_id"),
            # Clear per-cycle transients explicitly.
            "_position_command_id": None,
            "_position_decision": None,
            "_position_thesis": None,
            "_position_critic_notes": None,
        }
        config = ainvoke_config(
            session_id=self._session_id,
            graph_kind="position",
            cycle_id=cycle_id,
        )
        applied = None
        if self._evolution_runtime is not None:
            contract = str(resolved.get("contract_id") or position_id)
            origin = self._evolution_runtime.originating_configuration_for_contract(contract)
            if origin is None and parent_entry_cycle_id:
                origin = self._evolution_runtime.get_pinned(parent_entry_cycle_id)
            if origin is not None:
                applied = await self._evolution_runtime.apply_configuration_version(
                    cycle_id, origin
                )
            else:
                applied = await self._evolution_runtime.pin_and_apply_for_cycle(cycle_id)

        from joker.cognition.prompt_overrides import pinned_applied_configuration

        async def _invoke_position():
            if applied is not None:
                with pinned_applied_configuration(applied):
                    return await self._position_graph.ainvoke(state, config=config)
            return await self._position_graph.ainvoke(state, config=config)

        task = asyncio.create_task(_invoke_position())
        self._active_position_tasks.add(task)
        try:
            result_state = await task
            self._counters.last_success_at = datetime.now(timezone.utc)
            if self._deps.cycle_registry is not None:
                terminal_ok = self._cycle_reached_terminal_outcome(
                    result_state, graph_kind="position"
                )
                await self._deps.cycle_registry.upsert(
                    CognitiveCycleRecord(
                        session_id=self._session_id,
                        graph_kind="position",
                        cycle_id=cycle_id,
                        trigger_event_id=str(event.event_id),
                        snapshot_id=snapshot_id,
                        status="completed" if terminal_ok else "running",
                        checkpoint_thread_id=thread_id,
                        parent_entry_cycle_id=parent_entry_cycle_id,
                        original_strategy_id=str(
                            resolved.get("original_strategy_id")
                            or resolved.get("strategy_id")
                            or ""
                        )
                        or None,
                        original_proposal_id=str(resolved.get("proposal_id") or "") or None,
                        payload={
                            "configuration_version_id": (
                                str(applied.configuration_version_id)
                                if applied is not None
                                else None
                            )
                        },
                    )
                )
        finally:
            self._active_position_tasks.discard(task)
            self._counters.active_position_cycles = max(
                0, self._counters.active_position_cycles - 1
            )

    async def _run_order_work(self, event: DomainEvent) -> None:
        resolved = await self._resolve_provenance(event)
        client_order_id = str(resolved.get("client_order_id") or "")
        if not client_order_id:
            return
        await self._sync_objective_reservation(event, client_order_id=client_order_id)
        snapshot_id = str(resolved.get("snapshot_id") or resolved.get("market_snapshot_id") or "")
        order_projection: dict[str, Any] = {
            "client_order_id": client_order_id,
            "event_type": event.event_type.value,
            **{k: v for k, v in resolved.items() if k != "client_order_id"},
        }
        if self._deps.execution_runtime is not None:
            try:
                sync = await self._deps.execution_runtime.poll_order_status(client_order_id)
                if sync is not None:
                    contract = getattr(sync, "contract", None)
                    contract_payload = None
                    if contract is not None:
                        dump = getattr(contract, "model_dump", None)
                        contract_payload = dump(mode="json") if callable(dump) else str(contract)
                    order_projection.update(
                        {
                            "status": sync.status,
                            "quantity": sync.quantity,
                            "filled_quantity": getattr(sync, "filled_quantity", None),
                            "limit_price": getattr(sync, "limit_price", None),
                            "side": getattr(sync, "side", None),
                            "order_type": getattr(sync, "order_type", None)
                            or getattr(getattr(sync, "intent", None), "order_type", None),
                            "contract": contract_payload,
                            "_contract_obj": contract,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("order_projection_hydrate_failed", extra={"error": str(exc)})

        await self._resume_portfolio_for_order(client_order_id)
        await self._resume_pending_portfolio_reoptimizations()

        context: ContextPackage | None = None
        if snapshot_id and self._deps.snapshot_repo is not None:
            try:
                snapshot, data_quality, _surface, surface_slice = await load_snapshot_truth(
                    self._deps, snapshot_id
                )
                objective_context = None
                if self._deps.objective_state_loader is not None:
                    try:
                        from joker.objectives.schemas import state_to_context

                        obj_state = await self._deps.objective_state_loader()
                        objective_context = state_to_context(obj_state).model_dump_for_hash()
                    except Exception:
                        objective_context = None
                context = await assemble_role_context(
                    self._deps,
                    agent_role=AgentRole.ORDER_MANAGER,
                    session_id=self._session_id,
                    cycle_id=str(uuid4()),
                    snapshot=snapshot,
                    data_quality=data_quality,
                    option_surface_slice=surface_slice,
                    order_projection={
                        k: v for k, v in order_projection.items() if not str(k).startswith("_")
                    },
                    objective_context=objective_context,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("order_context_failed", extra={"error": str(exc)})

        if context is None:
            logger.info(
                "cognitive order event received without snapshot context",
                extra={"client_order_id": client_order_id},
            )
            return

        agent = OrderManagerAgent()
        applied = None
        if self._evolution_runtime is not None:
            contract = str(
                resolved.get("contract_id") or (order_projection or {}).get("contract_id") or ""
            )
            origin = None
            if contract:
                origin = self._evolution_runtime.originating_configuration_for_contract(contract)
            parent_cycle = str(resolved.get("cycle_id") or "")
            if origin is None and parent_cycle:
                origin = self._evolution_runtime.get_pinned(parent_cycle)
            if origin is not None:
                applied = await self._evolution_runtime.apply_configuration_version(
                    f"om:{client_order_id}", origin
                )
            else:
                applied = await self._evolution_runtime.pin_and_apply_for_cycle(
                    f"om:{client_order_id}"
                )
        from joker.cognition.prompt_overrides import pinned_applied_configuration

        if applied is not None:
            with pinned_applied_configuration(applied):
                decision = await agent.manage(
                    context,
                    self._router,
                    client_order_id=client_order_id,
                    order_projection=order_projection,
                )
        else:
            decision = await agent.manage(
                context,
                self._router,
                client_order_id=client_order_id,
                order_projection=order_projection,
            )
        decision_key = f"{client_order_id}:{decision.action}:{decision.rationale_summary}"
        if decision_key in self._order_decision_ids:
            return
        self._order_decision_ids.add(decision_key)
        if self._deps.order_management_repo is not None:
            await self._deps.order_management_repo.append(decision)
        await self._apply_order_decision(
            decision,
            order_projection=order_projection,
            trigger_event_id=str(event.event_id),
        )

    async def _apply_order_decision(
        self,
        decision: OrderManagementDecision,
        *,
        order_projection: dict[str, Any] | None = None,
        trigger_event_id: str | None = None,
    ) -> None:
        runtime = self._deps.execution_runtime
        if runtime is None:
            return
        action = decision.action
        if action == "continue_waiting":
            return
        source_state = str(
            (order_projection or {}).get("status")
            or (order_projection or {}).get("event_type")
            or "unknown"
        )
        from joker.persistence.order_management_actions import (
            OrderManagementActionRecord,
            make_order_management_action_key,
        )

        action_key = make_order_management_action_key(
            source_order_id=decision.client_order_id,
            source_order_state=source_state,
            trigger_event_id=str(trigger_event_id or ""),
            decision_id=str(decision.decision_id),
            action=str(action),
        )
        om_repo = self._deps.order_management_action_repo
        if om_repo is not None and await om_repo.has_key(action_key):
            logger.info(
                "order_management_action_idempotent_skip",
                extra={"action_key": action_key, "action": action},
            )
            return

        if action in {"cancel", "abandon"}:
            try:
                await runtime.cancel_order(client_order_id=decision.client_order_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "order_cancel_failed",
                    extra={
                        "client_order_id": decision.client_order_id,
                        "error": str(exc),
                    },
                )
                return
            if om_repo is not None:
                await om_repo.record(
                    OrderManagementActionRecord(
                        action_key=action_key,
                        session_id=self._session_id,
                        source_order_id=decision.client_order_id,
                        action=str(action),
                        source_order_state=source_state,
                        trigger_event_id=str(trigger_event_id or ""),
                        decision_id=str(decision.decision_id),
                    )
                )
            return
        if action in {"replace", "reduce_quantity"}:
            if self._deps.order_action_gateway is None:
                logger.warning("order_replace_skipped_no_gateway")
                return
            from joker.runtime.order_action_gateway import (
                OrderActionKind,
                OrderActionRequest,
            )

            projection = order_projection or {}
            contract = projection.get("_contract_obj") or projection.get("contract")
            if isinstance(contract, dict):
                from joker.schemas.domain import OptionContract

                contract = OptionContract.model_validate(contract)
            if contract is None:
                logger.warning(
                    "order_replace_skipped_missing_contract",
                    extra={"client_order_id": decision.client_order_id},
                )
                return
            from joker.runtime.execution_runtime import contract_id_for

            side = str(projection.get("side") or "buy")
            qty = int(
                decision.new_quantity
                if decision.new_quantity is not None
                else projection.get("quantity") or 1
            )
            if action == "reduce_quantity":
                open_qty = int(projection.get("quantity") or qty)
                filled = int(projection.get("filled_quantity") or 0)
                remaining = max(0, open_qty - filled)
                if decision.new_quantity is not None:
                    qty = min(int(decision.new_quantity), remaining)
                else:
                    qty = max(1, remaining // 2) if remaining > 1 else remaining
                if qty <= 0:
                    return
            limit = (
                float(decision.new_limit_price)
                if decision.new_limit_price is not None
                else (
                    float(projection["limit_price"])
                    if projection.get("limit_price") is not None
                    else None
                )
            )
            new_client_id = f"{decision.client_order_id}:replace:{decision.decision_id}"
            result = await self._deps.order_action_gateway.submit(
                OrderActionRequest(
                    action=OrderActionKind.REPLACE,
                    snapshot_id=str(decision.snapshot_id),
                    contract_id=contract_id_for(contract),
                    side=side,  # type: ignore[arg-type]
                    quantity=qty,
                    client_order_id=new_client_id,
                    limit_price=limit,
                    order_type="limit" if limit is not None else "market",
                    decision_id=str(decision.decision_id),
                    cycle_id=str(decision.cycle_id),
                    replace_of_client_order_id=decision.client_order_id,
                )
            )
            if result.submitted and om_repo is not None:
                await om_repo.record(
                    OrderManagementActionRecord(
                        action_key=action_key,
                        session_id=self._session_id,
                        source_order_id=decision.client_order_id,
                        action=str(action),
                        source_order_state=source_state,
                        trigger_event_id=str(trigger_event_id or ""),
                        decision_id=str(decision.decision_id),
                        replacement_client_order_id=new_client_id,
                    )
                )
            return

    async def _provider_available(self, *, local_only: bool) -> bool:
        if self._registry is None:
            return True
        try:
            if local_only:
                for name in ("fake", "ollama"):
                    provider = self._registry._providers.get(name)  # noqa: SLF001
                    if provider is not None:
                        health = await provider.healthcheck()
                        if health.status == "healthy":
                            return True
                return False
            provider = self._registry._providers.get("openai")  # noqa: SLF001
            if provider is None:
                return False
            health = await provider.healthcheck()
            return health.status == "healthy"
        except Exception:
            return False


def build_default_repositories(db_path: str | Path) -> dict[str, Any]:
    """Construct default cognitive repositories for a session database."""
    store = CognitiveArtifactStore(db_path)
    return {
        "evidence_repo": EvidenceRepository(store),
        "world_model_repo": WorldModelRepository(store),
        "hypothesis_repo": HypothesisRepository(store),
        "strategy_repo": StrategyRepository(store),
        "debate_repo": DebateRepository(store),
        "decision_repo": DecisionRepository(store),
        "position_thesis_repo": PositionThesisRepository(store),
        "order_management_repo": OrderManagementRepository(store),
        "model_call_repo": ModelCallRepository(store),
    }
