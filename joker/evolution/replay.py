"""Task 2 cognitive graph replay with isolated execution-faithful fills."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from joker.cognition.prompt_overrides import pinned_applied_configuration
from joker.evolution.configuration_applicator import ConfigurationApplicator
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.replay_execution import ReplayExecutionRuntime
from joker.evolution.replay_market import build_truth_from_episode
from joker.evolution.replay_position_runtime import ReplayPositionRuntime
from joker.evolution.repositories import ConfigurationVersionRepository
from joker.evolution.schemas import CognitiveConfigurationVersion, TradingEpisode
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import ainvoke_config
from joker.models.router import ModelRouter


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
        checkpointer_path: Path | None = None,
        checkpointer_saver: AsyncSqliteSaver | None = None,
        random_seed: int = 42,
    ) -> None:
        self._template = template_deps
        self._configs = config_repo
        self._applicator = ConfigurationApplicator(policy_store)
        self._checkpointer_path = checkpointer_path
        self._checkpointer_saver = checkpointer_saver
        self._random_seed = random_seed
        self.replay_count = 0
        self.shadow_count = 0
        self._shadow_runtimes: dict[str, ReplayPositionRuntime] = {}

    def _isolated_deps(self, router: ModelRouter | None = None) -> CognitiveGraphDeps:
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
            projection_loader=None,
            provenance_registry=None,
            order_action_gateway=None,
            cycle_registry=None,
            order_management_action_repo=None,
        )

    def _quote_pair(self, episode: TradingEpisode) -> tuple[Decimal, Decimal]:
        mid = episode.entry_price or Decimal("1.00")
        half = Decimal("0.01")
        return mid - half, mid + half

    def _exit_quote_pair(self, episode: TradingEpisode) -> tuple[Decimal, Decimal]:
        mid = episode.exit_price or episode.entry_price or Decimal("1.00")
        half = Decimal("0.01")
        return mid - half, mid + half

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
        deps = self._isolated_deps()
        if deps.execution_runtime is not None or deps.submit_callback is not None:
            raise CognitiveReplayError(
                "replay deps must not expose execution_runtime or submit_callback"
            )
        if deps.order_action_gateway is not None:
            raise CognitiveReplayError(
                "replay deps must not expose order_action_gateway"
            )

        truth = build_truth_from_episode(episode, random_seed=self._random_seed + sample)
        if episode.contract_id:
            bid, ask = self._quote_pair(episode)
            truth = truth.model_copy(
                update={
                    "contract_quotes": {
                        episode.contract_id: {
                            "bid": str(bid),
                            "ask": str(ask),
                            "mid": str((bid + ask) / Decimal("2")),
                        }
                    }
                }
            )
        execution = ReplayExecutionRuntime(truth=truth)
        if episode.contract_id:
            execution.lock_surface({episode.contract_id})
        position_rt = ReplayPositionRuntime(
            execution=execution, configuration_version_id=configuration_version_id
        )

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
            session_id=deps.session_id,
            graph_kind="decision",
            cycle_id=cycle_id,
        )
        with pinned_applied_configuration(applied):
            result = await graph.ainvoke(state, config=config)

        meta = result.get("meta_decision")
        action = getattr(meta, "action", None)
        action_value = getattr(action, "value", str(action) if action else "unknown")
        proposal = result.get("execution_proposal")
        selected = None
        if proposal is not None:
            selected = getattr(proposal, "contract_id", None) or getattr(
                proposal, "selected_contract_id", None
            )
        if selected is None and episode.contract_id and action_value in {
            "execute",
            "probe",
            "EXECUTE",
            "PROBE",
        }:
            # Faithful surface: challenger may only trade the frozen episode contract.
            selected = episode.contract_id

        node_trace = result.get("node_trace") or []
        position_rt.model_call_ids = [
            str(getattr(n, "model_call_id", "") or "")
            for n in node_trace
            if getattr(n, "model_call_id", None)
        ]

        bid, ask = self._quote_pair(episode)
        entry = position_rt.simulate_entry_from_meta(
            action=action_value,
            contract_id=str(selected) if selected else None,
            bid=bid,
            ask=ask,
            idempotency_key=(
                f"{episode.episode_id}:{configuration_version_id}:{sample}:entry"
            ),
        )
        if entry.get("traded"):
            exit_bid, exit_ask = self._exit_quote_pair(episode)
            # Opposite-direction / exit simulation uses frozen exit quotes, not
            # historical champion PnL.
            if action_value in {"execute", "probe", "EXECUTE", "PROBE"}:
                position_rt.simulate_exit(
                    bid=exit_bid,
                    ask=exit_ask,
                    idempotency_key=(
                        f"{episode.episode_id}:{configuration_version_id}:{sample}:exit"
                    ),
                )
            else:
                position_rt.mark("replay_finalised")

        payload = position_rt.outcome_payload()
        payload.update(
            {
                "realised_pnl": Decimal(str(payload["realised_pnl"])),
                "model_calls": max(1, len(node_trace)),
                "cost_gbp": Decimal("0.01") * Decimal(max(1, len(node_trace))),
                "meta_decision_action": action_value,
                "ran_task2_graph": True,
                "sample": sample,
                "latency_ms": int(10 * max(1, len(node_trace))),
                "historical_pnl_attributed": False,
            }
        )
        return payload

    async def run_challenger_shadow(
        self,
        challenger: CognitiveConfigurationVersion,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute challenger decision graph + isolated shadow ledger (no broker)."""
        self.shadow_count += 1
        applied = await self._applicator.apply(challenger)
        deps = self._isolated_deps()
        if deps.execution_runtime is not None or deps.submit_callback is not None:
            raise CognitiveReplayError(
                "shadow deps must not expose execution_runtime or submit_callback"
            )
        snapshot_id = str(item.get("snapshot_id") or uuid4())
        assignment_key = (
            f"{challenger.configuration_version_id}:{item.get('assignment_id')}"
        )
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
            session_id=deps.session_id,
            graph_kind="decision",
            cycle_id=cycle_id,
        )
        with pinned_applied_configuration(applied):
            result = await graph.ainvoke(state, config=config)
        meta = result.get("meta_decision")
        action_value = getattr(getattr(meta, "action", None), "value", None)

        runtime = self._shadow_runtimes.get(assignment_key)
        if runtime is None:
            from joker.evolution.replay_market import ReplayEpisodeTruth

            truth = ReplayEpisodeTruth(
                episode_id=uuid4(),
                initial_snapshot_id=UUID(snapshot_id)
                if _is_uuid(snapshot_id)
                else uuid4(),
                random_seed=self._random_seed,
            )
            execution = ReplayExecutionRuntime(truth=truth)
            runtime = ReplayPositionRuntime(
                execution=execution,
                configuration_version_id=challenger.configuration_version_id,
            )
            self._shadow_runtimes[assignment_key] = runtime

        contract_id = str(item.get("contract_id") or "") or None
        bid = Decimal(str(item.get("bid", "1.00")))
        ask = Decimal(str(item.get("ask", "1.02")))
        if runtime.stage in {"truth_loaded", "entry_graph_completed"} and not runtime.traded:
            runtime.simulate_entry_from_meta(
                action=str(action_value or "no_trade"),
                contract_id=contract_id,
                bid=bid,
                ask=ask,
                idempotency_key=f"shadow-entry:{assignment_key}:{snapshot_id}",
            )
        elif runtime.traded and item.get("exit"):
            runtime.simulate_exit(
                bid=bid,
                ask=ask,
                idempotency_key=f"shadow-exit:{assignment_key}:{snapshot_id}",
            )

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
            "open_at_end": runtime.open_at_end,
            "realised_pnl": str(runtime.execution.realised_pnl()),
        }

    def restore_shadow_runtime(
        self, key: str, runtime: ReplayPositionRuntime
    ) -> None:
        self._shadow_runtimes[key] = runtime


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
        return True
    except Exception:
        return False
