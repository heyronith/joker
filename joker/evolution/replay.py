"""Task 2 cognitive graph replay against frozen Task 1 truth (no broker)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from joker.cognition.prompt_overrides import pinned_configuration_overrides
from joker.evolution.configuration_applicator import ConfigurationApplicator
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.repositories import ConfigurationVersionRepository
from joker.evolution.schemas import CognitiveConfigurationVersion, TradingEpisode
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
from joker.models.router import ModelRouter


class CognitiveReplayError(RuntimeError):
    pass


class CognitiveReplayService:
    """Replay champion/challenger configs through Task 2 graphs without submission."""

    def __init__(
        self,
        *,
        template_deps: CognitiveGraphDeps,
        config_repo: ConfigurationVersionRepository,
        policy_store: PolicyVersionStore,
        checkpointer_path: Path | None = None,
    ) -> None:
        self._template = template_deps
        self._configs = config_repo
        self._applicator = ConfigurationApplicator(policy_store)
        self._checkpointer_path = checkpointer_path
        self.replay_count = 0
        self.shadow_count = 0

    def _isolated_deps(self, router: ModelRouter | None = None) -> CognitiveGraphDeps:
        """Copy read-only deps; strip all broker/execution write paths."""
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
            # Ephemeral cognition: do not share durable artifact stores across
            # champion/challenger/sample replays (canned IDs would collide).
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
            checkpointer=None,
            data_quality_loader=base.data_quality_loader,
            projection_loader=None,
            provenance_registry=None,
            order_action_gateway=None,
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
        deps = self._isolated_deps()
        if deps.execution_runtime is not None or deps.submit_callback is not None:
            raise CognitiveReplayError(
                "replay deps must not expose execution_runtime or submit_callback"
            )
        if deps.order_action_gateway is not None:
            raise CognitiveReplayError(
                "replay deps must not expose order_action_gateway"
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
        checkpointer: CognitiveCheckpointer | None = None
        try:
            if self._checkpointer_path is not None:
                path = self._checkpointer_path.with_name(
                    f"{self._checkpointer_path.stem}_{configuration_version_id}.db"
                )
                checkpointer = CognitiveCheckpointer(path)
                saver = await checkpointer.open()
                deps.checkpointer = saver
            graph = build_cognitive_graph(deps)
            config = ainvoke_config(
                session_id=deps.session_id,
                graph_kind="decision",
                cycle_id=cycle_id,
            )
            with pinned_configuration_overrides(
                configuration_version_id=str(applied.configuration_version_id),
                prompt_overrides=applied.prompt_overrides,
                role_profiles=applied.role_profiles,
            ):
                result = await graph.ainvoke(state, config=config)
        finally:
            if checkpointer is not None:
                await checkpointer.close()

        meta = result.get("meta_decision")
        action = getattr(meta, "action", None)
        action_value = getattr(action, "value", str(action) if action else "unknown")
        # Attribute historical PnL only when the challenger would still trade;
        # no-trade / abandon challengers get zero PnL vs historical closed trades.
        historical = episode.realised_pnl or Decimal("0")
        if episode.action_class == "closed_trade":
            if action_value in {"execute", "probe", "EXECUTE", "PROBE"}:
                pnl = historical
            else:
                pnl = Decimal("0")
        elif episode.action_class == "no_trade":
            pnl = Decimal("0") if action_value not in {"execute", "probe"} else historical
        else:
            pnl = historical

        return {
            "realised_pnl": pnl,
            "model_calls": len(result.get("node_trace") or []) or 1,
            "cost_gbp": Decimal("0.01"),
            "broker_submit": False,
            "execution_runtime": False,
            "meta_decision_action": action_value,
            "configuration_version_id": str(configuration_version_id),
            "ran_task2_graph": True,
            "sample": sample,
        }

    async def run_challenger_shadow(
        self,
        challenger: CognitiveConfigurationVersion,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute challenger decision graph for a live snapshot without broker access."""
        self.shadow_count += 1
        applied = await self._applicator.apply(challenger)
        deps = self._isolated_deps()
        if deps.execution_runtime is not None or deps.submit_callback is not None:
            raise CognitiveReplayError(
                "shadow deps must not expose execution_runtime or submit_callback"
            )
        snapshot_id = str(item.get("snapshot_id") or uuid4())
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
        with pinned_configuration_overrides(
            configuration_version_id=str(applied.configuration_version_id),
            prompt_overrides=applied.prompt_overrides,
            role_profiles=applied.role_profiles,
        ):
            result = await graph.ainvoke(state, config=config)
        meta = result.get("meta_decision")
        return {
            "action": "challenger_shadow_decision",
            "meta_decision_action": getattr(
                getattr(meta, "action", None), "value", None
            ),
            "snapshot_id": snapshot_id,
            "shadow": True,
            "broker_submit": False,
            "execution_runtime": False,
            "ran_challenger_graph": True,
            "challenger_version_id": str(challenger.configuration_version_id),
            "configuration_hash": challenger.content_hash,
        }
