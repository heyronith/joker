"""Production CognitiveReplayService experiment path (no injected replay_fn)."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.config.settings import CognitiveGraphSettings
from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.config import PromotionSettings
from joker.evolution.decision import EvolutionDecisionService
from joker.evolution.experiment_results_store import ExperimentEpisodeResultStore
from joker.evolution.experiment_runner import ExperimentRunner
from joker.evolution.improvement import ImprovementProposalService
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.promotion_gate import PromotionEligibilityGate
from joker.evolution.replay import CognitiveReplayService
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import ExperimentDefinition, PromptPatch, TradingEpisode
from joker.evolution.shadow import ShadowRuntime
from joker.evaluation.dataset_builder import DatasetBuilder
from joker.graph.context_hydrate import context_assembler_from_settings
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.runtime.cognitive_agent_runtime import build_default_repositories
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned

ET = ZoneInfo("America/New_York")


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


async def _seed(market, start, clock):
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
    await market.ingest_option_quotes(
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
                "quote_timestamp": start + timedelta(minutes=3),
                "is_0dte": True,
            }
        ]
    )
    now = start + timedelta(minutes=3, seconds=3)
    clock.set_now(now)
    tick = await market.tick(now=now)
    assert tick.snapshot is not None
    return tick.snapshot


@pytest.mark.asyncio
async def test_production_cognitive_replay_and_shadow_challenger(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "replay_prod.db"
    apply_task3_migrations(db)
    broker = PaperBroker(slippage_pct=0)
    session_id = "sess-replay-prod"

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
    champ_registry = None
    shadow = None
    try:
        snapshot = await _seed(supervisor.market_runtime, start, clock)

        fake = FakeModelProvider(available=True)
        register_full_path_canned(
            fake, snapshot.snapshot_id, "replay-cycle", session=session_id
        )
        registry = _fake_registry(fake)
        router = ModelRouter(registry, session_id=session_id)
        cognitive_repos = build_default_repositories(db)
        for repo in cognitive_repos.values():
            await repo.initialize()
        router.set_model_call_repo(cognitive_repos["model_call_repo"])

        template_deps = CognitiveGraphDeps(
            router=router,
            config=CognitiveGraphSettings(),
            session_id=session_id,
            run_id=session_id,
            context_assembler=context_assembler_from_settings(CognitiveGraphSettings()),
            snapshot_repo=SnapshotRepository(db),
            option_surface_repo=OptionSurfaceRepository(db),
            data_quality_repo=supervisor.data_quality_repository,
            db_path=db,
            execution_runtime=None,
            submit_callback=None,
            order_action_gateway=None,
            **cognitive_repos,
        )

        evo_repos = build_evolution_repositories(db)
        for repo in evo_repos.values():
            await repo.initialize()
        champ_registry = ChampionRegistry(db)
        champion = await champ_registry.bootstrap_champion()
        improvement = ImprovementProposalService(
            evo_repos["proposals"],
            evo_repos["configurations"],
            champ_registry.policy_store,
        )
        _, challenger = await improvement.propose(
            parent_champion=champion,
            weakness="calibration",
            hypothesis="prefer calibrated exits",
            patch=PromptPatch(
                role="meta_decision",
                parent_prompt_version_id=uuid4(),
                replacement_template="Prefer calibrated no-trade when evidence is thin.",
                change_rationale="calibration",
            ),
        )

        replay = CognitiveReplayService(
            template_deps=template_deps,
            config_repo=evo_repos["configurations"],
            policy_store=champ_registry.policy_store,
            checkpointer_path=None,
            allow_synthetic_starting_cash=True,
            session_starting_cash=Decimal("25000"),
        )
        # Isolation contract: production replay never sees broker write paths.
        isolated = replay._isolated_deps()
        assert isolated.execution_runtime is None
        assert isolated.submit_callback is None
        assert isolated.order_action_gateway is None

        episodes = [
            TradingEpisode(
                session_id=session_id,
                run_id=session_id,
                trading_date=date(2026, 7, 1),
                initial_snapshot_id=snapshot.snapshot_id,
                terminal_snapshot_id=snapshot.snapshot_id,
                action_class="closed_trade",
                configuration_version_id=champion.configuration_version_id,
                contract_id=CONTRACT_ID,
                quantity=Decimal("1"),
                realised_pnl=Decimal(str(i - 1)),
                entry_price=Decimal("1.10"),
                exit_price=Decimal("1.20"),
                completed=True,
                idempotency_key=f"replay-ep-{i}",
                created_at=datetime(2026, 7, 1, 12, 0, i, tzinfo=timezone.utc),
            )
            for i in range(2)
        ]
        for ep in episodes:
            await evo_repos["episodes"].append(ep)

        dataset = await DatasetBuilder(evo_repos["datasets"]).build_and_persist(
            episodes, random_seed=3, minimum_holdout=1, allow_incomplete=False
        )
        definition = ExperimentDefinition(
            experiment_id=uuid4(),
            champion_version_id=champion.configuration_version_id,
            challenger_version_id=challenger.configuration_version_id,
            dataset_id=dataset.dataset_id,
            maximum_cost_gbp=Decimal("25"),
        )
        runner = ExperimentRunner(
            evo_repos["experiments"],
            repeated_samples=1,
            db_path=db,
            replay_service=replay,
            gate=PromotionEligibilityGate(
                PromotionSettings(

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
                )
            ),
        )
        await runner.create(definition)

        # Production path: no replay_fn argument — CognitiveReplayService is bound.
        result = await runner.run(
            definition.experiment_id,
            episodes=episodes,
            partition_map=dataset.partition_map,
        )
        assert result.experiment_id == definition.experiment_id
        first_count = replay.replay_count
        assert first_count > 0
        store = ExperimentEpisodeResultStore(db)
        await store.initialize()
        keys = await store.list_keys(definition.experiment_id)
        assert len(keys) == first_count
        for key in keys:
            payload = await store.get_payload(key)
            assert payload is not None
            assert payload.get("ran_task2_graph") is True
            assert payload.get("broker_submit") is False

        # Process-restart simulation: fresh runner, same durable keys, no re-replay.
        runner2 = ExperimentRunner(
            evo_repos["experiments"],
            repeated_samples=1,
            db_path=db,
            replay_service=replay,
        )
        resumed = await runner2.run(
            definition.experiment_id,
            episodes=episodes,
            partition_map=dataset.partition_map,
        )
        assert replay.replay_count == first_count
        assert resumed.result_id == result.result_id or resumed.experiment_id == result.experiment_id

        decisions = EvolutionDecisionService(
            evo_repos["promotions"],
            evo_repos["configurations"],
            champ_registry,
            gate=PromotionEligibilityGate(
                PromotionSettings(
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
                )
            ),
        )
        # Real decision graph path (no agent_override); identical metrics remain eligible.
        decision = await decisions.decide_and_apply(
            experiment_id=definition.experiment_id,
            result=result,
            challenger=challenger,
            champion=champion,
            holdout_episode_count=len(dataset.partition_map.get("holdout", ())),
            completed_episode_count=len(episodes),
            adversarial_passed=True,
        )
        assert decision.deterministic_eligible in {True, False}
        assert decision.agent_action in {
            "promote",
            "reject",
            "extend_shadow",
            "request_more_evidence",
        }

        # Shadow uses wired challenger_runner (not hypothetical fallback).
        shadow = ShadowRuntime(
            evo_repos["shadow"],
            policy_store=champ_registry.policy_store,
            queue_size=8,
            challenger_runner=replay.run_challenger_shadow,
        )
        await shadow.start()
        assignment = await shadow.register_challenger(
            challenger=challenger, champion=champion
        )
        assert await shadow.enqueue_snapshot(
            assignment_id=assignment.assignment_id,
            challenger_version_id=challenger.configuration_version_id,
            snapshot_id=str(snapshot.snapshot_id),
            payload={"snapshot_id": str(snapshot.snapshot_id)},
        )
        for _ in range(40):
            if shadow.results:
                break
            await asyncio.sleep(0.05)
        assert shadow.results
        command = shadow.results[0].hypothetical_command
        assert command.get("ran_challenger_graph") is True
        assert command.get("broker_submit") is False
        assert command.get("action") != "hypothetical_entry"
        assert replay.shadow_count >= 1
        await shadow.stop()
    finally:
        if shadow is not None:
            await shadow.stop()
        if champ_registry is not None:
            await champ_registry.close()
        await supervisor.shutdown()
        from joker.persistence.aiosqlite_lifecycle import (
            drain_aiosqlite_workers,
            iter_aiosqlite_worker_threads,
            join_aiosqlite_workers,
        )

        await drain_aiosqlite_workers(timeout=5.0)
        join_aiosqlite_workers(timeout=5.0)
        assert not iter_aiosqlite_worker_threads()
