"""Cached replay must not invent ran_position_graph / ran_order_management."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.config.settings import CognitiveGraphSettings
from joker.evolution.adversarial_fixtures import ADVERSARIAL_DEFINITIONS
from joker.evolution.adversarial_runners import _evaluate_replay_invariants
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.replay import CognitiveReplayService, execution_flags_from_workflow
from joker.evolution.replay_market import ReplayEpisodeTruth
from joker.evolution.replay_store import ReplayExecutionStore, replay_key
from joker.evolution.replay_truth import ReplayMarketFrame, ReplayContractQuote
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import TradingEpisode
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig


def test_execution_flags_from_no_trade_workflow():
    ran_task2, ran_pos, ran_om = execution_flags_from_workflow(
        {
            "entry_graph_completed": True,
            "entry_action_submitted": False,
            "frames": {},
            "final_result_persisted": True,
        }
    )
    assert ran_task2 is True
    assert ran_pos is False
    assert ran_om is False


def test_execution_flags_from_position_and_om_frames():
    ran_task2, ran_pos, ran_om = execution_flags_from_workflow(
        {
            "entry_graph_completed": True,
            "entry_order_management_ran": False,
            "frames": {
                "1": {
                    "position_graph_completed": True,
                    "order_management_completed": True,
                    "order_management_ran": True,
                    "order_management_thread_ids": ["om:1"],
                }
            },
        }
    )
    assert ran_task2 is True
    assert ran_pos is True
    assert ran_om is True


def test_om_stage_completed_without_ran_is_not_om_execution():
    """order_management_completed alone (no threads / ran flag) is not evidence."""
    _, _, ran_om = execution_flags_from_workflow(
        {
            "entry_graph_completed": True,
            "frames": {
                "1": {
                    "order_management_completed": True,
                    "order_management_thread_ids": [],
                    "position_graph_completed": False,
                }
            },
        }
    )
    assert ran_om is False


def test_adv_23_cannot_pass_without_genuine_position_graph():
    payload = {
        "ran_task2_graph": True,
        "ran_position_graph": False,
        "broker_submit": False,
        "traded": True,
        "realised_pnl": "-1.0",
        "open_at_end": False,
        "integrity_findings": (),
        "calibration_pairs": (),
        "calibration_sample_count": 0,
        "entry_confidence": "0.5",
        "meta_decision_action": "execute",
    }
    definition = next(d for d in ADVERSARIAL_DEFINITIONS if d.scenario_id == "adv_23")
    satisfied, _ = _evaluate_replay_invariants(
        definition.expected_invariants,
        payload=payload,
    )
    assert "regime_shift_handled" not in satisfied

    # Reconstructing from a no-trade cached workflow must also fail adv_23.
    flags = execution_flags_from_workflow(
        {
            "entry_graph_completed": True,
            "frames": {},
            "final_result_persisted": True,
        }
    )
    cached_payload = {
        **payload,
        "ran_task2_graph": flags[0],
        "ran_position_graph": flags[1],
        "ran_order_management": flags[2],
        "traded": False,
        "open_at_end": False,
    }
    satisfied2, _ = _evaluate_replay_invariants(
        definition.expected_invariants,
        payload=cached_payload,
    )
    assert "regime_shift_handled" not in satisfied2


@pytest.mark.asyncio
async def test_cached_no_trade_reload_preserves_false_execution_flags(tmp_path):
    apply_task3_migrations(tmp_path / "c.db")
    repos = build_evolution_repositories(tmp_path / "c.db")
    for repo in repos.values():
        await repo.initialize()
    from joker.evolution.champion_registry import ChampionRegistry

    reg = ChampionRegistry(tmp_path / "c.db")
    champ = await reg.bootstrap_champion()
    assert champ is not None

    snap = uuid4()
    contract = ReplayContractQuote(
        contract_id="SPY:2026-07-01:500.0:call",
        symbol="SPY",
        expiry="2026-07-01",
        strike=Decimal("500"),
        option_type="call",
        is_0dte=True,
        bid=Decimal("1.00"),
        ask=Decimal("1.20"),
    )
    frame = ReplayMarketFrame(
        snapshot_id=snap,
        timestamp=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
        data_quality_id=uuid4(),
        option_surface_id=uuid4(),
        underlying_bid=Decimal("500"),
        underlying_ask=Decimal("500.1"),
        underlying_last=Decimal("500"),
        contracts=(contract,),
    )
    episode = TradingEpisode(
        session_id="cache-flags",
        run_id="cache-flags",
        trading_date=date(2026, 7, 1),
        initial_snapshot_id=snap,
        terminal_snapshot_id=snap,
        action_class="no_trade",
        configuration_version_id=champ.configuration_version_id,
        quantity=Decimal("0"),
        realised_pnl=Decimal("0"),
        completed=True,
        idempotency_key="cache-no-trade-1",
        snapshot_identity_status="verified",
        entry_decision_timestamp=frame.timestamp,
        terminal_event_timestamp=frame.timestamp,
        terminal_event_id=uuid4(),
        market_event_ids=(),
    )

    class TruthStub:
        async def load_for_episode(self, ep):
            return ReplayEpisodeTruth(
                episode_id=ep.episode_id,
                session_id=ep.session_id,
                trading_date=ep.trading_date,
                initial_snapshot_id=snap,
                terminal_snapshot_id=snap,
                starting_cash=Decimal("25000"),
                frames=(frame,),
            )

    store = ReplayExecutionStore(tmp_path / "exec.db")
    await store.initialize()
    exp_id = uuid4()
    key = replay_key(exp_id, episode.episode_id, champ.configuration_version_id, 1)
    await store.save_checkpoint(
        key=key,
        experiment_id=str(exp_id),
        episode_id=str(episode.episode_id),
        configuration_version_id=str(champ.configuration_version_id),
        sample_number=1,
        status="completed",
        frame_index=0,
        cash=Decimal("25000"),
        realised_pnl=Decimal("0"),
        orders={},
        fills=[],
        positions={},
        submitted_keys=set(),
        entry_cycle_id="entry-cycle",
        entry_order_id=None,
        entry_decision_completed=True,
        extra={
            "entry_graph_completed": True,
            "entry_action_submitted": False,
            "entry_action_value": "abandon",
            "entry_confidence": "0.3",
            "frames": {},
            "final_result_persisted": True,
        },
    )

    fake = FakeModelProvider(available=True)
    router = ModelRouter(
        ModelRegistry(ModelsConfig(), providers={"fake": fake, "ollama": fake, "openai": fake}),
        session_id="cache-flags",
    )
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="cache-flags",
        run_id="cache-flags",
        context_assembler=None,
        snapshot_repo=None,
        option_surface_repo=None,
        data_quality_repo=None,
        db_path=tmp_path / "c.db",
        execution_runtime=None,
        submit_callback=None,
    )
    replay = CognitiveReplayService(
        template_deps=deps,
        config_repo=repos["configurations"],
        policy_store=PolicyVersionStore(tmp_path / "c.db"),
        truth_loader=TruthStub(),  # type: ignore[arg-type]
        execution_store=store,
        allow_synthetic_starting_cash=True,
        session_starting_cash=Decimal("25000"),
    )
    payload = await replay.replay_episode(
        experiment_id=exp_id,
        episode=episode,
        configuration_version_id=champ.configuration_version_id,
        sample=1,
    )
    assert payload.get("resumed") is True
    assert payload["ran_task2_graph"] is True
    assert payload["ran_position_graph"] is False
    assert payload["ran_order_management"] is False
    assert payload["traded"] is False


@pytest.mark.asyncio
async def test_cached_reload_without_om_stays_false(tmp_path):
    apply_task3_migrations(tmp_path / "c.db")
    repos = build_evolution_repositories(tmp_path / "c.db")
    for repo in repos.values():
        await repo.initialize()
    from joker.evolution.champion_registry import ChampionRegistry

    reg = ChampionRegistry(tmp_path / "c.db")
    champ = await reg.bootstrap_champion()
    snap = uuid4()
    frame = ReplayMarketFrame(
        snapshot_id=snap,
        timestamp=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
        data_quality_id=uuid4(),
        option_surface_id=uuid4(),
        underlying_bid=Decimal("500"),
        underlying_ask=Decimal("500.1"),
        underlying_last=Decimal("500"),
        contracts=(
            ReplayContractQuote(
                contract_id="SPY:2026-07-01:500.0:call",
                symbol="SPY",
                expiry="2026-07-01",
                strike=Decimal("500"),
                option_type="call",
                is_0dte=True,
                bid=Decimal("1.00"),
                ask=Decimal("1.20"),
            ),
        ),
    )
    episode = TradingEpisode(
        session_id="cache-om",
        run_id="cache-om",
        trading_date=date(2026, 7, 1),
        initial_snapshot_id=snap,
        terminal_snapshot_id=snap,
        action_class="closed_trade",
        configuration_version_id=champ.configuration_version_id,
        quantity=Decimal("1"),
        realised_pnl=Decimal("-1"),
        completed=True,
        idempotency_key="cache-om-1",
        snapshot_identity_status="verified",
        entry_decision_timestamp=frame.timestamp,
        terminal_event_timestamp=frame.timestamp,
        terminal_event_id=uuid4(),
        market_event_ids=(),
    )

    class TruthStub:
        async def load_for_episode(self, ep):
            return ReplayEpisodeTruth(
                episode_id=ep.episode_id,
                session_id=ep.session_id,
                trading_date=ep.trading_date,
                initial_snapshot_id=snap,
                terminal_snapshot_id=snap,
                starting_cash=Decimal("25000"),
                frames=(frame, frame),
            )

    store = ReplayExecutionStore(tmp_path / "exec.db")
    await store.initialize()
    exp_id = uuid4()
    key = replay_key(exp_id, episode.episode_id, champ.configuration_version_id, 1)
    await store.save_checkpoint(
        key=key,
        experiment_id=str(exp_id),
        episode_id=str(episode.episode_id),
        configuration_version_id=str(champ.configuration_version_id),
        sample_number=1,
        status="completed",
        frame_index=1,
        cash=Decimal("24900"),
        realised_pnl=Decimal("-1"),
        orders={},
        fills=[],
        positions={},
        submitted_keys=set(),
        entry_cycle_id="entry-cycle",
        entry_order_id="entry-1",
        entry_decision_completed=True,
        extra={
            "entry_graph_completed": True,
            "entry_action_submitted": True,
            "entry_action_value": "execute",
            "entry_selected_contract": "SPY:2026-07-01:500.0:call",
            "entry_confidence": "0.55",
            # Position ran, but OM never executed (completed stage with empty threads).
            "frames": {
                "1": {
                    "position_graph_completed": True,
                    "order_management_completed": True,
                    "order_management_thread_ids": [],
                    "action_submitted": True,
                    "execution_checkpointed": True,
                }
            },
            "final_result_persisted": True,
        },
    )

    fake = FakeModelProvider(available=True)
    router = ModelRouter(
        ModelRegistry(ModelsConfig(), providers={"fake": fake, "ollama": fake, "openai": fake}),
        session_id="cache-om",
    )
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="cache-om",
        run_id="cache-om",
        context_assembler=None,
        snapshot_repo=None,
        option_surface_repo=None,
        data_quality_repo=None,
        db_path=tmp_path / "c.db",
        execution_runtime=None,
        submit_callback=None,
    )
    replay = CognitiveReplayService(
        template_deps=deps,
        config_repo=repos["configurations"],
        policy_store=PolicyVersionStore(tmp_path / "c.db"),
        truth_loader=TruthStub(),  # type: ignore[arg-type]
        execution_store=store,
        allow_synthetic_starting_cash=True,
        session_starting_cash=Decimal("25000"),
    )
    payload = await replay.replay_episode(
        experiment_id=exp_id,
        episode=episode,
        configuration_version_id=champ.configuration_version_id,
        sample=1,
    )
    assert payload["ran_task2_graph"] is True
    assert payload["ran_position_graph"] is True
    assert payload["ran_order_management"] is False
