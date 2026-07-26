"""ModelRouter-backed Task 3 evaluation and champion pin application."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.agent_schemas import EvaluatorAgentScores
from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.config import EvolutionSettings
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.runtime import EvolutionRuntime
from joker.evolution.schemas import TradingEpisode
from joker.evaluation.agentic_graph import AgenticEvaluationGraphRunner, EVALUATOR_ROLES
from joker.events.bus import InProcessAsyncEventBus
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig


@pytest.mark.asyncio
async def test_agentic_evaluation_invokes_model_router(tmp_path) -> None:
    db = tmp_path / "agentic_eval.db"
    apply_task3_migrations(db)
    repos = build_evolution_repositories(db)
    await repos["evaluations"].initialize()

    fake = FakeModelProvider(available=True)
    scores = EvaluatorAgentScores(
        thesis_quality=Decimal("0.7"),
        evidence_grounding_score=Decimal("0.8"),
        calibration_score=Decimal("0.75"),
        execution_quality=Decimal("0.6"),
        efficiency_score=Decimal("0.5"),
        avoidable_error_codes=(),
    )
    for role in EVALUATOR_ROLES:
        fake.set_canned_for_role(role, scores)
    registry = ModelRegistry(ModelsConfig(), providers={"ollama": fake, "openai": fake})
    router = ModelRouter(registry, session_id="eval-sess")

    episode = TradingEpisode(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        initial_snapshot_id=uuid4(),
        action_class="closed_trade",
        configuration_version_id=uuid4(),
        quantity=Decimal("1"),
        realised_pnl=Decimal("12"),
        entry_price=Decimal("1.0"),
        exit_price=Decimal("1.2"),
        completed=True,
        idempotency_key="agentic-1",
    )
    runner = AgenticEvaluationGraphRunner(
        repos["evaluations"],
        router=router,
        checkpointer_path=tmp_path / "eval_ckpt.db",
    )
    evaluation = await runner.evaluate(episode)
    assert evaluation.thesis_quality == Decimal("0.7")
    assert evaluation.evidence_grounding_score == Decimal("0.8")
    assert {c.request.role for c in fake.calls} >= set(EVALUATOR_ROLES)


@pytest.mark.asyncio
async def test_evolution_runtime_pins_champion_for_cycle(tmp_path) -> None:
    db = tmp_path / "pin.db"
    apply_task3_migrations(db)
    bus = InProcessAsyncEventBus()
    settings = EvolutionSettings(enabled=True)
    runtime = EvolutionRuntime(
        db_path=db,
        settings=settings,
        session_id="sess",
        run_id="run",
        event_bus=bus,
    )
    await runtime.start()
    applied = await runtime.pin_and_apply_for_cycle("cycle-1")
    assert applied is not None
    assert runtime.get_pinned("cycle-1") == applied.configuration_version_id
    assert applied.prompt_overrides
    champ = await runtime.configuration_for_new_cycle()
    assert champ is not None
    await runtime.shutdown()
