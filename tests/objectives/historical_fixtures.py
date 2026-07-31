"""Shared fixtures for factual historical-EV tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

from joker.evolution.schemas import EpisodeEvaluation, TradingEpisode
from joker.objectives.config import HistoricalOutcomeSettings
from joker.objectives.historical_outcomes import HistoricalOutcomeService
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations


def make_closed_episode(
    *,
    pnl: Decimal,
    as_of: datetime,
    hours_before: int = 24,
    direction: str = "bullish",
    episode_id: UUID | None = None,
    config_id: UUID | None = None,
    completed: bool = True,
    findings: tuple[str, ...] = (),
    sample: int | None = None,
    entry_price: Decimal = Decimal("1.10"),
) -> SimpleNamespace:
    """Build a TradingEpisode-like object plus attached evaluation for seeding."""
    eid = episode_id or uuid4()
    cfg = config_id or uuid4()
    entry_ts = as_of - timedelta(hours=hours_before + 1)
    term_ts = as_of - timedelta(hours=hours_before)
    entry_event = uuid4()
    term_event = uuid4()
    snap = uuid4()
    episode = TradingEpisode(
        episode_id=eid,
        session_id=f"hist-{eid}",
        run_id=f"run-{eid}",
        trading_date=date(2026, 6, 1),
        parent_strategy_id=uuid4(),
        initial_snapshot_id=snap,
        terminal_snapshot_id=uuid4(),
        direction=direction,  # type: ignore[arg-type]
        action_class="closed_trade",
        entry_price=entry_price,
        exit_price=entry_price + (pnl / Decimal("100")),
        quantity=Decimal("1"),
        realised_pnl=pnl,
        holding_seconds=1800,
        market_regime_tags=("trend",),
        entry_decision_event_id=entry_event,
        entry_decision_timestamp=entry_ts,
        terminal_event_id=term_event,
        terminal_event_timestamp=term_ts,
        configuration_version_id=cfg,
        completed=completed,
        completeness_findings=findings,
    )
    evaluation = EpisodeEvaluation(
        episode_id=eid,
        evaluator_version="3.2.0",
        outcome_quality=Decimal("0.8"),
        configuration_version_id=cfg,
        valid=True,
        created_at=term_ts + timedelta(minutes=5),
    )
    ns = SimpleNamespace(**episode.model_dump())
    # Restore UUID/datetime types lost through dump for attribute access helpers
    for field in (
        "episode_id",
        "parent_strategy_id",
        "initial_snapshot_id",
        "terminal_snapshot_id",
        "entry_decision_event_id",
        "terminal_event_id",
        "configuration_version_id",
    ):
        setattr(ns, field, getattr(episode, field))
    ns.entry_decision_timestamp = entry_ts
    ns.terminal_event_timestamp = term_ts
    ns.realised_pnl = pnl
    ns.evaluation = evaluation
    if sample is not None:
        ns.sample = sample
    return ns


def seed_positive_history(
    service: HistoricalOutcomeService,
    *,
    as_of: datetime,
    n: int = 20,
    pnl: Decimal = Decimal("12.00"),
    direction: str = "bullish",
) -> list[SimpleNamespace]:
    episodes = [
        make_closed_episode(
            pnl=pnl,
            as_of=as_of,
            hours_before=24 + i,
            direction=direction,
        )
        for i in range(n)
    ]
    service.seed_episodes_for_tests(episodes)
    return episodes


def make_hist_service(
    tmp_path,
    *,
    minimum_samples_for_ev: int = 20,
    require_lcb: bool = True,
) -> tuple[HistoricalOutcomeService, ObjectiveRepository]:
    db = tmp_path / "obj_hist.db"
    apply_objective_migrations(db)
    repo = ObjectiveRepository(db)
    settings = HistoricalOutcomeSettings(
        minimum_samples_for_ev=minimum_samples_for_ev,
        minimum_effective_sample_size=min(15, minimum_samples_for_ev),
        require_lower_confidence_bound_positive=require_lcb,
        use_similarity_weighting=True,
        require_same_strategy_family=True,
    )
    svc = HistoricalOutcomeService(settings=settings, repository=repo)
    return svc, repo
