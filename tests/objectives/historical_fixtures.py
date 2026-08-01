"""Shared fixtures for factual historical-EV tests.

Production-path helpers persist through Task-3 repositories — they do not
mutate private HistoricalOutcomeService fields.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

from joker.evolution.repositories import (
    DatasetRepository,
    EpisodeEvaluationRepository,
    TradingEpisodeRepository,
    build_evolution_repositories,
)
from joker.evolution.schemas import EpisodeEvaluation, EvaluationDataset, TradingEpisode
from joker.objectives.config import HistoricalOutcomeSettings
from joker.objectives.historical_outcomes import (
    HistoricalOutcomeService,
    build_historical_outcome_service_from_evolution_repos,
)
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations


def make_closed_episode(
    *,
    pnl: Decimal,
    as_of: datetime,
    hours_before: int = 24,
    direction: str = "bullish",
    strategy_family: str = "breakout_continuation",
    episode_id: UUID | None = None,
    config_id: UUID | None = None,
    completed: bool = True,
    findings: tuple[str, ...] = (),
    entry_price: Decimal = Decimal("1.10"),
    pattern_ids: tuple[UUID, ...] | None = None,
    regime_labels: tuple[str, ...] = ("trend",),
    liquidity_bucket: str = "normal",
    volatility_bucket: str = "elevated",
    option_type: str = "call",
    session_phase: str | None = None,
) -> tuple[TradingEpisode, EpisodeEvaluation]:
    """Build a factual closed TradingEpisode + matching EpisodeEvaluation."""
    eid = episode_id or uuid4()
    cfg = config_id or uuid4()
    entry_ts = as_of - timedelta(hours=hours_before + 1)
    term_ts = as_of - timedelta(hours=hours_before)
    entry_event = uuid4()
    term_event = uuid4()
    snap = uuid4()
    patterns = pattern_ids if pattern_ids is not None else (uuid4(),)
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
        market_regime_tags=regime_labels,
        strategy_family=strategy_family,
        pattern_ids=patterns,
        option_type=option_type,
        session_phase=session_phase,
        volatility_bucket=volatility_bucket,
        liquidity_bucket=liquidity_bucket,
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
    return episode, evaluation


async def persist_positive_history(
    *,
    episode_repo: TradingEpisodeRepository,
    evaluation_repo: EpisodeEvaluationRepository,
    as_of: datetime,
    n: int = 20,
    pnl: Decimal = Decimal("12.00"),
    direction: str = "bullish",
    strategy_family: str = "breakout_continuation",
    shared_pattern_id: UUID | None = None,
) -> list[tuple[TradingEpisode, EpisodeEvaluation]]:
    """Persist n factual closed episodes + evaluations via production repos."""
    pattern = shared_pattern_id or uuid4()
    out: list[tuple[TradingEpisode, EpisodeEvaluation]] = []
    for i in range(n):
        episode, evaluation = make_closed_episode(
            pnl=pnl,
            as_of=as_of,
            hours_before=24 + i,
            direction=direction,
            strategy_family=strategy_family,
            pattern_ids=(pattern,),
        )
        await episode_repo.append(episode)
        await evaluation_repo.append(evaluation)
        out.append((episode, evaluation))
    return out


async def make_repo_backed_hist_service(
    tmp_path: Path,
    *,
    minimum_samples_for_ev: int = 20,
    require_lcb: bool = True,
) -> tuple[
    HistoricalOutcomeService,
    ObjectiveRepository,
    TradingEpisodeRepository,
    EpisodeEvaluationRepository,
    DatasetRepository,
]:
    """Build a historical service wired to real Task-3 repositories."""
    db = tmp_path / "hist_evo.db"
    apply_objective_migrations(db)
    obj_repo = ObjectiveRepository(db)
    evo = build_evolution_repositories(db)
    for repo in evo.values():
        await repo.initialize()
    settings = HistoricalOutcomeSettings(
        minimum_samples_for_ev=minimum_samples_for_ev,
        minimum_effective_sample_size=min(15, minimum_samples_for_ev),
        require_lower_confidence_bound_positive=require_lcb,
        use_similarity_weighting=True,
        require_same_strategy_family=True,
        minimum_similarity=0.10,
    )
    svc = build_historical_outcome_service_from_evolution_repos(
        episode_repo=evo["episodes"],
        evaluation_repo=evo["evaluations"],
        dataset_repo=evo["datasets"],
        settings=settings,
        repository=obj_repo,
    )
    return (
        svc,
        obj_repo,
        evo["episodes"],  # type: ignore[return-value]
        evo["evaluations"],  # type: ignore[return-value]
        evo["datasets"],  # type: ignore[return-value]
    )


def make_hist_service(
    tmp_path,
    *,
    minimum_samples_for_ev: int = 20,
    require_lcb: bool = True,
) -> tuple[HistoricalOutcomeService, ObjectiveRepository]:
    """Legacy sync helper for pure unit tests (empty loaders until await persist)."""
    db = tmp_path / "obj_hist.db"
    apply_objective_migrations(db)
    repo = ObjectiveRepository(db)
    settings = HistoricalOutcomeSettings(
        minimum_samples_for_ev=minimum_samples_for_ev,
        minimum_effective_sample_size=min(15, minimum_samples_for_ev),
        require_lower_confidence_bound_positive=require_lcb,
        use_similarity_weighting=True,
        require_same_strategy_family=True,
        minimum_similarity=0.10,
    )
    svc = HistoricalOutcomeService(settings=settings, repository=repo)
    return svc, repo


async def persist_dataset_with_episodes(
    dataset_repo: DatasetRepository,
    *,
    episode_ids: tuple[UUID, ...],
    partition: str = "train",
    time_end: datetime | None = None,
) -> EvaluationDataset:
    ds = EvaluationDataset(
        episode_ids=episode_ids,
        partition_map={partition: episode_ids},
        time_end=time_end or datetime.now(timezone.utc) + timedelta(days=1),
        construction_timestamp=datetime.now(timezone.utc),
    )
    await dataset_repo.append(ds)
    return ds
