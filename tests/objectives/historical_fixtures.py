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


async def persist_compiler_produced_history(
    *,
    episode_repo: TradingEpisodeRepository,
    evaluation_repo: EpisodeEvaluationRepository,
    as_of: datetime,
    n: int = 20,
    pnl: Decimal = Decimal("15.00"),
    strategy_family: str = "breakout_continuation",
) -> list[tuple[TradingEpisode, EpisodeEvaluation]]:
    """Persist episodes produced by EpisodeCompiler (production provenance path)."""
    from types import SimpleNamespace

    from joker.cognition.schemas import (
        AgentRole,
        EntryPlan,
        ExecutionPlan,
        ExitPlan,
        InvalidationPlan,
        MarketDirection,
        StrategyHypothesis,
        StrategyLegCandidate,
    )
    from joker.evolution.episode_compiler import EpisodeCompiler
    from joker.evolution.event_horizon import Task1EventHorizon, Task1HorizonEvent
    from joker.evolution.lifecycle import PositionLifecycleResolver
    from tests.evolution.projection_helpers import (
        FakeExecutionProjection,
        closed_trade_projection,
    )

    pattern = uuid4()
    out: list[tuple[TradingEpisode, EpisodeEvaluation]] = []
    contract = "SPY:2026-07-01:500.0:call"

    class _Horizon:
        async def load(self, **kwargs):
            start = kwargs["start_timestamp"]
            end = kwargs["end_timestamp"]
            raw_e1 = kwargs.get("entry_decision_event_id")
            raw_e2 = kwargs.get("terminal_event_id")
            if not isinstance(raw_e1, UUID):
                raise AssertionError(
                    "horizon loader requires factual entry_decision_event_id"
                )
            if not isinstance(raw_e2, UUID):
                raise AssertionError(
                    "horizon loader requires factual terminal_event_id"
                )
            return Task1EventHorizon(
                session_id=kwargs["session_id"],
                events=(
                    Task1HorizonEvent(
                        event_id=raw_e1,
                        event_type="COGNITIVE_CYCLE_STARTED",
                        exchange_timestamp=start,
                        sequence=1,
                    ),
                    Task1HorizonEvent(
                        event_id=raw_e2,
                        event_type="POSITION_CLOSED",
                        exchange_timestamp=end,
                        sequence=2,
                    ),
                ),
                market_event_ids=(raw_e1, raw_e2),
            )

    for i in range(n):
        entry_id = f"entry-{i}"
        exit_id = f"exit-{i}"
        strategy_id = uuid4()
        entry_anchor = uuid4()
        terminal_anchor = uuid4()
        strategy = StrategyHypothesis(
            session_id="hist",
            snapshot_id=uuid4(),
            cycle_id=f"c-{i}",
            prompt_version="1.0.0",
            model_call_id=uuid4(),
            strategy_id=strategy_id,
            source_hypothesis_ids=(pattern,),
            name="bull",
            market_thesis="t",
            direction=MarketDirection.BULLISH,
            strategy_family=strategy_family,
            candidate_legs=(
                StrategyLegCandidate(
                    contract_id=contract,
                    side="buy",
                    option_type="call",
                    strike=Decimal("500"),
                    quantity=1,
                    rationale="primary",
                ),
            ),
            entry_plan=EntryPlan(entry_style="immediate", preferred_order_type="limit"),
            execution_plan=ExecutionPlan(
                max_quote_age_seconds=5,
                partial_fill_policy="accept",
                replacement_policy="none",
            ),
            exit_plan=ExitPlan(stop_conditions=("stop",)),
            invalidation_plan=InvalidationPlan(conditions=("inv",)),
            expected_horizon_seconds=600,
            confidence=0.7,
            novelty_score=0.5,
            agent_role=AgentRole.BULLISH_INVENTOR,
        )

        class _StratRepo:
            async def get_by_id(self, sid):
                return strategy if str(sid) == str(strategy_id) else None

        class _Prov:
            async def get_by_client_order_id(self, coid: str):
                return SimpleNamespace(
                    client_order_id=coid,
                    strategy_id=str(strategy_id),
                    proposal_id=str(uuid4()),
                    decision_id=str(uuid4()),
                    cycle_id=f"c-{i}",
                    snapshot_id=str(uuid4()),
                    contract_id=contract,
                    kind="entry" if coid == entry_id else "exit",
                    causation_event_id=str(entry_anchor) if coid == entry_id else None,
                    extra={
                        "position_lifecycle_id": f"hist:{entry_id}:{contract}",
                        "originating_entry_client_order_id": entry_id,
                        "causation_event_id": str(entry_anchor)
                        if coid == entry_id
                        else None,
                    },
                )

            async def list_by_lifecycle_id(self, _lid: str):
                return []

        prov = _Prov()
        compiler = EpisodeCompiler(
            episode_repo,
            provenance=prov,
            event_horizon_loader=_Horizon(),
            strategy_repo=_StratRepo(),
        )
        compiler._lifecycle = PositionLifecycleResolver(provenance=prov)
        projection = closed_trade_projection(
            contract_id=contract,
            entry_id=entry_id,
            exit_id=exit_id,
            realised_pnl=pnl,
            entry_price=Decimal("1.00"),
            exit_price=Decimal("1.00") + (pnl / Decimal("100")),
            session_id="hist",
        )
        # Align projection PnL with resolver when prices imply different value.
        # Prefer explicit realised matching closed_trade_projection default path.
        entry_ts = as_of - timedelta(hours=24 + i, minutes=10)
        term_ts = as_of - timedelta(hours=24 + i)
        episode = await compiler.compile_from_position_closed(
            session_id="hist",
            run_id=f"run-{i}",
            trading_date=date(2026, 7, 1),
            configuration_version_id=uuid4(),
            event_payload={
                "contract_id": contract,
                "client_order_id": exit_id,
                "position_lifecycle_id": f"hist:{entry_id}:{contract}",
            },
            event_id=str(terminal_anchor),
            execution=FakeExecutionProjection(projection),
            initial_snapshot_id=uuid4(),
            terminal_snapshot_id=uuid4(),
            entry_cycle_id=f"c-{i}",
            entry_decision_timestamp=entry_ts,
            terminal_event_timestamp=term_ts,
        )
        assert episode.strategy_family == strategy_family
        assert episode.entry_decision_event_id == entry_anchor
        assert episode.terminal_event_id == terminal_anchor
        assert episode.completed is True
        evaluation = EpisodeEvaluation(
            episode_id=episode.episode_id,
            evaluator_version="3.2.0",
            outcome_quality=Decimal("0.8"),
            configuration_version_id=episode.configuration_version_id,
            valid=True,
            created_at=term_ts + timedelta(minutes=5),
        )
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
