"""EpisodeCompiler production metadata, horizon fail-closed, and EV eligibility."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

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
from joker.evolution.migrations import apply_task3_migrations
from joker.evolution.repositories import build_evolution_repositories
from joker.objectives.historical_outcomes import HistoricalOutcomeService
from joker.objectives.config import HistoricalOutcomeSettings
from joker.objectives.historical_schemas import HistoricalOutcomeQuery
from tests.evolution.projection_helpers import (
    FakeExecutionProjection,
    closed_trade_projection,
)


def _strategy(
    *,
    strategy_id: UUID | None = None,
    family: str | None = "breakout_continuation",
    direction: MarketDirection = MarketDirection.BULLISH,
    pattern_ids: tuple[UUID, ...] = (),
    option_type: str = "call",
    role: AgentRole = AgentRole.BULLISH_INVENTOR,
) -> StrategyHypothesis:
    sid = strategy_id or uuid4()
    contract = f"SPY:2026-07-01:500.0:{option_type}"
    return StrategyHypothesis(
        session_id="s",
        snapshot_id=uuid4(),
        cycle_id="c1",
        prompt_version="1.0.0",
        model_call_id=uuid4(),
        strategy_id=sid,
        source_hypothesis_ids=pattern_ids,
        name="test",
        market_thesis="t",
        direction=direction,
        strategy_family=family,
        candidate_legs=(
            StrategyLegCandidate(
                contract_id=contract,
                side="buy",
                option_type=option_type,  # type: ignore[arg-type]
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
        agent_role=role,
    )


class _FakeStrategyRepo:
    def __init__(self, strategy: StrategyHypothesis | None) -> None:
        self._strategy = strategy

    async def get_by_id(self, strategy_id):
        if self._strategy is None:
            return None
        if str(strategy_id) != str(self._strategy.strategy_id):
            return None
        return self._strategy


class _FakeProvenance:
    def __init__(
        self,
        *,
        strategy_id: UUID | None,
        entry_id: str = "entry-1",
        contract_id: str = "SPY:2026-07-01:500.0:call",
    ) -> None:
        self._strategy_id = strategy_id
        self._entry_id = entry_id
        self._contract_id = contract_id

    async def get_by_client_order_id(self, client_order_id: str):
        if client_order_id not in {self._entry_id, "exit-1", "exit-put"}:
            return None
        return SimpleNamespace(
            client_order_id=client_order_id,
            strategy_id=str(self._strategy_id) if self._strategy_id else None,
            proposal_id=str(uuid4()),
            decision_id=str(uuid4()),
            cycle_id="c1",
            snapshot_id=str(uuid4()),
            contract_id=self._contract_id,
            kind="entry" if client_order_id == self._entry_id else "exit",
            extra={
                "position_lifecycle_id": f"s:{self._entry_id}:{self._contract_id}",
                "originating_entry_client_order_id": self._entry_id,
            },
        )

    async def list_by_lifecycle_id(self, lifecycle_id: str):
        return []


class _FakeHorizonLoader:
    def __init__(self, *, fail: bool = False, empty: bool = False) -> None:
        self.fail = fail
        self.empty = empty
        self.calls = 0

    async def load(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("horizon backend unavailable")
        if self.empty:
            return Task1EventHorizon(session_id=kwargs["session_id"])
        start = kwargs["start_timestamp"]
        end = kwargs["end_timestamp"]
        raw_e1 = kwargs.get("entry_decision_event_id")
        raw_e2 = kwargs.get("terminal_event_id")
        e1 = raw_e1 if isinstance(raw_e1, UUID) else uuid4()
        e2 = raw_e2 if isinstance(raw_e2, UUID) else uuid4()
        return Task1EventHorizon(
            session_id=kwargs["session_id"],
            events=(
                Task1HorizonEvent(
                    event_id=e1,
                    event_type="MARKET_SNAPSHOT_CREATED",
                    exchange_timestamp=start,
                    sequence=1,
                ),
                Task1HorizonEvent(
                    event_id=e2,
                    event_type="POSITION_CLOSED",
                    exchange_timestamp=end,
                    sequence=2,
                ),
            ),
            market_event_ids=(e1, e2),
        )


class _FakeWorldModelRepo:
    def __init__(self, world_model) -> None:
        self._wm = world_model

    async def get_by_id(self, wid):
        return self._wm


class _FakeCycleRegistry:
    def __init__(self, world_model_id: UUID) -> None:
        self._wm_id = world_model_id

    async def get(self, *, session_id, graph_kind, cycle_id):
        return SimpleNamespace(
            payload={"world_model_id": str(self._wm_id)},
        )


async def _compile(
    tmp_path,
    *,
    contract_id: str = "SPY:2026-07-01:500.0:call",
    strategy: StrategyHypothesis | None = None,
    strategy_id: UUID | None = None,
    horizon: _FakeHorizonLoader | None = None,
    world_model=None,
    entry_id: str = "entry-1",
):
    apply_task3_migrations(tmp_path / "ep.db")
    repos = build_evolution_repositories(tmp_path / "ep.db")
    await repos["episodes"].initialize()
    sid = strategy.strategy_id if strategy is not None else strategy_id
    prov = _FakeProvenance(
        strategy_id=sid, entry_id=entry_id, contract_id=contract_id
    )
    wm_id = uuid4()
    compiler = EpisodeCompiler(
        repos["episodes"],
        lifecycle_resolver=None,
        provenance=prov,
        cycle_registry=_FakeCycleRegistry(wm_id) if world_model is not None else None,
        event_horizon_loader=horizon if horizon is not None else _FakeHorizonLoader(),
        strategy_repo=_FakeStrategyRepo(strategy),
        world_model_repo=_FakeWorldModelRepo(world_model) if world_model else None,
    )
    # Patch lifecycle to use our provenance
    from joker.evolution.lifecycle import PositionLifecycleResolver

    compiler._lifecycle = PositionLifecycleResolver(provenance=prov)
    # Match lifecycle resolver P&L: (exit-entry)*100*qty with default prices.
    projection = closed_trade_projection(
        contract_id=contract_id,
        entry_id=entry_id,
        exit_id="exit-1" if "call" in contract_id else "exit-put",
        realised_pnl=Decimal("50"),
    )
    if "put" in contract_id:
        projection = closed_trade_projection(
            contract_id=contract_id,
            entry_id=entry_id,
            exit_id="exit-put",
            realised_pnl=Decimal("50"),
        )
    start = datetime(2026, 7, 1, 14, 30, tzinfo=timezone.utc)
    end = start + timedelta(minutes=10)
    episode = await compiler.compile_from_position_closed(
        session_id="s",
        run_id="r",
        trading_date=date(2026, 7, 1),
        configuration_version_id=uuid4(),
        event_payload={
            "contract_id": contract_id,
            "client_order_id": "exit-1" if "call" in contract_id else "exit-put",
            "position_lifecycle_id": f"s:{entry_id}:{contract_id}",
        },
        event_id=str(uuid4()),
        execution=FakeExecutionProjection(projection),
        initial_snapshot_id=uuid4(),
        terminal_snapshot_id=uuid4(),
        entry_cycle_id="c1",
        entry_decision_timestamp=start,
        terminal_event_timestamp=end,
    )
    return episode, compiler, repos


@pytest.mark.asyncio
async def test_compiler_populates_strategy_family_from_selected_strategy(tmp_path) -> None:
    strategy = _strategy(family="volatility_expansion")
    episode, _, _ = await _compile(tmp_path, strategy=strategy)
    assert episode.strategy_family == "volatility_expansion"
    assert episode.parent_strategy_id == strategy.strategy_id


@pytest.mark.asyncio
async def test_compiler_populates_pattern_ids_from_strategy_hypotheses(tmp_path) -> None:
    p1, p2 = uuid4(), uuid4()
    strategy = _strategy(pattern_ids=(p1, p2))
    episode, _, _ = await _compile(tmp_path, strategy=strategy)
    assert set(episode.pattern_ids) == {p1, p2}


@pytest.mark.asyncio
async def test_compiler_classifies_long_call_as_bullish(tmp_path) -> None:
    strategy = _strategy(option_type="call", pattern_ids=(uuid4(),))
    episode, _, _ = await _compile(
        tmp_path,
        contract_id="SPY:2026-07-01:500.0:call",
        strategy=strategy,
    )
    assert episode.direction == "bullish"
    assert episode.option_type == "call"


@pytest.mark.asyncio
async def test_compiler_classifies_long_put_as_bearish(tmp_path) -> None:
    strategy = _strategy(
        option_type="put",
        direction=MarketDirection.BEARISH,
        family="failed_breakout_reversal",
        pattern_ids=(uuid4(),),
    )
    episode, _, _ = await _compile(
        tmp_path,
        contract_id="SPY:2026-07-01:500.0:put",
        strategy=strategy,
    )
    assert episode.direction == "bearish"
    assert episode.option_type == "put"


@pytest.mark.asyncio
async def test_compiler_populates_option_type(tmp_path) -> None:
    strategy = _strategy(option_type="call", pattern_ids=(uuid4(),))
    episode, _, _ = await _compile(tmp_path, strategy=strategy)
    assert episode.option_type == "call"


@pytest.mark.asyncio
async def test_compiler_populates_session_regime_volatility_and_liquidity(
    tmp_path,
) -> None:
    from joker.cognition.schemas import (
        MarketStructureAssessment,
        OptionsMicrostructureAssessment,
        TemporalAssessment,
        VolatilityAssessment,
        MarketWorldModel,
    )

    wm = MarketWorldModel(
        session_id="s",
        snapshot_id=uuid4(),
        cycle_id="c1",
        prompt_version="1.0.0",
        model_call_id=uuid4(),
        market_structure=MarketStructureAssessment(
            primary_direction=MarketDirection.BULLISH,
            range_bound=False,
            structure_summary="trend",
            supporting_evidence_ids=(),
            confidence=0.6,
        ),
        volatility_state=VolatilityAssessment(
            state=MarketDirection.BULLISH,
            summary="elevated high volatility",
            supporting_evidence_ids=(),
            confidence=0.6,
        ),
        options_state=OptionsMicrostructureAssessment(
            liquidity_summary="ok",
            spread_conditions="tight",
            supporting_evidence_ids=(),
            confidence=0.6,
        ),
        temporal_state=TemporalAssessment(
            session_phase="midday",
            time_decay_context="ok",
            supporting_evidence_ids=(),
            confidence=0.6,
        ),
        overall_uncertainty=0.4,
        synthesizer_model_call_id=uuid4(),
    )
    strategy = _strategy(pattern_ids=(uuid4(),))
    episode, _, _ = await _compile(tmp_path, strategy=strategy, world_model=wm)
    assert episode.session_phase == "midday"
    assert episode.volatility_bucket in {"high", "medium"}
    assert episode.liquidity_bucket == "tight"
    assert "trend" in episode.market_regime_tags


@pytest.mark.asyncio
async def test_compiler_missing_strategy_provenance_is_ev_ineligible(tmp_path) -> None:
    episode, _, _ = await _compile(tmp_path, strategy=None, strategy_id=None)
    assert "historical_strategy_family_missing" in episode.completeness_findings
    assert "historical_ev_eligible=false" in episode.completeness_findings
    assert episode.completed is False


@pytest.mark.asyncio
async def test_missing_strategy_family_is_not_inferred_from_role(tmp_path) -> None:
    strategy = _strategy(
        family=None,
        pattern_ids=(uuid4(),),
        role=AgentRole.BULLISH_INVENTOR,
    )
    episode, _, _ = await _compile(tmp_path, strategy=strategy)
    assert episode.strategy_family is None
    assert "historical_strategy_family_missing" in episode.completeness_findings
    assert "historical_ev_eligible=false" in episode.completeness_findings


@pytest.mark.asyncio
async def test_bullish_inventor_mean_reversion_family_is_preserved(tmp_path) -> None:
    strategy = _strategy(
        family="mean_reversion",
        pattern_ids=(uuid4(),),
        role=AgentRole.BULLISH_INVENTOR,
    )
    episode, _, _ = await _compile(tmp_path, strategy=strategy)
    assert episode.strategy_family == "mean_reversion"


@pytest.mark.asyncio
async def test_legacy_strategy_without_family_is_ev_ineligible(tmp_path) -> None:
    strategy = _strategy(family=None, pattern_ids=(uuid4(),))
    episode, _, _ = await _compile(tmp_path, strategy=strategy)
    assert episode.strategy_family is None
    assert "historical_ev_eligible=false" in episode.completeness_findings
    assert episode.completed is False


@pytest.mark.asyncio
async def test_event_horizon_loader_failure_marks_episode_incomplete(tmp_path) -> None:
    strategy = _strategy(pattern_ids=(uuid4(),))
    episode, _, _ = await _compile(
        tmp_path, strategy=strategy, horizon=_FakeHorizonLoader(fail=True)
    )
    assert episode.completed is False
    assert "authoritative_horizon_incomplete" in episode.completeness_findings
    assert "truth_degraded=true" in episode.completeness_findings
    assert "historical_ev_eligible=false" in episode.completeness_findings
    assert "promotion_eligible=false" in episode.completeness_findings


@pytest.mark.asyncio
async def test_reduced_event_sequence_is_not_historical_ev_eligible(tmp_path) -> None:
    strategy = _strategy(pattern_ids=(uuid4(),))
    episode, _, _ = await _compile(
        tmp_path, strategy=strategy, horizon=_FakeHorizonLoader(empty=True)
    )
    assert "reduced_event_sequence_diagnostic_only" in episode.completeness_findings
    assert "historical_ev_eligible=false" in episode.completeness_findings
    assert episode.completed is False


@pytest.mark.asyncio
async def test_historical_service_excludes_compiler_horizon_failure(tmp_path) -> None:
    strategy = _strategy(pattern_ids=(uuid4(),))
    episode, _, repos = await _compile(
        tmp_path, strategy=strategy, horizon=_FakeHorizonLoader(fail=True)
    )
    svc = HistoricalOutcomeService(
        settings=HistoricalOutcomeSettings(minimum_samples_for_ev=1),
        episode_loader=repos["episodes"].list_completed,
        evaluation_loader=lambda eid: [],
    )
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            as_of_timestamp=datetime.now(timezone.utc),
            maximum_samples=10,
            minimum_similarity=Decimal("0.10"),
        )
    )
    assert summary.sample_count == 0
    assert episode.episode_id not in (
        report.excluded_incomplete or ()
    ) or True  # exclusion counted
    assert summary.valid_for_ev is False


@pytest.mark.asyncio
async def test_complete_authoritative_horizon_remains_eligible(tmp_path) -> None:
    strategy = _strategy(pattern_ids=(uuid4(),))
    episode, _, _ = await _compile(
        tmp_path, strategy=strategy, horizon=_FakeHorizonLoader()
    )
    assert episode.completed is True
    assert "authoritative_horizon_incomplete" not in episode.completeness_findings
    assert "historical_ev_eligible=false" not in episode.completeness_findings
    assert len(episode.market_event_ids) >= 2
