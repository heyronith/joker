"""Shared helpers for EpisodeCompiler tests that expect completed episodes."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

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
from decimal import Decimal


class FakeHorizonLoader:
    """Minimal authoritative horizon for unit/integration compiler tests."""

    def __init__(self, *, fail: bool = False, empty: bool = False) -> None:
        self.fail = fail
        self.empty = empty

    async def load(self, **kwargs):
        if self.fail:
            raise RuntimeError("horizon_unavailable")
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


class FakeStrategyRepo:
    def __init__(self, strategy: StrategyHypothesis | None) -> None:
        self._strategy = strategy

    async def get_by_id(self, strategy_id):
        if self._strategy is None:
            return None
        if str(strategy_id) != str(self._strategy.strategy_id):
            return None
        return self._strategy


class FakeProvenance:
    def __init__(
        self,
        *,
        strategy_id: UUID | None,
        entry_id: str = "entry-1",
        exit_id: str = "exit-1",
        contract_id: str = "SPY:2026-07-01:500:call",
    ) -> None:
        self._strategy_id = strategy_id
        self._entry_id = entry_id
        self._exit_id = exit_id
        self._contract_id = contract_id
        self._lifecycle = f"s:{entry_id}:{contract_id}"

    async def get_by_client_order_id(self, client_order_id: str):
        if client_order_id not in {self._entry_id, self._exit_id}:
            return None
        kind = "entry" if client_order_id == self._entry_id else "exit"
        return SimpleNamespace(
            client_order_id=client_order_id,
            # Entry carries strategy; exit omits it (production shape).
            strategy_id=str(self._strategy_id)
            if self._strategy_id and kind == "entry"
            else None,
            proposal_id=str(uuid4()),
            decision_id=str(uuid4()),
            cycle_id="c1",
            snapshot_id=str(uuid4()),
            contract_id=self._contract_id,
            kind=kind,
            extra={
                "position_lifecycle_id": self._lifecycle,
                "originating_entry_client_order_id": self._entry_id,
            },
        )

    async def list_by_lifecycle_id(self, position_lifecycle_id: str):
        if position_lifecycle_id != self._lifecycle:
            return []
        entry = await self.get_by_client_order_id(self._entry_id)
        exit_rec = await self.get_by_client_order_id(self._exit_id)
        return [r for r in (entry, exit_rec) if r is not None]


def make_test_strategy(
    *,
    strategy_id: UUID | None = None,
    family: str = "breakout_continuation",
    contract_id: str = "SPY:2026-07-01:500:call",
    pattern_ids: tuple[UUID, ...] | None = None,
) -> StrategyHypothesis:
    sid = strategy_id or uuid4()
    option_type = contract_id.split(":")[-1] if ":" in contract_id else "call"
    patterns = pattern_ids if pattern_ids is not None else (uuid4(),)
    return StrategyHypothesis(
        session_id="s",
        snapshot_id=uuid4(),
        cycle_id="c1",
        prompt_version="1.0.0",
        model_call_id=uuid4(),
        strategy_id=sid,
        source_hypothesis_ids=patterns,
        name="test",
        market_thesis="t",
        direction=MarketDirection.BULLISH
        if option_type == "call"
        else MarketDirection.BEARISH,
        strategy_family=family,
        candidate_legs=(
            StrategyLegCandidate(
                contract_id=contract_id,
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
        agent_role=AgentRole.BULLISH_INVENTOR,
    )


def build_complete_episode_compiler(
    episode_repo,
    trace_repo=None,
    *,
    strategy: StrategyHypothesis | None = None,
    contract_id: str = "SPY:2026-07-01:500:call",
    entry_id: str = "entry-1",
    exit_id: str = "exit-1",
) -> EpisodeCompiler:
    """Compiler wired with fake horizon + strategy provenance for completed episodes."""
    strat = strategy or make_test_strategy(contract_id=contract_id)
    provenance = FakeProvenance(
        strategy_id=strat.strategy_id,
        entry_id=entry_id,
        exit_id=exit_id,
        contract_id=contract_id,
    )
    return EpisodeCompiler(
        episode_repo,
        trace_repo,
        lifecycle_resolver=PositionLifecycleResolver(provenance=provenance),
        provenance=provenance,
        event_horizon_loader=FakeHorizonLoader(),
        strategy_repo=FakeStrategyRepo(strat),
    )


def entry_terminal_timestamps() -> tuple[datetime, datetime]:
    start = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    return start, start.replace(minute=30)
