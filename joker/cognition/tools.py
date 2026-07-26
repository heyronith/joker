"""Read-only cognitive tool boundary over Task 1 truth."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from joker.cognition.schemas import (
    AgentDataRequest,
    AgentEvidence,
    DebateReview,
    MetaDecision,
    PatternHypothesis,
    StrategyHypothesis,
)
from joker.market.bars import MarketBar
from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.market.quality import DataQualityReport
from joker.market.snapshots import MarketSnapshot


class CognitiveReadTools(Protocol):
    """Controlled read-only access to market and cognitive state."""

    async def get_market_snapshot(self, snapshot_id: UUID | str) -> MarketSnapshot | None:
        """Fetch a market snapshot by ID."""
        ...

    async def get_bars(
        self,
        *,
        snapshot_id: UUID | str,
        timeframe: str,
        limit: int | None = None,
    ) -> tuple[MarketBar, ...]:
        """Return bars for a snapshot and timeframe ('1m' or '5m')."""
        ...

    async def query_option_surface(
        self,
        *,
        snapshot_id: UUID | str,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[OptionContractSnapshot, ...]:
        """Return a bounded option-surface slice."""
        ...

    async def get_data_quality(
        self, snapshot_id: UUID | str
    ) -> DataQualityReport | None:
        """Return data-quality report for a snapshot."""
        ...

    async def get_order_projection(
        self, client_order_id: str
    ) -> dict[str, Any] | None:
        """Return read-only order projection."""
        ...

    async def get_position_projection(
        self, position_id: str
    ) -> dict[str, Any] | None:
        """Return read-only position projection."""
        ...

    async def list_session_hypotheses(
        self, session_id: str
    ) -> tuple[PatternHypothesis, ...]:
        """List pattern hypotheses for the session."""
        ...

    async def get_strategy(
        self, strategy_id: UUID | str
    ) -> StrategyHypothesis | None:
        """Fetch a strategy hypothesis by ID."""
        ...

    async def get_decision_provenance(
        self, decision_id: UUID | str
    ) -> dict[str, Any] | None:
        """Return decision provenance bundle (decision, reviews, evidence refs)."""
        ...

    async def fulfill_data_request(
        self, request: AgentDataRequest, *, snapshot_id: UUID | str
    ) -> dict[str, Any]:
        """Validate and fulfil a bounded agent data request."""
        ...


class InMemoryCognitiveReadTools:
    """In-memory CognitiveReadTools for tests."""

    def __init__(
        self,
        *,
        snapshots: dict[str, MarketSnapshot] | None = None,
        surfaces: dict[str, OptionSurfaceSnapshot] | None = None,
        data_quality: dict[str, DataQualityReport] | None = None,
        orders: dict[str, dict[str, Any]] | None = None,
        positions: dict[str, dict[str, Any]] | None = None,
        hypotheses: dict[str, list[PatternHypothesis]] | None = None,
        strategies: dict[str, StrategyHypothesis] | None = None,
        decisions: dict[str, MetaDecision] | None = None,
        reviews: dict[str, list[DebateReview]] | None = None,
        evidence: dict[str, list[AgentEvidence]] | None = None,
    ) -> None:
        self._snapshots = snapshots or {}
        self._surfaces = surfaces or {}
        self._data_quality = data_quality or {}
        self._orders = orders or {}
        self._positions = positions or {}
        self._hypotheses = hypotheses or {}
        self._strategies = strategies or {}
        self._decisions = decisions or {}
        self._reviews = reviews or {}
        self._evidence = evidence or {}

    async def get_market_snapshot(self, snapshot_id: UUID | str) -> MarketSnapshot | None:
        return self._snapshots.get(str(snapshot_id))

    async def get_bars(
        self,
        *,
        snapshot_id: UUID | str,
        timeframe: str,
        limit: int | None = None,
    ) -> tuple[MarketBar, ...]:
        snapshot = await self.get_market_snapshot(snapshot_id)
        if snapshot is None:
            return ()
        bars = snapshot.bars_1m if timeframe == "1m" else snapshot.bars_5m
        if limit is not None:
            return bars[-limit:]
        return bars

    async def query_option_surface(
        self,
        *,
        snapshot_id: UUID | str,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[OptionContractSnapshot, ...]:
        snapshot = await self.get_market_snapshot(snapshot_id)
        if snapshot is None or snapshot.option_surface_id is None:
            return ()
        surface = self._surfaces.get(str(snapshot.option_surface_id))
        if surface is None:
            return ()
        contracts = surface.contracts
        params = parameters or {}
        max_rows = int(params.get("max_rows", 80))
        option_type = params.get("option_type")
        if option_type:
            contracts = tuple(c for c in contracts if c.option_type == option_type)
        strike_min = params.get("strike_min")
        strike_max = params.get("strike_max")
        if strike_min is not None or strike_max is not None:
            filtered = []
            for contract in contracts:
                if strike_min is not None and contract.strike < strike_min:
                    continue
                if strike_max is not None and contract.strike > strike_max:
                    continue
                filtered.append(contract)
            contracts = tuple(filtered)
        return contracts[:max_rows]

    async def get_data_quality(
        self, snapshot_id: UUID | str
    ) -> DataQualityReport | None:
        return self._data_quality.get(str(snapshot_id))

    async def get_order_projection(
        self, client_order_id: str
    ) -> dict[str, Any] | None:
        return self._orders.get(client_order_id)

    async def get_position_projection(
        self, position_id: str
    ) -> dict[str, Any] | None:
        return self._positions.get(position_id)

    async def list_session_hypotheses(
        self, session_id: str
    ) -> tuple[PatternHypothesis, ...]:
        return tuple(self._hypotheses.get(session_id, []))

    async def get_strategy(
        self, strategy_id: UUID | str
    ) -> StrategyHypothesis | None:
        return self._strategies.get(str(strategy_id))

    async def get_decision_provenance(
        self, decision_id: UUID | str
    ) -> dict[str, Any] | None:
        decision = self._decisions.get(str(decision_id))
        if decision is None:
            return None
        review_list = self._reviews.get(str(decision_id), [])
        evidence_ids = list(decision.supporting_evidence_ids) + list(
            decision.contradicting_evidence_ids
        )
        evidence_items: list[AgentEvidence] = []
        for eid in evidence_ids:
            for items in self._evidence.values():
                for item in items:
                    if item.evidence_id == eid:
                        evidence_items.append(item)
        return {
            "decision": decision.model_dump(mode="json"),
            "reviews": [r.model_dump(mode="json") for r in review_list],
            "evidence": [e.model_dump(mode="json") for e in evidence_items],
        }

    async def fulfill_data_request(
        self, request: AgentDataRequest, *, snapshot_id: UUID | str
    ) -> dict[str, Any]:
        if request.request_type == "bars":
            timeframe = str(request.parameters.get("timeframe", "1m"))
            limit = request.parameters.get("limit")
            bars = await self.get_bars(
                snapshot_id=snapshot_id,
                timeframe=timeframe,
                limit=int(limit) if limit is not None else None,
            )
            return {"bars": [b.model_dump(mode="json") for b in bars]}
        if request.request_type == "option_surface_slice":
            rows = await self.query_option_surface(
                snapshot_id=snapshot_id, parameters=request.parameters
            )
            return {"contracts": [c.model_dump(mode="json") for c in rows]}
        if request.request_type == "data_quality":
            report = await self.get_data_quality(snapshot_id)
            return {"data_quality": report.model_dump(mode="json") if report else None}
        if request.request_type == "ledger_order":
            order_id = str(request.parameters.get("client_order_id", ""))
            return {"order": await self.get_order_projection(order_id)}
        if request.request_type == "ledger_position":
            position_id = str(request.parameters.get("position_id", ""))
            return {"position": await self.get_position_projection(position_id)}
        if request.request_type == "active_hypotheses":
            session_id = str(request.parameters.get("session_id", ""))
            hypotheses = await self.list_session_hypotheses(session_id)
            return {
                "hypotheses": [h.model_dump(mode="json") for h in hypotheses],
            }
        return {"error": f"unsupported request_type={request.request_type!r}"}

    def seed_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Test helper to register a snapshot."""
        self._snapshots[str(snapshot.snapshot_id)] = snapshot

    def seed_surface(self, surface: OptionSurfaceSnapshot) -> None:
        """Test helper to register an option surface."""
        self._surfaces[str(surface.surface_id)] = surface

    def seed_data_quality(
        self, snapshot_id: UUID | str, report: DataQualityReport
    ) -> None:
        """Test helper to register data quality for a snapshot."""
        self._data_quality[str(snapshot_id)] = report
