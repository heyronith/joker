"""Request-bound FakeModelProvider canned path for LivePaperRunner acceptance."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID, uuid4

from joker.cognition.schemas import (
    AgentEvidence,
    AgentRole,
    DebateReview,
    EntryPlan,
    EvidenceReference,
    ExecutionLeg,
    ExecutionPlan,
    ExecutionProposal,
    ExitPlan,
    InvalidationPlan,
    MarketDirection,
    MarketStructureAssessment,
    MarketWorldModel,
    MetaDecision,
    MetaDecisionAction,
    OptionsMicrostructureAssessment,
    PatternHypothesis,
    PositionAction,
    PositionThesisVersion,
    RegimeHypothesis,
    StrategyHypothesis,
    StrategyLegCandidate,
    TemporalAssessment,
    VolatilityAssessment,
    OrderManagementDecision,
)
from joker.models.fake_provider import FakeModelProvider
from joker.models.schemas import ModelRequest, utc_now


def _ref(snapshot_id) -> EvidenceReference:
    return EvidenceReference(
        snapshot_id=snapshot_id,
        source_type="underlying",
        source_id="SPY",
        observed_at=utc_now(),
        value_summary="live acceptance ref",
    )


def register_request_bound_canned(
    fake: FakeModelProvider,
    *,
    session: str,
    contract_id: str,
    position_action: PositionAction = PositionAction.HOLD,
) -> None:
    """Register factories so every response binds to the live request snapshot/cycle."""

    for role in (
        AgentRole.MARKET_STRUCTURE,
        AgentRole.VOLATILITY,
        AgentRole.OPTIONS_MICROSTRUCTURE,
        AgentRole.TEMPORAL_CONTEXT,
        AgentRole.ANOMALY,
    ):

        def _evidence_factory(
            request: ModelRequest, *, _role: AgentRole = role
        ) -> AgentEvidence:
            sid = request.snapshot_id or uuid4()
            return AgentEvidence(
                session_id=session,
                snapshot_id=sid,
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or str(uuid4()),
                agent_role=_role,
                claim=f"{_role.value} claim",
                direction=MarketDirection.BULLISH,
                confidence=0.7,
                supporting_references=(_ref(sid),),
            )

        fake.set_role_factory(role.value, _evidence_factory)

    def _world_model_from_request(request: ModelRequest) -> MarketWorldModel:
        sid = request.snapshot_id or uuid4()
        raw_ids = request.context_payload.get("evidence_ids") or []
        bound: tuple[UUID, ...] = tuple(UUID(str(x)) for x in raw_ids) or (uuid4(),)
        refs = bound[:3]
        return MarketWorldModel(
            session_id=session,
            snapshot_id=sid,
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or str(uuid4()),
            regime_hypotheses=(
                RegimeHypothesis(
                    label="bullish-continuation",
                    direction=MarketDirection.BULLISH,
                    confidence=0.62,
                    supporting_evidence_ids=refs,
                    rationale="Structure and volatility evidence align bullish",
                ),
            ),
            market_structure=MarketStructureAssessment(
                primary_direction=MarketDirection.BULLISH,
                structure_summary="Higher lows with reclaim of VWAP",
                supporting_evidence_ids=refs[:2],
                confidence=0.6,
            ),
            volatility_state=VolatilityAssessment(
                state=MarketDirection.VOLATILITY_COMPRESSION,
                summary="IV compressed vs morning",
                supporting_evidence_ids=refs[:1],
                confidence=0.55,
            ),
            options_state=OptionsMicrostructureAssessment(
                liquidity_summary="ATM spreads acceptable",
                spread_conditions="tight",
                supporting_evidence_ids=refs[:1],
                confidence=0.58,
            ),
            temporal_state=TemporalAssessment(
                session_phase="regular",
                time_decay_context="mid-session 0DTE",
                supporting_evidence_ids=refs[:1],
                confidence=0.5,
            ),
            evidence_ids=bound,
            unresolved_questions=("Does volume confirm breakout?",),
            overall_uncertainty=0.4,
            synthesizer_model_call_id=request.request_id,
        )

    fake.set_role_factory(
        AgentRole.WORLD_MODEL_SYNTHESISER.value, _world_model_from_request
    )

    for role, name in (
        (AgentRole.PATTERN_MINER, "breakout"),
        (AgentRole.SEQUENCE_ANALYST, "sequence"),
        (AgentRole.ANALOGY_RETRIEVER, "analogy"),
    ):

        def _pattern_factory(
            request: ModelRequest,
            *,
            _role: AgentRole = role,
            _name: str = name,
        ) -> PatternHypothesis:
            sid = request.snapshot_id or uuid4()
            return PatternHypothesis(
                session_id=session,
                snapshot_id=sid,
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or str(uuid4()),
                name=_name,
                description=f"{_name} pattern",
                direction=MarketDirection.BULLISH,
                expected_horizon_seconds=300,
                novelty_score=0.5,
                confidence=0.55,
                agent_role=_role,
                supporting_evidence_ids=(uuid4(), uuid4()),
            )

        fake.set_role_factory(role.value, _pattern_factory)

    for role_name, agent_role, family in (
        ("bullish_inventor", AgentRole.BULLISH_INVENTOR, "breakout_continuation"),
        ("bearish_inventor", AgentRole.BEARISH_INVENTOR, "failed_breakout_reversal"),
        ("neutral_advocate", AgentRole.NEUTRAL_ADVOCATE, "mean_reversion"),
    ):

        def _strategy_factory(
            request: ModelRequest,
            *,
            _role_name: str = role_name,
            _agent_role: AgentRole = agent_role,
            _family: str = family,
        ) -> StrategyHypothesis:
            sid = request.snapshot_id or uuid4()
            leg = StrategyLegCandidate(
                contract_id=contract_id,
                side="buy",
                option_type="call",
                strike=Decimal("500"),
                quantity=1,
                rationale="ATM call from surface",
            )
            return StrategyHypothesis(
                session_id=session,
                snapshot_id=sid,
                strategy_id=uuid4(),
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or str(uuid4()),
                name=_role_name,
                market_thesis=f"{_role_name} thesis",
                direction=MarketDirection.BULLISH
                if "bull" in _role_name
                else (
                    MarketDirection.BEARISH
                    if "bear" in _role_name
                    else MarketDirection.NEUTRAL
                ),
                strategy_family=_family,
                source_hypothesis_ids=(uuid4(),),
                candidate_legs=(leg,),
                entry_plan=EntryPlan(
                    entry_style="immediate", preferred_order_type="limit"
                ),
                execution_plan=ExecutionPlan(
                    max_quote_age_seconds=60,
                    partial_fill_policy="wait",
                    replacement_policy="none",
                ),
                exit_plan=ExitPlan(stop_conditions=("stop",)),
                invalidation_plan=InvalidationPlan(conditions=("inv",)),
                expected_horizon_seconds=600,
                confidence=0.65,
                novelty_score=0.5,
                agent_role=_agent_role,
            )

        fake.set_role_factory(role_name, _strategy_factory)

    for role in (
        "strategy_advocate",
        "falsifier",
        "historical_critic",
        "execution_critic",
        "alternative_explanation",
    ):

        def _debate_factory(
            request: ModelRequest, *, _role: str = role
        ) -> DebateReview:
            candidates = request.context_payload.get("candidate_strategies") or []
            strategy_id = uuid4()
            for raw in candidates:
                if isinstance(raw, dict) and raw.get("strategy_id"):
                    strategy_id = UUID(str(raw["strategy_id"]))
                    break
            return DebateReview(
                strategy_id=strategy_id,
                snapshot_id=request.snapshot_id or uuid4(),
                cycle_id=request.cycle_id or str(uuid4()),
                reviewer_role=AgentRole(_role),
                verdict="support",
                confidence=0.6,
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
            )

        fake.set_role_factory(role, _debate_factory)

    def _meta_decision_from_request(request: ModelRequest) -> MetaDecision:
        candidates = request.context_payload.get("candidate_strategies") or []
        scores = request.context_payload.get("objective_strategy_scores") or []
        valid_ids = {
            str(s.get("strategy_id"))
            for s in scores
            if isinstance(s, dict)
            and s.get("strategy_id") is not None
            and s.get("valid")
            and not s.get("is_no_trade")
        }
        selected: UUID | None = None
        if valid_ids:
            for raw in candidates:
                if not isinstance(raw, dict) or not raw.get("strategy_id"):
                    continue
                if str(raw["strategy_id"]) in valid_ids:
                    selected = UUID(str(raw["strategy_id"]))
                    break
        if selected is None:
            for raw in candidates:
                if isinstance(raw, dict) and raw.get("strategy_id"):
                    selected = UUID(str(raw["strategy_id"]))
                    break
        if scores and not valid_ids:
            selected = None
        action = (
            MetaDecisionAction.EXECUTE
            if selected is not None
            else MetaDecisionAction.ABANDON
        )
        return MetaDecision(
            session_id=session,
            snapshot_id=request.snapshot_id or uuid4(),
            decision_id=uuid4(),
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or str(uuid4()),
            action=action,
            selected_strategy_id=selected,
            confidence=0.7,
            rationale_summary="execute test" if selected is not None else "wait/abandon",
        )

    fake.set_role_factory("meta_decision", _meta_decision_from_request)

    def _entry_factory(request: ModelRequest) -> ExecutionProposal:
        sid = request.snapshot_id or uuid4()
        strategy_id = uuid4()
        candidates = request.context_payload.get("candidate_strategies") or []
        for raw in candidates:
            if isinstance(raw, dict) and raw.get("strategy_id"):
                strategy_id = UUID(str(raw["strategy_id"]))
                break
        return ExecutionProposal(
            proposal_id=uuid4(),
            decision_id=uuid4(),
            strategy_id=strategy_id,
            session_id=session,
            cycle_id=request.cycle_id or str(uuid4()),
            snapshot_id=sid,
            action="execute",
            legs=(
                ExecutionLeg(
                    contract_id=contract_id,
                    side="buy",
                    quantity=1,
                    limit_price=Decimal("1.10"),
                    sequence_order=0,
                    max_quote_age_seconds=3600,
                    replacement_policy="none",
                    partial_fill_policy="wait",
                ),
            ),
            order_type="limit",
            time_in_force="day",
            entry_rationale="test entry",
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
        )

    fake.set_role_factory("entry_tactician", _entry_factory)

    def _thesis_factory(request: ModelRequest) -> PositionThesisVersion:
        sid = request.snapshot_id or uuid4()
        return PositionThesisVersion(
            position_id=contract_id,
            contract_id=contract_id,
            session_id=session,
            snapshot_id=sid,
            original_strategy_id=uuid4(),
            current_thesis="thesis holds"
            if position_action == PositionAction.HOLD
            else "exit now",
            recommended_action=position_action,
            recommended_quantity=1,
            recommended_limit_price=Decimal("1.20"),
            confidence=0.7,
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
        )

    fake.set_role_factory("position_thesis", _thesis_factory)
    fake.set_role_factory("position_decision", _thesis_factory)

    def _om_factory(request: ModelRequest) -> OrderManagementDecision:
        return OrderManagementDecision(
            session_id=session,
            snapshot_id=request.snapshot_id or uuid4(),
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or str(uuid4()),
            client_order_id="order-1",
            action="continue_waiting",  # type: ignore[arg-type]
            rationale_summary="wait",
        )

    fake.set_role_factory("order_manager", _om_factory)
