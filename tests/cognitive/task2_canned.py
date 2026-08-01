"""Shared canned FakeModelProvider outputs for Task 2 active-path tests."""

from __future__ import annotations

from datetime import date
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

CONTRACT_ID = "SPY:2026-07-01:500.0:call"


def _ref(snapshot_id) -> EvidenceReference:
    return EvidenceReference(
        snapshot_id=snapshot_id,
        source_type="underlying",
        source_id="SPY",
        observed_at=utc_now(),
        value_summary="test ref",
    )


def register_full_path_canned(
    fake: FakeModelProvider,
    snapshot_id,
    cycle_id: str,
    *,
    session: str = "sess-t2",
    contract_id: str = CONTRACT_ID,
    position_action: PositionAction = PositionAction.HOLD,
    order_action: str = "continue_waiting",
) -> UUID:
    """Register canned outputs for a complete cognitive path."""
    sid = snapshot_id
    mc = uuid4()
    evidence_ids: list[UUID] = []

    for role in (
        AgentRole.MARKET_STRUCTURE,
        AgentRole.VOLATILITY,
        AgentRole.OPTIONS_MICROSTRUCTURE,
        AgentRole.TEMPORAL_CONTEXT,
        AgentRole.ANOMALY,
    ):
        ev = AgentEvidence(
            session_id=session,
            snapshot_id=sid,
            prompt_version="2.0.0",
            model_call_id=mc,
            cycle_id=cycle_id,
            agent_role=role,
            claim=f"{role.value} claim",
            direction=MarketDirection.BULLISH,
            confidence=0.7,
            supporting_references=(_ref(sid),),
        )
        evidence_ids.append(ev.evidence_id)
        fake.set_canned_for_role(role.value, ev)

    eids = tuple(evidence_ids)

    def _world_model_from_request(request: ModelRequest) -> MarketWorldModel:
        raw_ids = request.context_payload.get("evidence_ids") or []
        bound: tuple[UUID, ...] = tuple(UUID(str(x)) for x in raw_ids) or eids
        refs = bound[:3] if bound else eids[:3]
        return MarketWorldModel(
            session_id=session,
            snapshot_id=request.snapshot_id or sid,
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or cycle_id,
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
                supporting_evidence_ids=refs[:2] if refs else (),
                confidence=0.6,
            ),
            volatility_state=VolatilityAssessment(
                state=MarketDirection.VOLATILITY_COMPRESSION,
                summary="IV compressed vs morning",
                supporting_evidence_ids=refs[1:3] if len(refs) > 1 else refs,
                confidence=0.55,
            ),
            options_state=OptionsMicrostructureAssessment(
                liquidity_summary="ATM spreads acceptable",
                spread_conditions="tight",
                supporting_evidence_ids=refs[2:4] if len(refs) > 2 else refs,
                confidence=0.58,
            ),
            temporal_state=TemporalAssessment(
                session_phase="regular",
                time_decay_context="mid-session 0DTE",
                supporting_evidence_ids=refs[3:5] if len(refs) > 3 else refs,
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
        fake.set_canned_for_role(
            role.value,
            PatternHypothesis(
                session_id=session,
                snapshot_id=sid,
                prompt_version="2.0.0",
                model_call_id=mc,
                cycle_id=cycle_id,
                name=name,
                description=f"{name} pattern",
                direction=MarketDirection.BULLISH,
                expected_horizon_seconds=300,
                novelty_score=0.5,
                confidence=0.55,
                agent_role=role,
                supporting_evidence_ids=eids[:2],
            ),
        )

    strategy_id = uuid4()
    strategies_by_role: dict[str, StrategyHypothesis] = {}
    for role_name, agent_role in (
        ("bullish_inventor", AgentRole.BULLISH_INVENTOR),
        ("bearish_inventor", AgentRole.BEARISH_INVENTOR),
        ("neutral_advocate", AgentRole.NEUTRAL_ADVOCATE),
    ):
        sid_strategy = strategy_id if role_name == "bullish_inventor" else uuid4()
        leg = StrategyLegCandidate(
            contract_id=contract_id,
            side="buy",
            option_type="call",
            strike=Decimal("500"),
            quantity=1,
            rationale="ATM call from surface",
        )
        strategies_by_role[role_name] = StrategyHypothesis(
            session_id=session,
            snapshot_id=sid,
            strategy_id=sid_strategy,
            prompt_version="2.0.0",
            model_call_id=mc,
            cycle_id=cycle_id,
            name=role_name,
            market_thesis=f"{role_name} thesis",
            direction=MarketDirection.BULLISH
            if "bull" in role_name
            else (
                MarketDirection.BEARISH if "bear" in role_name else MarketDirection.NEUTRAL
            ),
            candidate_legs=(leg,),
            entry_plan=EntryPlan(entry_style="immediate", preferred_order_type="limit"),
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
            agent_role=agent_role,
        )
        fake.set_canned_for_role(role_name, strategies_by_role[role_name])

    for role in (
        "strategy_advocate",
        "falsifier",
        "historical_critic",
        "execution_critic",
        "alternative_explanation",
    ):
        fake.set_canned_for_role(
            role,
            DebateReview(
                strategy_id=strategy_id,
                snapshot_id=sid,
                cycle_id=cycle_id,
                reviewer_role=AgentRole(role),
                verdict="support",
                confidence=0.6,
                prompt_version="2.0.0",
                model_call_id=mc,
            ),
        )

    # Bind meta_decision to the strategies in *this* request. Concurrent
    # register_full_path_canned calls otherwise race static selected_strategy_id
    # against a different inventor response mid-graph.
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
        # Prefer a strategy with a valid objective score when scores are present.
        if valid_ids:
            for raw in candidates:
                if not isinstance(raw, dict) or not raw.get("strategy_id"):
                    continue
                sid_cand = str(raw["strategy_id"])
                if sid_cand in valid_ids:
                    selected = UUID(sid_cand)
                    break
        if selected is None:
            for raw in candidates:
                if isinstance(raw, dict) and raw.get("strategy_id"):
                    selected = UUID(str(raw["strategy_id"]))
                    break
        # When objective scores exist but none are valid trade scores, abandon.
        if scores and not valid_ids:
            selected = None
        action = (
            MetaDecisionAction.EXECUTE
            if selected is not None
            else MetaDecisionAction.ABANDON
        )
        return MetaDecision(
            session_id=session,
            snapshot_id=request.snapshot_id or sid,
            decision_id=uuid4(),
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or cycle_id,
            action=action,
            selected_strategy_id=selected,
            confidence=0.7,
            rationale_summary="execute test" if selected is not None else "abandon test",
        )

    fake.set_role_factory("meta_decision", _meta_decision_from_request)
    fake.set_canned_for_role(
        "entry_tactician",
        ExecutionProposal(
            proposal_id=uuid4(),
            decision_id=uuid4(),
            strategy_id=strategy_id,
            session_id=session,
            cycle_id=cycle_id,
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
            prompt_version="2.0.0",
            model_call_id=mc,
        ),
    )

    thesis = PositionThesisVersion(
        position_id=contract_id,
        contract_id=contract_id,
        session_id=session,
        snapshot_id=sid,
        original_strategy_id=strategy_id,
        current_thesis="thesis holds" if position_action == PositionAction.HOLD else "exit now",
        recommended_action=position_action,
        recommended_quantity=1,
        recommended_limit_price=Decimal("1.20"),
        confidence=0.7,
        prompt_version="2.0.0",
        model_call_id=mc,
    )
    # Preserve paper-path role factories when present. Static canned responses would
    # clear factories (set_canned_for_role pops them) and break EXIT/OM tracking.
    paper_ctl = getattr(fake, "_paper_path_state", None)
    if paper_ctl is not None:
        paper_ctl.session_id = session
        paper_ctl.contract_id = contract_id
        paper_ctl.position_action = position_action
    else:
        fake.set_canned_for_role("position_thesis", thesis)
        fake.set_canned_for_role(
            "position_decision",
            thesis.model_copy(
                update={
                    "thesis_version_id": uuid4(),
                    "recommended_action": position_action,
                }
            ),
        )
        fake.set_canned_for_role(
            "order_manager",
            OrderManagementDecision(
                session_id=session,
                snapshot_id=sid,
                prompt_version="2.0.0",
                model_call_id=mc,
                cycle_id=cycle_id,
                client_order_id="order-1",
                action=order_action,  # type: ignore[arg-type]
                rationale_summary=f"order action {order_action}",
            ),
        )
    return strategy_id
