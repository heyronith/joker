"""Deterministic FakeModelProvider path for adversarial mode runners.

Adversarial fixtures use synthetic snapshot IDs that are not covered by paper/
replay canned registrations. Mode runners install these request-bound factories
so cognitive graphs can complete without live models, while still observing
fail-closed / no-trade / exit behaviour from gateway and DQ outcomes.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from joker.cognition.schemas import (
    AgentEvidence,
    AgentRole,
    DebateReview,
    EntryPlan,
    EvidenceConflict,
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
)
from joker.models.schemas import ModelRequest, utc_now

DEFAULT_CONTRACT_ID = "SPY:2026-07-01:500.0:call"


def _ref(snapshot_id: UUID) -> EvidenceReference:
    return EvidenceReference(
        snapshot_id=snapshot_id,
        source_type="underlying",
        source_id="SPY",
        observed_at=utc_now(),
        value_summary="adversarial fixture ref",
    )


def install_adversarial_model_path(
    provider: Any,
    *,
    session_id: str,
    contract_id: str = DEFAULT_CONTRACT_ID,
    meta_action: MetaDecisionAction = MetaDecisionAction.ABANDON,
    position_action: PositionAction = PositionAction.HOLD,
) -> None:
    """Install request-bound factories for a complete offline cognitive path."""
    if provider is None or not hasattr(provider, "set_role_factory"):
        return

    def _evidence(request: ModelRequest) -> AgentEvidence:
        role_raw = request.role or AgentRole.MARKET_STRUCTURE.value
        try:
            role = AgentRole(role_raw)
        except ValueError:
            role = AgentRole.MARKET_STRUCTURE
        sid = request.snapshot_id or uuid4()
        return AgentEvidence(
            session_id=session_id,
            snapshot_id=sid,
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or "adv",
            agent_role=role,
            claim=f"{role.value} adversarial claim",
            direction=MarketDirection.NEUTRAL,
            confidence=0.55,
            supporting_references=(_ref(sid),),
        )

    for role in (
        AgentRole.MARKET_STRUCTURE,
        AgentRole.VOLATILITY,
        AgentRole.OPTIONS_MICROSTRUCTURE,
        AgentRole.TEMPORAL_CONTEXT,
        AgentRole.ANOMALY,
    ):
        provider.set_role_factory(role.value, _evidence)

    def _world(request: ModelRequest) -> MarketWorldModel:
        sid = request.snapshot_id or uuid4()
        raw_ids = request.context_payload.get("evidence_ids") or []
        bound: tuple[UUID, ...] = tuple(UUID(str(x)) for x in raw_ids) or (uuid4(),)
        refs = bound[:3]
        return MarketWorldModel(
            session_id=session_id,
            snapshot_id=sid,
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or "adv",
            regime_hypotheses=(
                RegimeHypothesis(
                    label="adversarial-neutral",
                    direction=MarketDirection.NEUTRAL,
                    confidence=0.5,
                    supporting_evidence_ids=refs,
                    rationale="adversarial fixture world model",
                ),
            ),
            market_structure=MarketStructureAssessment(
                primary_direction=MarketDirection.NEUTRAL,
                structure_summary="adversarial structure",
                supporting_evidence_ids=refs[:1],
                confidence=0.5,
            ),
            volatility_state=VolatilityAssessment(
                state=MarketDirection.VOLATILITY_COMPRESSION,
                summary="adversarial vol",
                supporting_evidence_ids=refs[:1],
                confidence=0.5,
            ),
            options_state=OptionsMicrostructureAssessment(
                liquidity_summary="adversarial options",
                spread_conditions="tight",
                supporting_evidence_ids=refs[:1],
                confidence=0.5,
            ),
            temporal_state=TemporalAssessment(
                session_phase="regular",
                time_decay_context="0DTE adversarial",
                supporting_evidence_ids=refs[:1],
                confidence=0.5,
            ),
            evidence_ids=bound,
            unresolved_questions=(),
            overall_uncertainty=0.5,
            synthesizer_model_call_id=request.request_id,
        )

    provider.set_role_factory(AgentRole.WORLD_MODEL_SYNTHESISER.value, _world)

    def _pattern(request: ModelRequest) -> PatternHypothesis:
        sid = request.snapshot_id or uuid4()
        role_raw = request.role or AgentRole.PATTERN_MINER.value
        try:
            role = AgentRole(role_raw)
        except ValueError:
            role = AgentRole.PATTERN_MINER
        return PatternHypothesis(
            session_id=session_id,
            snapshot_id=sid,
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or "adv",
            name="adversarial-pattern",
            description="adversarial pattern",
            direction=MarketDirection.NEUTRAL,
            expected_horizon_seconds=300,
            novelty_score=0.4,
            confidence=0.45,
            agent_role=role,
            supporting_evidence_ids=(),
        )

    for role in (
        AgentRole.PATTERN_MINER,
        AgentRole.SEQUENCE_ANALYST,
        AgentRole.ANALOGY_RETRIEVER,
    ):
        provider.set_role_factory(role.value, _pattern)

    def _strategy(request: ModelRequest) -> StrategyHypothesis:
        sid = request.snapshot_id or uuid4()
        role_raw = request.role or "bullish_inventor"
        role_map = {
            "bullish_inventor": AgentRole.BULLISH_INVENTOR,
            "bearish_inventor": AgentRole.BEARISH_INVENTOR,
            "neutral_advocate": AgentRole.NEUTRAL_ADVOCATE,
        }
        agent_role = role_map.get(role_raw, AgentRole.NEUTRAL_ADVOCATE)
        direction = (
            MarketDirection.BULLISH
            if "bull" in role_raw
            else (
                MarketDirection.BEARISH
                if "bear" in role_raw
                else MarketDirection.NEUTRAL
            )
        )
        return StrategyHypothesis(
            session_id=session_id,
            snapshot_id=sid,
            strategy_id=uuid4(),
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or "adv",
            name=role_raw,
            market_thesis=f"{role_raw} adversarial thesis",
            direction=direction,
            candidate_legs=(
                StrategyLegCandidate(
                    contract_id=contract_id,
                    side="buy",
                    option_type="call",
                    strike=Decimal("500"),
                    quantity=1,
                    rationale="adversarial ATM",
                ),
            ),
            entry_plan=EntryPlan(entry_style="immediate", preferred_order_type="limit"),
            execution_plan=ExecutionPlan(
                max_quote_age_seconds=60,
                partial_fill_policy="wait",
                replacement_policy="none",
            ),
            exit_plan=ExitPlan(stop_conditions=("stop",)),
            invalidation_plan=InvalidationPlan(conditions=("inv",)),
            expected_horizon_seconds=600,
            confidence=0.5,
            novelty_score=0.4,
            agent_role=agent_role,
        )

    for role_name in ("bullish_inventor", "bearish_inventor", "neutral_advocate"):
        provider.set_role_factory(role_name, _strategy)

    def _debate(request: ModelRequest) -> DebateReview:
        sid = request.snapshot_id or uuid4()
        role_raw = request.role or "falsifier"
        try:
            reviewer = AgentRole(role_raw)
        except ValueError:
            reviewer = AgentRole.FALSIFIER
        return DebateReview(
            strategy_id=uuid4(),
            snapshot_id=sid,
            cycle_id=request.cycle_id or "adv",
            reviewer_role=reviewer,
            verdict="support",
            confidence=0.5,
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
        )

    for role in (
        "strategy_advocate",
        "falsifier",
        "historical_critic",
        "execution_critic",
        "alternative_explanation",
    ):
        provider.set_role_factory(role, _debate)

    def _meta(request: ModelRequest) -> MetaDecision:
        sid = request.snapshot_id or uuid4()
        selected: UUID | None = None
        if meta_action in {MetaDecisionAction.EXECUTE, MetaDecisionAction.PROBE}:
            candidates = request.context_payload.get("candidate_strategies") or []
            for raw in candidates:
                if isinstance(raw, dict) and raw.get("strategy_id"):
                    selected = UUID(str(raw["strategy_id"]))
                    break
            if selected is None:
                selected = uuid4()
        return MetaDecision(
            session_id=session_id,
            snapshot_id=sid,
            decision_id=uuid4(),
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
            cycle_id=request.cycle_id or "adv",
            action=meta_action,
            selected_strategy_id=selected,
            confidence=0.4,
            rationale_summary=f"adversarial meta:{meta_action.value}",
        )

    provider.set_role_factory("meta_decision", _meta)

    def _entry(request: ModelRequest) -> ExecutionProposal:
        sid = request.snapshot_id or uuid4()
        return ExecutionProposal(
            proposal_id=uuid4(),
            decision_id=uuid4(),
            strategy_id=uuid4(),
            session_id=session_id,
            cycle_id=request.cycle_id or "adv",
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
            entry_rationale="adversarial entry",
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
        )

    provider.set_role_factory("entry_tactician", _entry)

    def _position(request: ModelRequest) -> PositionThesisVersion:
        sid = request.snapshot_id or uuid4()
        return PositionThesisVersion(
            thesis_version_id=uuid4(),
            position_id=contract_id,
            contract_id=contract_id,
            session_id=session_id,
            snapshot_id=sid,
            original_strategy_id=uuid4(),
            current_thesis=(
                "exit" if position_action == PositionAction.EXIT else "hold adversarial"
            ),
            recommended_action=position_action,
            recommended_quantity=1,
            recommended_limit_price=Decimal("1.20"),
            confidence=0.6,
            prompt_version=request.prompt_version or "2.0.0",
            model_call_id=request.request_id,
        )

    # Do not clobber scenario-specific position factories already installed.
    if "position_thesis" not in getattr(provider, "_role_factories", {}):
        provider.set_role_factory("position_thesis", _position)
    if "position_decision" not in getattr(provider, "_role_factories", {}):
        provider.set_role_factory("position_decision", _position)


def install_scenario_specific_observations(
    provider: Any,
    *,
    session_id: str,
    stimulus: dict[str, Any],
    contract_id: str = DEFAULT_CONTRACT_ID,
) -> None:
    """Overlay role factories that emit distinctive, observable artefacts.

    Generic ABANDON alone must not satisfy scenario-specific invariants. These
    factories plant concrete fields (evidence_conflicts, opposing reviews, wide
    spreads, empty evidence links, overfit failure modes) that the entry runner
    can observe from the graph result.
    """
    if provider is None or not hasattr(provider, "set_role_factory"):
        return

    if stimulus.get("conflicting_evidence"):
        def _world_conflict(request: ModelRequest) -> MarketWorldModel:
            sid = request.snapshot_id or uuid4()
            raw_ids = request.context_payload.get("evidence_ids") or []
            bound: tuple[UUID, ...] = tuple(UUID(str(x)) for x in raw_ids)
            eid_a = bound[0] if bound else None
            eid_b = bound[1] if len(bound) > 1 else eid_a
            conflicts = ()
            if eid_a is not None and eid_b is not None:
                conflicts = (
                    EvidenceConflict(
                        claim_a="structure bullish",
                        claim_b="volatility bearish",
                        evidence_ids_a=(eid_a,),
                        evidence_ids_b=(eid_b,),
                        resolution_status="unresolved",
                    ),
                )
            return MarketWorldModel(
                session_id=session_id,
                snapshot_id=sid,
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or "adv",
                regime_hypotheses=(
                    RegimeHypothesis(
                        label="conflicted-regime",
                        direction=MarketDirection.NEUTRAL,
                        confidence=0.4,
                        supporting_evidence_ids=bound[:2],
                        rationale="structure and volatility disagree",
                    ),
                ),
                market_structure=MarketStructureAssessment(
                    primary_direction=MarketDirection.BULLISH,
                    structure_summary="structure bullish",
                    supporting_evidence_ids=bound[:1],
                    confidence=0.7,
                ),
                volatility_state=VolatilityAssessment(
                    state=MarketDirection.BEARISH,
                    summary="volatility bearish — conflicts with structure",
                    supporting_evidence_ids=bound[1:2] or bound[:1],
                    confidence=0.7,
                ),
                options_state=OptionsMicrostructureAssessment(
                    liquidity_summary="ok",
                    spread_conditions="tight",
                    supporting_evidence_ids=(),
                    confidence=0.5,
                ),
                temporal_state=TemporalAssessment(
                    session_phase="regular",
                    time_decay_context="0DTE",
                    supporting_evidence_ids=(),
                    confidence=0.5,
                ),
                evidence_ids=bound,
                evidence_conflicts=conflicts,
                unresolved_questions=("resolve structure vs volatility",),
                overall_uncertainty=0.8,
                synthesizer_model_call_id=request.request_id,
            )

        def _meta_conflict(request: ModelRequest) -> MetaDecision:
            raw_ids = request.context_payload.get("evidence_ids") or []
            bound: tuple[UUID, ...] = tuple(UUID(str(x)) for x in raw_ids)
            return MetaDecision(
                session_id=session_id,
                snapshot_id=request.snapshot_id or uuid4(),
                decision_id=uuid4(),
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or "adv",
                action=MetaDecisionAction.ABANDON,
                selected_strategy_id=None,
                confidence=0.25,
                rationale_summary="abandon due to unresolved evidence conflict",
                contradicting_evidence_ids=bound[:2],
            )

        provider.set_role_factory(AgentRole.WORLD_MODEL_SYNTHESISER.value, _world_conflict)
        provider.set_role_factory("meta_decision", _meta_conflict)

    if stimulus.get("false_consensus"):
        def _debate_oppose(request: ModelRequest) -> DebateReview:
            role_raw = request.role or "falsifier"
            try:
                reviewer = AgentRole(role_raw)
            except ValueError:
                reviewer = AgentRole.FALSIFIER
            verdict = "oppose" if role_raw in {"falsifier", "historical_critic"} else "support"
            return DebateReview(
                strategy_id=uuid4(),
                snapshot_id=request.snapshot_id or uuid4(),
                cycle_id=request.cycle_id or "adv",
                reviewer_role=reviewer,
                verdict=verdict,  # type: ignore[arg-type]
                confidence=0.7,
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                claims=("repeated identical claims do not constitute consensus",),
            )

        def _meta_consensus(request: ModelRequest) -> MetaDecision:
            raw_ids = request.context_payload.get("evidence_ids") or []
            bound: tuple[UUID, ...] = tuple(UUID(str(x)) for x in raw_ids)
            return MetaDecision(
                session_id=session_id,
                snapshot_id=request.snapshot_id or uuid4(),
                decision_id=uuid4(),
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or "adv",
                action=MetaDecisionAction.ABANDON,
                selected_strategy_id=None,
                confidence=0.3,
                rationale_summary="resist false consensus from repeated evidence",
                contradicting_evidence_ids=bound[:1],
            )

        for role in (
            "strategy_advocate",
            "falsifier",
            "historical_critic",
            "execution_critic",
            "alternative_explanation",
        ):
            provider.set_role_factory(role, _debate_oppose)
        provider.set_role_factory("meta_decision", _meta_consensus)

    if stimulus.get("thin_liquidity"):
        def _world_thin(request: ModelRequest) -> MarketWorldModel:
            sid = request.snapshot_id or uuid4()
            return MarketWorldModel(
                session_id=session_id,
                snapshot_id=sid,
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or "adv",
                regime_hypotheses=(
                    RegimeHypothesis(
                        label="bullish-thin-liquidity",
                        direction=MarketDirection.BULLISH,
                        confidence=0.55,
                        supporting_evidence_ids=(),
                        rationale="bullish price with thin liquidity",
                    ),
                ),
                market_structure=MarketStructureAssessment(
                    primary_direction=MarketDirection.BULLISH,
                    structure_summary="price bullish",
                    supporting_evidence_ids=(),
                    confidence=0.6,
                ),
                volatility_state=VolatilityAssessment(
                    state=MarketDirection.NEUTRAL,
                    summary="vol neutral",
                    supporting_evidence_ids=(),
                    confidence=0.5,
                ),
                options_state=OptionsMicrostructureAssessment(
                    liquidity_summary="thin book — wide spreads",
                    spread_conditions="wide",
                    supporting_evidence_ids=(),
                    confidence=0.8,
                ),
                temporal_state=TemporalAssessment(
                    session_phase="regular",
                    time_decay_context="0DTE",
                    supporting_evidence_ids=(),
                    confidence=0.5,
                ),
                evidence_ids=(),
                unresolved_questions=(),
                overall_uncertainty=0.7,
                synthesizer_model_call_id=request.request_id,
            )

        def _meta_thin(request: ModelRequest) -> MetaDecision:
            return MetaDecision(
                session_id=session_id,
                snapshot_id=request.snapshot_id or uuid4(),
                decision_id=uuid4(),
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or "adv",
                action=MetaDecisionAction.ABANDON,
                selected_strategy_id=None,
                confidence=0.2,
                rationale_summary="reject entry — thin liquidity / wide spreads",
            )

        provider.set_role_factory(AgentRole.WORLD_MODEL_SYNTHESISER.value, _world_thin)
        provider.set_role_factory("meta_decision", _meta_thin)

    if stimulus.get("unsupported_reasoning"):
        def _meta_unsupported(request: ModelRequest) -> MetaDecision:
            return MetaDecision(
                session_id=session_id,
                snapshot_id=request.snapshot_id or uuid4(),
                decision_id=uuid4(),
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or "adv",
                action=MetaDecisionAction.ABANDON,
                selected_strategy_id=None,
                confidence=0.15,
                rationale_summary="reject unsupported reasoning without evidence links",
                supporting_evidence_ids=(),
            )

        def _strategy_unsupported(request: ModelRequest) -> StrategyHypothesis:
            sid = request.snapshot_id or uuid4()
            role_raw = request.role or "bullish_inventor"
            return StrategyHypothesis(
                session_id=session_id,
                snapshot_id=sid,
                strategy_id=uuid4(),
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or "adv",
                name="unsupported-claim",
                market_thesis="unsupported claim with no evidence",
                direction=MarketDirection.BULLISH,
                candidate_legs=(
                    StrategyLegCandidate(
                        contract_id=contract_id,
                        side="buy",
                        option_type="call",
                        strike=Decimal("500"),
                        quantity=1,
                        rationale="ungrounded",
                    ),
                ),
                entry_plan=EntryPlan(entry_style="immediate", preferred_order_type="limit"),
                execution_plan=ExecutionPlan(
                    max_quote_age_seconds=60,
                    partial_fill_policy="wait",
                    replacement_policy="none",
                ),
                exit_plan=ExitPlan(stop_conditions=("stop",)),
                invalidation_plan=InvalidationPlan(conditions=("inv",)),
                expected_horizon_seconds=60,
                confidence=0.9,
                novelty_score=0.1,
                agent_role=AgentRole.BULLISH_INVENTOR,
                # Deliberately empty — unsupported by evidence.
                supporting_evidence_ids=(),
            )

        for role_name in ("bullish_inventor", "bearish_inventor", "neutral_advocate"):
            provider.set_role_factory(role_name, _strategy_unsupported)
        provider.set_role_factory("meta_decision", _meta_unsupported)

    if stimulus.get("narrow_overfit"):
        def _debate_overfit(request: ModelRequest) -> DebateReview:
            role_raw = request.role or "falsifier"
            try:
                reviewer = AgentRole(role_raw)
            except ValueError:
                reviewer = AgentRole.FALSIFIER
            return DebateReview(
                strategy_id=uuid4(),
                snapshot_id=request.snapshot_id or uuid4(),
                cycle_id=request.cycle_id or "adv",
                reviewer_role=reviewer,
                verdict="oppose",
                confidence=0.75,
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                identified_failure_modes=("narrow_overfit", "curve_fit_to_single_window"),
                claims=("strategy overfits a narrow regime window",),
            )

        def _meta_overfit(request: ModelRequest) -> MetaDecision:
            return MetaDecision(
                session_id=session_id,
                snapshot_id=request.snapshot_id or uuid4(),
                decision_id=uuid4(),
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or "adv",
                action=MetaDecisionAction.ABANDON,
                selected_strategy_id=None,
                confidence=0.2,
                rationale_summary="reject narrowly overfit strategy",
            )

        for role in (
            "strategy_advocate",
            "falsifier",
            "historical_critic",
            "execution_critic",
            "alternative_explanation",
        ):
            provider.set_role_factory(role, _debate_overfit)
        provider.set_role_factory("meta_decision", _meta_overfit)

    if stimulus.get("justified_no_trade"):
        def _meta_justified(request: ModelRequest) -> MetaDecision:
            return MetaDecision(
                session_id=session_id,
                snapshot_id=request.snapshot_id or uuid4(),
                decision_id=uuid4(),
                prompt_version=request.prompt_version or "2.0.0",
                model_call_id=request.request_id,
                cycle_id=request.cycle_id or "adv",
                action=MetaDecisionAction.ABANDON,
                selected_strategy_id=None,
                confidence=0.35,
                rationale_summary="justified no-trade: edge insufficient after review",
                supporting_evidence_ids=(uuid4(),),
            )

        provider.set_role_factory("meta_decision", _meta_justified)
