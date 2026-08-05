"""Typed, non-authoritative review context for provisional portfolios."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ContractCandidateReviewSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int
    evaluation_id: UUID
    strategy_id: UUID
    contract_id: str
    quantity: int
    capital_required: Decimal
    probability_goal: Decimal | None
    lower_probability_bound: Decimal | None
    estimate_type: str
    relative_spread: Decimal
    liquidity_score: float
    uncertainty_reasons: tuple[str, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()


class PortfolioCandidateReviewSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int
    portfolio_id: UUID
    component_contract_ids: tuple[str, ...]
    component_quantities: tuple[int, ...]
    capital_deployed: Decimal
    probability_goal: Decimal | None
    lower_probability_bound: Decimal | None
    concentration_penalty: Decimal
    liquidity_penalty: Decimal
    shared_scenario_grid_hash: str | None
    failure_modes: tuple[str, ...] = ()
    selected: bool = False


class PortfolioReviewContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target_portfolio_decision_id: UUID
    snapshot_id: UUID
    evaluated_objective_version: int
    ranked_contract_candidates: tuple[ContractCandidateReviewSummary, ...]
    ranked_portfolio_candidates: tuple[PortfolioCandidateReviewSummary, ...]
    selected_candidate: PortfolioCandidateReviewSummary | None
    wait_candidate: dict[str, Any]
    remaining_goal_gap_usd: Decimal
    time_remaining_seconds: int
    deadline_exchange_time: str | None
    shared_scenario_summary: dict[str, Any]
    failure_modes: tuple[str, ...]
    evidence_ids: tuple[UUID, ...]


class PortfolioDebateReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    review_id: UUID
    reviewer_role: str
    target_portfolio_decision_id: UUID
    selected_portfolio_id: UUID | None
    verdict: str
    confidence: float
    claims: tuple[str, ...]
    identified_failure_modes: tuple[str, ...]
    required_revisions: tuple[str, ...]
    finalizer_recommendation: Literal["preserve", "wait", "reoptimize"]


def build_portfolio_review_context(
    *,
    state: dict[str, Any],
    limit: int,
) -> PortfolioReviewContext | None:
    decision = state.get("_target_portfolio_decision")
    if not isinstance(decision, dict) or not decision.get("decision_id"):
        return None
    outcomes = {
        str(item.get("estimate_id")): item
        for item in (state.get("_contract_outcomes") or [])
        if item.get("estimate_id")
    }
    contract_rows = list(state.get("_quantity_grid") or [])
    contract_rows.sort(
        key=lambda row: (
            Decimal(str(row.get("probability_goal") or "-1")),
            Decimal(str(row.get("lower_probability_bound") or "-1")),
            -Decimal(str(row.get("capital_required") or "0")),
            str(row.get("contract_id") or ""),
            -int(row.get("quantity") or 0),
        ),
        reverse=True,
    )
    contract_summaries: list[ContractCandidateReviewSummary] = []
    evidence_ids: set[UUID] = set()
    for rank, row in enumerate(contract_rows[: max(1, int(limit))], start=1):
        outcome = outcomes.get(str(row.get("outcome_estimate_id")), {})
        row_evidence = tuple(
            UUID(str(value)) for value in outcome.get("evidence_ids") or ()
        )
        evidence_ids.update(row_evidence)
        contract_summaries.append(
            ContractCandidateReviewSummary(
                rank=rank,
                evaluation_id=UUID(str(row["evaluation_id"])),
                strategy_id=UUID(str(row["strategy_id"])),
                contract_id=str(row["contract_id"]),
                quantity=int(row["quantity"]),
                capital_required=Decimal(str(row["capital_required"])),
                probability_goal=(
                    Decimal(str(row["probability_goal"]))
                    if row.get("probability_goal") is not None
                    else None
                ),
                lower_probability_bound=(
                    Decimal(str(row["lower_probability_bound"]))
                    if row.get("lower_probability_bound") is not None
                    else None
                ),
                estimate_type=str(row.get("estimate_type") or "unknown"),
                relative_spread=Decimal(str(row.get("relative_spread") or "1")),
                liquidity_score=float(row.get("liquidity_score") or 0),
                uncertainty_reasons=tuple(
                    str(value)
                    for value in outcome.get("uncertainty_reasons") or ()
                ),
                evidence_ids=row_evidence,
            )
        )

    portfolio_rows = list(state.get("_portfolio_grid") or [])
    portfolio_rows.sort(
        key=lambda row: (
            Decimal(str(row.get("probability_goal") or "-1")),
            Decimal(str(row.get("lower_probability_bound") or "-1")),
            -Decimal(str(row.get("concentration_penalty") or "1")),
            str(row.get("portfolio_id") or ""),
        ),
        reverse=True,
    )
    portfolio_summaries = tuple(
        PortfolioCandidateReviewSummary(
            rank=rank,
            portfolio_id=UUID(str(row["portfolio_id"])),
            component_contract_ids=tuple(row.get("component_contract_ids") or ()),
            component_quantities=tuple(
                int(value) for value in row.get("component_quantities") or ()
            ),
            capital_deployed=Decimal(str(row.get("capital_deployed") or "0")),
            probability_goal=(
                Decimal(str(row["probability_goal"]))
                if row.get("probability_goal") is not None
                else None
            ),
            lower_probability_bound=(
                Decimal(str(row["lower_probability_bound"]))
                if row.get("lower_probability_bound") is not None
                else None
            ),
            concentration_penalty=Decimal(
                str(row.get("concentration_penalty") or "0")
            ),
            liquidity_penalty=Decimal(str(row.get("liquidity_penalty") or "0")),
            shared_scenario_grid_hash=row.get("shared_scenario_grid_hash"),
            failure_modes=tuple(row.get("reason_codes") or ()),
            selected=bool(row.get("selected")),
        )
        for rank, row in enumerate(
            portfolio_rows[: max(1, int(limit))], start=1
        )
    )
    selected_id = str(decision.get("selected_portfolio_id") or "")
    selected = next(
        (
            portfolio
            for portfolio in portfolio_summaries
            if str(portfolio.portfolio_id) == selected_id
        ),
        None,
    )
    objective = dict(state.get("_objective_context") or {})
    shared_grid = dict(state.get("_shared_underlying_scenario_grid") or {})
    return PortfolioReviewContext(
        target_portfolio_decision_id=UUID(str(decision["decision_id"])),
        snapshot_id=UUID(str(decision["snapshot_id"])),
        evaluated_objective_version=int(
            decision.get("evaluated_objective_version")
            or decision.get("objective_version")
            or 0
        ),
        ranked_contract_candidates=tuple(contract_summaries),
        ranked_portfolio_candidates=portfolio_summaries,
        selected_candidate=selected,
        wait_candidate={
            "action": "wait",
            "probability_goal": decision.get("wait_probability_goal"),
            "reason_codes": ["explicit_wait_candidate"],
        },
        remaining_goal_gap_usd=Decimal(
            str(objective.get("required_profit_remaining_usd") or "0")
        ),
        time_remaining_seconds=int(decision.get("time_remaining_seconds") or 0),
        deadline_exchange_time=objective.get("deadline_exchange_time"),
        shared_scenario_summary={
            "grid_hash": shared_grid.get("grid_hash"),
            "horizon_seconds": shared_grid.get("horizon_seconds"),
            "generation_method": shared_grid.get("generation_method"),
            "scenario_count": len(shared_grid.get("scenarios") or ()),
        },
        failure_modes=tuple(
            sorted(
                {
                    reason
                    for portfolio in portfolio_summaries
                    for reason in portfolio.failure_modes
                }
            )
        ),
        evidence_ids=tuple(sorted(evidence_ids, key=str)),
    )


def portfolio_review_from_debate(
    review: Any,
    context: PortfolioReviewContext,
) -> PortfolioDebateReview:
    verdict = str(review.verdict)
    if verdict in {"oppose", "request_revision", "execution_concern"}:
        recommendation: Literal["preserve", "wait", "reoptimize"] = "reoptimize"
    elif verdict in {"request_more_evidence", "insufficient_information"}:
        recommendation = "wait"
    else:
        recommendation = "preserve"
    return PortfolioDebateReview(
        review_id=review.review_id,
        reviewer_role=str(review.reviewer_role),
        target_portfolio_decision_id=context.target_portfolio_decision_id,
        selected_portfolio_id=(
            context.selected_candidate.portfolio_id
            if context.selected_candidate is not None
            else None
        ),
        verdict=verdict,
        confidence=float(review.confidence),
        claims=tuple(review.claims),
        identified_failure_modes=tuple(review.identified_failure_modes),
        required_revisions=tuple(review.required_revisions),
        finalizer_recommendation=recommendation,
    )
