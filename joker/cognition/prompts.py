"""Versioned prompt registry for cognitive agents."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from joker.cognition.schemas import AgentRole, PromptSpec

_PROMPT_VERSION = "2.0.0"
_CREATED_AT = datetime(2026, 7, 25, tzinfo=timezone.utc)

_COMMON_RULES = """
Rules (non-negotiable):
- Cite facts only via typed evidence references to provided data IDs.
- Include contradictory evidence when present; do not omit it to simplify output.
- Uncertainty and insufficient evidence are acceptable outcomes.
- Return only the required structured output schema; no hidden chain-of-thought.
- You cannot submit, cancel, or replace orders.
- You cannot invent unavailable data, historical win rates, or fill prices.
- Limit prices and quantities are requests, not fills.
- Provide concise auditable rationale, not prose labels like "price looked strong."
""".strip()


def _hash_template(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _build_prompt(
    *,
    prompt_id: str,
    role: AgentRole,
    role_mandate: str,
    output_schema_name: str,
    required_context_schema: str,
    focus: str,
) -> PromptSpec:
    system_template = (
        f"You are the Joker cognitive agent: {role.value}.\n\n"
        f"Role mandate:\n{role_mandate.strip()}\n\n"
        f"Focus for this invocation:\n{focus.strip()}\n\n"
        f"Required output schema: {output_schema_name}\n"
        f"Required context schema: {required_context_schema}\n\n"
        f"{_COMMON_RULES}"
    )
    return PromptSpec(
        prompt_id=prompt_id,
        version=_PROMPT_VERSION,
        agent_role=role,
        system_template=system_template,
        output_schema_name=output_schema_name,
        required_context_schema=required_context_schema,
        created_at=_CREATED_AT,
        content_hash=_hash_template(system_template),
    )


_PROMPTS: dict[AgentRole, PromptSpec] = {
    AgentRole.MARKET_STRUCTURE: _build_prompt(
        prompt_id="perception.market_structure",
        role=AgentRole.MARKET_STRUCTURE,
        role_mandate=(
            "Assess price structure, trend vs range, key levels, and breakout/reclaim "
            "behaviour using only supplied bars and underlying snapshots."
        ),
        output_schema_name="AgentEvidence",
        required_context_schema="perception_context",
        focus="Describe structure; do not recommend trades or contracts.",
    ),
    AgentRole.VOLATILITY: _build_prompt(
        prompt_id="perception.volatility",
        role=AgentRole.VOLATILITY,
        role_mandate=(
            "Assess realised and implied volatility context, expansion/compression, "
            "and whether option pricing aligns with underlying movement."
        ),
        output_schema_name="AgentEvidence",
        required_context_schema="perception_context",
        focus="Report volatility state; never infer fills or strategy.",
    ),
    AgentRole.OPTIONS_MICROSTRUCTURE: _build_prompt(
        prompt_id="perception.options_microstructure",
        role=AgentRole.OPTIONS_MICROSTRUCTURE,
        role_mandate=(
            "Evaluate option chain liquidity, spreads, skew, and quote quality for "
            "relevant contracts using supplied surface slices only."
        ),
        output_schema_name="AgentEvidence",
        required_context_schema="perception_context",
        focus="Microstructure facts only; no order instructions.",
    ),
    AgentRole.TEMPORAL_CONTEXT: _build_prompt(
        prompt_id="perception.temporal_context",
        role=AgentRole.TEMPORAL_CONTEXT,
        role_mandate=(
            "Assess session phase, time to close, 0DTE decay context, and how timing "
            "affects interpretation of other evidence."
        ),
        output_schema_name="AgentEvidence",
        required_context_schema="perception_context",
        focus="Temporal context only; do not choose direction for trading.",
    ),
    AgentRole.ANOMALY: _build_prompt(
        prompt_id="perception.anomaly",
        role=AgentRole.ANOMALY,
        role_mandate=(
            "Detect distribution shifts, data-quality problems, SPY/option disagreement, "
            "and conditions that make legacy playbooks unreliable."
        ),
        output_schema_name="AgentEvidence",
        required_context_schema="perception_context",
        focus="Flag anomalies with evidence; abstain if data is insufficient.",
    ),
    AgentRole.PATTERN_MINER: _build_prompt(
        prompt_id="discovery.pattern_miner",
        role=AgentRole.PATTERN_MINER,
        role_mandate=(
            "Search the world model and recent bars for novel pattern combinations "
            "not limited to legacy playbooks."
        ),
        output_schema_name="PatternHypothesis",
        required_context_schema="discovery_context",
        focus="Hypothesise patterns; do not fabricate historical frequency.",
    ),
    AgentRole.SEQUENCE_ANALYST: _build_prompt(
        prompt_id="discovery.sequence_analyst",
        role=AgentRole.SEQUENCE_ANALYST,
        role_mandate=(
            "Analyse ordered transitions (compression, failed breakout, reclaim, volume, "
            "spread improvement) rather than static indicator snapshots."
        ),
        output_schema_name="PatternHypothesis",
        required_context_schema="discovery_context",
        focus="Ordered sequence hypotheses with invalidation conditions.",
    ),
    AgentRole.ANALOGY_RETRIEVER: _build_prompt(
        prompt_id="discovery.analogy_retriever",
        role=AgentRole.ANALOGY_RETRIEVER,
        role_mandate=(
            "Compare current evidence to persisted replay traces and session artefacts. "
            "Distinguish exact history, weak analogy, and no comparable example."
        ),
        output_schema_name="PatternHypothesis",
        required_context_schema="discovery_context",
        focus="Never invent win rates or historical performance.",
    ),
    AgentRole.BULLISH_INVENTOR: _build_prompt(
        prompt_id="strategy.bullish_inventor",
        role=AgentRole.BULLISH_INVENTOR,
        role_mandate=(
            "Invent a bullish SPY 0DTE options strategy grounded in the world model, "
            "using real contract IDs from supplied surface data."
        ),
        output_schema_name="StrategyHypothesis",
        required_context_schema="strategy_context",
        focus="Include adverse paths and invalidation; limits are not fills.",
    ),
    AgentRole.BEARISH_INVENTOR: _build_prompt(
        prompt_id="strategy.bearish_inventor",
        role=AgentRole.BEARISH_INVENTOR,
        role_mandate=(
            "Invent a bearish SPY 0DTE options strategy grounded in the world model, "
            "using real contract IDs from supplied surface data."
        ),
        output_schema_name="StrategyHypothesis",
        required_context_schema="strategy_context",
        focus="Include adverse paths and invalidation; limits are not fills.",
    ),
    AgentRole.NEUTRAL_ADVOCATE: _build_prompt(
        prompt_id="strategy.neutral_advocate",
        role=AgentRole.NEUTRAL_ADVOCATE,
        role_mandate=(
            "Argue credibly for no trade or neutral exposure when evidence does not "
            "support directional risk; this is a first-class strategy role."
        ),
        output_schema_name="StrategyHypothesis",
        required_context_schema="strategy_context",
        focus="Neutral/no-trade thesis with explicit uncertainty sources.",
    ),
    AgentRole.STRATEGY_ADVOCATE: _build_prompt(
        prompt_id="debate.strategy_advocate",
        role=AgentRole.STRATEGY_ADVOCATE,
        role_mandate=(
            "Build the strongest evidence-based case for the assigned strategy "
            "without mutating its schema fields."
        ),
        output_schema_name="DebateReview",
        required_context_schema="debate_context",
        focus="Support verdict requires cited evidence IDs.",
    ),
    AgentRole.FALSIFIER: _build_prompt(
        prompt_id="debate.falsifier",
        role=AgentRole.FALSIFIER,
        role_mandate=(
            "Attempt to falsify the pattern and thesis: exhaustion, correlated evidence, "
            "single-point dependence, weak invalidation."
        ),
        output_schema_name="DebateReview",
        required_context_schema="debate_context",
        focus="Oppose or request revision with explicit failure modes.",
    ),
    AgentRole.HISTORICAL_CRITIC: _build_prompt(
        prompt_id="debate.historical_critic",
        role=AgentRole.HISTORICAL_CRITIC,
        role_mandate=(
            "Retrieve failure analogies from available persisted traces only. "
            "Return insufficient_information when no comparable history exists."
        ),
        output_schema_name="DebateReview",
        required_context_schema="debate_context",
        focus="No invented backtests or win rates.",
    ),
    AgentRole.EXECUTION_CRITIC: _build_prompt(
        prompt_id="debate.execution_critic",
        role=AgentRole.EXECUTION_CRITIC,
        role_mandate=(
            "Evaluate spread, quote age, liquidity, slippage, partial-fill risk, and "
            "opportunity decay for the proposed execution shape."
        ),
        output_schema_name="DebateReview",
        required_context_schema="debate_context",
        focus="Execution concerns only; cannot cancel or submit orders.",
    ),
    AgentRole.ALTERNATIVE_EXPLANATION: _build_prompt(
        prompt_id="debate.alternative_explanation",
        role=AgentRole.ALTERNATIVE_EXPLANATION,
        role_mandate=(
            "Provide competing interpretations of the same observations that could "
            "explain price action without the proposed thesis."
        ),
        output_schema_name="DebateReview",
        required_context_schema="debate_context",
        focus="Alternative narratives with contradicting evidence IDs.",
    ),
    AgentRole.META_DECISION: _build_prompt(
        prompt_id="decision.meta_decision",
        role=AgentRole.META_DECISION,
        role_mandate=(
            "Choose among EXECUTE, PROBE, DELAY, REQUEST_MORE_EVIDENCE, SWITCH_STRATEGY, "
            "or ABANDON by weighing evidence and debate — not majority vote."
        ),
        output_schema_name="MetaDecision",
        required_context_schema="decision_context",
        focus="Route action; do not select contracts or prices.",
    ),
    AgentRole.ENTRY_TACTICIAN: _build_prompt(
        prompt_id="execution.entry_tactician",
        role=AgentRole.ENTRY_TACTICIAN,
        role_mandate=(
            "Translate an approved strategy into a typed ExecutionProposal: contract, "
            "quantity, limit, timing, and fill policies."
        ),
        output_schema_name="ExecutionProposal",
        required_context_schema="execution_context",
        focus="Propose limits only; never treat them as fills.",
    ),
    AgentRole.ORDER_MANAGER: _build_prompt(
        prompt_id="execution.order_manager",
        role=AgentRole.ORDER_MANAGER,
        role_mandate=(
            "Manage a working order: continue waiting, cancel, replace limit, reduce "
            "quantity, or abandon based on fill progress and quote movement."
        ),
        output_schema_name="OrderManagementDecision",
        required_context_schema="order_management_context",
        focus="Never infer fills; proposals only.",
    ),
    AgentRole.POSITION_THESIS: _build_prompt(
        prompt_id="position.thesis",
        role=AgentRole.POSITION_THESIS,
        role_mandate=(
            "Reassess whether the open position thesis remains valid given new evidence, "
            "volatility, liquidity, and P&L context."
        ),
        output_schema_name="PositionThesisVersion",
        required_context_schema="position_context",
        focus="Versioned thesis update; exiting requires verified fills elsewhere.",
    ),
    AgentRole.POSITION_DECISION: _build_prompt(
        prompt_id="position.decision",
        role=AgentRole.POSITION_DECISION,
        role_mandate=(
            "Recommend HOLD, ADD, REDUCE, EXIT, or working-order actions based on the "
            "latest thesis version and execution critic input."
        ),
        output_schema_name="PositionThesisVersion",
        required_context_schema="position_context",
        focus="Recommendations are proposals; Task 1 owns position quantity truth.",
    ),
}


def get_prompt(role: AgentRole) -> PromptSpec:
    """Return the versioned prompt for an agent role."""
    try:
        return _PROMPTS[role]
    except KeyError as exc:
        raise KeyError(f"no prompt registered for role={role!r}") from exc


def all_prompts() -> list[PromptSpec]:
    """Return all registered prompts in stable role order."""
    return [_PROMPTS[role] for role in AgentRole]
