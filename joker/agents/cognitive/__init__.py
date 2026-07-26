"""Task 2 cognitive agents — perception through execution."""

from joker.agents.cognitive.base import CognitiveAgent
from joker.agents.cognitive.debate import (
    DEBATE_ROLES,
    AlternativeExplanationAgent,
    ExecutionCriticAgent,
    FalsifierAgent,
    HistoricalCriticAgent,
    StrategyAdvocateAgent,
    debate_agent_for,
    debate_context_for_strategy,
    run_debate_panel,
)
from joker.agents.cognitive.decision import MetaDecisionAgent, validate_meta_decision
from joker.agents.cognitive.discovery import (
    DISCOVERY_ROLES,
    AnalogyRetrieverAgent,
    PatternMinerAgent,
    SequenceAnalystAgent,
    discovery_agent_for,
    run_discovery_swarm,
    select_pattern_hypotheses,
)
from joker.agents.cognitive.execution import (
    EntryTacticianAgent,
    ExecutionCommandCompiler,
    ExecutionProposalValidator,
    ExecutionValidationConfig,
    ProvenancedExecutionCommand,
    parse_contract_id,
)
from joker.agents.cognitive.order_management import OrderManagerAgent
from joker.agents.cognitive.perception import (
    PERCEPTION_ROLES,
    AnomalyAgent,
    MarketStructureAgent,
    OptionsMicrostructureAgent,
    TemporalContextAgent,
    VolatilityAgent,
    perception_agent_for,
    run_perception_swarm,
)
from joker.agents.cognitive.position import PositionDecisionAgent, PositionThesisAgent
from joker.agents.cognitive.runner import run_agent_with_optional_data_request
from joker.agents.cognitive.strategy import (
    NOVEL_STRATEGY_NAME_EXAMPLES,
    STRATEGY_INVENTOR_ROLES,
    BearishInventorAgent,
    BullishInventorAgent,
    NeutralAdvocateAgent,
    is_novel_strategy_name,
    run_strategy_inventors,
    strategy_agent_for,
)

__all__ = [
    "AlternativeExplanationAgent",
    "AnalogyRetrieverAgent",
    "AnomalyAgent",
    "BearishInventorAgent",
    "BullishInventorAgent",
    "CognitiveAgent",
    "DEBATE_ROLES",
    "DISCOVERY_ROLES",
    "EntryTacticianAgent",
    "ExecutionCommandCompiler",
    "ExecutionCriticAgent",
    "ExecutionProposalValidator",
    "ExecutionValidationConfig",
    "FalsifierAgent",
    "HistoricalCriticAgent",
    "MarketStructureAgent",
    "MetaDecisionAgent",
    "NOVEL_STRATEGY_NAME_EXAMPLES",
    "NeutralAdvocateAgent",
    "OptionsMicrostructureAgent",
    "OrderManagerAgent",
    "PERCEPTION_ROLES",
    "PatternMinerAgent",
    "PositionDecisionAgent",
    "PositionThesisAgent",
    "ProvenancedExecutionCommand",
    "SequenceAnalystAgent",
    "STRATEGY_INVENTOR_ROLES",
    "StrategyAdvocateAgent",
    "TemporalContextAgent",
    "VolatilityAgent",
    "debate_agent_for",
    "debate_context_for_strategy",
    "discovery_agent_for",
    "is_novel_strategy_name",
    "parse_contract_id",
    "perception_agent_for",
    "run_agent_with_optional_data_request",
    "run_debate_panel",
    "run_discovery_swarm",
    "run_perception_swarm",
    "run_strategy_inventors",
    "select_pattern_hypotheses",
    "strategy_agent_for",
    "validate_meta_decision",
]
