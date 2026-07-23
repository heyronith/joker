"""OpenAI-backed agent council using structured LLM outputs."""

from __future__ import annotations

import json
from datetime import date

from joker.agents.llm_client import LLMClient, LLMClientError
from joker.agents.prompts import (
    CRITIC_ROLE,
    MARKET_REGIME_ROLE,
    OPTIONS_LIQUIDITY_ROLE,
    PRICE_ACTION_ROLE,
    RISK_NARRATOR_ROLE,
    SYNTHESIZER_ROLE,
    agent_system_prompt,
)
from joker.agents.security import validate_agent_output_type, validate_output_does_not_loosen_risk
from joker.compliance.openai_audit import audit_and_sanitize_openai_context
from joker.config.settings import AgentSettings
from joker.schemas.domain import (
    AgentCouncilDecision,
    AgentOpinion,
    DayMemoryBundle,
    Playbook,
    TechnicalFeatures,
)


class AgentError(Exception):
    pass


class OpenAIAgentCouncil:
    """Agent council backed by OpenAI structured outputs."""

    mock_mode = False

    def __init__(self, llm_client: LLMClient, settings: AgentSettings | None = None) -> None:
        self.llm = llm_client
        self.settings = settings or AgentSettings()

    def _complete_opinion(
        self,
        agent_name: str,
        role: str,
        context: dict,
    ) -> AgentOpinion:
        safe_context, _audit = audit_and_sanitize_openai_context(context, prompt_type=agent_name)
        prompt = f"Context JSON:\n{json.dumps(safe_context, default=str)}\n\nReturn AgentOpinion."
        try:
            result = self.llm.complete_structured(
                prompt,
                AgentOpinion,
                metadata={"agent": agent_name},
                system_prompt=agent_system_prompt(agent_name, role),
                timeout_seconds=float(self.settings.council_timeout_seconds),
            )
        except LLMClientError as exc:
            raise AgentError(f"{agent_name} failed: {exc}") from exc
        opinion = result.result
        if opinion.agent_name != agent_name:
            opinion = opinion.model_copy(update={"agent_name": agent_name})
        validate_agent_output_type(opinion)
        validate_output_does_not_loosen_risk(opinion)
        return opinion

    def _complete_playbook(
        self,
        trading_day: date,
        opinions: list[AgentOpinion],
        *,
        max_trades: int = 1,
        memory: DayMemoryBundle | None = None,
    ) -> Playbook:
        max_enabled = max(2, max_trades)
        context = {
            "trading_day": trading_day.isoformat(),
            "opinions": [o.model_dump(mode="json") for o in opinions],
            "max_trades_per_day": max_trades,
            "max_enabled_setups": max_enabled,
        }
        if memory is not None:
            from joker.memory import memory_prompt_dict

            context["day_memory"] = memory_prompt_dict(memory)
        safe_context, _audit = audit_and_sanitize_openai_context(
            context,
            prompt_type="SynthesizerAgent",
        )
        prompt = f"Context JSON:\n{json.dumps(safe_context, default=str)}\n\nReturn Playbook."
        try:
            result = self.llm.complete_structured(
                prompt,
                Playbook,
                metadata={"agent": "SynthesizerAgent"},
                system_prompt=agent_system_prompt("SynthesizerAgent", SYNTHESIZER_ROLE),
                timeout_seconds=float(self.settings.council_timeout_seconds),
            )
        except LLMClientError as exc:
            raise AgentError(f"SynthesizerAgent failed: {exc}") from exc
        playbook = result.result
        validate_agent_output_type(playbook)
        validate_output_does_not_loosen_risk(playbook)
        if not playbook.setups:
            raise AgentError("SynthesizerAgent returned playbook with no setups")
        return playbook.model_copy(update={"trading_day": trading_day})

    def run_premarket(
        self,
        run_id: str,
        trading_day: date,
        features: TechnicalFeatures,
        max_loss: float,
        max_trades: int,
        spread_pct: float = 5.0,
        memory: DayMemoryBundle | None = None,
    ) -> tuple[AgentCouncilDecision, Playbook]:
        feature_ctx = features.model_dump(mode="json")
        mem_ctx = None
        if memory is not None:
            from joker.memory import memory_prompt_dict

            mem_ctx = memory_prompt_dict(memory)
        opinions = [
            self._complete_opinion(
                "MarketRegimeAgent",
                MARKET_REGIME_ROLE,
                {"features": feature_ctx, "day_memory": mem_ctx},
            ),
            self._complete_opinion(
                "PriceActionAgent",
                PRICE_ACTION_ROLE,
                {"features": feature_ctx, "day_memory": mem_ctx},
            ),
            self._complete_opinion(
                "OptionsLiquidityAgent",
                OPTIONS_LIQUIDITY_ROLE,
                {"spread_pct": spread_pct},
            ),
            self._complete_opinion(
                "RiskNarratorAgent",
                RISK_NARRATOR_ROLE,
                {
                    "max_loss_usd": max_loss,
                    "trades_remaining": max_trades,
                    "day_memory": mem_ctx,
                },
            ),
        ]
        critic_ctx = {
            "opinions": [o.model_dump(mode="json") for o in opinions],
            "day_memory": mem_ctx,
        }
        opinions.append(
            self._complete_opinion("CriticAgent", CRITIC_ROLE, critic_ctx)
        )
        playbook = self._complete_playbook(
            trading_day, opinions, max_trades=max_trades, memory=memory
        )
        decision = AgentCouncilDecision(
            run_id=run_id,
            timestamp=features.as_of,
            opinions=opinions,
            synthesis_summary=playbook.summary,
            playbook_id=playbook.playbook_id,
        )
        return decision, playbook
