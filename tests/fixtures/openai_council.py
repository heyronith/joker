"""Mock OpenAI council responses for offline replay tests."""

from __future__ import annotations

from datetime import date

from joker.agents.llm_client import MockLLMClient
from joker.schemas.domain import AgentOpinion, MarketRegime, Playbook, PlaybookSetup


def build_valid_openai_mock_client(trading_day: date | None = None) -> MockLLMClient:
    day = trading_day or date(2026, 7, 1)
    client = MockLLMClient()

    client.set_response(
        AgentOpinion,
        AgentOpinion(
            agent_name="MarketRegimeAgent",
            summary="Trend up with momentum",
            confidence=0.75,
            regime=MarketRegime.TREND_UP,
        ),
        agent="MarketRegimeAgent",
    )
    client.set_response(
        AgentOpinion,
        AgentOpinion(
            agent_name="PriceActionAgent",
            summary="Price above VWAP",
            confidence=0.7,
        ),
        agent="PriceActionAgent",
    )
    client.set_response(
        AgentOpinion,
        AgentOpinion(
            agent_name="OptionsLiquidityAgent",
            summary="Spreads acceptable",
            confidence=0.65,
        ),
        agent="OptionsLiquidityAgent",
    )
    client.set_response(
        AgentOpinion,
        AgentOpinion(
            agent_name="RiskNarratorAgent",
            summary="Budget intact",
            confidence=0.9,
        ),
        agent="RiskNarratorAgent",
    )
    client.set_response(
        AgentOpinion,
        AgentOpinion(
            agent_name="CriticAgent",
            summary="Council consensus acceptable",
            confidence=0.8,
        ),
        agent="CriticAgent",
    )
    client.set_response(
        Playbook,
        Playbook(
            trading_day=day,
            title=f"OpenAI SPY 0DTE Plan — {day.isoformat()}",
            summary="Bullish bias with defined risk",
            setups=[
                PlaybookSetup(
                    name="VWAP reclaim call",
                    direction="long_call",
                    entry_conditions=["Price reclaims VWAP", "Momentum confirmation"],
                    stop_rule="50% premium stop",
                    take_profit_rule="100% premium target",
                ),
            ],
        ),
        agent="SynthesizerAgent",
    )
    return client


def build_weak_critic_mock_client(trading_day: date | None = None) -> MockLLMClient:
    client = build_valid_openai_mock_client(trading_day)
    client.set_response(
        AgentOpinion,
        AgentOpinion(
            agent_name="CriticAgent",
            summary="Low confidence from multiple agents — weak plan",
            confidence=0.3,
        ),
        agent="CriticAgent",
    )
    return client


def build_invalid_playbook_mock_client(trading_day: date | None = None) -> MockLLMClient:
    day = trading_day or date(2026, 7, 1)
    client = build_valid_openai_mock_client(day)
    client.set_response(
        Playbook,
        Playbook(
            trading_day=day,
            title="Bad plan",
            summary="Guaranteed profit on every trade",
            setups=[
                PlaybookSetup(
                    name="Bad",
                    direction="long_call",
                    entry_conditions=[],
                    stop_rule="",
                    take_profit_rule="",
                )
            ],
        ),
        agent="SynthesizerAgent",
    )
    return client
