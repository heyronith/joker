"""Deterministic mock agents for offline tests and mock_agents mode."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from joker.schemas.domain import (
    AgentCouncilDecision,
    AgentOpinion,
    MarketRegime,
    Playbook,
    PlaybookSetup,
    TechnicalFeatures,
)


class AgentCouncilProtocol(Protocol):
    mock_mode: bool

    def run_premarket(
        self,
        run_id: str,
        trading_day: date,
        features: TechnicalFeatures,
        max_loss: float,
        max_trades: int,
        spread_pct: float = 5.0,
        memory: Any | None = None,
    ) -> tuple[AgentCouncilDecision, Playbook]: ...


class MockMarketRegimeAgent:
    name = "MarketRegimeAgent"

    def run(self, features: TechnicalFeatures) -> AgentOpinion:
        regime = MarketRegime.UNKNOWN
        if features.trend_label == "trend_up":
            regime = MarketRegime.TREND_UP
        elif features.trend_label == "trend_down":
            regime = MarketRegime.TREND_DOWN
        elif features.trend_label == "chop":
            regime = MarketRegime.CHOP
        return AgentOpinion(
            agent_name=self.name,
            summary=f"Regime assessed as {regime.value}",
            confidence=0.7,
            regime=regime,
        )


class MockPriceActionAgent:
    name = "PriceActionAgent"

    def run(self, features: TechnicalFeatures) -> AgentOpinion:
        dist = features.distance_from_vwap_pct
        summary = "Price near VWAP" if dist is not None and abs(dist) < 0.1 else "Price extended"
        return AgentOpinion(
            agent_name=self.name,
            summary=summary,
            confidence=0.65,
        )


class MockOptionsLiquidityAgent:
    name = "OptionsLiquidityAgent"

    def run(self, spread_pct: float) -> AgentOpinion:
        summary = "Liquidity acceptable" if spread_pct <= 10 else "Liquidity poor"
        return AgentOpinion(
            agent_name=self.name,
            summary=summary,
            confidence=0.6,
            metadata={"notes": f"spread_pct={spread_pct}"},
        )


class MockRiskNarratorAgent:
    name = "RiskNarratorAgent"

    def run(self, max_loss: float, trades_remaining: int) -> AgentOpinion:
        return AgentOpinion(
            agent_name=self.name,
            summary=f"Daily loss budget ${max_loss:.0f}, {trades_remaining} trades remaining",
            confidence=0.9,
        )


class MockCriticAgent:
    name = "CriticAgent"

    def run(self, opinions: list[AgentOpinion]) -> AgentOpinion:
        low_conf = [o.agent_name for o in opinions if o.confidence < 0.5]
        summary = "Council consensus acceptable"
        if low_conf:
            summary = f"Low confidence from: {', '.join(low_conf)}"
        return AgentOpinion(
            agent_name=self.name,
            summary=summary,
            confidence=0.75,
        )


class MockSynthesizerAgent:
    name = "SynthesizerAgent"

    def run(
        self,
        run_id: str,
        trading_day: date,
        opinions: list[AgentOpinion],
    ) -> Playbook:
        regime_opinion = next((o for o in opinions if o.regime), None)
        direction = "long_call"
        if regime_opinion and regime_opinion.regime == MarketRegime.TREND_DOWN:
            direction = "long_put"
        setup = PlaybookSetup(
            name="Primary 0DTE setup",
            direction=direction,  # type: ignore[arg-type]
            entry_conditions=["VWAP reclaim", "Momentum confirmation"],
            stop_rule="50% premium stop",
            take_profit_rule="100% premium target",
            require_trend="trend_up" if direction == "long_call" else "trend_down",
            vwap_side="above" if direction == "long_call" else "below",
            min_vwap_distance_pct=0.02,
            min_momentum_pct=0.1 if direction == "long_call" else -0.1,
            stop_pct=0.5,
            take_profit_pct=1.0,
        )
        return Playbook(
            trading_day=trading_day,
            title=f"Daily SPY 0DTE Plan — {trading_day.isoformat()}",
            summary="; ".join(o.summary for o in opinions),
            setups=[setup],
        )


class MockAgentCouncil:
    """Deterministic local agent council — no OpenAI calls."""

    mock_mode = True

    def __init__(self) -> None:
        self.market_regime = MockMarketRegimeAgent()
        self.price_action = MockPriceActionAgent()
        self.options_liquidity = MockOptionsLiquidityAgent()
        self.risk_narrator = MockRiskNarratorAgent()
        self.critic = MockCriticAgent()
        self.synthesizer = MockSynthesizerAgent()

    def run_premarket(
        self,
        run_id: str,
        trading_day: date,
        features: TechnicalFeatures,
        max_loss: float,
        max_trades: int,
        spread_pct: float = 5.0,
        memory: Any | None = None,
    ) -> tuple[AgentCouncilDecision, Playbook]:
        _ = memory
        opinions = [
            self.market_regime.run(features),
            self.price_action.run(features),
            self.options_liquidity.run(spread_pct),
            self.risk_narrator.run(max_loss, max_trades),
        ]
        opinions.append(self.critic.run(opinions))
        playbook = self.synthesizer.run(run_id, trading_day, opinions)
        decision = AgentCouncilDecision(
            run_id=run_id,
            timestamp=features.as_of,
            opinions=opinions,
            synthesis_summary=playbook.summary,
            playbook_id=playbook.playbook_id,
        )
        return decision, playbook
