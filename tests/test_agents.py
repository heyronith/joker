"""Phase 7 agent council tests — updated for Phase 15."""

from __future__ import annotations

import json

import pytest

from joker.agents.council import AgentError, BaseAgent, create_agent_council
from joker.agents.communicator import CommunicatorAgent
from joker.agents.mock_agents import MockAgentCouncil
from joker.config.settings import AppSettings
from joker.features.engine import FeatureEngine
from joker.schemas.domain import AgentOpinion, Playbook
from tests.fixtures.domain import make_snapshot


def test_agents_run_on_mocked_input() -> None:
    council = MockAgentCouncil()
    engine = FeatureEngine(max_age_seconds=99999)
    features = engine.compute(make_snapshot(price=550.0))
    opinion = council.market_regime.run(features)
    assert opinion.agent_name == "MarketRegimeAgent"
    assert opinion.confidence >= 0.0


def test_malformed_model_output_rejected() -> None:
    class TestAgent(BaseAgent):
        name = "TestAgent"

    with pytest.raises(AgentError, match="invalid JSON"):
        TestAgent().parse_output("not json", AgentOpinion)


def test_missing_fields_rejected() -> None:
    class TestAgent(BaseAgent):
        name = "TestAgent"

    with pytest.raises(AgentError, match="validation failed"):
        TestAgent().parse_output(json.dumps({"agent_name": "x"}), AgentOpinion)


def test_synthesizer_creates_playbook_with_required_rules() -> None:
    council = MockAgentCouncil()
    from datetime import date

    opinions = [AgentOpinion(agent_name="a", summary="s", confidence=0.8)]
    pb = council.synthesizer.run("run-1", date.today(), opinions)
    assert isinstance(pb, Playbook)
    assert pb.setups[0].stop_rule
    assert pb.setups[0].take_profit_rule


def test_communicator_answers_system_question() -> None:
    agent = CommunicatorAgent()
    reply = agent.answer(
        "what is the system doing?",
        {
            "mode": "PAPER",
            "trade_state": "IDLE",
            "council_status": "idle",
            "market_price": None,
        },
    )
    assert "PAPER" in reply
    assert "unavailable" in reply.lower()


def test_agents_cannot_call_broker_directly() -> None:
    import inspect
    import joker.agents.openai_agents as mod

    source = inspect.getsource(mod)
    assert "submit_order" not in source
    assert "RiskGovernor" not in source


def test_council_premarket_produces_playbook_and_decision() -> None:
    council = MockAgentCouncil()
    from datetime import date

    features = FeatureEngine(max_age_seconds=99999).compute(make_snapshot())
    decision, playbook = council.run_premarket(
        "run-1", date.today(), features, max_loss=500, max_trades=3
    )
    assert decision.playbook_id == playbook.playbook_id
    assert len(decision.opinions) == 5


def test_factory_defaults_to_mock_agents() -> None:
    app = AppSettings.model_validate({})
    council = create_agent_council(app)
    assert council.mock_mode is True
