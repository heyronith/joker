"""OpenAI-backed replay flow tests (mocked client, no real API)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from joker.agents.council_analysis import analyze_council
from joker.config.settings import AppSettings
from joker.runtime.replay_runner import ReplayRunConfig, ReplayRunner
from joker.schemas.domain import AgentCouncilDecision, AgentOpinion
from tests.fixtures.openai_council import (
    build_invalid_playbook_mock_client,
    build_valid_openai_mock_client,
    build_weak_critic_mock_client,
)


@pytest.fixture
def settings(tmp_path: Path) -> AppSettings:
    return AppSettings.model_validate(
        {
            "mode": "PAPER",
            "db_path": str(tmp_path / "joker.db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
            "agents": {"mock_agents": False},
        }
    )


def test_openai_replay_with_mocked_client(settings: AppSettings, synthetic_replay_path: Path) -> None:
    client = build_valid_openai_mock_client(date(2026, 7, 1))
    result = ReplayRunner(settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            deterministic=True,
            mock_agents=False,
            llm_client=client,
        )
    )
    assert result.summary.events_processed > 0
    assert result.report_path is not None
    assert result.report_path.exists()
    assert result.playbook is not None
    assert result.playbook_validation is not None
    assert result.playbook_validation.approved is True
    content = result.report_path.read_text()
    assert "openai agents" in content.lower()
    assert "Synthetic replay" in content or "synthetic replay" in content.lower()


def test_mock_agents_replay_still_works(settings: AppSettings, synthetic_replay_path: Path) -> None:
    result = ReplayRunner(settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            deterministic=True,
            mock_agents=True,
        )
    )
    assert result.summary.events_processed > 0
    content = result.report_path.read_text() if result.report_path else ""
    assert "mock agents" in content.lower()


def test_invalid_playbook_fails_closed(settings: AppSettings, synthetic_replay_path: Path) -> None:
    client = build_invalid_playbook_mock_client(date(2026, 7, 1))
    result = ReplayRunner(settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            deterministic=True,
            mock_agents=False,
            llm_client=client,
        )
    )
    assert any("playbook_validation_failed" in f for f in result.failures)
    assert result.summary.trades_entered == 0
    assert result.report_path is not None
    content = result.report_path.read_text()
    assert "Playbook Validation" in content


def test_weak_critic_flagged(settings: AppSettings, synthetic_replay_path: Path) -> None:
    client = build_weak_critic_mock_client(date(2026, 7, 1))
    result = ReplayRunner(settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            deterministic=True,
            mock_agents=False,
            llm_client=client,
        )
    )
    assert result.council_analysis is not None
    assert result.council_analysis.council_blocked is True
    content = result.report_path.read_text() if result.report_path else ""
    assert "Critic" in content or "council_blocked" in str(result.failures)


def test_council_disagreement_in_report() -> None:
    decision = AgentCouncilDecision(
        run_id="r1",
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        opinions=[
            AgentOpinion(agent_name="MarketRegimeAgent", summary="up", confidence=0.8, regime=__import__("joker.schemas.domain", fromlist=["MarketRegime"]).MarketRegime.TREND_UP),
            AgentOpinion(agent_name="PriceActionAgent", summary="down", confidence=0.3, regime=__import__("joker.schemas.domain", fromlist=["MarketRegime"]).MarketRegime.TREND_DOWN),
            AgentOpinion(agent_name="CriticAgent", summary="Low confidence from PriceActionAgent", confidence=0.75),
        ],
        synthesis_summary="Mixed signals",
    )
    analysis = analyze_council(decision)
    assert analysis.conflicting_regimes is True
    assert "PriceActionAgent" in analysis.low_confidence_agents


def test_replay_reproducible_openai_mock(settings: AppSettings, synthetic_replay_path: Path, tmp_path: Path) -> None:
    client = build_valid_openai_mock_client(date(2026, 7, 1))
    s1 = AppSettings.model_validate({**settings.model_dump(), "db_path": str(tmp_path / "a.db"), "event_log_dir": str(tmp_path / "la"), "reports_dir": str(tmp_path / "ra")})
    s2 = AppSettings.model_validate({**settings.model_dump(), "db_path": str(tmp_path / "b.db"), "event_log_dir": str(tmp_path / "lb"), "reports_dir": str(tmp_path / "rb")})
    r1 = ReplayRunner(s1).run(ReplayRunConfig(replay_path=synthetic_replay_path, deterministic=True, mock_agents=False, llm_client=client, skip_premarket=True))
    r2 = ReplayRunner(s2).run(ReplayRunConfig(replay_path=synthetic_replay_path, deterministic=True, mock_agents=False, llm_client=client, skip_premarket=True))
    assert r1.summary.signals_detected == r2.summary.signals_detected
    assert r1.summary.trades_entered == r2.summary.trades_entered
