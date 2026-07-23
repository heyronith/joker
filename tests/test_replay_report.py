"""Replay postmarket report content tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from joker.config.settings import AppSettings
from joker.runtime.replay_runner import ReplayRunConfig, ReplayRunner
from joker.reporting.replay_report import SYNTHETIC_WARNING
from tests.fixtures.openai_council import build_valid_openai_mock_client


@pytest.fixture
def settings(tmp_path: Path) -> AppSettings:
    return AppSettings.model_validate(
        {
            "db_path": str(tmp_path / "joker.db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
            "agents": {"mock_agents": False},
        }
    )


def test_synthetic_warning_in_report(settings: AppSettings, synthetic_replay_path: Path) -> None:
    client = build_valid_openai_mock_client(date(2026, 7, 1))
    result = ReplayRunner(settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            deterministic=True,
            mock_agents=False,
            llm_client=client,
        )
    )
    content = result.report_path.read_text()
    assert SYNTHETIC_WARNING.split("**")[1].strip(":") in content or "Synthetic replay" in content


def test_agent_mode_in_report(settings: AppSettings, synthetic_replay_path: Path, tmp_path: Path) -> None:
    client = build_valid_openai_mock_client(date(2026, 7, 1))
    openai_settings = settings.model_copy(
        update={"reports_dir": str(tmp_path / "reports_openai")}
    )
    openai_result = ReplayRunner(openai_settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            deterministic=True,
            mock_agents=False,
            llm_client=client,
        )
    )
    mock_settings = settings.model_copy(
        update={
            "db_path": str(tmp_path / "mock.db"),
            "reports_dir": str(tmp_path / "reports_mock"),
        }
    )
    mock_result = ReplayRunner(mock_settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            deterministic=True,
            mock_agents=True,
        )
    )
    assert "openai agents" in openai_result.report_path.read_text().lower()
    assert "mock agents" in mock_result.report_path.read_text().lower()


def test_playbook_validation_in_report(settings: AppSettings, synthetic_replay_path: Path) -> None:
    client = build_valid_openai_mock_client(date(2026, 7, 1))
    result = ReplayRunner(settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            deterministic=True,
            mock_agents=False,
            llm_client=client,
        )
    )
    assert "Playbook Validation" in result.report_path.read_text()


def test_risk_rejection_and_exits_in_report(
    settings: AppSettings, synthetic_replay_path: Path
) -> None:
    result = ReplayRunner(settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            deterministic=True,
            mock_agents=True,
            skip_premarket=True,
        )
    )
    content = result.report_path.read_text()
    assert "Risk Governor Decisions" in content
    assert "Exit Decisions" in content or result.summary.trades_exited == 0
