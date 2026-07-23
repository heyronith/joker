"""Replay failure handling tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from joker.config.settings import AppSettings
from joker.runtime.replay_errors import EmptyReplayFailure, ReplayLoadFailure
from joker.runtime.replay_runner import ReplayRunConfig, ReplayRunner


def test_empty_replay_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("# empty\n")
    settings = AppSettings.model_validate(
        {"db_path": str(tmp_path / "db"), "event_log_dir": str(tmp_path / "logs"), "reports_dir": str(tmp_path / "reports")}
    )
    with pytest.raises(ReplayLoadFailure, match="no events"):
        ReplayRunner(settings).run(ReplayRunConfig(replay_path=path, skip_premarket=True))


def test_corrupt_replay_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.jsonl"
    path.write_text("{not valid json\n")
    settings = AppSettings.model_validate(
        {"db_path": str(tmp_path / "db"), "event_log_dir": str(tmp_path / "logs"), "reports_dir": str(tmp_path / "reports")}
    )
    with pytest.raises(ReplayLoadFailure):
        ReplayRunner(settings).run(ReplayRunConfig(replay_path=path))


def test_invalid_playbook_blocks_trading(settings_like, synthetic_replay_path, tmp_path) -> None:
    from tests.fixtures.openai_council import build_invalid_playbook_mock_client

    settings = AppSettings.model_validate(
        {
            "db_path": str(tmp_path / "db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
            "agents": {"mock_agents": False},
        }
    )
    client = build_invalid_playbook_mock_client()
    result = ReplayRunner(settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            mock_agents=False,
            llm_client=client,
            deterministic=True,
        )
    )
    assert result.summary.trades_entered == 0
    assert result.report_path is not None
    logs = list((tmp_path / "logs").glob("*.jsonl"))
    assert logs
    raw = logs[0].read_text()
    assert "replay.failure" in raw or "playbook.validation" in raw


def test_openai_timeout_logged(settings_like, synthetic_replay_path, tmp_path) -> None:
    from joker.agents.llm_client import MockLLMClient

    client = MockLLMClient(delay_seconds=999.0)
    settings = AppSettings.model_validate(
        {
            "db_path": str(tmp_path / "db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
            "agents": {"mock_agents": False},
        }
    )
    result = ReplayRunner(settings).run(
        ReplayRunConfig(
            replay_path=synthetic_replay_path,
            mock_agents=False,
            llm_client=client,
            deterministic=True,
        )
    )
    assert any("openai_council_failed" in f for f in result.failures)
    assert result.report_path is not None


@pytest.fixture
def settings_like(tmp_path: Path) -> AppSettings:
    return AppSettings.model_validate(
        {"db_path": str(tmp_path / "db"), "event_log_dir": str(tmp_path / "logs"), "reports_dir": str(tmp_path / "reports")}
    )
