"""Test fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from joker.data.synthetic_replay import write_synthetic_replay


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_replay_path(tmp_path: Path) -> Path:
    return write_synthetic_replay(tmp_path / "spy_0dte_synthetic_day.jsonl")


@pytest.fixture(autouse=True)
def _default_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests have minimal required env unless overridden."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-unit-tests-only")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("JOKER_CONFIG", "config/paper.yaml")
