"""Tracked all-task paper profile safety invariants."""

from __future__ import annotations

from pathlib import Path

from joker.app.safety import SafetyMode
from joker.config.loader import load_app_settings
from joker.models.schemas import ModelsConfig


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "config" / "paper-all-tasks.yaml"


def test_paper_all_tasks_profile_invariants() -> None:
    assert PROFILE.is_file()
    app, _env = load_app_settings(PROFILE)

    assert app.mode is SafetyMode.PAPER
    assert app.live_trading_enabled is False
    assert app.data.default_provider == "webull"
    assert app.cognitive_graph.enabled is True
    assert app.agents.mock_agents is False
    assert app.cognitive_graph.legacy_fallback_enabled is False
    assert app.evolution.enabled is True
    assert isinstance(app.models, ModelsConfig)
    assert app.models.ollama.enabled is True
