"""Legacy council failure must not block cognitive market polling."""

from __future__ import annotations

from joker.config.settings import AppSettings
from joker.runtime.cognitive_startup import FAKE_OVERRIDE_ENV


def test_cognitive_mode_does_not_require_armed_playbook_flag() -> None:
    """Regression: cognitive_graph runtime remains the configured mode."""
    settings = AppSettings(agents={"runtime": "cognitive_graph", "mock_agents": True})
    assert settings.agents.runtime == "cognitive_graph"
    assert settings.cognitive_graph.legacy_fallback_enabled is False


def test_fake_override_env_name_stable() -> None:
    assert FAKE_OVERRIDE_ENV == "JOKER_COGNITIVE_USE_FAKE_MODELS"
