"""ModelsConfig must be typed at AppSettings load time."""

from __future__ import annotations

from pathlib import Path

import pytest

from joker.config.loader import load_app_settings
from joker.config.settings import AppSettings
from joker.models.schemas import ModelProfileConfig, ModelsConfig, default_model_profiles


ROOT = Path(__file__).resolve().parents[2]


def test_paper_yaml_loads_models_as_models_config() -> None:
    app, _env = load_app_settings(ROOT / "config" / "paper.yaml")
    assert isinstance(app.models, ModelsConfig)
    assert app.models.ollama.enabled is True
    assert app.models.openai.enabled is False
    assert app.models.profiles
    for name, profile in app.models.profiles.items():
        assert isinstance(profile, ModelProfileConfig), name


def test_default_settings_create_default_model_profiles() -> None:
    app = AppSettings()
    assert isinstance(app.models, ModelsConfig)
    defaults = default_model_profiles()
    assert set(app.models.profiles) == set(defaults)
    for name, profile in app.models.profiles.items():
        assert isinstance(profile, ModelProfileConfig)
        assert profile.provider == defaults[name].provider


def test_malformed_models_config_fails_at_startup() -> None:
    with pytest.raises((TypeError, ValueError)):
        AppSettings.model_validate({"models": "not-a-mapping"})


def test_none_models_coerces_to_models_config() -> None:
    app = AppSettings.model_validate({"models": None})
    assert isinstance(app.models, ModelsConfig)
