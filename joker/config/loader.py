"""YAML configuration loading and merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from joker.config.settings import AppSettings, EnvSettings


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config_path(config_path: str | Path, project_root: Path | None = None) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    root = project_root or Path.cwd()
    return root / path


def load_app_settings(
    config_path: str | Path | None = None,
    project_root: Path | None = None,
    env: EnvSettings | None = None,
) -> tuple[AppSettings, EnvSettings]:
    """Load merged settings from default.yaml, profile YAML, and environment."""
    root = project_root or Path.cwd()
    env_settings = env or EnvSettings()  # type: ignore[call-arg]

    default_path = root / "config" / "default.yaml"
    profile_path = resolve_config_path(
        config_path or env_settings.joker_config,
        project_root=root,
    )

    merged = merge_configs(load_yaml_config(default_path), load_yaml_config(profile_path))

    if env_settings.joker_data_dir:
        merged["data_dir"] = env_settings.joker_data_dir
        merged.setdefault("db_path", f"{env_settings.joker_data_dir}/joker.db")
        merged.setdefault(
            "event_log_dir", f"{env_settings.joker_data_dir}/logs/jsonl"
        )
        merged.setdefault("reports_dir", f"{env_settings.joker_data_dir}/reports")

    if env_settings.joker_log_level:
        merged.setdefault("logging", {})
        if isinstance(merged["logging"], dict):
            merged["logging"]["level"] = env_settings.joker_log_level

    app_settings = AppSettings.model_validate(merged)
    return app_settings, env_settings
