"""ObjectiveSettings config loading."""

from __future__ import annotations

from pathlib import Path

from joker.config.loader import load_app_settings


def test_paper_all_tasks_loads_objective_section() -> None:
    root = Path(__file__).resolve().parents[2]
    app, _env = load_app_settings(root / "config" / "paper-all-tasks.yaml")
    assert app.objective.enabled is True
    assert app.objective.require_deadline is True
    assert app.objective.require_total_loss_acknowledgement is True
    assert app.objective.sizing.prohibit_loss_multiplier is True
    assert app.objective.feasibility.minimum_samples_for_numeric_probability == 20
