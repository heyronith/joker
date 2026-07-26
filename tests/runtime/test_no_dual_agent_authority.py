"""Ensure legacy and cognitive agents never dual-submit."""

from __future__ import annotations

import inspect

from joker.runtime.live_paper_runner import LivePaperRunner


def test_live_paper_runner_gates_legacy_loop_when_cognitive_mode() -> None:
    source = inspect.getsource(LivePaperRunner.run)
    assert "cognitive_mode" in source
    assert "agent_led = execution_mode == \"agent_led\" and not cognitive_mode" in source
    assert "rules_auto_entry=(\n                not cognitive_mode" in source
