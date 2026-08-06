"""Ensure legacy and cognitive agents never dual-submit."""

from __future__ import annotations

import inspect

from joker.runtime.live_paper_runner import LivePaperRunner


def test_live_paper_runner_gates_legacy_loop_when_cognitive_mode() -> None:
    source = inspect.getsource(LivePaperRunner.run)
    assert "cognitive_mode" in source
    assert "agent_led = execution_mode == \"agent_led\" and not cognitive_mode" in source
    assert "rules_auto_entry=(\n                not cognitive_mode" in source


def test_live_paper_runner_two_phase_cognitive_startup_ordering() -> None:
    """Task 1 ExecutionRuntime must bind before CognitiveAgentRuntime.start/resume."""
    source = inspect.getsource(LivePaperRunner.run)
    assert "recovery_only_mode = bool(config.reconciliation_only_recovery)" in source
    assert "start_agent=not (recovery_only_mode or cognitive_mode)" in source
    assert "bind_cognitive_graph_to_task1(" in source
    assert "task1_bridge.start_agent()" in source
    assert "live_paper_cognitive_session_id" in source
    bind_idx = source.index("bind_cognitive_graph_to_task1(")
    agent_idx = source.index("task1_bridge.start_agent()")
    assert bind_idx < agent_idx
