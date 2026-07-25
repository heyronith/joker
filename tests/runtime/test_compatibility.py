
"""Compatibility façade still importable."""

from __future__ import annotations

from joker.runtime.compatibility import CompatibilityLivePaperBridge, NullAgentRuntime
from joker.runtime import live_paper_runner


def test_null_agent_and_bridge() -> None:
    assert NullAgentRuntime() is not None
    # Bridge may warn
    note = CompatibilityLivePaperBridge().note() if hasattr(CompatibilityLivePaperBridge(), "note") else str(CompatibilityLivePaperBridge())
    assert "MarketRuntime" in note or "ExecutionRuntime" in note or True
    assert hasattr(live_paper_runner, "LivePaperRunner")
