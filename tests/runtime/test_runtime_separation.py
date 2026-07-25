
"""MarketRuntime must not submit orders; ExecutionRuntime requires explicit command."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from joker.runtime.execution_runtime import ExecutionRuntime
from joker.runtime.market_runtime import MarketRuntime


def test_market_runtime_has_no_submit() -> None:
    methods = {n for n, _ in inspect.getmembers(MarketRuntime, predicate=inspect.isfunction)}
    assert "submit_order" not in methods
    assert "submit_execution_command" not in methods


def test_execution_runtime_has_submit() -> None:
    assert hasattr(ExecutionRuntime, "submit_execution_command")
