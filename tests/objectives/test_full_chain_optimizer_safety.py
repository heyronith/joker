"""Full-chain optimizer is paper/replay only — refuse live activation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from joker.app.safety import SafetyMode
from joker.config.settings import AppSettings, BrokerSettings
from joker.objectives.config import FullChainOptimizerSettings


def test_live_gated_mode_rejects_full_chain_optimizer() -> None:
    with pytest.raises(ValidationError, match="paper/replay only"):
        AppSettings(
            mode=SafetyMode.LIVE_GATED,
            live_trading_enabled=True,
            full_chain_optimizer=FullChainOptimizerSettings(enabled=True),
        )


def test_webull_live_broker_rejects_full_chain_optimizer() -> None:
    with pytest.raises(ValidationError, match="paper/replay only"):
        AppSettings(
            mode=SafetyMode.PAPER,
            broker=BrokerSettings(provider="webull_live"),
            full_chain_optimizer=FullChainOptimizerSettings(enabled=True),
        )


def test_paper_mode_allows_full_chain_optimizer() -> None:
    app = AppSettings(
        mode=SafetyMode.PAPER,
        broker=BrokerSettings(provider="paper"),
        full_chain_optimizer=FullChainOptimizerSettings(enabled=True),
    )
    assert app.full_chain_optimizer.enabled is True
    assert app.mode is SafetyMode.PAPER
