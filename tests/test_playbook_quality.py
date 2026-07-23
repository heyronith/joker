"""Playbook quality validator tests."""

from __future__ import annotations

from datetime import date

import pytest

from joker.schemas.domain import Playbook, PlaybookSetup, RiskConfig
from joker.strategy.playbook_quality import (
    PlaybookQualityReason,
    PlaybookQualityValidator,
    trim_playbook_enabled_setups,
)


def _valid_playbook() -> Playbook:
    return Playbook(
        trading_day=date(2026, 7, 1),
        title="SPY 0DTE Plan",
        summary="Defined risk only",
        setups=[
            PlaybookSetup(
                name="Call setup",
                direction="long_call",
                entry_conditions=["VWAP reclaim"],
                stop_rule="50% premium stop",
                take_profit_rule="100% premium target",
            )
        ],
    )


def test_valid_openai_playbook_passes() -> None:
    result = PlaybookQualityValidator().validate(_valid_playbook())
    assert result.approved is True


def test_missing_stop_rejected() -> None:
    pb = _valid_playbook()
    pb.setups[0] = pb.setups[0].model_copy(update={"stop_rule": ""})
    result = PlaybookQualityValidator().validate(pb)
    assert PlaybookQualityReason.MISSING_STOP in result.reason_codes


def test_missing_take_profit_rejected() -> None:
    pb = _valid_playbook()
    pb.setups[0] = pb.setups[0].model_copy(update={"take_profit_rule": ""})
    result = PlaybookQualityValidator().validate(pb)
    assert PlaybookQualityReason.MISSING_TAKE_PROFIT in result.reason_codes


def test_short_option_language_rejected() -> None:
    pb = _valid_playbook()
    pb = pb.model_copy(update={"summary": "Use short call spreads for income"})
    result = PlaybookQualityValidator().validate(pb)
    assert PlaybookQualityReason.SHORT_OPTION_LANGUAGE in result.reason_codes or PlaybookQualityReason.SPREAD_LANGUAGE in result.reason_codes


def test_guaranteed_profit_rejected() -> None:
    pb = _valid_playbook()
    pb = pb.model_copy(update={"summary": "This plan has guaranteed profit"})
    result = PlaybookQualityValidator().validate(pb)
    assert PlaybookQualityReason.GUARANTEED_PROFIT in result.reason_codes


def test_empty_entry_conditions_rejected() -> None:
    pb = _valid_playbook()
    pb.setups[0] = pb.setups[0].model_copy(update={"entry_conditions": []})
    result = PlaybookQualityValidator().validate(pb)
    assert PlaybookQualityReason.EMPTY_ENTRY_CONDITIONS in result.reason_codes


def test_loosen_risk_rejected() -> None:
    pb = _valid_playbook()
    pb = pb.model_copy(update={"summary": "Increase max_daily_loss today"})
    cfg = RiskConfig(
        max_daily_loss_usd=500,
        max_trades_per_day=3,
        max_open_positions=1,
        max_premium_usd=200,
        max_spread_pct=15,
        quote_max_age_seconds=30,
    )
    result = PlaybookQualityValidator(cfg).validate(pb)
    assert PlaybookQualityReason.RISK_CONFIG_CONFLICT in result.reason_codes


def test_critic_blocked_rejected() -> None:
    result = PlaybookQualityValidator().validate(_valid_playbook(), critic_blocked=True)
    assert PlaybookQualityReason.CRITIC_BLOCKED in result.reason_codes


def test_two_enabled_setups_allowed_with_max_one_trade_per_day() -> None:
    cfg = RiskConfig(
        max_daily_loss_usd=100,
        max_trades_per_day=1,
        max_open_positions=1,
        max_premium_usd=100,
        max_spread_pct=15,
        quote_max_age_seconds=30,
    )
    pb = _valid_playbook()
    pb = pb.model_copy(
        update={
            "setups": [
                *pb.setups,
                PlaybookSetup(
                    name="Put setup",
                    direction="long_put",
                    entry_conditions=["VWAP rejection"],
                    stop_rule="50% premium stop",
                    take_profit_rule="100% premium target",
                ),
            ]
        }
    )
    result = PlaybookQualityValidator(cfg).validate(pb)
    assert result.approved is True


def test_three_enabled_setups_rejected() -> None:
    cfg = RiskConfig(
        max_daily_loss_usd=100,
        max_trades_per_day=1,
        max_open_positions=1,
        max_premium_usd=100,
        max_spread_pct=15,
        quote_max_age_seconds=30,
    )
    pb = _valid_playbook()
    extra = [
        PlaybookSetup(
            name=f"Setup {i}",
            direction="long_call",
            entry_conditions=["test"],
            stop_rule="50% premium stop",
            take_profit_rule="100% premium target",
        )
        for i in range(3)
    ]
    pb = pb.model_copy(update={"setups": extra})
    result = PlaybookQualityValidator(cfg).validate(pb)
    assert PlaybookQualityReason.TOO_MANY_SETUPS in result.reason_codes


def test_trim_playbook_disables_excess_setups() -> None:
    cfg = RiskConfig(
        max_daily_loss_usd=100,
        max_trades_per_day=1,
        max_open_positions=1,
        max_premium_usd=100,
        max_spread_pct=15,
        quote_max_age_seconds=30,
    )
    setups = [
        PlaybookSetup(
            name=f"S{i}",
            direction="long_call",
            entry_conditions=["x"],
            stop_rule="50%",
            take_profit_rule="100%",
        )
        for i in range(4)
    ]
    pb = _valid_playbook().model_copy(update={"setups": setups})
    trimmed = trim_playbook_enabled_setups(pb, cfg)
    assert sum(1 for s in trimmed.setups if s.enabled) == 2
    assert PlaybookQualityValidator(cfg).validate(trimmed).approved is True
