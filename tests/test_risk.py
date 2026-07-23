"""Phase 5 risk governor tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from joker.app.safety import SafetyMode
from joker.risk.governor import RiskGovernor, RiskReasonCode
from joker.schemas.domain import RiskConfig
from tests.fixtures.domain import make_candidate, make_contract, make_daily_state, make_quote


@pytest.fixture
def governor() -> RiskGovernor:
    config = RiskConfig(
        max_daily_loss_usd=500,
        max_trades_per_day=3,
        max_open_positions=1,
        max_premium_usd=200,
        max_spread_pct=15,
        quote_max_age_seconds=30,
    )
    return RiskGovernor(config, SafetyMode.PAPER)


def test_valid_candidate_passes(governor: RiskGovernor) -> None:
    decision = governor.evaluate(make_candidate(), make_daily_state())
    assert decision.approved is True


def test_wrong_symbol_rejected(governor: RiskGovernor) -> None:
    c = make_candidate(contract=make_contract(symbol="QQQ"))
    decision = governor.evaluate(c, make_daily_state())
    assert RiskReasonCode.WRONG_SYMBOL in decision.reason_codes


def test_non_0dte_rejected(governor: RiskGovernor) -> None:
    with pytest.raises(Exception):
        make_contract(is_0dte=False)


def test_stale_quote_rejected(governor: RiskGovernor) -> None:
    old = datetime.now(timezone.utc) - timedelta(seconds=120)
    quote = make_quote(timestamp=old)
    c = make_candidate(quote=quote)
    decision = governor.evaluate(c, make_daily_state())
    assert RiskReasonCode.STALE_QUOTE in decision.reason_codes


def test_wide_spread_rejected(governor: RiskGovernor) -> None:
    quote = make_quote(bid=1.0, ask=2.0)
    c = make_candidate(quote=quote)
    decision = governor.evaluate(c, make_daily_state())
    assert RiskReasonCode.WIDE_SPREAD in decision.reason_codes


def test_no_stop_rejected(governor: RiskGovernor) -> None:
    c = make_candidate(stop_price=0)
    decision = governor.evaluate(c, make_daily_state())
    assert RiskReasonCode.NO_STOP in decision.reason_codes


def test_max_daily_loss_rejected(governor: RiskGovernor) -> None:
    state = make_daily_state(daily_pnl_usd=-600)
    decision = governor.evaluate(make_candidate(), state)
    assert RiskReasonCode.MAX_DAILY_LOSS in decision.reason_codes


def test_duplicate_order_rejected(governor: RiskGovernor) -> None:
    c = make_candidate()
    governor.evaluate(c, make_daily_state())
    decision = governor.evaluate(c, make_daily_state())
    assert RiskReasonCode.DUPLICATE_ORDER in decision.reason_codes


def test_kill_switch_blocks(governor: RiskGovernor) -> None:
    state = make_daily_state(kill_switch=True)
    decision = governor.evaluate(make_candidate(), state)
    assert RiskReasonCode.KILL_SWITCH in decision.reason_codes


def test_agent_override_ignored(governor: RiskGovernor) -> None:
    state = make_daily_state(kill_switch=True)
    decision = governor.evaluate(make_candidate(), state, agent_override=True)
    assert decision.approved is False


def _agent_led_governor(**overrides) -> RiskGovernor:
    cfg = dict(
        max_daily_loss_usd=50,
        max_trades_per_day=1,
        max_open_positions=1,
        max_premium_usd=50,
        max_spread_pct=5,
        quote_max_age_seconds=30,
        policy="agent_led",
    )
    cfg.update(overrides)
    return RiskGovernor(RiskConfig(**cfg), SafetyMode.PAPER)


def test_agent_led_allows_wide_spread_and_high_premium() -> None:
    gov = _agent_led_governor()
    quote = make_quote(bid=1.0, ask=3.0)  # wide spread; premium > soft cap
    c = make_candidate(quote=quote, entry_limit_price=2.0)
    decision = gov.evaluate(c, make_daily_state())
    assert decision.approved is True
    assert decision.policy == "agent_led"
    assert RiskReasonCode.WIDE_SPREAD not in decision.reason_codes
    assert RiskReasonCode.MAX_PREMIUM not in decision.reason_codes


def test_agent_led_ignores_max_trades_and_daily_loss() -> None:
    gov = _agent_led_governor()
    state = make_daily_state(daily_pnl_usd=-999, trades_count=99)
    decision = gov.evaluate(make_candidate(), state)
    assert decision.approved is True
    assert RiskReasonCode.MAX_DAILY_LOSS not in decision.reason_codes
    assert RiskReasonCode.MAX_TRADES not in decision.reason_codes


def test_agent_led_still_blocks_kill_switch() -> None:
    gov = _agent_led_governor()
    decision = gov.evaluate(make_candidate(), make_daily_state(kill_switch=True))
    assert decision.approved is False
    assert RiskReasonCode.KILL_SWITCH in decision.reason_codes


def test_agent_led_still_blocks_no_stop() -> None:
    gov = _agent_led_governor()
    decision = gov.evaluate(make_candidate(stop_price=0), make_daily_state())
    assert RiskReasonCode.NO_STOP in decision.reason_codes
