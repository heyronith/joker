"""Authoritative execution validation and 0DTE derivation tests."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.agents.cognitive.execution import (
    AuthoritativeMarketTruth,
    ExecutionProposalValidator,
    parse_contract_id,
    validate_and_compile_proposal,
)
from joker.cognition.exceptions import CognitiveValidationError
from joker.cognition.schemas import ExecutionLeg, ExecutionProposal
from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.market.quality import DataQualityReport, DataQualitySeverity
from joker.market.snapshots import MarketSnapshot, UnderlyingSnapshot


def _snapshot(*, trading_date: date, surface_id=None) -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id=uuid4(),
        exchange_time=datetime(trading_date.year, trading_date.month, trading_date.day, 15, 0, tzinfo=timezone.utc),
        trading_date=trading_date,
        underlying=UnderlyingSnapshot(
            symbol="SPY",
            exchange_time=datetime(trading_date.year, trading_date.month, trading_date.day, 15, 0, tzinfo=timezone.utc),
            last=Decimal("500"),
        ),
        option_surface_id=surface_id,
        data_quality_id=uuid4(),
    )


def _proposal(contract_id: str, snapshot_id, *, max_quote_age: int = 60) -> ExecutionProposal:
    return ExecutionProposal(
        proposal_id=uuid4(),
        decision_id=uuid4(),
        strategy_id=uuid4(),
        session_id="s",
        cycle_id="c",
        snapshot_id=snapshot_id,
        action="execute",
        legs=(
            ExecutionLeg(
                contract_id=contract_id,
                side="buy",
                quantity=1,
                limit_price=Decimal("1.10"),
                sequence_order=0,
                max_quote_age_seconds=max_quote_age,
                replacement_policy="none",
                partial_fill_policy="wait",
            ),
        ),
        order_type="limit",
        time_in_force="day",
        entry_rationale="test",
        prompt_version="2.0.0",
        model_call_id=uuid4(),
    )


def test_parse_contract_id_derives_0dte_from_trading_date() -> None:
    today = date(2026, 7, 25)
    ok = parse_contract_id("SPY:2026-07-25:500.0:call", trading_date=today)
    assert ok.is_0dte is True
    with pytest.raises(CognitiveValidationError, match="trading date|0DTE"):
        parse_contract_id("SPY:2026-07-26:500.0:call", trading_date=today)


def test_future_expiry_labelled_as_0dte_rejected() -> None:
    trading = date(2026, 7, 25)
    surface_id = uuid4()
    snapshot = _snapshot(trading_date=trading, surface_id=surface_id)
    contract_id = "SPY:2026-07-26:500.0:call"
    surface = OptionSurfaceSnapshot(
        surface_id=surface_id,
        exchange_time=snapshot.exchange_time,
        trading_date=trading,
        underlying_symbol="SPY",
        contracts=(
            OptionContractSnapshot(
                contract_id=contract_id,
                symbol="SPY",
                expiry=date(2026, 7, 26),
                strike=Decimal("500"),
                option_type="call",
                bid=Decimal("1.0"),
                ask=Decimal("1.2"),
                quote_timestamp=snapshot.exchange_time,
                quote_age_ms=100,
            ),
        ),
    )
    proposal = _proposal(contract_id, snapshot.snapshot_id)
    truth = AuthoritativeMarketTruth(
        snapshot=snapshot,
        data_quality=DataQualityReport(
            severity=DataQualitySeverity.OK,
            usable_for_execution=True,
        ),
        option_surface=surface,
    )
    with pytest.raises(CognitiveValidationError, match="trading date|0DTE"):
        ExecutionProposalValidator().validate(proposal, truth=truth)


def test_invented_contract_absent_from_surface_rejected() -> None:
    trading = date(2026, 7, 25)
    surface_id = uuid4()
    snapshot = _snapshot(trading_date=trading, surface_id=surface_id)
    surface = OptionSurfaceSnapshot(
        surface_id=surface_id,
        exchange_time=snapshot.exchange_time,
        trading_date=trading,
        underlying_symbol="SPY",
        contracts=(
            OptionContractSnapshot(
                contract_id="SPY:2026-07-25:500.0:call",
                symbol="SPY",
                expiry=trading,
                strike=Decimal("500"),
                option_type="call",
                bid=Decimal("1.0"),
                ask=Decimal("1.2"),
                quote_timestamp=snapshot.exchange_time,
                quote_age_ms=100,
            ),
        ),
    )
    proposal = _proposal("SPY:2026-07-25:999.0:call", snapshot.snapshot_id)
    truth = AuthoritativeMarketTruth(
        snapshot=snapshot,
        data_quality=DataQualityReport(
            severity=DataQualitySeverity.OK,
            usable_for_execution=True,
        ),
        option_surface=surface,
    )
    with pytest.raises(CognitiveValidationError, match="absent from the option surface"):
        ExecutionProposalValidator().validate(proposal, truth=truth)


def test_stale_data_quality_rejected() -> None:
    trading = date(2026, 7, 25)
    surface_id = uuid4()
    snapshot = _snapshot(trading_date=trading, surface_id=surface_id)
    contract_id = "SPY:2026-07-25:500.0:call"
    surface = OptionSurfaceSnapshot(
        surface_id=surface_id,
        exchange_time=snapshot.exchange_time,
        trading_date=trading,
        underlying_symbol="SPY",
        contracts=(
            OptionContractSnapshot(
                contract_id=contract_id,
                symbol="SPY",
                expiry=trading,
                strike=Decimal("500"),
                option_type="call",
                bid=Decimal("1.0"),
                ask=Decimal("1.2"),
                quote_timestamp=snapshot.exchange_time,
                quote_age_ms=100,
            ),
        ),
    )
    proposal = _proposal(contract_id, snapshot.snapshot_id)
    truth = AuthoritativeMarketTruth(
        snapshot=snapshot,
        data_quality=DataQualityReport(
            severity=DataQualitySeverity.CRITICAL,
            usable_for_reasoning=False,
            usable_for_execution=False,
        ),
        option_surface=surface,
    )
    with pytest.raises(CognitiveValidationError, match="data quality"):
        validate_and_compile_proposal(proposal, truth=truth)
