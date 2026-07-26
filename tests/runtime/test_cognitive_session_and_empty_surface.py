"""Tests for stable cognitive session identity and empty surface unavailability."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.config.settings import EnvSettings
from joker.data.webull_options_provider import SurfaceFetchResult, WebullOptionsDataProvider
from joker.market.quality import DataQualityCode
from joker.runtime.cognitive_session import (
    LOCAL_PAPER_ACCOUNT_IDENTITY,
    exchange_trading_date,
    hash_paper_account_id,
    live_paper_cognitive_session_id,
    paper_account_identity,
    stable_cognitive_session_id,
)
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock, SystemExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID

ET = ZoneInfo("America/New_York")


def test_stable_session_same_account_and_exchange_date() -> None:
    day = date(2026, 7, 1)
    a = stable_cognitive_session_id(
        trading_date=day,
        account_identity="webull:abc123",
        mode="paper",
    )
    b = stable_cognitive_session_id(
        trading_date=day,
        account_identity="webull:abc123",
        mode="paper",
    )
    assert a == b
    assert a == "cog:paper:webull:abc123:2026-07-01"


def test_different_webull_accounts_produce_different_identities() -> None:
    env_a = EnvSettings.model_construct(webull_paper_account_id="ACCOUNT_A")
    env_b = EnvSettings.model_construct(webull_paper_account_id="ACCOUNT_B")
    id_a = paper_account_identity(broker_kind="webull_paper", env=env_a)
    id_b = paper_account_identity(broker_kind="webull_paper", env=env_b)
    assert id_a != id_b
    assert id_a.startswith("webull:")
    assert hash_paper_account_id("ACCOUNT_A") in id_a
    day = date(2026, 7, 1)
    assert live_paper_cognitive_session_id(
        broker_kind="webull_paper", env=env_a, trading_date=day
    ) != live_paper_cognitive_session_id(
        broker_kind="webull_paper", env=env_b, trading_date=day
    )


def test_local_paper_uses_explicit_local_identity() -> None:
    env = EnvSettings.model_construct(webull_paper_account_id="SHOULD_NOT_BE_USED")
    assert (
        paper_account_identity(broker_kind="local_paper", env=env)
        == LOCAL_PAPER_ACCOUNT_IDENTITY
    )


def test_exchange_trading_date_not_host_today() -> None:
    """Host calendar date must not override the exchange trading date."""
    host_today = date(2026, 7, 4)  # Saturday
    exchange_day = date(2026, 7, 2)  # XNYS session under frozen clock
    calendar = MarketCalendar()
    clock = FrozenExchangeClock(
        datetime(2026, 7, 2, 15, 0, tzinfo=ET), calendar=calendar
    )
    assert clock.trading_date() == exchange_day
    assert clock.trading_date() != host_today
    env = EnvSettings.model_construct(webull_paper_account_id="ACCT")
    sid = live_paper_cognitive_session_id(
        broker_kind="webull_paper",
        env=env,
        trading_date=clock.trading_date(),
    )
    assert sid.endswith(f":{exchange_day.isoformat()}")
    assert host_today.isoformat() not in sid
    # Helper itself is exchange-clock based (America/New_York), not a bare host today.
    assert exchange_trading_date(calendar=calendar) == SystemExchangeClock(
        calendar=calendar
    ).trading_date()


def test_zero_discovered_contracts_are_unavailable() -> None:
    result = SurfaceFetchResult(
        snapshots=[],
        discovered_count=0,
        selected_count=0,
        fetched_count=0,
        failed_batches=("zero_contracts_discovered",),
        complete=False,
        trading_date=date(2026, 7, 1),
    )
    findings = result.to_data_quality_findings()
    assert findings
    assert findings[0].code == DataQualityCode.OPTION_SURFACE_UNAVAILABLE
    assert findings[0].severity.value == "error"


@pytest.mark.asyncio
async def test_empty_fetch_blocks_execution_despite_prior_surface(tmp_path) -> None:
    """Prior surface must not remain execution-usable after an empty discovery."""
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "empty_surface.db"
    broker = PaperBroker(slippage_pct=0)
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id="empty-surface",
            broker_account_id="paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
    )
    await supervisor.start(start_agent=False)
    market = supervisor.market_runtime
    assert market is not None

    # Persist a valid surface first.
    for i in range(3):
        ts = start + timedelta(minutes=i, seconds=5)
        clock.set_now(ts)
        await market.ingest_underlying_quote(
            symbol="SPY",
            bid=Decimal("499.90"),
            ask=Decimal("500.10"),
            last=Decimal("500") + Decimal(i),
            source_timestamp=ts,
            received_timestamp=ts,
        )
    await market.ingest_option_quotes(
        [
            {
                "contract_id": CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "1.00",
                "ask": "1.20",
                "last": "1.10",
                "quote_timestamp": start + timedelta(minutes=3),
                "is_0dte": True,
            }
        ]
    )
    first = await market.tick(now=start + timedelta(minutes=3, seconds=3))
    assert first.snapshot is not None
    assert first.quality is not None
    assert first.quality.usable_for_execution is True
    assert market.latest_surface is not None

    class _EmptyApi:
        CONTRACT_DISCOVERY_VERIFIED = True
        SNAPSHOT_VERIFIED = True

        def find_option_contracts(self, symbol, expiration):
            return []

        def get_option_snapshots(self, batch):
            return []

    provider = WebullOptionsDataProvider(
        env=EnvSettings.model_construct(),
        api=_EmptyApi(),  # type: ignore[arg-type]
    )
    fetch = provider.fetch_surface_snapshots(500.0, trading_date=date(2026, 7, 1))
    assert fetch.complete is False
    assert fetch.discovered_count == 0
    findings = fetch.to_data_quality_findings()
    assert findings
    assert findings[0].code == DataQualityCode.OPTION_SURFACE_UNAVAILABLE
    market.enqueue_quality_findings(findings)

    later = start + timedelta(minutes=6)
    clock.set_now(later)
    await market.ingest_underlying_quote(
        symbol="SPY",
        bid=Decimal("500.00"),
        ask=Decimal("500.20"),
        last=Decimal("500.10"),
        source_timestamp=later,
        received_timestamp=later,
    )
    second = await market.tick(now=later + timedelta(seconds=3))
    assert second.quality is not None
    assert second.quality.usable_for_execution is False
    assert any(
        f.code == DataQualityCode.OPTION_SURFACE_UNAVAILABLE
        for f in second.quality.findings
    )
    # Prior surface object may still exist, but execution must be blocked.
    assert market.latest_surface is not None
    await supervisor.shutdown()
