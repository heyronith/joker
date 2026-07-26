"""Live paper runner tests — injected Webull doubles only (no network)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from joker.app.safety import SafetyMode
from joker.config.settings import AppSettings, EnvSettings
from joker.data.freshness import FreshnessConfig, evaluate_quote_freshness
from joker.data.webull_api import MockWebullMarketApi, WebullCandle, WebullQuote
from joker.data.webull_options_api import MockWebullOptionsMarketApi
from joker.runtime.live_paper_runner import (
    LivePaperError,
    LivePaperRunConfig,
    LivePaperRunner,
)
from joker.schemas.domain import Playbook, PlaybookSetup
from joker.schemas.options_data import OptionContractMetadata, OptionSnapshot


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings.model_validate(
        {
            "mode": "PAPER",
            "live_trading_enabled": False,
            "db_path": str(tmp_path / "joker.db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
            "data_dir": str(tmp_path),
            "agents": {
                "mock_agents": True,
                "intraday_enabled": True,
                "intraday_interval_seconds": 0.5,
                "max_intraday_calls_per_session": 2,
                "postmarket_learner_enabled": True,
                "memory_lookback_days": 5,
            },
            "data": {"default_provider": "webull", "quote_poll_interval_seconds": 0.2},
            "risk": {
                "allow_delayed_quotes": True,
                "feed_max_silence_seconds": 60,
                "delayed_quote_max_age_seconds": 900,
                "quote_max_age_seconds": 30,
                "max_premium_usd": 500,
            },
        }
    )


def _candles(n: int = 10) -> list[WebullCandle]:
    base = _now() - timedelta(minutes=n)
    out: list[WebullCandle] = []
    price = 550.0
    for i in range(n):
        ts = base + timedelta(minutes=i)
        out.append(
            WebullCandle(
                timestamp=ts,
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price + 0.2,
                volume=1000,
            )
        )
        price += 0.3
    return out


def _env() -> EnvSettings:
    return EnvSettings(  # type: ignore[arg-type]
        OPENAI_API_KEY="sk-test-key-for-unit-tests-only",
        OPENAI_MODEL="gpt-5.4-mini",
        WEBULL_APP_KEY="k",
        WEBULL_APP_SECRET="s",
        WEBULL_MARKET_DATA_ENABLED=True,
        WEBULL_LIVE_TRADING_ENABLED=False,
        WEBULL_ACCESS_TOKEN="tok",
    )


def _quote(*, delayed: bool = True) -> WebullQuote:
    # Exchange timestamp can be delayed; received_at freshness is what matters.
    return WebullQuote(
        symbol="SPY",
        price=553.0,
        bid=552.9,
        ask=553.1,
        timestamp=_now() - timedelta(minutes=10),
        delayed=delayed,
    )


def _option_contract(option_type: str, strike: float) -> OptionContractMetadata:
    today = date.today()
    return OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=today,
        strike=strike,
        option_type=option_type,  # type: ignore[arg-type]
        contract_id=f"SPY{today.strftime('%y%m%d')}{'C' if option_type == 'call' else 'P'}{int(strike * 1000):08d}",
        source="webull_opra",
    )


def _option_snap(option_type: str, strike: float) -> OptionSnapshot:
    contract = _option_contract(option_type, strike)
    return OptionSnapshot(
        contract=contract,
        bid=1.0,
        ask=1.1,
        mid=1.05,
        spread_pct=9.5,
        quote_timestamp=_now() - timedelta(minutes=10),
        delayed=True,
        source="webull_opra",
        is_synthetic=False,
    )


def _approved_playbook() -> Playbook:
    return Playbook(
        trading_day=date.today(),
        title="SPY 0DTE live paper test",
        summary="Injected SPY playbook for tests",
        setups=[
            PlaybookSetup(
                name="SPY trend call",
                direction="long_call",
                entry_conditions=["VWAP reclaim"],
                stop_rule="50% premium stop",
                take_profit_rule="100% premium target",
                require_trend="any",
                vwap_side="either",
                min_momentum_pct=0.0,
                stop_pct=0.5,
                take_profit_pct=1.0,
            ),
        ],
        approved=True,
    )


def test_delayed_quote_freshness_uses_received_at() -> None:
    now = _now()
    exchange_old = now - timedelta(minutes=12)
    received = now - timedelta(seconds=2)
    ok = evaluate_quote_freshness(
        quote_timestamp=exchange_old,
        reference_time=now,
        delayed=True,
        received_at=received,
        config=FreshnessConfig(allow_delayed_quotes=True),
    )
    assert ok.ok is True

    silent = evaluate_quote_freshness(
        quote_timestamp=exchange_old,
        reference_time=now,
        delayed=True,
        received_at=now - timedelta(seconds=120),
        config=FreshnessConfig(allow_delayed_quotes=True, feed_max_silence_seconds=60),
    )
    assert silent.ok is False
    assert silent.reason == "FEED_SILENT"


def test_live_paper_rejects_live_trading_flag(tmp_path: Path) -> None:
    settings = _settings(tmp_path).model_copy(update={"live_trading_enabled": False})
    # Mode LIVE_GATED blocked
    settings = settings.model_copy(update={"mode": SafetyMode.LIVE_GATED})
    runner = LivePaperRunner(settings, _env())
    with pytest.raises(LivePaperError, match="PAPER"):
        runner.run(
            LivePaperRunConfig(
                duration_seconds=1,
                approved_playbook=_approved_playbook(),
                webull_api=MockWebullMarketApi(quote=_quote(), candles=_candles()),
            )
        )


def test_live_paper_end_to_end_with_injected_webull(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBULL_MARKET_DATA_ENABLED", "true")
    # Force local PaperBroker — do not pick up real WEBULL_PAPER_* from developer .env
    monkeypatch.setenv("WEBULL_PAPER_TRADING_ENABLED", "false")
    monkeypatch.delenv("WEBULL_PAPER_ACCOUNT_ID", raising=False)
    from joker.broker.interface import PaperBroker
    from joker.data import webull_capability

    monkeypatch.setattr(webull_capability, "capability_usable_for_shadow", lambda: True)

    quote = _quote(delayed=True)
    stream = [quote, quote, quote]
    market_api = MockWebullMarketApi(
        quote=quote,
        candles=_candles(12),
        stream_quotes=stream,
    )
    call = _option_snap("call", 553.0)
    put = _option_snap("put", 553.0)
    options_api = MockWebullOptionsMarketApi(
        contracts=[call.contract, put.contract],
        snapshots={call.contract.contract_id: call, put.contract.contract_id: put},
    )

    runner = LivePaperRunner(_settings(tmp_path), _env())
    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=3,
            mock_agents=True,
            approved_playbook=_approved_playbook(),
            webull_api=market_api,
            webull_options_api=options_api,
            require_options=True,
            broker=PaperBroker(initial_balance=25000.0, slippage_pct=0.0),
        )
    )

    assert not result.errors, result.errors
    assert result.summary is not None
    assert result.summary.is_synthetic is False
    assert result.events_processed >= 1
    assert result.report_path is not None
    assert result.report_path.exists()


def test_live_paper_fails_without_real_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from joker.data.webull_errors import WebullApiError

    # Force local PaperBroker — do not pick up real WEBULL_PAPER_* from developer .env
    monkeypatch.setenv("WEBULL_PAPER_TRADING_ENABLED", "false")
    monkeypatch.delenv("WEBULL_PAPER_ACCOUNT_ID", raising=False)

    market_api = MockWebullMarketApi(
        fail_snapshot=WebullApiError("No snapshot"),
        candles=[],
    )
    runner = LivePaperRunner(_settings(tmp_path), _env())
    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=1,
            approved_playbook=_approved_playbook(),
            webull_api=market_api,
            require_options=False,
        )
    )
    assert result.errors or result.failures
    assert any("warmup_failed" in e for e in (result.errors + result.failures))
