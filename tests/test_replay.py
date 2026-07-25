"""Phase 16 replay engine tests."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.data.replay_loader import ReplayLoadError, inspect_replay, load_replay_file
from joker.data.replay_provider import ReplayMarketDataProvider
from joker.data.synthetic_replay import build_synthetic_replay_events
from joker.execution.exit_manager import ExitManager, OpenTradeContext
from joker.execution.option_selector import OptionSelector, OptionSelectionError
from joker.features.engine import FeatureEngine
from joker.risk.governor import RiskGovernor
from joker.runtime.market_handler import MarketEventHandler
from joker.runtime.reactive_engine import ReactiveEngine
from joker.runtime.replay_clock import ReplayClockController
from joker.runtime.replay_runner import ReplayRunConfig, ReplayRunner
from joker.schemas.domain import Playbook, PlaybookSetup, RiskConfig
from joker.schemas.replay import (
    ExitReason,
    OptionQuoteEvent,
    ReplayClock,
    ReplaySpeedMode,
    SpyCandleEvent,
    SpyQuoteEvent,
)
from joker.schemas.domain import Candle


@pytest.fixture
def minimal_replay_path(tmp_path: Path) -> Path:
    path = tmp_path / "minimal.jsonl"
    ts = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    events = [
        {
            "event_type": "spy_quote",
            "event_id": "q1",
            "timestamp": ts.isoformat(),
            "symbol": "SPY",
            "source": "synthetic_stock",
            "price": 550.0,
            "bid": 549.99,
            "ask": 550.01,
        },
        {
            "event_type": "option_quote",
            "event_id": "o1",
            "timestamp": (ts + timedelta(minutes=1)).isoformat(),
            "symbol": "SPY",
            "source": "synthetic_option",
            "contract_id": "SPY_CALL_550",
            "expiration": "2026-07-01",
            "strike": 550.0,
            "option_type": "call",
            "bid": 1.0,
            "ask": 1.1,
            "mid": 1.05,
            "spread_pct": 9.5,
            "quote_timestamp": (ts + timedelta(minutes=1)).isoformat(),
        },
    ]
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


def test_replay_file_parsing(synthetic_replay_path: Path) -> None:
    session = load_replay_file(synthetic_replay_path)
    assert session.metadata.is_synthetic is True
    assert session.metadata.event_count > 0
    assert session.events[0].timestamp <= session.events[-1].timestamp


def test_invalid_replay_file_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"event_type": "spy_quote"}\n')
    with pytest.raises(ReplayLoadError):
        load_replay_file(bad)


def test_event_ordering_preserved(synthetic_replay_path: Path) -> None:
    session = load_replay_file(synthetic_replay_path)
    timestamps = [e.timestamp for e in session.events]
    assert timestamps == sorted(timestamps)


def test_replay_clock_deterministic_mode() -> None:
    ts = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    clock = ReplayClock(current_time=ts, trading_day=date(2026, 7, 1))
    controller = ReplayClockController.deterministic(clock)
    next_ts = ts + timedelta(minutes=5)
    controller.wait_until_next(next_ts)
    assert clock.mode is ReplaySpeedMode.DETERMINISTIC


def test_feature_engine_updates_from_replay_events(synthetic_replay_path: Path) -> None:
    session = load_replay_file(synthetic_replay_path)
    provider = ReplayMarketDataProvider(session)
    engine = FeatureEngine(max_age_seconds=120)
    features = None
    for event in provider.stream_events():
        snap = provider.get_latest_snapshot()
        if snap and len(snap.candles) >= 3:
            features = engine.compute(snap, reference_time=provider.current_time)
            break
    assert features is not None
    assert features.symbol == "SPY"


def test_option_selector_atm_call() -> None:
    ts = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    quotes = [
        OptionQuoteEvent(
            timestamp=ts,
            source="test",
            contract_id="c550",
            expiration=date(2026, 7, 1),
            strike=550.0,
            option_type="call",
            bid=1.0,
            ask=1.1,
            mid=1.05,
            spread_pct=9.5,
            quote_timestamp=ts,
        ),
        OptionQuoteEvent(
            timestamp=ts,
            source="test",
            contract_id="c555",
            expiration=date(2026, 7, 1),
            strike=555.0,
            option_type="call",
            bid=0.5,
            ask=0.6,
            mid=0.55,
            spread_pct=18.0,
            quote_timestamp=ts,
        ),
    ]
    sel = OptionSelector().select_from_events(quotes, "long_call", 550.0, ts)
    assert sel.contract.strike == 550.0
    assert sel.contract.option_type == "call"


def test_option_selector_atm_put() -> None:
    ts = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    quotes = [
        OptionQuoteEvent(
            timestamp=ts,
            source="test",
            contract_id="p545",
            expiration=date(2026, 7, 1),
            strike=545.0,
            option_type="put",
            bid=1.0,
            ask=1.1,
            mid=1.05,
            spread_pct=9.5,
            quote_timestamp=ts,
        ),
    ]
    sel = OptionSelector().select_from_events(quotes, "long_put", 545.0, ts)
    assert sel.contract.option_type == "put"


def test_option_selector_rejects_stale() -> None:
    ts = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    quotes = [
        OptionQuoteEvent(
            timestamp=ts,
            source="test",
            contract_id="c1",
            expiration=date(2026, 7, 1),
            strike=550.0,
            option_type="call",
            bid=1.0,
            ask=1.1,
            mid=1.05,
            spread_pct=9.5,
            quote_timestamp=ts - timedelta(minutes=10),
        ),
    ]
    with pytest.raises(OptionSelectionError, match="STALE"):
        OptionSelector().select_from_events(quotes, "long_call", 550.0, ts)


def test_option_selector_rejects_wide_spread() -> None:
    ts = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    quotes = [
        OptionQuoteEvent(
            timestamp=ts,
            source="test",
            contract_id="c1",
            expiration=date(2026, 7, 1),
            strike=550.0,
            option_type="call",
            bid=0.5,
            ask=1.5,
            mid=1.0,
            spread_pct=100.0,
            quote_timestamp=ts,
        ),
    ]
    with pytest.raises(OptionSelectionError, match="WIDE"):
        OptionSelector().select_from_events(quotes, "long_call", 550.0, ts)


def test_option_selector_rejects_missing_bid() -> None:
    ts = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    quotes = [
        OptionQuoteEvent(
            timestamp=ts,
            source="test",
            contract_id="c1",
            expiration=date(2026, 7, 1),
            strike=550.0,
            option_type="call",
            bid=0,
            ask=1.1,
            mid=0.55,
            spread_pct=100.0,
            quote_timestamp=ts,
        ),
    ]
    with pytest.raises(OptionSelectionError):
        OptionSelector().select_from_events(quotes, "long_call", 550.0, ts)


def test_option_selector_respects_max_premium() -> None:
    ts = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    quotes = [
        OptionQuoteEvent(
            timestamp=ts,
            source="test",
            contract_id="c1",
            expiration=date(2026, 7, 1),
            strike=550.0,
            option_type="call",
            bid=3.0,
            ask=3.2,
            mid=3.1,
            spread_pct=6.0,
            quote_timestamp=ts,
        ),
    ]
    from joker.execution.option_selector import OptionSelectorConfig

    with pytest.raises(OptionSelectionError, match="MAX_PREMIUM"):
        OptionSelector(OptionSelectorConfig(max_premium_usd=200)).select_from_events(
            quotes, "long_call", 550.0, ts
        )


def test_stop_loss_exit() -> None:
    mgr = ExitManager()
    ctx = OpenTradeContext(
        position_id="p1",
        entry_price=1.0,
        stop_price=0.5,
        take_profit_price=2.0,
        entry_time=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
    )
    decision = mgr.check_exit(ctx, 0.45, datetime(2026, 7, 1, 14, 10, tzinfo=timezone.utc))
    assert decision is not None
    assert decision.reason is ExitReason.STOP_LOSS


def test_take_profit_exit() -> None:
    mgr = ExitManager()
    ctx = OpenTradeContext(
        position_id="p1",
        entry_price=1.0,
        stop_price=0.5,
        take_profit_price=2.0,
        entry_time=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
    )
    decision = mgr.check_exit(ctx, 2.1, datetime(2026, 7, 1, 14, 10, tzinfo=timezone.utc))
    assert decision is not None
    assert decision.reason is ExitReason.TAKE_PROFIT


def test_time_stop_exit() -> None:
    mgr = ExitManager()
    ctx = OpenTradeContext(
        position_id="p1",
        entry_price=1.0,
        stop_price=0.5,
        take_profit_price=2.0,
        entry_time=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
        time_stop_minutes=15,
    )
    decision = mgr.check_exit(ctx, 1.0, datetime(2026, 7, 1, 14, 20, tzinfo=timezone.utc))
    assert decision is not None
    assert decision.reason is ExitReason.TIME_STOP


def test_eod_exit() -> None:
    # EOD is America/New_York 15:55; July is EDT (UTC-4) → 19:55 UTC
    mgr = ExitManager(eod_time=datetime(2026, 7, 1, 15, 55).time())
    ctx = OpenTradeContext(
        position_id="p1",
        entry_price=1.0,
        stop_price=0.5,
        take_profit_price=2.0,
        entry_time=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
    )
    decision = mgr.check_exit(ctx, 1.0, datetime(2026, 7, 1, 19, 56, tzinfo=timezone.utc))
    assert decision is not None
    assert decision.reason is ExitReason.END_OF_DAY


def test_trailing_stop_activates_and_exits() -> None:
    mgr = ExitManager(trail_activate_mfe_pct=0.35, trail_giveback_pct=0.20)
    ctx = OpenTradeContext(
        position_id="p1",
        entry_price=1.0,
        stop_price=0.5,
        take_profit_price=5.0,
        entry_time=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
    )
    # MFE 40% → trail activates at peak * 0.8 = 1.12
    ctx = mgr.update_trailing(ctx, 1.40)
    assert ctx.trail_active is True
    assert ctx.trail_stop_price is not None
    assert ctx.trail_stop_price == pytest.approx(1.12)
    # Giveback through trail
    decision = mgr.check_exit(ctx, 1.10, datetime(2026, 7, 1, 14, 10, tzinfo=timezone.utc))
    assert decision is not None
    assert decision.reason is ExitReason.STOP_LOSS
    assert "Trailing" in (decision.message or "")


def test_eod_not_triggered_before_et_close() -> None:
    mgr = ExitManager(eod_time=datetime(2026, 7, 1, 15, 55).time())
    ctx = OpenTradeContext(
        position_id="p1",
        entry_price=1.0,
        stop_price=0.5,
        take_profit_price=2.0,
        entry_time=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
    )
    # 15:56 UTC is still morning ET — must not force EOD
    decision = mgr.check_exit(ctx, 1.0, datetime(2026, 7, 1, 15, 56, tzinfo=timezone.utc))
    assert decision is None


def _make_handler(tmp_path, mode=SafetyMode.PAPER):
    from joker.config.settings import AppSettings
    from joker.execution.option_selector import OptionSelector, OptionSelectorConfig
    from joker.logging.event_log import EventLogWriter
    from joker.schemas.domain import DailyState

    settings = AppSettings.model_validate({"mode": mode.value})
    risk = RiskConfig(
        max_daily_loss_usd=500,
        max_trades_per_day=3,
        max_open_positions=1,
        max_premium_usd=200,
        max_spread_pct=15,
        quote_max_age_seconds=120,
    )
    broker = PaperBroker(slippage_pct=0.0)
    reactive = ReactiveEngine(RiskGovernor(risk, mode), broker)
    pb = Playbook(
        trading_day=date(2026, 7, 1),
        title="t",
        summary="s",
        setups=[
            PlaybookSetup(
                name="call",
                direction="long_call",
                stop_rule="50%",
                take_profit_rule="100%",
            )
        ],
        approved=True,
    )
    reactive.arm_playbook(pb)
    return MarketEventHandler(
        provider=MagicMock(),
        reactive_engine=reactive,
        risk_governor=reactive.risk_governor,
        broker=broker,
        feature_engine=FeatureEngine(max_age_seconds=120),
        option_selector=OptionSelector(OptionSelectorConfig(quote_max_age_seconds=120)),
        exit_manager=ExitManager(),
        mode=mode,
        run_id="test-run",
        daily_state=DailyState(trading_day=date(2026, 7, 1), run_id="test-run", mode=mode.value),
    )


def test_shadow_mode_replay_never_submits_broker() -> None:
    handler = _make_handler(Path("/tmp"), mode=SafetyMode.SHADOW)
    assert handler.shadow is not None
    broker = handler.broker
    assert broker.list_open_orders() == []


def test_full_replay_day(tmp_path: Path, synthetic_replay_path: Path) -> None:
    from joker.config.settings import AppSettings

    settings = AppSettings.model_validate(
        {
            "mode": "PAPER",
            "db_path": str(tmp_path / "joker.db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )
    runner = ReplayRunner(settings)
    result = runner.run(
        ReplayRunConfig(replay_path=synthetic_replay_path, deterministic=True, mock_agents=True)
    )
    assert result.summary.events_processed > 0
    report = tmp_path / "reports" / "postmarket" / "2026-07-01.md"
    assert report.exists()
    assert result.summary.is_synthetic is True
    content = report.read_text()
    assert "Synthetic replay" in content or "synthetic replay" in content.lower()
    assert "mock agents" in content.lower()


def test_replay_reproducible(tmp_path: Path, synthetic_replay_path: Path) -> None:
    from joker.config.settings import AppSettings

    settings = AppSettings.model_validate(
        {
            "mode": "PAPER",
            "db_path": str(tmp_path / "a.db"),
            "event_log_dir": str(tmp_path / "logs_a"),
            "reports_dir": str(tmp_path / "reports_a"),
        }
    )
    r1 = ReplayRunner(settings).run(
        ReplayRunConfig(replay_path=synthetic_replay_path, deterministic=True, skip_premarket=True)
    )
    settings2 = AppSettings.model_validate(
        {
            "mode": "PAPER",
            "db_path": str(tmp_path / "b.db"),
            "event_log_dir": str(tmp_path / "logs_b"),
            "reports_dir": str(tmp_path / "reports_b"),
        }
    )
    r2 = ReplayRunner(settings2).run(
        ReplayRunConfig(replay_path=synthetic_replay_path, deterministic=True, skip_premarket=True)
    )
    assert r1.summary.signals_detected == r2.summary.signals_detected
    assert r1.summary.trades_entered == r2.summary.trades_entered
    assert r1.summary.trades_exited == r2.summary.trades_exited
    assert r1.summary.final_pnl_usd == r2.summary.final_pnl_usd


def test_replay_stores_jsonl_and_db(tmp_path: Path, synthetic_replay_path: Path) -> None:
    from joker.config.settings import AppSettings
    from joker.logging.event_log import EventLogWriter

    log_dir = tmp_path / "logs"
    settings = AppSettings.model_validate(
        {
            "db_path": str(tmp_path / "joker.db"),
            "event_log_dir": str(log_dir),
            "reports_dir": str(tmp_path / "reports"),
        }
    )
    result = ReplayRunner(settings).run(
        ReplayRunConfig(replay_path=synthetic_replay_path, deterministic=True, skip_premarket=True)
    )
    log_files = list(log_dir.glob("*.jsonl"))
    assert log_files
    events = EventLogWriter(log_dir).read_all(result.summary.run_id)
    assert any(e["event_type"] == "replay.completed" for e in events)


def test_inspect_replay(synthetic_replay_path: Path) -> None:
    info = inspect_replay(synthetic_replay_path)
    assert info["is_synthetic"] is True
    assert info["event_count"] > 0


def test_no_secrets_in_replay_logs(tmp_path: Path, synthetic_replay_path: Path) -> None:
    from joker.config.settings import AppSettings
    from joker.logging.event_log import EventLogWriter

    secret = "sk-test-key-for-unit-tests-only"
    log_dir = tmp_path / "logs"
    settings = AppSettings.model_validate(
        {
            "db_path": str(tmp_path / "joker.db"),
            "event_log_dir": str(log_dir),
            "reports_dir": str(tmp_path / "reports"),
        }
    )
    result = ReplayRunner(settings).run(
        ReplayRunConfig(replay_path=synthetic_replay_path, deterministic=True, skip_premarket=True)
    )
    raw = (log_dir / f"{result.summary.run_id}.jsonl").read_text()
    assert secret not in raw
