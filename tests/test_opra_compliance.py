"""Phase 21A — OPRA data-governance compliance tests."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from joker.compliance.data_classification import (
    DataClassification,
    SOURCE_WEBULL_OPRA,
    SOURCE_WEBULL_STOCK,
    classify_market_event,
    is_opra_source,
    is_stock_source,
    policy_for,
)
from joker.compliance.openai_audit import audit_and_sanitize_openai_context
from joker.compliance.opra_sanitizer import (
    RawOpraViolationError,
    assert_no_raw_opra,
    redact_opra_values,
    sanitize_for_openai,
    sanitize_for_persistence,
    snapshot_to_safe_metadata,
)
from joker.compliance.opra_scanner import discover_db_paths, quarantine_opra_artifacts, scan_local_opra
from joker.data.options_capture import capture_options_snapshot
from joker.data.options_normalizer import normalize_webull_option_snapshot, snapshot_to_quote_event
from joker.data.webull_market_provider import WebullMarketDataProvider
from joker.data.webull_api import MockWebullMarketApi, WebullQuote
from joker.config.settings import EnvSettings
from joker.logging.event_log import EventLogWriter
from joker.schemas.options_data import OptionContractMetadata, OptionSnapshot
from joker.schemas.replay import OptionQuoteEvent, SpyQuoteEvent
from joker.storage.database import Database
from joker.storage.models import RiskDecisionRecord
from joker.tui.state import DashboardState
from tests.fixtures.domain import make_candidate
from joker.app.safety import SafetyMode
from joker.runtime.shadow import ShadowRuntime
from joker.broker.interface import PaperBroker
from joker.schemas.domain import RiskDecision


def _raw_snapshot() -> OptionSnapshot:
    contract = OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=date.today(),
        strike=545.0,
        option_type="call",
        contract_id="SPY260706C00545000",
    )
    return normalize_webull_option_snapshot(
        contract,
        {
            "bid": 1.2,
            "ask": 1.3,
            "volume": 100,
            "openInterest": 500,
            "delta": 0.45,
            "quote_time": datetime.now(timezone.utc).isoformat(),
        },
    )


def test_is_opra_source_strict() -> None:
    assert is_opra_source("webull_opra")
    assert is_opra_source("opra")
    assert is_opra_source("us_option")
    assert is_opra_source("webull_option")
    assert not is_opra_source("webull_stock")
    assert not is_opra_source("webull")
    assert not is_opra_source("synthetic_webull_option")


def test_is_stock_source() -> None:
    assert is_stock_source("webull_stock")
    assert is_stock_source("webull")
    assert is_stock_source("synthetic_stock")
    assert not is_stock_source("webull_opra")


def test_classify_market_event_stock_vs_opra() -> None:
    stock = {
        "event_type": "spy_quote",
        "source": SOURCE_WEBULL_STOCK,
        "price": 550.0,
        "bid": 549.9,
        "ask": 550.1,
    }
    assert classify_market_event(stock) == DataClassification.STOCK_MARKET_DATA

    ambiguous = {"source": "webull", "event_type": "spy_quote", "price": 550.0, "bid": 1.0}
    assert classify_market_event(ambiguous) == DataClassification.STOCK_MARKET_DATA

    opra = {
        "event_type": "option_quote",
        "source": SOURCE_WEBULL_OPRA,
        "option_type": "call",
        "strike": 545.0,
        "bid": 1.2,
        "ask": 1.3,
    }
    assert classify_market_event(opra) == DataClassification.RAW_OPRA

    webull_alone_option = {
        "source": "webull",
        "option_type": "call",
        "strike": 545.0,
        "bid": 1.0,
        "ask": 1.1,
    }
    assert classify_market_event(webull_alone_option) == DataClassification.RAW_OPRA


def test_webull_stock_not_sanitized_on_persist() -> None:
    stock_event = {
        "event_type": "market.quote",
        "source": SOURCE_WEBULL_STOCK,
        "data_classification": DataClassification.STOCK_MARKET_DATA.value,
        "price": 550.0,
        "bid": 549.9,
        "ask": 550.1,
    }
    safe = sanitize_for_persistence(stock_event)
    assert safe["bid"] == 549.9
    assert safe["ask"] == 550.1


def test_sanitizer_removes_opra_values() -> None:
    snap = _raw_snapshot()
    payload = snap.model_dump(mode="json")
    safe = sanitize_for_persistence(payload)
    assert_no_raw_opra(safe)
    assert "bid" not in safe
    assert safe.get("spread_check") in ("PASS", "FAIL")
    assert "contract_id" not in safe
    assert safe.get("contract_role") == "ATM_CALL"


def test_sanitizer_preserves_decision_metadata() -> None:
    meta = {
        "spread_check": "PASS",
        "freshness_check": "FAIL",
        "candidate_created": True,
        "risk_reason_code": ["MAX_TRADES"],
        "selected_direction": "long_call",
    }
    safe = sanitize_for_persistence(meta)
    assert safe["spread_check"] == "PASS"
    assert safe["candidate_created"] is True


def test_raw_opra_cannot_be_persisted() -> None:
    with pytest.raises(RawOpraViolationError):
        assert_no_raw_opra(_raw_snapshot().model_dump(mode="json"))


def test_raw_opra_cannot_be_sent_to_openai() -> None:
    snap = _raw_snapshot().model_dump(mode="json")
    safe = sanitize_for_openai(snap)
    assert_no_raw_opra(safe)


def test_openai_may_include_stock_regime_not_opra() -> None:
    context = {
        "features": {"trend_label": "trend_up", "distance_from_vwap_pct": 0.5},
        "source": SOURCE_WEBULL_STOCK,
        "bid": 549.9,
        "ask": 550.1,
    }
    safe, audit = audit_and_sanitize_openai_context(context, prompt_type="MarketRegimeAgent")
    assert safe["features"]["trend_label"] == "trend_up"
    assert safe["bid"] == 549.9
    assert audit.sanitized is True

    opra_context = {"bid": 1.2, "source": SOURCE_WEBULL_OPRA, "option_type": "call", "strike": 545.0}
    safe_opra, audit_opra = audit_and_sanitize_openai_context(
        opra_context, prompt_type="OptionsLiquidityAgent"
    )
    assert audit_opra.raw_opra_detected is True
    assert_no_raw_opra(safe_opra)


def test_synthetic_replay_option_quote_may_persist() -> None:
    event = OptionQuoteEvent(
        timestamp=datetime.now(timezone.utc),
        source="synthetic_option",
        contract_id="SYN",
        expiration=date.today(),
        strike=545.0,
        option_type="call",
        bid=1.0,
        ask=1.1,
        mid=1.05,
        spread_pct=9.5,
        quote_timestamp=datetime.now(timezone.utc),
        is_synthetic=True,
        data_classification=DataClassification.SYNTHETIC_DATA.value,
    )
    payload = event.model_dump(mode="json")
    safe = sanitize_for_persistence(payload)
    assert safe["bid"] == 1.0


def test_webull_quote_event_marked_raw_opra() -> None:
    snap = _raw_snapshot()
    event = snapshot_to_quote_event(snap)
    assert event.source == "webull_opra"
    assert event.data_classification == DataClassification.RAW_OPRA.value
    assert event.persist_allowed is False
    safe = sanitize_for_persistence(event.model_dump(mode="json"))
    assert_no_raw_opra(safe)
    assert "contract_id" not in safe


def test_spy_quote_event_from_webull_stock() -> None:
    env = EnvSettings(
        OPENAI_API_KEY="sk-test",
        OPENAI_MODEL="gpt-test",
        WEBULL_APP_KEY="k",
        WEBULL_APP_SECRET="s",
        WEBULL_MARKET_DATA_ENABLED=True,
    )
    api = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, 549.9, 550.1, datetime.now(timezone.utc))
    )
    provider = WebullMarketDataProvider(env, api=api)
    event = provider.fetch_snapshot_event()
    assert event.source == SOURCE_WEBULL_STOCK
    assert classify_market_event(event.model_dump(mode="json")) == DataClassification.STOCK_MARKET_DATA


def test_event_log_writer_sanitizes_opra(tmp_path: Path) -> None:
    writer = EventLogWriter(tmp_path)
    writer.append(
        run_id="run-1",
        mode="shadow",
        source="webull_opra",
        event_type="option.snapshot",
        payload=_raw_snapshot().model_dump(mode="json"),
    )
    events = writer.read_all("run-1")
    assert_no_raw_opra(events[0]["payload"])


def test_sqlite_save_sanitizes_opra_payload(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    db.initialize()
    record = RiskDecisionRecord(
        run_id="r1",
        candidate_id="c1",
        approved=False,
        reason_codes=["STALE_QUOTE"],
        payload=_raw_snapshot().model_dump(mode="json"),
    )
    saved = db.save(record)
    assert_no_raw_opra(saved.payload)
    assert "contract_id" not in saved.payload


def test_capture_options_snapshot_writes_safe_metadata(tmp_path: Path) -> None:
    snap = _raw_snapshot()

    class FakeProvider:
        verified = True

        def market_today(self) -> date:
            return date.today()

        def discover_contracts(self, symbol: str, exp: date) -> list:
            return [snap.contract]

        def discover_osi_candidates(self, price: float, exp: date) -> list:
            return [snap.contract]

        def select_atm_candidates(self, contracts, price, exp):
            from types import SimpleNamespace

            return SimpleNamespace(atm_call=snap.contract, atm_put=None)

        def fetch_snapshot(self, contract):
            return snap

    class FakeStock:
        def get_snapshot(self, symbol: str):
            from types import SimpleNamespace

            return SimpleNamespace(price=545.0)

    env = EnvSettings()
    path, summary = capture_options_snapshot(
        env,
        captures_dir=tmp_path,
        options_provider=FakeProvider(),  # type: ignore[arg-type]
        stock_api=FakeStock(),
    )
    text = path.read_text()
    assert '"bid": 1.2' not in text
    assert "spread_check" in text
    assert "SPY260706" not in text
    assert summary["underlying_price_available"] is True


def test_tui_ephemeral_vs_persisted_state() -> None:
    state = DashboardState(
        call_bid=1.2,
        call_ask=1.3,
        call_mid=1.25,
        call_spread_pct=8.0,
        option_quote_timestamp="2026-07-01T15:00:00Z",
        selected_call_contract="SPY260706C00545000",
    )
    ephemeral = state.display_state_ephemeral()
    assert ephemeral["call_bid"] == 1.2
    assert ephemeral["selected_call_contract"] == "SPY260706C00545000"
    persisted = state.persisted_state_safe()
    assert "call_bid" not in persisted
    assert "selected_call_contract" not in persisted
    assert persisted["call_contract_selected"] is True
    assert persisted["spread_check"] in ("PASS", "FAIL")


def test_shadow_runtime_persist_metadata_has_no_prices() -> None:
    runtime = ShadowRuntime(mode=SafetyMode.SHADOW)
    candidate = make_candidate()
    record = runtime.record_candidate(
        candidate,
        RiskDecision(candidate_id=candidate.candidate_id, approved=True),
        PaperBroker(),
    )
    runtime.simulate_outcome(record, exit_price=2.0)
    meta = record.persist_metadata()
    assert_no_raw_opra(meta)
    assert meta["shadow_result_label"] == "WIN"
    assert "simulated_pnl" not in meta


def test_redact_opra_values() -> None:
    text = '{"bid": 1.23, "ask": 1.25}'
    redacted = redact_opra_values(text)
    assert "1.23" not in redacted
    assert "REDACTED_OPRA" in redacted


def test_scanner_flags_opra_not_stock(tmp_path: Path) -> None:
    cap_dir = tmp_path / "data" / "captures"
    cap_dir.mkdir(parents=True)
    stock_file = cap_dir / "stock.jsonl"
    stock_file.write_text(
        json.dumps(
            {
                "source": SOURCE_WEBULL_STOCK,
                "event_type": "spy_quote",
                "bid": 549.9,
                "ask": 550.1,
                "data_classification": DataClassification.STOCK_MARKET_DATA.value,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    opra_file = cap_dir / "opra.jsonl"
    opra_file.write_text(
        json.dumps(
            {
                "source": SOURCE_WEBULL_OPRA,
                "event_type": "option.snapshot",
                "bid": 1.2,
                "ask": 1.3,
                "option_type": "call",
                "strike": 545.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = scan_local_opra(root=tmp_path)
    grouped = result.by_category()
    assert len(grouped["possible_raw_opra"]) >= 1
    assert any("opra.jsonl" in f.path for f in grouped["possible_raw_opra"])
    stock_findings = grouped.get("stock_data_not_opra", [])
    assert any("stock.jsonl" in f.path for f in stock_findings) or not result.violations


def test_scanner_ignores_synthetic_option_replay(tmp_path: Path) -> None:
    replay_dir = tmp_path / "data" / "replays"
    replay_dir.mkdir(parents=True)
    path = replay_dir / "synthetic.jsonl"
    path.write_text(
        json.dumps(
            {
                "event_type": "option_quote",
                "source": "synthetic_option",
                "is_synthetic": True,
                "data_classification": DataClassification.SYNTHETIC_DATA.value,
                "bid": 1.0,
                "ask": 1.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = scan_local_opra(root=tmp_path)
    assert not result.violations
    assert any(f.category == "synthetic_ignored" for f in result.findings)


def test_scanner_auto_discovers_sqlite_db(tmp_path: Path) -> None:
    db_path = tmp_path / "data" / "joker.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"")
    discovered = discover_db_paths(tmp_path, tmp_path / "data" / "other.db")
    assert any(p.name == "joker.db" for p in discovered)


def test_quarantine_command_moves_suspicious_files(tmp_path: Path) -> None:
    bad_dir = tmp_path / "data" / "captures"
    bad_dir.mkdir(parents=True)
    bad_file = bad_dir / "webull_violation.jsonl"
    bad_file.write_text(
        json.dumps(
            {
                "source": SOURCE_WEBULL_OPRA,
                "bid": 2.5,
                "event_type": "option.snapshot",
                "option_type": "call",
                "strike": 545.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scan = scan_local_opra(root=tmp_path)
    dest = quarantine_opra_artifacts(scan, root=tmp_path, delete=False)
    assert not bad_file.exists()
    assert (dest / bad_file.name).exists()
    assert (dest / "manifest.json").exists()


def test_secrets_redaction_still_works_in_event_log(tmp_path: Path) -> None:
    writer = EventLogWriter(tmp_path, redact_keys=["api_key"])
    writer.append(
        run_id="run-secret",
        mode="paper",
        source="test",
        event_type="config.loaded",
        payload={"api_key": "sk-testsecretvalue1234567890"},
    )
    events = writer.read_all("run-secret")
    assert events[0]["payload"]["api_key"] == "[REDACTED]"


def test_data_classification_policies() -> None:
    raw_policy = policy_for(DataClassification.RAW_OPRA)
    assert raw_policy.persist_allowed is False
    assert raw_policy.openai_allowed is False
    stock_policy = policy_for(DataClassification.STOCK_MARKET_DATA)
    assert stock_policy.persist_allowed is True
