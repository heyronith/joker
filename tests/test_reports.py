"""Phase 13 report generator tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from joker.reporting.generator import ReportGenerator, compute_metrics
from joker.storage.database import ensure_database
from joker.storage.models import RiskDecisionRecord, TradeCandidateRecord


def test_report_from_fixture_db(tmp_path: Path) -> None:
    db = ensure_database(tmp_path / "test.db")
    run_id = db.create_run(mode="PAPER", trading_day=date(2026, 7, 1)).run_id
    db.save(
        TradeCandidateRecord(
            run_id=run_id,
            candidate_id="c1",
            payload={"symbol": "SPY"},
        )
    )
    db.save(
        RiskDecisionRecord(
            run_id=run_id,
            candidate_id="c1",
            approved=False,
            reason_codes=["WIDE_SPREAD"],
        )
    )
    gen = ReportGenerator(db, tmp_path / "reports")
    path = gen.generate_postmarket(run_id, date(2026, 7, 1))
    assert path.exists()
    content = path.read_text()
    assert "WIDE_SPREAD" in content or "Skipped" in content


def test_no_trades_day_report(tmp_path: Path) -> None:
    db = ensure_database(tmp_path / "test.db")
    run_id = db.create_run(mode="PAPER", trading_day=date(2026, 7, 2)).run_id
    gen = ReportGenerator(db, tmp_path / "reports")
    path = gen.generate_postmarket(run_id, date(2026, 7, 2))
    assert "Trades: 0" in path.read_text()


def test_performance_metrics() -> None:
    metrics = compute_metrics(
        candidates=[1, 2],
        risk_decisions=[
            {"approved": True, "reason_codes": []},
            {"approved": False, "reason_codes": ["NO_STOP"]},
        ],
    )
    assert metrics.candidate_count == 2
    assert metrics.trade_count == 1
    assert metrics.skipped_trades == 1
