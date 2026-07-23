"""Phase 14 end-to-end paper day replay."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from joker.agents.council import AgentCouncil
from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.config.settings import AppSettings
from joker.features.engine import FeatureEngine
from joker.logging.event_log import EventLogWriter
from joker.reporting.generator import ReportGenerator
from joker.risk.governor import RiskGovernor
from joker.runtime.premarket import PremarketWorkflow
from joker.runtime.reactive_engine import ReactiveEngine
from joker.runtime.run_manager import RunManager
from joker.schemas.domain import RiskConfig
from joker.storage.database import ensure_database
from joker.storage.models import RiskDecisionRecord
from tests.fixtures.domain import make_candidate, make_daily_state, make_quote, make_snapshot


def test_full_paper_day_replay(tmp_path: Path) -> None:
    db = ensure_database(tmp_path / "joker.db")
    event_log = EventLogWriter(tmp_path / "logs")
    settings = AppSettings.model_validate(
        {
            "mode": "PAPER",
            "reports_dir": str(tmp_path / "reports"),
            "db_path": str(tmp_path / "joker.db"),
        }
    )
    run_manager = RunManager(db, event_log, settings)
    run_id = run_manager.start_run(trading_day=date(2026, 7, 1))

    features = FeatureEngine(max_age_seconds=999999).compute(make_snapshot())
    premarket = PremarketWorkflow(db, event_log, settings, AgentCouncil())
    playbook = premarket.run(run_id, date(2026, 7, 1), features)
    approved = premarket.approve_playbook(run_id, playbook)

    config = RiskConfig(
        max_daily_loss_usd=settings.risk.max_daily_loss_usd,
        max_trades_per_day=settings.risk.max_trades_per_day,
        max_open_positions=settings.risk.max_open_positions,
        max_premium_usd=settings.risk.max_premium_usd,
        max_spread_pct=settings.risk.max_spread_pct,
        quote_max_age_seconds=300,
    )
    broker = PaperBroker(
        initial_balance=settings.paper.initial_balance_usd,
        slippage_pct=0.0,
    )
    engine = ReactiveEngine(RiskGovernor(config, SafetyMode.PAPER), broker)
    engine.arm_playbook(approved)

    candidate = make_candidate(
        run_id=run_id,
        quote=make_quote(bid=0.50, ask=0.55),
        entry_limit_price=0.55,
    )
    decision = engine.on_signal(candidate, make_daily_state(run_id=run_id))
    db.save(
        RiskDecisionRecord(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            approved=decision.approved,
            reason_codes=decision.reason_codes,
            payload=decision.model_dump(mode="json"),
        )
    )

    run_manager.end_run(run_id)
    report = ReportGenerator(db, tmp_path / "reports").generate_postmarket(
        run_id, date(2026, 7, 1)
    )

    events = event_log.read_all(run_id)
    assert len(events) >= 2
    assert decision.approved is True
    assert report.exists()
    assert (tmp_path / "reports" / "premarket" / "2026-07-01.md").exists()

    # Reproducibility: replay with same inputs yields same risk decision
    gov2 = RiskGovernor(config, SafetyMode.PAPER)
    decision2 = gov2.evaluate(
        make_candidate(
            run_id=run_id,
            quote=make_quote(bid=0.50, ask=0.55),
            entry_limit_price=0.55,
        ),
        make_daily_state(run_id=run_id),
    )
    assert decision2.approved == decision.approved
