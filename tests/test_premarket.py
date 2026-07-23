"""Phase 8 premarket workflow tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from joker.config.settings import AppSettings
from joker.features.engine import FeatureEngine
from joker.logging.event_log import EventLogWriter
from joker.runtime.premarket import PremarketWorkflow
from joker.storage.database import ensure_database
from tests.fixtures.domain import make_snapshot


def test_premarket_run_creates_playbook(tmp_path: Path) -> None:
    db = ensure_database(tmp_path / "test.db")
    event_log = EventLogWriter(tmp_path / "logs")
    settings = AppSettings.model_validate(
        {"reports_dir": str(tmp_path / "reports"), "mode": "PAPER"}
    )
    workflow = PremarketWorkflow(db, event_log, settings)
    features = FeatureEngine(max_age_seconds=99999).compute(make_snapshot())
    playbook = workflow.run("run-pre", date(2026, 7, 1), features)
    assert playbook.playbook_id
    report = tmp_path / "reports" / "premarket" / "2026-07-01.md"
    assert report.exists()


def test_approval_state_stored(tmp_path: Path) -> None:
    db = ensure_database(tmp_path / "test.db")
    event_log = EventLogWriter(tmp_path / "logs")
    settings = AppSettings.model_validate({"mode": "PAPER"})
    workflow = PremarketWorkflow(db, event_log, settings)
    features = FeatureEngine(max_age_seconds=99999).compute(make_snapshot())
    pb = workflow.run("run-pre", date.today(), features)
    approved = workflow.approve_playbook("run-pre", pb)
    assert approved.approved is True


def test_unapproved_playbook_cannot_arm() -> None:
    from joker.runtime.reactive_engine import ReactiveEngine, StateMachineError
    from joker.risk.governor import RiskGovernor
    from joker.broker.interface import PaperBroker
    from joker.schemas.domain import Playbook, PlaybookSetup, RiskConfig
    from joker.app.safety import SafetyMode
    from datetime import date

    pb = Playbook(
        trading_day=date.today(),
        title="t",
        summary="s",
        setups=[
            PlaybookSetup(
                name="s",
                direction="long_call",
                stop_rule="x",
                take_profit_rule="y",
            )
        ],
        approved=False,
    )
    engine = ReactiveEngine(
        RiskGovernor(
            RiskConfig(
                max_daily_loss_usd=500,
                max_trades_per_day=3,
                max_open_positions=1,
                max_premium_usd=200,
                max_spread_pct=15,
                quote_max_age_seconds=30,
            ),
            SafetyMode.PAPER,
        ),
        PaperBroker(),
    )
    try:
        engine.arm_playbook(pb)
        raised = False
    except StateMachineError:
        raised = True
    assert raised
