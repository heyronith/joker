"""Focused tests for one-hour paper goal-test timing, classification, and safety."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from joker.cli.paper_goal_timing import (
    DEFAULT_OBJECTIVE_DURATION_MINUTES,
    DEFAULT_RECONCILIATION_RECOVERY_MINUTES,
    PaperGoalTimingError,
    resolve_reconciliation_only_timing,
    resolve_paper_goal_timing,
)
from joker.objectives.deadline import deadline_from_duration_minutes
from joker.runtime.paper_goal_result import (
    classify_paper_goal,
    contains_secrets,
    redact_mapping,
)
from joker.time.calendar import MarketCalendar


ET = ZoneInfo("America/New_York")


def _regular_now(hour: int = 10, minute: int = 0) -> datetime:
    """Pick a known regular-session weekday (Tuesday 2026-08-04)."""
    return datetime(2026, 8, 4, hour, minute, tzinfo=ET)


def test_objective_duration_defaults_to_60_minutes() -> None:
    timing = resolve_paper_goal_timing(
        objective_duration_minutes=None,
        target_deadline=None,
        duration_minutes=None,
        now=_regular_now(10, 0),
        require_regular_session=True,
    )
    assert timing.objective_duration_minutes == DEFAULT_OBJECTIVE_DURATION_MINUTES
    assert timing.runtime_duration_minutes == DEFAULT_OBJECTIVE_DURATION_MINUTES
    assert timing.objective_source == "default"


def test_objective_deadline_uses_exchange_clock() -> None:
    now = _regular_now(10, 15)
    deadline = deadline_from_duration_minutes(60, exchange_tz="America/New_York", now=now)
    assert deadline.tzinfo is not None
    assert deadline == now + timedelta(minutes=60)
    timing = resolve_paper_goal_timing(
        objective_duration_minutes=60,
        target_deadline=None,
        duration_minutes=None,
        now=now,
    )
    assert timing.objective_deadline == deadline
    assert timing.exchange_now == now


def test_objective_duration_and_absolute_deadline_are_mutually_exclusive() -> None:
    with pytest.raises(PaperGoalTimingError, match="mutually exclusive"):
        resolve_paper_goal_timing(
            objective_duration_minutes=60,
            target_deadline="15:30 ET",
            duration_minutes=60,
            now=_regular_now(10, 0),
        )


def test_runtime_cannot_end_before_objective_deadline() -> None:
    with pytest.raises(PaperGoalTimingError, match="duration-minutes must be >="):
        resolve_paper_goal_timing(
            objective_duration_minutes=60,
            target_deadline=None,
            duration_minutes=45,
            now=_regular_now(10, 0),
        )


def test_objective_window_must_fit_before_market_close() -> None:
    # 15:30 + 60m = 16:30 > 16:00 close
    now = _regular_now(15, 30)
    with pytest.raises(PaperGoalTimingError) as exc_info:
        resolve_paper_goal_timing(
            objective_duration_minutes=60,
            target_deadline=None,
            duration_minutes=None,
            now=now,
        )
    msg = str(exc_info.value)
    assert "does not fit before regular-session" in msg
    assert "current exchange time=" in msg
    assert "requested deadline=" in msg
    assert "regular-session close=" in msg
    assert "maximum valid duration=" in msg
    assert now.isoformat() in msg


def test_duration_and_absolute_deadline_mutually_exclusive() -> None:
    with pytest.raises(PaperGoalTimingError, match="mutually exclusive"):
        resolve_paper_goal_timing(
            objective_duration_minutes=45,
            target_deadline="11:30 ET",
            duration_minutes=None,
            now=_regular_now(10, 0),
        )


def test_absolute_deadline_path_resolves() -> None:
    timing = resolve_paper_goal_timing(
        objective_duration_minutes=None,
        target_deadline="15:30 ET",
        duration_minutes=None,
        now=_regular_now(10, 0),
    )
    assert timing.objective_source == "absolute_deadline"
    assert timing.objective_deadline.hour == 15
    assert timing.objective_deadline.minute == 30
    assert timing.runtime_duration_minutes == timing.objective_duration_minutes


def test_resume_near_close_uses_persisted_shorter_deadline() -> None:
    now = _regular_now(15, 35)
    persisted = now + timedelta(minutes=20)
    timing = resolve_paper_goal_timing(
        objective_duration_minutes=None,
        target_deadline=persisted.isoformat(),
        duration_minutes=None,
        now=now,
    )
    assert timing.objective_deadline == persisted
    assert timing.objective_duration_minutes == 20
    assert timing.runtime_duration_minutes == 20


def test_reconciliation_only_resume_uses_bounded_runtime_not_new_objective_window() -> None:
    now = _regular_now(15, 0)
    original_deadline = now - timedelta(minutes=10)
    timing = resolve_reconciliation_only_timing(
        original_deadline=original_deadline,
        duration_minutes=None,
        now=now,
    )
    assert timing.objective_deadline == original_deadline
    assert timing.objective_duration_minutes == 0
    assert timing.runtime_duration_minutes == DEFAULT_RECONCILIATION_RECOVERY_MINUTES
    assert timing.objective_source == "reconciliation_only"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("objective_duration_minutes", 30.0),
        ("target_deadline", "15:30 ET"),
        ("authorized_capital", 250.0),
        ("target_profit_pct", 10.0),
        ("max_concurrent_positions", 2),
    ],
)
def test_resume_rejects_new_objective_mutation_flags(field, value) -> None:
    from joker.cli.session_confirm import validate_resume_mutation_flags

    kwargs = {
        "objective_duration_minutes": None,
        "target_deadline": None,
        "authorized_capital": None,
        "target_profit_pct": None,
        "max_concurrent_positions": None,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="rejects new-objective mutation flags"):
        validate_resume_mutation_flags(**kwargs)


@pytest.mark.asyncio
async def test_cli_values_create_confirmed_objective(tmp_path: Path) -> None:
    from joker.app.safety import SafetyMode
    from joker.cli.session_confirm import confirm_session_objective
    from joker.config.settings import AppSettings
    from joker.objectives.config import ObjectiveSettings
    from joker.persistence.migrations import apply_task1_migrations

    db = tmp_path / "t1.db"
    apply_task1_migrations(db)
    deadline = _regular_now(11, 0) + timedelta(minutes=60)
    app = AppSettings(
        mode=SafetyMode.PAPER,
        live_trading_enabled=False,
        objective=ObjectiveSettings(enabled=True),
    )
    bundle = await confirm_session_objective(
        app,
        session_id="paper-test-1",
        db_path=db,
        authorized_usd=250.0,
        target_profit_pct=10.0,
        deadline_exchange_time=deadline,
        max_concurrent_positions=1,
        acknowledge_total_loss=True,
        yes=True,
    )
    assert bundle.objective_id
    assert bundle.objective_service is not None
    assert bundle.capital_budget.authorized_usd == 250.0
    assert float(bundle.capital_budget.plan.target_profit_pct) == 10.0
    state = await bundle.objective_service.get_state()
    assert state.status == "active"
    assert state.deadline_exchange_time == deadline


@pytest.mark.asyncio
async def test_interactive_values_create_confirmed_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from joker.app.safety import SafetyMode
    from joker.cli.session_confirm import confirm_session_objective
    from joker.config.settings import AppSettings
    from joker.objectives.config import ObjectiveSettings
    from joker.persistence.migrations import apply_task1_migrations

    db = tmp_path / "t1.db"
    apply_task1_migrations(db)
    deadline = _regular_now(11, 0) + timedelta(minutes=60)
    prompts = iter([100.0, 20.0, 1])
    confirms = iter([True, True])  # total-loss ack, arm objective

    monkeypatch.setattr(
        "joker.cli.session_confirm.typer.prompt",
        lambda *a, **k: next(prompts),
    )
    monkeypatch.setattr(
        "joker.cli.session_confirm.typer.confirm",
        lambda *a, **k: next(confirms),
    )
    app = AppSettings(
        mode=SafetyMode.PAPER,
        live_trading_enabled=False,
        objective=ObjectiveSettings(enabled=True),
    )
    bundle = await confirm_session_objective(
        app,
        session_id="paper-test-interactive",
        db_path=db,
        deadline_exchange_time=deadline,
        yes=False,
    )
    assert bundle.objective_id
    assert bundle.capital_budget.authorized_usd == 100.0


def test_paper_test_requires_webull_paper() -> None:
    from joker.broker.factory import webull_paper_env_ready
    from joker.config.settings import EnvSettings

    env = EnvSettings.model_validate(
        {
            "WEBULL_PAPER_TRADING_ENABLED": False,
            "WEBULL_LIVE_TRADING_ENABLED": False,
        }
    )
    assert webull_paper_env_ready(env) is False


def test_paper_test_never_uses_live_broker() -> None:
    from joker.app.safety import SafetyMode
    from joker.broker.webull_live import WebullLiveClient, WebullLiveConfigError
    from joker.config.settings import AppSettings, EnvSettings

    env = EnvSettings.model_validate({"WEBULL_LIVE_TRADING_ENABLED": False})
    with pytest.raises(WebullLiveConfigError):
        WebullLiveClient(
            env,
            app_settings=AppSettings(
                mode=SafetyMode.PAPER, live_trading_enabled=False
            ),
            skip_account_list_check=True,
            capture_only=True,
        )


@pytest.mark.asyncio
async def test_deadline_blocks_new_entries(tmp_path: Path) -> None:
    from joker.cli.session_confirm import confirm_session_objective
    from joker.app.safety import SafetyMode
    from joker.config.settings import AppSettings
    from joker.objectives.config import ObjectiveSettings
    from joker.objectives.service import ObjectiveServiceError
    from joker.persistence.migrations import apply_task1_migrations

    db = tmp_path / "t1.db"
    apply_task1_migrations(db)
    # Deadline already in the past relative to recompute `now`
    past = _regular_now(10, 0)
    deadline = past + timedelta(minutes=1)
    app = AppSettings(
        mode=SafetyMode.PAPER,
        objective=ObjectiveSettings(enabled=True, stop_new_entries_at_deadline=True),
    )
    bundle = await confirm_session_objective(
        app,
        session_id="paper-deadline-block",
        db_path=db,
        authorized_usd=100,
        target_profit_pct=10,
        deadline_exchange_time=deadline,
        max_concurrent_positions=1,
        acknowledge_total_loss=True,
        yes=True,
    )
    state = await bundle.objective_service.recompute_from_truth(
        now=deadline + timedelta(seconds=5)
    )
    assert state.status == "deadline_reached"
    assert state.entries_paused is True
    with pytest.raises(ObjectiveServiceError, match="deadline"):
        bundle.objective_service._assert_entry_allowed(state)


@pytest.mark.asyncio
async def test_deadline_allows_existing_position_exit(tmp_path: Path) -> None:
    """Deadline pauses entries but does not invent a force-close."""
    from joker.cli.session_confirm import confirm_session_objective
    from joker.app.safety import SafetyMode
    from joker.config.settings import AppSettings
    from joker.objectives.config import ObjectiveSettings
    from joker.persistence.migrations import apply_task1_migrations

    db = tmp_path / "t1.db"
    apply_task1_migrations(db)
    past = _regular_now(10, 0)
    deadline = past + timedelta(minutes=1)
    app = AppSettings(
        mode=SafetyMode.PAPER,
        objective=ObjectiveSettings(enabled=True, stop_new_entries_at_deadline=True),
    )
    bundle = await confirm_session_objective(
        app,
        session_id="paper-deadline-exit",
        db_path=db,
        authorized_usd=100,
        target_profit_pct=10,
        deadline_exchange_time=deadline,
        max_concurrent_positions=1,
        acknowledge_total_loss=True,
        yes=True,
    )
    state = await bundle.objective_service.recompute_from_truth(
        open_position_count=1,
        now=deadline + timedelta(seconds=5),
    )
    assert state.status == "deadline_reached"
    assert state.open_position_count == 1
    # Exit path remains available via normal position management (not blocked here).


def test_goal_achieved_classification() -> None:
    klass, _ = classify_paper_goal(
        ending_realized_pnl_usd=25.0,
        target_profit_usd=20.0,
        open_positions_remaining=0,
        working_orders_remaining=0,
        reconciliation_clean=True,
        deadline_reached=False,
        system_operational=True,
    )
    assert klass == "PAPER_OBJECTIVE_ACHIEVED"


def test_goal_missed_no_trade_classification() -> None:
    klass, reason = classify_paper_goal(
        ending_realized_pnl_usd=0.0,
        target_profit_usd=20.0,
        open_positions_remaining=0,
        working_orders_remaining=0,
        reconciliation_clean=True,
        deadline_reached=True,
        system_operational=True,
    )
    assert klass == "PAPER_OBJECTIVE_MISSED"
    assert "below_target" in reason


def test_inconclusive_on_unresolved_order() -> None:
    klass, reason = classify_paper_goal(
        ending_realized_pnl_usd=0.0,
        target_profit_usd=20.0,
        open_positions_remaining=0,
        working_orders_remaining=1,
        reconciliation_clean=True,
        deadline_reached=True,
        system_operational=True,
    )
    assert klass == "PAPER_OBJECTIVE_INCONCLUSIVE"
    assert "unresolved_working_order" in reason


def test_evidence_contains_no_secrets() -> None:
    payload = redact_mapping(
        {
            "account_id": "SECRETACCOUNT123",
            "paper_account_hash": "40933deecf5db239",
            "access_token": "super-secret-token",
            "nested": {"webull_app_secret": "x", "ok": 1},
        }
    )
    secrets = contains_secrets(payload)
    assert secrets == []
    assert "account_id" not in payload
    assert "access_token" not in payload
    assert payload["paper_account_hash"] == "40933deecf5db239"
    assert payload["nested"] == {"ok": 1}


def test_require_regular_session_rejects_closed() -> None:
    # Sunday
    sunday = datetime(2026, 8, 2, 12, 0, tzinfo=ET)
    with pytest.raises(PaperGoalTimingError, match="regular market session"):
        resolve_paper_goal_timing(
            objective_duration_minutes=60,
            target_deadline=None,
            duration_minutes=None,
            now=sunday,
            require_regular_session=True,
        )


def test_new_objective_fails_closed_when_unfinished_session_exists() -> None:
    from joker.cli.session_confirm import validate_objective_session_action

    with pytest.raises(ValueError, match="unfinished objective"):
        validate_objective_session_action(
            "new", has_definition=True, latest_status="active"
        )


def test_resume_and_new_objective_are_explicitly_distinct() -> None:
    from joker.cli.session_confirm import validate_objective_session_action

    assert (
        validate_objective_session_action(
            "resume", has_definition=True, latest_status="paused"
        )
        == "resume"
    )
    assert (
        validate_objective_session_action(
            "new", has_definition=True, latest_status="target_reached"
        )
        == "new"
    )
    assert (
        validate_objective_session_action(
            "resume",
            has_definition=True,
            latest_status="deadline_reached",
            unresolved_work=True,
        )
        == "reconciliation_only"
    )
    with pytest.raises(ValueError, match="no unfinished objective"):
        validate_objective_session_action(
            "resume", has_definition=False, latest_status=None
        )


@pytest.mark.asyncio
async def test_confirmed_objective_recovers_for_stable_cli_session(tmp_path: Path) -> None:
    from joker.app.safety import SafetyMode
    from joker.cli.session_confirm import (
        confirm_session_objective,
        recover_session_objective_bundle,
    )
    from joker.config.settings import AppSettings
    from joker.objectives.config import ObjectiveSettings
    from joker.persistence.migrations import apply_task1_migrations

    db = tmp_path / "t1.db"
    apply_task1_migrations(db)
    app = AppSettings(
        mode=SafetyMode.PAPER,
        live_trading_enabled=False,
        objective=ObjectiveSettings(enabled=True),
    )
    created = await confirm_session_objective(
        app,
        session_id="cog:paper:local_paper:2026-08-05",
        db_path=db,
        authorized_usd=250.0,
        target_profit_pct=10.0,
        deadline_exchange_time=datetime(2026, 8, 5, 15, 30, tzinfo=ET),
        confirmed_at_exchange_time=datetime(2026, 8, 5, 10, 0, tzinfo=ET),
        max_concurrent_positions=2,
        acknowledge_total_loss=True,
        yes=True,
    )
    recovered = await recover_session_objective_bundle(
        app,
        session_id="cog:paper:local_paper:2026-08-05",
        db_path=db,
    )

    assert recovered is not None
    assert recovered.objective_id == created.objective_id
    assert recovered.capital_budget.authorized_usd == 250.0
    assert recovered.capital_budget.plan.max_concurrent_positions == 2


@pytest.mark.asyncio
async def test_past_deadline_objective_is_recomputed_before_session_action(
    tmp_path: Path,
) -> None:
    from joker.app.safety import SafetyMode
    from joker.cli.session_confirm import (
        confirm_session_objective,
        recover_session_objective_bundle,
        validate_objective_session_action,
    )
    from joker.config.settings import AppSettings
    from joker.objectives.config import ObjectiveSettings
    from joker.objectives.repository import ObjectiveRepository
    from joker.persistence.migrations import apply_task1_migrations

    db = tmp_path / "t1.db"
    apply_task1_migrations(db)
    app = AppSettings(
        mode=SafetyMode.PAPER,
        live_trading_enabled=False,
        objective=ObjectiveSettings(enabled=True),
    )
    session_id = "cog:paper:local_paper:2026-08-04"
    deadline = _regular_now(10, 30)
    await confirm_session_objective(
        app,
        session_id=session_id,
        db_path=db,
        authorized_usd=250.0,
        target_profit_pct=10.0,
        deadline_exchange_time=deadline,
        confirmed_at_exchange_time=_regular_now(10, 0),
        max_concurrent_positions=1,
        acknowledge_total_loss=True,
        yes=True,
    )
    await recover_session_objective_bundle(
        app,
        session_id=session_id,
        db_path=db,
        now=_regular_now(10, 31),
    )
    state = ObjectiveRepository(db).latest_state_for_session(session_id)
    assert state is not None and state.status == "deadline_reached"
    with pytest.raises(ValueError, match="no unfinished objective"):
        validate_objective_session_action(
            "resume", has_definition=True, latest_status=state.status
        )


@pytest.mark.asyncio
async def test_failed_reoptimization_blocks_new_objective_until_resolved(tmp_path) -> None:
    from joker.cli.session_confirm import has_unresolved_portfolio_work
    from joker.persistence.cognitive_execution_provenance import (
        CognitiveExecutionProvenanceRegistry,
        PortfolioExecutionOwner,
        PortfolioReoptimizationRequestRecord,
        PortfolioReoptimizationStatus,
        stable_reoptimization_request_id,
    )

    db = tmp_path / "provenance.db"
    registry = CognitiveExecutionProvenanceRegistry(db)
    await registry.initialize()
    owner = PortfolioExecutionOwner("session-a", "local_paper", "2026-08-04")
    request_id = stable_reoptimization_request_id(
        session_id=owner.session_id,
        broker_account_identity=owner.broker_account_identity,
        trading_date=owner.trading_date,
        original_portfolio_decision_id="decision-old",
        remaining_authorized_tuple_ids=("tuple-old",),
    )
    request = await registry.portfolio_reoptimizations.enqueue(
        PortfolioReoptimizationRequestRecord(
            request_id=request_id,
            session_id=owner.session_id,
            origin_run_id="run-a",
            broker_account_identity=owner.broker_account_identity,
            trading_date=owner.trading_date,
            original_portfolio_decision_id="decision-old",
            already_filled_tuple_ids=("tuple-filled",),
            open_positions=({"contract_id": "contract-open", "quantity": 1},),
            remaining_authorized_tuple_ids=("tuple-old",),
            reason_codes=("capital_changed",),
            latest_objective_state={"version": 2},
            latest_objective_version=2,
            latest_snapshot_id="snapshot-new",
            created_exchange_time=_regular_now(10, 0).isoformat(),
        )
    )
    running = await registry.portfolio_reoptimizations.begin_attempt(
        request.request_id,
        owner=owner,
        current_run_id="run-a",
        attempt_exchange_time=_regular_now(10, 0).isoformat(),
    )
    await registry.portfolio_reoptimizations.fail_attempt(
        attempt=running,
        failed_at=_regular_now(10, 0).isoformat(),
        failure_reason="validation_failed",
    )
    assert await has_unresolved_portfolio_work(
        provenance_db_path=db,
        session_id=owner.session_id,
        broker_account_identity=owner.broker_account_identity,
        trading_date=owner.trading_date,
    )
    await registry.portfolio_reoptimizations.resolve_failed(
        request.request_id,
        resolved_at=_regular_now(10, 1).isoformat(),
        resolved_by="operator:test",
        resolution_reason="reconciled",
    )
    assert not await has_unresolved_portfolio_work(
        provenance_db_path=db,
        session_id=owner.session_id,
        broker_account_identity=owner.broker_account_identity,
        trading_date=owner.trading_date,
    )
