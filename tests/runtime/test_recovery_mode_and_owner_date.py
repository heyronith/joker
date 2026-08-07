"""Acceptance coverage for explicit recovery-mode and persisted owner-date invariants."""

from __future__ import annotations

import inspect
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from joker.cli import paper as paper_cli
from joker.config.settings import AppSettings, EnvSettings
from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
    PortfolioComponentStatus,
    PortfolioExecutionComponentRecord,
    PortfolioExecutionOwner,
    PortfolioReoptimizationRequestRecord,
    PortfolioReoptimizationStatus,
    stable_reoptimization_request_id,
)
from joker.runtime.live_paper_runner import LivePaperRunConfig, LivePaperRunner
from joker.runtime.portfolio_owner import (
    PortfolioOwnerDateConflictError,
    resolve_persisted_portfolio_owner,
)
from joker.runtime.recovery_mode import RecoveryMode, recovery_mode_value


NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc).isoformat()
PRIOR_DATE = "2026-08-05"


def _config_for_mode(mode: str) -> LivePaperRunConfig:
    return LivePaperRunConfig(
        recovery_mode=mode,
        reconciliation_only_recovery=mode in {"reconciliation_only", "broker_only"},
        recovery_owner_trading_date=PRIOR_DATE,
        cognitive_session_id_override=f"cog:paper:acct:{PRIOR_DATE}",
    )


def test_cli_normal_mode_reaches_runner_as_normal() -> None:
    source = inspect.getsource(paper_cli)
    assert "recovery_mode=recovery_mode," in source
    config = _config_for_mode("normal")
    assert recovery_mode_value(config) is RecoveryMode.NORMAL
    assert config.recovery_mode is RecoveryMode.NORMAL
    assert config.reconciliation_only_recovery is False


def test_cli_reconciliation_only_reaches_runner_unchanged() -> None:
    config = _config_for_mode("reconciliation_only")
    assert recovery_mode_value(config) is RecoveryMode.RECONCILIATION_ONLY
    assert config.recovery_mode is RecoveryMode.RECONCILIATION_ONLY
    assert config.reconciliation_only_recovery is True


def test_cli_broker_only_reaches_runner_unchanged() -> None:
    config = _config_for_mode("broker_only")
    assert recovery_mode_value(config) is RecoveryMode.BROKER_ONLY
    assert config.recovery_mode is RecoveryMode.BROKER_ONLY
    # Compatibility boolean may be true without overwriting broker_only.
    assert config.reconciliation_only_recovery is True


def _webull_recovery_runner(tmp_path: Path) -> tuple[LivePaperRunner, object, EnvSettings]:
    from joker.broker.webull_trade_api import MockWebullTradeApi

    app = AppSettings(db_path=str(tmp_path / "app.db"))
    app = app.model_copy(
        update={
            "broker": app.broker.model_copy(update={"provider": "webull_paper"}),
            "agents": app.agents.model_copy(update={"runtime": "cognitive_graph"}),
        }
    )
    env = EnvSettings(
        _env_file=None,
        OPENAI_API_KEY="test-ci-key-not-real",
        WEBULL_MARKET_DATA_ENABLED=False,
        WEBULL_LIVE_TRADING_ENABLED=False,
        WEBULL_PAPER_TRADING_ENABLED=True,
        WEBULL_PAPER_ACCOUNT_ID="PAPER_ACCT_1",
        WEBULL_APP_KEY="paper-key",
        WEBULL_APP_SECRET="paper-secret",
        WEBULL_ACCESS_TOKEN="paper-token",
    )
    return LivePaperRunner(app, env), MockWebullTradeApi(account_id="PAPER_ACCT_1"), env


def test_broker_only_emits_operator_resolution_required(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    from joker.runtime.cognitive_session import paper_account_identity

    runner, api, env = _webull_recovery_runner(tmp_path)
    account_identity = paper_account_identity(broker_kind="webull_paper", env=env)
    session_id = f"cog:paper:{account_identity}:{PRIOR_DATE}"
    observed: list[tuple[str, dict]] = []

    async def _seed_unresolved() -> None:
        task1_db = Path(runner.app_settings.db_path).parent / "joker_task1.db"
        provenance = CognitiveExecutionProvenanceRegistry(
            task1_db.with_name(task1_db.stem + "_cognitive_provenance.db")
        )
        await provenance.initialize()
        owner = PortfolioExecutionOwner(
            session_id=session_id,
            broker_account_identity=account_identity,
            trading_date=PRIOR_DATE,
        )
        await provenance.portfolio_executions.authorize(
            PortfolioExecutionComponentRecord(
                session_id=owner.session_id,
                origin_run_id="run-a",
                broker_account_identity=owner.broker_account_identity,
                trading_date=owner.trading_date,
                target_portfolio_decision_id="decision-a",
                selected_portfolio_id="portfolio-a",
                authorized_position_tuple_id="tuple-b",
                component_index=1,
                component_count=2,
                strategy_id="strategy-b",
                contract_id="contract-b",
                authorized_quantity=1,
                capital_allocation=Decimal("100"),
                client_order_id="client-b",
                status=PortfolioComponentStatus.AUTHORIZED,
                remaining_quantity=1,
                original_decision_snapshot_id="snapshot-a",
                evaluated_objective_version=1,
                evaluated_timestamp=NOW,
            )
        )

    asyncio.run(_seed_unresolved())

    original_run = LivePaperRunner._run_reconciliation_only_recovery

    def wrapped(self, **kwargs):
        log = kwargs["log"]

        def capturing_log(event: str, payload: dict) -> None:
            observed.append((event, payload))
            log(event, payload)

        kwargs = dict(kwargs)
        kwargs["log"] = capturing_log
        return original_run(self, **kwargs)

    monkeypatch.setattr(LivePaperRunner, "_run_reconciliation_only_recovery", wrapped)

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=0.1,
            mock_agents=True,
            require_options=False,
            recovery_mode="broker_only",
            recovery_owner_trading_date=PRIOR_DATE,
            reconciliation_only_recovery=True,
            trade_api=api,
            cognitive_session_id_override=session_id,
            objective_service=None,
        )
    )
    assert result.feed_health == "RECOVERY_ONLY"
    assert any(name == "broker_only.operator_resolution_required" for name, _ in observed)


def test_broker_only_does_not_bind_or_invent_objective_service(
    tmp_path: Path, monkeypatch
) -> None:
    runner, api, _env = _webull_recovery_runner(tmp_path)
    bound: list[object] = []

    monkeypatch.setattr(
        "joker.runtime.session_supervisor.SessionSupervisor.bind_objective_service",
        lambda self, service: bound.append(service),
    )
    monkeypatch.setattr(
        "joker.runtime.objective_recovery.recover_session_objective",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("broker_only must not invent objective recovery")
        ),
    )
    monkeypatch.setattr(
        "joker.models.router.ModelRouter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("broker_only must not invent objective/model infrastructure")
        ),
    )

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=0.1,
            mock_agents=True,
            require_options=False,
            recovery_mode="broker_only",
            recovery_owner_trading_date=PRIOR_DATE,
            reconciliation_only_recovery=True,
            trade_api=api,
            cognitive_session_id_override=f"cog:paper:webull:test:{PRIOR_DATE}",
            objective_service=None,
        )
    )
    assert result.feed_health == "RECOVERY_ONLY"
    assert bound == []


def test_next_day_recovery_uses_prior_owner_date() -> None:
    owner = resolve_persisted_portfolio_owner(
        session_id=f"cog:paper:acct:{PRIOR_DATE}",
        broker_account_identity="paper-acct",
        explicit_trading_date=PRIOR_DATE,
    )
    assert owner.trading_date == PRIOR_DATE
    assert owner.trading_date != date(2026, 8, 6).isoformat()


def test_post_close_recovery_uses_persisted_owner_date() -> None:
    owner = resolve_persisted_portfolio_owner(
        session_id="legacy-session",
        broker_account_identity="paper-acct",
        explicit_trading_date=PRIOR_DATE,
        candidate_trading_date=PRIOR_DATE,
    )
    assert owner.trading_date == PRIOR_DATE


def test_weekend_recovery_uses_prior_session_owner() -> None:
    owner = resolve_persisted_portfolio_owner(
        session_id=f"cog:paper:acct:{PRIOR_DATE}",
        broker_account_identity="paper-acct",
        candidate_trading_date=PRIOR_DATE,
    )
    assert owner.session_id.endswith(PRIOR_DATE)
    assert owner.trading_date == PRIOR_DATE


def test_owner_date_mismatch_fails_closed() -> None:
    with pytest.raises(PortfolioOwnerDateConflictError, match="conflict"):
        resolve_persisted_portfolio_owner(
            session_id=f"cog:paper:acct:{PRIOR_DATE}",
            broker_account_identity="paper-acct",
            explicit_trading_date="2026-07-01",
            candidate_trading_date=PRIOR_DATE,
        )


@pytest.mark.asyncio
async def test_prior_session_components_are_found_by_runner(tmp_path: Path) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    owner = PortfolioExecutionOwner(
        session_id=f"cog:paper:acct:{PRIOR_DATE}",
        broker_account_identity="paper-acct",
        trading_date=PRIOR_DATE,
    )
    await registry.portfolio_executions.authorize(
        PortfolioExecutionComponentRecord(
            session_id=owner.session_id,
            origin_run_id="run-a",
            broker_account_identity=owner.broker_account_identity,
            trading_date=owner.trading_date,
            target_portfolio_decision_id="decision-a",
            selected_portfolio_id="portfolio-a",
            authorized_position_tuple_id="tuple-b",
            component_index=1,
            component_count=2,
            strategy_id="strategy-b",
            contract_id="contract-b",
            authorized_quantity=1,
            capital_allocation=Decimal("100"),
            client_order_id="client-b",
            status=PortfolioComponentStatus.AUTHORIZED,
            remaining_quantity=1,
            original_decision_snapshot_id="snapshot-a",
            evaluated_objective_version=1,
            evaluated_timestamp=NOW,
        )
    )
    found = await registry.portfolio_executions.list_resumable(
        session_id=owner.session_id,
        broker_account_identity=owner.broker_account_identity,
        trading_date=owner.trading_date,
    )
    # Next-day clock date must not be required to discover prior-session work.
    assert [item.authorized_position_tuple_id for item in found] == ["tuple-b"]
    assert (
        await registry.portfolio_executions.list_resumable(
            session_id=owner.session_id,
            broker_account_identity=owner.broker_account_identity,
            trading_date="2026-08-06",
        )
        == []
    )


@pytest.mark.asyncio
async def test_prior_session_requests_are_found_by_runner(tmp_path: Path) -> None:
    registry = CognitiveExecutionProvenanceRegistry(tmp_path / "prov.db")
    await registry.initialize()
    owner = PortfolioExecutionOwner(
        session_id=f"cog:paper:acct:{PRIOR_DATE}",
        broker_account_identity="paper-acct",
        trading_date=PRIOR_DATE,
    )
    remaining = ("tuple-b",)
    request_id = stable_reoptimization_request_id(
        session_id=owner.session_id,
        broker_account_identity=owner.broker_account_identity,
        trading_date=owner.trading_date,
        original_portfolio_decision_id="decision-a",
        remaining_authorized_tuple_ids=remaining,
    )
    await registry.portfolio_reoptimizations.enqueue(
        PortfolioReoptimizationRequestRecord(
            request_id=request_id,
            session_id=owner.session_id,
            origin_run_id="run-a",
            broker_account_identity=owner.broker_account_identity,
            trading_date=owner.trading_date,
            original_portfolio_decision_id="decision-a",
            already_filled_tuple_ids=("tuple-a",),
            open_positions=(),
            remaining_authorized_tuple_ids=remaining,
            reason_codes=("material_truth_changed",),
            latest_objective_state={"version": 1},
            latest_objective_version=1,
            latest_snapshot_id="snapshot-b",
            created_exchange_time=NOW,
            status=PortfolioReoptimizationStatus.PENDING,
        )
    )
    pending = await registry.portfolio_reoptimizations.list_pending(
        session_id=owner.session_id,
        broker_account_identity=owner.broker_account_identity,
        trading_date=owner.trading_date,
    )
    assert [item.request_id for item in pending] == [request_id]
    assert (
        await registry.portfolio_reoptimizations.list_pending(
            session_id=owner.session_id,
            broker_account_identity=owner.broker_account_identity,
            trading_date="2026-08-06",
        )
        == []
    )
