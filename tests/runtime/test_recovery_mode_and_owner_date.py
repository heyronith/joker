"""Explicit recovery-mode propagation and persisted owner-date safety."""

from __future__ import annotations

import inspect
from datetime import date

import pytest

from joker.cli import paper as paper_cli
from joker.runtime.live_paper_runner import LivePaperRunConfig
from joker.runtime.portfolio_owner import (
    PortfolioOwnerDateConflictError,
    resolve_persisted_portfolio_owner,
)
from joker.runtime.recovery_mode import RecoveryMode, recovery_mode_value


def test_cli_paper_passes_explicit_recovery_mode() -> None:
    source = inspect.getsource(paper_cli)
    assert "recovery_mode=recovery_mode," in source
    assert "recovery_owner_trading_date=(" in source


@pytest.mark.parametrize(
    ("recovery_mode", "expected"),
    [
        ("normal", RecoveryMode.NORMAL),
        ("reconciliation_only", RecoveryMode.RECONCILIATION_ONLY),
        ("broker_only", RecoveryMode.BROKER_ONLY),
    ],
)
def test_live_paper_run_config_preserves_explicit_recovery_mode(
    recovery_mode: str,
    expected: RecoveryMode,
) -> None:
    config = LivePaperRunConfig(
        recovery_mode=recovery_mode,
        reconciliation_only_recovery=recovery_mode
        in {"reconciliation_only", "broker_only"},
        recovery_owner_trading_date="2026-08-05",
    )
    assert recovery_mode_value(config) is expected
    assert config.recovery_mode is expected


def test_legacy_boolean_alone_still_maps_to_reconciliation_only() -> None:
    config = LivePaperRunConfig(reconciliation_only_recovery=True)
    assert config.recovery_mode is RecoveryMode.RECONCILIATION_ONLY


def test_broker_only_is_not_silently_converted_to_reconciliation_only() -> None:
    config = LivePaperRunConfig(
        recovery_mode="broker_only",
        reconciliation_only_recovery=True,
    )
    assert config.recovery_mode is RecoveryMode.BROKER_ONLY
    assert recovery_mode_value(config) is RecoveryMode.BROKER_ONLY


def test_persisted_owner_uses_session_embedded_date() -> None:
    owner = resolve_persisted_portfolio_owner(
        session_id="cog:paper:acct:2026-08-05",
        broker_account_identity="paper-acct",
    )
    assert owner.trading_date == "2026-08-05"


def test_persisted_owner_uses_explicit_date_when_session_lacks_date() -> None:
    owner = resolve_persisted_portfolio_owner(
        session_id="legacy-session",
        broker_account_identity="paper-acct",
        explicit_trading_date="2026-07-01",
    )
    assert owner.trading_date == "2026-07-01"


def test_persisted_owner_rejects_conflicting_dates() -> None:
    with pytest.raises(PortfolioOwnerDateConflictError, match="conflict"):
        resolve_persisted_portfolio_owner(
            session_id="cog:paper:acct:2026-08-05",
            broker_account_identity="paper-acct",
            explicit_trading_date=date(2026, 7, 1),
            candidate_trading_date="2026-08-05",
        )


def test_persisted_owner_fails_closed_without_any_persisted_date() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        resolve_persisted_portfolio_owner(
            session_id="legacy-session",
            broker_account_identity="paper-acct",
        )


def test_recovery_runner_does_not_use_clock_trading_date_for_owner() -> None:
    source = inspect.getsource(
        __import__(
            "joker.runtime.live_paper_runner", fromlist=["LivePaperRunner"]
        ).LivePaperRunner._run_reconciliation_only_recovery
    )
    assert "resolve_persisted_portfolio_owner" in source
    assert "supervisor.clock.trading_date()" not in source
