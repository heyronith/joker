"""LIVE_GATED agentic trading runner — same cognition as paper, live broker only."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from joker.app.safety import SafetyMode
from joker.broker.factory import create_live_broker
from joker.broker.reconciliation import BrokerReconciliationService, ReconciliationReport
from joker.broker.webull_live import WebullLiveClient
from joker.config.settings import AppSettings, EnvSettings
from joker.objectives.service import SessionObjectiveService
from joker.persistence.broker_submission_journal import SyncBrokerSubmissionJournal
from joker.runtime.cognitive_session_factory import (
    PreparedTradingSession,
    prepare_cognitive_live_session,
)
from joker.runtime.live_activation import LiveActivation

logger = logging.getLogger(__name__)


class LiveTradingError(Exception):
    pass


@dataclass(frozen=True)
class LiveTradingHealth:
    mode: str
    account_id_hash: str
    broker_connected: bool
    market_data_healthy: bool
    option_surface_healthy: bool
    objective_confirmed: bool
    historical_ev_available: bool
    reconciliation_clean: bool
    unknown_submissions: int
    working_orders: int
    open_positions: int
    entries_permitted: bool
    degraded_reasons: tuple[str, ...]


class LiveTradingRunner:
    """Connect production WebullLiveClient to the shared agentic session."""

    def __init__(
        self,
        *,
        app_settings: AppSettings,
        env: EnvSettings,
        objective_service: SessionObjectiveService,
        activation: LiveActivation,
        trade_api: Any | None = None,
        capture_only: bool = False,
        db_path: Path | None = None,
    ) -> None:
        if app_settings.mode is not SafetyMode.LIVE_GATED:
            raise LiveTradingError("LiveTradingRunner requires mode LIVE_GATED")
        if not app_settings.live_trading_enabled:
            raise LiveTradingError("LiveTradingRunner requires live_trading_enabled")
        if not env.webull_live_trading_enabled:
            raise LiveTradingError(
                "LiveTradingRunner requires WEBULL_LIVE_TRADING_ENABLED=true"
            )
        if not activation.is_active():
            raise LiveTradingError("LiveActivation is expired or inactive")
        if not bool(getattr(app_settings.cognitive_graph, "enabled", True)):
            raise LiveTradingError("LiveTradingRunner requires cognitive graph enabled")
        if not bool(getattr(app_settings.evolution, "enabled", False)):
            raise LiveTradingError("LiveTradingRunner requires evolution.enabled")
        if not bool(getattr(app_settings.objective, "enabled", False)):
            raise LiveTradingError("LiveTradingRunner requires objective.enabled")

        self.app_settings = app_settings
        self.env = env
        self.objective_service = objective_service
        self.activation = activation
        self._trade_api = trade_api
        self._capture_only = capture_only
        self._db_path = Path(db_path or app_settings.db_path)
        self.session: PreparedTradingSession | None = None
        self.last_reconciliation: ReconciliationReport | None = None
        self._entries_permitted = False
        self._degraded_reasons: list[str] = []

    async def start(
        self,
        *,
        fake_model_provider: Any | None = None,
        start_cognitive_agent: bool = True,
        clock: Any | None = None,
    ) -> PreparedTradingSession:
        """Authenticate, reconcile, restore, then permit cognition."""
        state = await self.objective_service.get_state()
        if str(getattr(state, "status", "")) in {
            "pending_confirmation",
            "draft",
        }:
            raise LiveTradingError("LiveTradingRunner requires confirmed objective")

        journal = SyncBrokerSubmissionJournal(self._db_path)
        broker = create_live_broker(
            self.app_settings,
            self.env,
            trade_api=self._trade_api,
            journal_db_path=self._db_path,
            capture_only=self._capture_only,
            skip_account_list_check=self._trade_api is not None,
        )
        if not isinstance(broker, WebullLiveClient):
            raise LiveTradingError("create_live_broker did not return WebullLiveClient")
        if broker.account_id_hash != self.activation.account_id_hash:
            raise LiveTradingError(
                "LiveActivation account_id_hash does not match live broker account"
            )

        # Startup reconciliation before cognitive entry workers.
        truth = broker.get_account_truth()
        svc = BrokerReconciliationService(
            broker=broker,
            journal=journal,
            account_id_hash=broker.account_id_hash,
        )
        report = svc.reconcile(
            local_orders=[],
            local_positions=[],
            account_truth=truth,
        )
        self.last_reconciliation = report
        if report.degraded:
            self._degraded_reasons = [f.kind for f in report.findings]
            # Still start session for position management / exits, but block entries.
            self._entries_permitted = False
            logger.error(
                "live_startup_reconciliation_degraded",
                extra={"findings": self._degraded_reasons},
            )
        else:
            self._entries_permitted = not bool(self.app_settings.risk.kill_switch)

        if self.app_settings.risk.kill_switch:
            self._entries_permitted = False
            self._degraded_reasons.append("kill_switch")

        session = await prepare_cognitive_live_session(
            app_settings=self.app_settings,
            objective_service=self.objective_service,
            broker=broker,
            db_path=self._db_path,
            fake_model_provider=fake_model_provider,
            clock=clock,
            start_cognitive_agent=start_cognitive_agent and self._entries_permitted,
        )
        # Ensure historical-EV dependencies present.
        if session.historical_outcome_service is None:
            raise LiveTradingError(
                "LiveTradingRunner requires historical-EV dependencies"
            )
        if not self._entries_permitted and start_cognitive_agent:
            # Position management may still run via agent when entries blocked.
            await session.bridge.astart_agent()
        self.session = session
        return session

    def health(self) -> LiveTradingHealth:
        broker = self.session.broker if self.session else None
        account_hash = getattr(broker, "account_id_hash", self.activation.account_id_hash)
        working = 0
        positions = 0
        connected = False
        if isinstance(broker, WebullLiveClient):
            try:
                working = len(broker.list_open_orders())
                positions = len(broker.list_positions())
                connected = True
            except Exception:
                connected = False
        report = self.last_reconciliation
        unknown = report.unknown_submissions if report else 0
        clean = bool(report.clean) if report else False
        obj_ok = False
        hist_ok = self.session is not None and self.session.historical_outcome_service is not None
        if self.session is not None:
            # Confirmed objective checked at start; re-check status asynchronously elsewhere.
            obj_ok = True
        return LiveTradingHealth(
            mode="LIVE_GATED",
            account_id_hash=account_hash,
            broker_connected=connected,
            market_data_healthy=True,
            option_surface_healthy=True,
            objective_confirmed=obj_ok,
            historical_ev_available=hist_ok,
            reconciliation_clean=clean,
            unknown_submissions=unknown,
            working_orders=working,
            open_positions=positions,
            entries_permitted=self._entries_permitted and clean,
            degraded_reasons=tuple(self._degraded_reasons),
        )

    async def shutdown(self) -> None:
        if self.session is not None:
            await self.session.shutdown()
            self.session = None
