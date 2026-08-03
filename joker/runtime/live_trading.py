"""LIVE_GATED agentic trading runner — same cognition as paper, live broker only."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from joker.app.safety import SafetyMode
from joker.broker.events import OrderDetailPollingEventSource
from joker.broker.factory import create_live_broker
from joker.broker.reconciliation import BrokerReconciliationService, ReconciliationReport
from joker.broker.webull_live import WebullLiveClient
from joker.config.settings import AppSettings, EnvSettings
from joker.objectives.service import SessionObjectiveService
from joker.persistence.broker_submission_journal import SyncBrokerSubmissionJournal
from joker.runtime.cognitive_session import live_gated_cognitive_session_id
from joker.runtime.cognitive_session_factory import (
    PreparedTradingSession,
    prepare_cognitive_live_session,
)
from joker.runtime.entry_permission import EntryPermissionState
from joker.runtime.live_activation import LiveActivation
from joker.runtime.live_market_data_loop import LiveMarketDataLoop
from joker.schemas.domain import BrokerOrder, Position

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
        stock_api: Any | None = None,
        options_api: Any | None = None,
        capture_only: bool = False,
        db_path: Path | None = None,
        poll_interval_seconds: float = 2.0,
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
        self._stock_api = stock_api
        self._options_api = options_api
        self._capture_only = capture_only
        self._db_path = Path(db_path or app_settings.db_path)
        self._poll_interval_seconds = poll_interval_seconds
        self.session: PreparedTradingSession | None = None
        self.last_reconciliation: ReconciliationReport | None = None
        self.entry_permission = EntryPermissionState(permitted=False, reasons=("startup",))
        self._poller: OrderDetailPollingEventSource | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._market_loop: LiveMarketDataLoop | None = None
        self._market_task: asyncio.Task[None] | None = None
        self._market_stop: asyncio.Event | None = None
        self._market_data_healthy = False
        self._option_surface_healthy = False
        self.session_id: str | None = None

    async def start(
        self,
        *,
        fake_model_provider: Any | None = None,
        start_cognitive_agent: bool = True,
        start_evolution_workers: bool = True,
        start_market_loop: bool | None = None,
        clock: Any | None = None,
        session_id: str | None = None,
    ) -> PreparedTradingSession:
        """Restore Task-1 truth, reconcile, then start agents."""
        state = await self.objective_service.get_state()
        if str(getattr(state, "status", "")) in {
            "pending_confirmation",
            "draft",
        }:
            raise LiveTradingError("LiveTradingRunner requires confirmed objective")

        obj_id = getattr(state, "objective_id", None)
        if obj_id is None or str(obj_id) != str(self.activation.objective_id):
            raise LiveTradingError(
                "LiveActivation.objective_id does not match confirmed objective"
            )
        obj_capital = Decimal(str(getattr(state, "authorised_capital_usd", "0") or "0"))
        if obj_capital != self.activation.authorized_capital_usd:
            raise LiveTradingError(
                "LiveActivation.authorized_capital_usd does not match "
                f"objective authorised_capital_usd ({obj_capital})"
            )

        resolved_session_id = session_id or live_gated_cognitive_session_id(
            account_id_hash=self.activation.account_id_hash,
        )
        self.session_id = resolved_session_id

        journal = SyncBrokerSubmissionJournal(self._db_path)
        broker = create_live_broker(
            self.app_settings,
            self.env,
            trade_api=self._trade_api,
            activation=self.activation,
            journal_db_path=self._db_path,
            capture_only=self._capture_only,
            skip_account_list_check=self._trade_api is not None,
            session_id=resolved_session_id,
            objective_id=str(self.activation.objective_id),
        )
        if not isinstance(broker, WebullLiveClient):
            raise LiveTradingError("create_live_broker did not return WebullLiveClient")
        if broker.account_id_hash != self.activation.account_id_hash:
            raise LiveTradingError(
                "LiveActivation account_id_hash does not match live broker account"
            )

        # Construct shared session with all agent/evolution workers stopped.
        self.entry_permission.block("startup_reconciliation")
        session = await prepare_cognitive_live_session(
            app_settings=self.app_settings,
            objective_service=self.objective_service,
            broker=broker,
            db_path=self._db_path,
            session_id=resolved_session_id,
            fake_model_provider=fake_model_provider,
            clock=clock,
            start_cognitive_agent=False,
            start_evolution_workers=False,
            entry_permission=self.entry_permission,
        )
        if session.historical_outcome_service is None:
            raise LiveTradingError(
                "LiveTradingRunner requires historical-EV dependencies"
            )

        exec_rt = session.bridge.execution_runtime
        await exec_rt.restore_order_mappings()
        projection = await exec_rt.project_session()
        local_orders = _working_orders_from_projection(projection)
        local_positions = _open_positions_from_projection(projection)

        # Load objective reservations/exposure before reconcile.
        await self.objective_service.get_state()

        truth = broker.get_account_truth()
        svc = BrokerReconciliationService(
            broker=broker,
            journal=journal,
            account_id_hash=broker.account_id_hash,
        )
        # First pass against actual persisted projection — never empty lists by default.
        _ = svc.reconcile(
            local_orders=local_orders,
            local_positions=local_positions,
            account_truth=truth,
        )
        resolved = svc.resolve_unknown_submissions()
        for rec in resolved:
            await exec_rt.resolve_submission_unknown(
                rec.client_order_id,
                side=str(rec.side or "buy"),
                quantity=Decimal(rec.quantity or 0) if rec.quantity else None,
                contract_id=rec.contract_id,
            )

        # Append-only Task-1 reconciliation corrections against broker truth.
        task1_report = await exec_rt.run_reconciliation()
        if not task1_report.is_consistent:
            await exec_rt.apply_reconciliation_corrections(task1_report)

        # Second broker↔local reconcile after unknown resolution + corrections.
        projection = await exec_rt.project_session()
        local_orders = _working_orders_from_projection(projection)
        local_positions = _open_positions_from_projection(projection)
        truth = broker.get_account_truth()
        report = svc.reconcile(
            local_orders=local_orders,
            local_positions=local_positions,
            account_truth=truth,
        )
        self.last_reconciliation = report

        reasons: list[str] = []
        if report.degraded or report.entries_blocked:
            reasons.extend(f.kind for f in report.findings)
            self.entry_permission.block(*reasons or ("reconciliation_degraded",))
            logger.error(
                "live_startup_reconciliation_degraded",
                extra={"findings": list(self.entry_permission.reasons)},
            )
        elif bool(self.app_settings.risk.kill_switch):
            self.entry_permission.block("kill_switch")
        else:
            self.entry_permission.allow()

        self._refresh_market_health(session)

        # Start order-detail polling after reconciliation; owned by this session.
        await self._start_order_poller(session, broker)

        run_market = (
            (not self._capture_only)
            if start_market_loop is None
            else bool(start_market_loop)
        )
        if run_market:
            await self._start_market_loop(session)

        # Restore position lifecycles / start workers after truth is restored.
        if start_cognitive_agent:
            # Position management starts even when entries are blocked.
            await session.bridge.astart_agent()
        if start_evolution_workers:
            await session.evolution_runtime.start_workers()
            await session.evolution_runtime.resume()

        self.session = session
        return session

    async def _start_market_loop(self, session: PreparedTradingSession) -> None:
        """Authenticate Webull market providers, warm snapshot, then poll."""
        from joker.runtime.live_market_data_loop import LiveMarketDataError

        loop = LiveMarketDataLoop(
            app_settings=self.app_settings,
            env=self.env,
            stock_api=self._stock_api,
            options_api=self._options_api,
            require_options=False,
            source_label="live_gated",
        )
        try:
            loop.authenticate()
            await loop.awarmup(session.bridge)
            # One immediate poll so health reflects a real observation.
            await loop.apoll_once(session.bridge)
        except LiveMarketDataError as exc:
            loop.close()
            raise LiveTradingError(f"live market-data loop failed: {exc}") from exc
        self._market_loop = loop
        self._market_stop = asyncio.Event()
        self._market_task = asyncio.create_task(
            loop.run(
                session.bridge,
                poll_interval_seconds=self.app_settings.data.quote_poll_interval_seconds,
                stop_event=self._market_stop,
            )
        )
        self._market_data_healthy = loop.observations_received > 0
        self._option_surface_healthy = bool(loop.last_surface_complete)

    async def _start_order_poller(
        self, session: PreparedTradingSession, broker: WebullLiveClient
    ) -> None:
        exec_rt = session.bridge.execution_runtime
        journal = broker.journal

        def _client_ids() -> list[str]:
            if journal is None:
                return list(getattr(exec_rt, "_client_to_broker", {}).keys())
            ids: list[str] = []
            for status in (
                "submission_started",
                "submission_unknown",
                "accepted",
                "partially_filled",
                "previewed",
            ):
                for rec in journal.list_by_status(
                    status, account_id_hash=broker.account_id_hash  # type: ignore[arg-type]
                ):
                    ids.append(rec.client_order_id)
            return list(dict.fromkeys(ids))

        async def _on_event(event: Any) -> None:
            order = broker.get_order(event.client_order_id)
            if order is None:
                return
            update = broker.to_order_update(order)
            await exec_rt.on_broker_update(update, client_order_id=event.client_order_id)

        self._poller = OrderDetailPollingEventSource(
            broker,
            client_order_ids=_client_ids,
            on_event=_on_event,
            poll_interval_seconds=self._poll_interval_seconds,
        )
        self._poll_task = asyncio.create_task(self._poller.start())

    def _refresh_market_health(self, session: PreparedTradingSession) -> None:
        if self._market_loop is not None and self._market_loop.observations_received > 0:
            self._market_data_healthy = True
            self._option_surface_healthy = bool(self._market_loop.last_surface_complete)
            return

        market = getattr(session.bridge, "supervisor", None)
        runtime = getattr(market, "market_runtime", None) if market else None
        if runtime is not None:
            healthy = getattr(runtime, "is_healthy", None)
            if callable(healthy):
                try:
                    self._market_data_healthy = bool(healthy())
                except Exception:
                    self._market_data_healthy = False
            else:
                last = getattr(runtime, "last_snapshot", None) or getattr(
                    runtime, "latest_snapshot", None
                )
                self._market_data_healthy = last is not None
        else:
            self._market_data_healthy = False

        # Option surface: prefer latest persisted snapshot completeness.
        try:
            dq = getattr(market, "data_quality_repository", None) if market else None
            if dq is not None:
                # Presence of a quality repo is not health; require explicit OK signal.
                status = getattr(market, "last_data_quality_status", None)
                if status is not None:
                    self._option_surface_healthy = str(status).lower() in {
                        "ok",
                        "healthy",
                        "usable",
                    }
                else:
                    self._option_surface_healthy = False
            else:
                self._option_surface_healthy = False
        except Exception:
            self._option_surface_healthy = False

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
        if self.session is not None:
            self._refresh_market_health(self.session)
        report = self.last_reconciliation
        unknown = report.unknown_submissions if report else 0
        clean = bool(report.clean) if report else False
        obj_ok = False
        hist_ok = (
            self.session is not None and self.session.historical_outcome_service is not None
        )
        if self.session is not None:
            obj_ok = True
        permitted, reasons = self.entry_permission.as_tuple()
        return LiveTradingHealth(
            mode="LIVE_GATED",
            account_id_hash=account_hash,
            broker_connected=connected,
            market_data_healthy=self._market_data_healthy,
            option_surface_healthy=self._option_surface_healthy,
            objective_confirmed=obj_ok,
            historical_ev_available=hist_ok,
            reconciliation_clean=clean,
            unknown_submissions=unknown,
            working_orders=working,
            open_positions=positions,
            entries_permitted=permitted,
            degraded_reasons=reasons,
        )

    async def shutdown(self) -> None:
        if self._market_stop is not None:
            self._market_stop.set()
        if self._market_task is not None:
            self._market_task.cancel()
            try:
                await asyncio.wait_for(self._market_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            self._market_task = None
        if self._market_loop is not None:
            self._market_loop.close()
            self._market_loop = None
        self._market_stop = None
        if self._poller is not None:
            await self._poller.stop()
            self._poller = None
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await asyncio.wait_for(self._poll_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            self._poll_task = None
        if self.session is not None:
            await self.session.shutdown()
            self.session = None


def _working_orders_from_projection(projection: Any) -> list[BrokerOrder]:
    orders: list[BrokerOrder] = []
    raw = getattr(projection, "orders", None) or {}
    if isinstance(raw, dict):
        values = raw.values()
    else:
        values = list(raw or [])
    for item in values:
        status = str(getattr(item, "status", "") or "").lower()
        if status not in {
            "submitted",
            "accepted",
            "partially_filled",
            "open",
            "pending",
            "working",
        }:
            continue
        # Projection order lifecycles are not BrokerOrder; pass through for id compare.
        orders.append(item)  # type: ignore[arg-type]
    return orders


def _open_positions_from_projection(projection: Any) -> list[Position]:
    positions: list[Position] = []
    raw = getattr(projection, "positions", None) or {}
    if isinstance(raw, dict):
        values = raw.values()
    else:
        values = list(raw or [])
    for item in values:
        qty = int(getattr(item, "quantity", 0) or 0)
        if qty == 0:
            continue
        positions.append(item)  # type: ignore[arg-type]
    return positions
