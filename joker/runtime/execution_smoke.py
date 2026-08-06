"""Sandbox-only execution smoke — one Task-1-gated place+cancel on Webull paper.

Real-money / prod / uat trade endpoints are rejected before any broker call.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from joker.app.safety import SafetyMode
from joker.broker.factory import (
    BrokerFactoryError,
    BrokerSelection,
    resolve_live_paper_broker,
)
from joker.broker.interface import PaperBroker
from joker.broker.webull import WebullClient
from joker.broker.webull_trade_api import (
    WebullTradeConfigError,
    account_looks_like_live_brokerage,
    validate_webull_paper_trade_env,
)
from joker.config.settings import AppSettings, EnvSettings
from joker.ledger.schemas import LedgerEventType
from joker.runtime.execution_runtime import ExecutionCommand, contract_id_for
from joker.schemas.domain import OptionContract, OrderIntent

logger = logging.getLogger(__name__)

SANDBOX_TRADE_ENV = "sandbox"
MAX_QUOTE_AGE_SECONDS = 10.0
MIN_ASK_USD = 0.10
SMOKE_LIMIT_PRICE = 0.01
CANCEL_POLL_TIMEOUT_SECONDS = 30.0
# Webull sandbox rejects new 0DTE opening orders after 15:40 America/New_York.
ODTE_OPEN_CUTOFF_ET = (15, 40)


class ExecutionSmokeError(RuntimeError):
    """Fail-closed smoke gate or lifecycle failure."""


@dataclass
class ExecutionSmokeResult:
    """Structured, redactable smoke outcome."""

    passed: bool
    broker_kind: str
    trade_api_env: str
    client_order_id: str | None = None
    broker_order_id: str | None = None
    order_status: str | None = None
    cancel_status: str | None = None
    contract_id: str | None = None
    initial_open_orders: int = 0
    initial_positions: int = 0
    final_open_orders: int = 0
    final_positions: int = 0
    fill_detected: bool = False
    flattened: bool = False
    ledger_event_types: list[str] = field(default_factory=list)
    task1_active: bool = False
    task2_healthy: bool = False
    task3_enabled: bool = False
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def redacted_summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "broker_kind": self.broker_kind,
            "trade_api_env": self.trade_api_env,
            "client_order_id": _redact_id(self.client_order_id),
            "broker_order_id": _redact_id(self.broker_order_id),
            "order_status": self.order_status,
            "cancel_status": self.cancel_status,
            "contract_id": self.contract_id,
            "initial_open_orders": self.initial_open_orders,
            "initial_positions": self.initial_positions,
            "final_open_orders": self.final_open_orders,
            "final_positions": self.final_positions,
            "fill_detected": self.fill_detected,
            "flattened": self.flattened,
            "ledger_event_types": list(self.ledger_event_types),
            "task1_active": self.task1_active,
            "task2_healthy": self.task2_healthy,
            "task3_enabled": self.task3_enabled,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


def _redact_id(value: str | None) -> str | None:
    if not value:
        return value
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def require_odte_open_window(*, now_et: datetime | None = None) -> None:
    """Fail closed when Webull will reject new 0DTE opening orders."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    now = now_et or datetime.now(tz=et)
    if now.tzinfo is None:
        now = now.replace(tzinfo=et)
    else:
        now = now.astimezone(et)
    cutoff = now.replace(
        hour=ODTE_OPEN_CUTOFF_ET[0],
        minute=ODTE_OPEN_CUTOFF_ET[1],
        second=0,
        microsecond=0,
    )
    if now >= cutoff:
        raise ExecutionSmokeError(
            "execution-smoke cannot open new SPY 0DTE orders after 15:40 America/New_York "
            f"(now={now.isoformat()}); Webull returns OAUTH_OPENAPI_OPTION_TICKER_NEARLY_EXPIRED"
        )


def effective_trade_api_env(env: EnvSettings) -> str:
    """Return the trade HTTP environment after WEBULL_TRADE_* overlay."""
    trade = env.trade_credentials_env()
    return str(trade.webull_api_env or "").strip().lower()


def require_sandbox_trade_environment(env: EnvSettings) -> str:
    """Refuse prod, uat, or ambiguous trade hosts. Only sandbox is allowed."""
    api_env = effective_trade_api_env(env)
    if api_env != SANDBOX_TRADE_ENV:
        raise ExecutionSmokeError(
            f"execution-smoke requires WEBULL_TRADE_API_ENV=sandbox "
            f"(effective trade env={api_env!r}); prod/uat/ambiguous hosts are prohibited"
        )
    return api_env


def require_non_live_account(env: EnvSettings, *, accounts: list[dict[str, Any]]) -> None:
    """Fail if the configured account classifies as live brokerage."""
    account_id = str(env.webull_paper_account_id or "").strip()
    trade_env = effective_trade_api_env(env)
    matched = False
    for row in accounts:
        if not isinstance(row, dict):
            continue
        if str(row.get("account_id") or "") != account_id:
            continue
        matched = True
        if account_looks_like_live_brokerage(row, api_env=trade_env):
            raise ExecutionSmokeError(
                "configured WEBULL_PAPER_ACCOUNT_ID classifies as a live brokerage account; "
                "execution-smoke refuses live-classified accounts"
            )
    if not matched and trade_env != SANDBOX_TRADE_ENV:
        raise ExecutionSmokeError(
            "configured paper account was not found among broker accounts for classification"
        )


def require_webull_paper_selection(
    app_settings: AppSettings,
    env: EnvSettings,
    *,
    trade_api: object | None = None,
    broker: object | None = None,
) -> BrokerSelection:
    """Resolve broker and assert webull_paper identity with no PaperBroker fallback."""
    require_sandbox_trade_environment(env)
    if app_settings.mode is not SafetyMode.PAPER:
        raise ExecutionSmokeError("execution-smoke requires mode=PAPER")
    if app_settings.live_trading_enabled or env.webull_live_trading_enabled:
        raise ExecutionSmokeError("execution-smoke refuses live trading flags")
    provider = (app_settings.broker.provider or "").strip().lower()
    if provider not in {"webull_paper", "webull"}:
        raise ExecutionSmokeError(
            f"execution-smoke requires broker.provider=webull_paper (got {provider!r})"
        )
    try:
        validate_webull_paper_trade_env(env)
    except WebullTradeConfigError as exc:
        raise ExecutionSmokeError(str(exc)) from exc
    try:
        selection = resolve_live_paper_broker(
            app_settings, env, trade_api=trade_api, broker=broker  # type: ignore[arg-type]
        )
    except BrokerFactoryError as exc:
        raise ExecutionSmokeError(str(exc)) from exc
    if selection.kind != "webull_paper":
        raise ExecutionSmokeError(
            f"resolved broker kind must be webull_paper, got {selection.kind!r}"
        )
    if isinstance(selection.client, PaperBroker):
        raise ExecutionSmokeError("PaperBroker must never be present in execution-smoke")
    if not isinstance(selection.client, WebullClient):
        raise ExecutionSmokeError(
            f"resolved broker must be WebullClient, got {type(selection.client).__name__}"
        )
    return selection


def _quote_age_seconds(snapshot: Any, *, now: datetime) -> float | None:
    ts = getattr(snapshot, "quote_timestamp", None)
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (now - ts).total_seconds())


def select_smoke_contract(
    *,
    call_snap: Any | None,
    put_snap: Any | None,
    now: datetime | None = None,
    max_age_seconds: float = MAX_QUOTE_AGE_SECONDS,
    min_ask: float = MIN_ASK_USD,
) -> tuple[OptionContract, Any]:
    """Pick a fresh SPY 0DTE contract with ask >= min_ask."""
    now = now or datetime.now(timezone.utc)
    candidates: list[tuple[OptionContract, Any, float]] = []
    for snap in (call_snap, put_snap):
        if snap is None:
            continue
        meta = getattr(snap, "contract", None)
        if meta is None:
            continue
        contract_id = getattr(meta, "contract_id", None) or getattr(
            meta, "instrument_id", None
        )
        if not contract_id:
            continue
        bid = getattr(snap, "bid", None)
        ask = getattr(snap, "ask", None)
        if bid is None or ask is None:
            continue
        try:
            ask_f = float(ask)
        except (TypeError, ValueError):
            continue
        if ask_f < min_ask:
            continue
        age = _quote_age_seconds(snap, now=now)
        if age is None or age > max_age_seconds:
            continue
        contract = OptionContract(
            symbol=str(getattr(meta, "underlying_symbol", None) or "SPY"),
            expiration=meta.expiration,
            strike=float(meta.strike),
            option_type=meta.option_type,
            is_0dte=True,
        )
        candidates.append((contract, snap, ask_f))
    if not candidates:
        raise ExecutionSmokeError(
            "no eligible SPY 0DTE option quote (need contract_id, bid/ask, "
            f"ask>={min_ask}, quote age<{max_age_seconds}s)"
        )
    candidates.sort(key=lambda row: row[2])
    return candidates[0][0], candidates[0][1]


def build_smoke_execution_command(
    contract: OptionContract,
    *,
    client_order_id: str | None = None,
    limit_price: float = SMOKE_LIMIT_PRICE,
    quantity: int = 1,
    broker_account_id: str,
) -> ExecutionCommand:
    """Build the ExecutionCommand used by the production Task 1 path."""
    cid = client_order_id or f"smk{uuid4().hex[:29]}"
    if len(cid) > 32:
        raise ExecutionSmokeError("smoke client_order_id must be <= 32 characters")
    intent = OrderIntent(
        intent_id=cid,
        candidate_id="execution-smoke",
        contract=contract,
        side="buy",
        order_type="limit",
        quantity=quantity,
        limit_price=limit_price,
    )
    return ExecutionCommand(
        client_order_id=cid,
        intent=intent,
        broker_account_id=broker_account_id,
    )


class ExecutionSmokeRunner:
    """Orchestrates sandbox place+cancel through CompatibilityLivePaperBridge."""

    def __init__(
        self,
        app_settings: AppSettings,
        env: EnvSettings,
        *,
        require_sandbox: bool = False,
        confirm_place: bool = False,
        trade_api: object | None = None,
        broker: object | None = None,
        options_provider: object | None = None,
        market_provider: object | None = None,
        db_path: Path | None = None,
    ) -> None:
        self.app_settings = app_settings
        self.env = env
        self.require_sandbox = require_sandbox
        self.confirm_place = confirm_place
        self._trade_api = trade_api
        self._broker_override = broker
        self._options_provider = options_provider
        self._market_provider = market_provider
        self._db_path = db_path
        self._http_clients: list[Any] = []

    def _paper_account_identity(self) -> str:
        from joker.runtime.cognitive_session import paper_account_identity

        return paper_account_identity(broker_kind="webull_paper", env=self.env)

    def run(self) -> ExecutionSmokeResult:
        if not self.require_sandbox or not self.confirm_place:
            raise ExecutionSmokeError(
                "execution-smoke requires both --require-sandbox and --confirm-place"
            )

        result = ExecutionSmokeResult(passed=False, broker_kind="", trade_api_env="")
        bridge = None
        evolution = None
        try:
            trade_env = require_sandbox_trade_environment(self.env)
            result.trade_api_env = trade_env
            selection = require_webull_paper_selection(
                self.app_settings,
                self.env,
                trade_api=self._trade_api,
                broker=self._broker_override,
            )
            result.broker_kind = selection.kind
            broker = selection.client
            self._http_clients.append(broker)

            accounts = broker.list_accounts_raw()
            require_non_live_account(self.env, accounts=accounts)

            initial_orders = broker.list_open_orders()
            initial_positions = broker.list_positions()
            result.initial_open_orders = len(initial_orders)
            result.initial_positions = len(initial_positions)
            if initial_orders or initial_positions:
                raise ExecutionSmokeError(
                    f"sandbox must be flat before smoke "
                    f"(open_orders={len(initial_orders)}, positions={len(initial_positions)})"
                )

            bridge, cognitive, evolution = self._start_runtimes(broker)
            result.task1_active = (
                bridge.supervisor.market_runtime is not None
                and bridge.supervisor.execution_runtime is not None
            )
            if not result.task1_active:
                raise ExecutionSmokeError(
                    "Task 1 MarketRuntime/ExecutionRuntime not active"
                )
            health = bridge.run_coro(cognitive.health())
            if getattr(health, "status", "") == "unavailable" and not getattr(
                health, "local_provider_available", False
            ):
                raise ExecutionSmokeError(
                    f"Task 2 cognitive runtime unhealthy: {health.status}"
                )
            result.task2_healthy = True
            result.task3_enabled = bool(
                evolution is not None
                and getattr(evolution, "settings", None) is not None
                and evolution.settings.enabled
                and getattr(evolution, "_prepared", False)
            )
            if not result.task3_enabled:
                raise ExecutionSmokeError(
                    "Task 3 EvolutionRuntime must be enabled and started"
                )

            result.final_open_orders = result.initial_open_orders
            result.final_positions = result.initial_positions
            result.notes.append("pre_place_gates_ok")

            # Fail closed after proving runtimes/health: Webull rejects new 0DTE opens
            # after 15:40 ET (OAUTH_OPENAPI_OPTION_TICKER_NEARLY_EXPIRED).
            require_odte_open_window()

            market_provider = self._market_provider
            if market_provider is None:
                from joker.data.webull_market_provider import WebullMarketDataProvider

                market_provider = WebullMarketDataProvider(self.env)
                self._http_clients.append(market_provider)
            if not market_provider.authenticate():
                raise ExecutionSmokeError("Webull market-data authentication failed")
            snap = market_provider.get_latest_snapshot()
            if snap is None or not getattr(snap, "price", None):
                event = market_provider.fetch_snapshot_event()
                price = float(getattr(event, "price", 0) or 0)
            else:
                price = float(snap.price)
            if price <= 0:
                raise ExecutionSmokeError("unable to obtain SPY underlying price")

            options_provider = self._options_provider
            if options_provider is None:
                from joker.data.webull_options_provider import (
                    create_webull_options_provider,
                )

                options_provider = create_webull_options_provider(
                    self.env, app_settings=self.app_settings
                )
                self._http_clients.append(options_provider)
            options_provider.authenticate()
            call_snap, put_snap = options_provider.fetch_atm_snapshots(price)
            contract, _quote = select_smoke_contract(
                call_snap=call_snap, put_snap=put_snap
            )
            result.contract_id = contract_id_for(contract)

            command = build_smoke_execution_command(
                contract,
                broker_account_id=self._paper_account_identity(),
            )
            result.client_order_id = command.client_order_id

            order = bridge.submit_execution_command(command)
            result.broker_order_id = order.order_id
            result.order_status = order.status
            result.notes.append("order_acknowledged")

            ledger = self._ledger_types(bridge, command.client_order_id)
            result.ledger_event_types = ledger
            if LedgerEventType.ORDER_SUBMISSION_REQUESTED.value not in ledger:
                raise ExecutionSmokeError(
                    "ORDER_SUBMISSION_REQUESTED missing from ledger"
                )
            if LedgerEventType.BROKER_ORDER_ACCEPTED.value not in ledger:
                bridge.run_coro(
                    bridge.supervisor.execution_runtime.poll_order_status(  # type: ignore[union-attr]
                        command.client_order_id
                    )
                )
                ledger = self._ledger_types(bridge, command.client_order_id)
                result.ledger_event_types = ledger
            if LedgerEventType.BROKER_ORDER_ACCEPTED.value not in ledger:
                raise ExecutionSmokeError(
                    "broker acknowledgement not persisted in ledger"
                )

            if order.status in {"filled", "partial"}:
                result.fill_detected = True
                result.errors.append("unexpected fill on smoke order")
                self._flatten_if_needed(bridge, broker, contract, result)
                result.passed = False
                return result

            cancelled = bridge.cancel_order(client_order_id=command.client_order_id)
            result.cancel_status = cancelled.status
            deadline = time.monotonic() + CANCEL_POLL_TIMEOUT_SECONDS
            final = cancelled
            while time.monotonic() < deadline:
                polled = bridge.run_coro(
                    bridge.supervisor.execution_runtime.poll_order_status(  # type: ignore[union-attr]
                        command.client_order_id
                    )
                )
                if polled is not None:
                    final = polled
                    result.cancel_status = polled.status
                    if polled.status in {"cancelled", "rejected"}:
                        break
                    if polled.status in {"filled", "partial"}:
                        result.fill_detected = True
                        break
                time.sleep(0.5)

            ledger = self._ledger_types(bridge, command.client_order_id)
            result.ledger_event_types = ledger

            if result.fill_detected or final.status in {"filled", "partial"}:
                result.fill_detected = True
                result.errors.append("smoke order filled during cancel window")
                self._flatten_if_needed(bridge, broker, contract, result)
                result.passed = False
                return result

            if final.status != "cancelled":
                raise ExecutionSmokeError(
                    f"order did not reach cancelled within "
                    f"{CANCEL_POLL_TIMEOUT_SECONDS}s (status={final.status!r})"
                )
            if LedgerEventType.CANCELLATION.value not in ledger:
                raise ExecutionSmokeError("cancellation ledger event missing")

            report = bridge.run_coro(
                bridge.supervisor.execution_runtime.run_reconciliation()  # type: ignore[union-attr]
            )
            result.notes.append(
                f"reconciliation_consistent={getattr(report, 'is_consistent', None)}"
            )

            final_orders = broker.list_open_orders()
            final_positions = broker.list_positions()
            result.final_open_orders = len(final_orders)
            result.final_positions = len(final_positions)
            if final_orders or final_positions:
                raise ExecutionSmokeError(
                    f"sandbox not flat after cancel "
                    f"(open_orders={len(final_orders)}, positions={len(final_positions)})"
                )

            if any(
                t in result.ledger_event_types
                for t in (
                    LedgerEventType.PARTIAL_FILL.value,
                    LedgerEventType.FINAL_FILL.value,
                )
            ):
                result.fill_detected = True
                raise ExecutionSmokeError("fill ledger events present after smoke")

            result.passed = True
            result.notes.append("smoke_passed")
            return result
        except Exception as exc:
            result.errors.append(str(exc))
            result.passed = False
            logger.exception("execution_smoke_failed")
            return result
        finally:
            self._shutdown(bridge, evolution)

    def _start_runtimes(self, broker: WebullClient) -> tuple[Any, Any, Any]:
        import asyncio as _asyncio

        from joker.cognition.exceptions import CognitiveRuntimeConfigurationError
        from joker.evolution.runtime import EvolutionRuntime
        from joker.graph.context_hydrate import context_assembler_from_settings
        from joker.graph.graph_deps import CognitiveGraphDeps
        from joker.market.data_quality_store import DataQualityRepository
        from joker.market.option_surface import OptionSurfaceRepository
        from joker.market.snapshots import SnapshotRepository
        from joker.models.router import ModelRouter
        from joker.persistence.cognitive_execution_provenance import (
            CognitiveExecutionProvenanceRegistry,
        )
        from joker.runtime.cognitive_agent_runtime import (
            CognitiveAgentRuntime,
            build_default_repositories,
        )
        from joker.runtime.cognitive_binding import bind_cognitive_graph_to_task1
        from joker.runtime.cognitive_session import live_paper_cognitive_session_id
        from joker.runtime.cognitive_startup import validate_cognitive_providers
        from joker.runtime.compatibility import CompatibilityLivePaperBridge

        if not self.app_settings.cognitive_graph.enabled:
            raise ExecutionSmokeError(
                "cognitive_graph.enabled must be true for execution-smoke"
            )
        if not bool(getattr(self.app_settings.evolution, "enabled", False)):
            raise ExecutionSmokeError(
                "evolution.enabled must be true for execution-smoke"
            )

        try:
            startup = _asyncio.run(
                validate_cognitive_providers(
                    self.app_settings.models,
                    mock_agents=bool(self.app_settings.agents.mock_agents),
                )
            )
        except CognitiveRuntimeConfigurationError as exc:
            raise ExecutionSmokeError(str(exc)) from exc

        registry = startup.registry
        task1_db = self._db_path or (
            Path(self.app_settings.db_path).parent / "joker_task1_execution_smoke.db"
        )
        session_id = live_paper_cognitive_session_id(
            broker_kind="webull_paper", env=self.env
        )
        run_id = f"execution-smoke-{uuid4().hex[:12]}"
        model_router = ModelRouter(registry, session_id=session_id, model_call_repo=None)
        repos = build_default_repositories(task1_db)
        model_router.set_model_call_repo(repos["model_call_repo"])
        deps = CognitiveGraphDeps(
            router=model_router,
            config=self.app_settings.cognitive_graph,
            session_id=session_id,
            run_id=run_id,
            broker_account_identity=self._paper_account_identity(),
            context_assembler=context_assembler_from_settings(
                self.app_settings.cognitive_graph
            ),
            snapshot_repo=SnapshotRepository(task1_db),
            option_surface_repo=OptionSurfaceRepository(task1_db),
            data_quality_repo=DataQualityRepository(task1_db),
            db_path=task1_db,
            **repos,
        )
        cognitive = CognitiveAgentRuntime(
            session_id=session_id,
            run_id=run_id,
            router=model_router,
            config=self.app_settings.cognitive_graph,
            graph_deps=deps,
            registry=registry,
            checkpointer_path=task1_db.with_name(task1_db.stem + "_cognitive_ckpt.db"),
        )
        bridge = CompatibilityLivePaperBridge(
            broker=broker,
            db_path=task1_db,
            session_id=session_id,
            run_id=run_id,
            broker_account_id=self._paper_account_identity(),
            broker_account_identity=self._paper_account_identity(),
            agent_runtime=cognitive,
        )
        bridge.start(start_agent=False)
        provenance = CognitiveExecutionProvenanceRegistry(
            task1_db.with_name(task1_db.stem + "_cognitive_provenance.db")
        )
        bridge.run_coro(provenance.initialize())
        bind_cognitive_graph_to_task1(
            deps,
            bridge,
            data_quality_repo=bridge.supervisor.data_quality_repository,
            provenance_registry=provenance,
        )
        evolution = EvolutionRuntime(
            db_path=task1_db,
            settings=self.app_settings.evolution,
            session_id=session_id,
            run_id=run_id,
            event_bus=bridge.supervisor.event_bus,
            execution_runtime=bridge.execution_runtime,
            model_router=model_router,
            cognitive_graph_deps=deps,
        )
        bridge.run_coro(evolution.prepare())
        evolution.subscribe_events()
        cognitive.bind_evolution_runtime(evolution)
        bridge.start_agent()
        bridge.run_coro(evolution.start_workers())
        bridge.run_coro(evolution.resume())
        return bridge, cognitive, evolution

    def _ledger_types(self, bridge: Any, client_order_id: str) -> list[str]:
        store = bridge.supervisor.ledger_store
        if store is None:
            return []
        events = bridge.run_coro(store.get_by_session(bridge.session_id))
        types: list[str] = []
        for event in events:
            if getattr(event, "client_order_id", None) != client_order_id:
                continue
            et = getattr(event, "event_type", None)
            types.append(et.value if hasattr(et, "value") else str(et))
        return types

    def _flatten_if_needed(
        self,
        bridge: Any,
        broker: WebullClient,
        contract: OptionContract,
        result: ExecutionSmokeResult,
    ) -> None:
        positions = broker.list_positions()
        if not positions:
            result.final_open_orders = len(broker.list_open_orders())
            result.final_positions = 0
            return
        flatten_id = f"smoke-flatten-{uuid4().hex}"
        intent = OrderIntent(
            intent_id=flatten_id,
            candidate_id="execution-smoke-flatten",
            contract=contract,
            side="sell",
            order_type="limit",
            quantity=1,
            limit_price=0.01,
        )
        cmd = ExecutionCommand(
            client_order_id=flatten_id,
            intent=intent,
            broker_account_id=self._paper_account_identity(),
        )
        try:
            order = bridge.submit_execution_command(cmd)
            result.notes.append(
                f"flatten_submitted status={order.status} id={_redact_id(order.order_id)}"
            )
            result.flattened = True
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"flatten_failed: {exc}")
        result.final_open_orders = len(broker.list_open_orders())
        result.final_positions = len(broker.list_positions())

    def _shutdown(self, bridge: Any, evolution: Any) -> None:
        if evolution is not None and bridge is not None:
            try:
                shutdown = getattr(evolution, "shutdown", None)
                if callable(shutdown):
                    bridge.run_coro(shutdown())
            except Exception:
                logger.exception("execution_smoke_evolution_shutdown_failed")
        if bridge is not None:
            try:
                bridge.shutdown()
            except Exception:
                logger.exception("execution_smoke_bridge_shutdown_failed")
        for client in list(self._http_clients):
            closer = getattr(client, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.exception("execution_smoke_client_close_failed")
        self._http_clients.clear()
