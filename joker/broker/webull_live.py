"""Production Webull live broker — distinct from paper WebullClient.

Never falls back to paper credentials or paper account IDs.
Does not place orders merely because the class exists; construction requires
LIVE_GATED + live flags + exact production account match.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal

from joker.app.safety import SafetyMode
from joker.broker.account_truth import (
    BrokerAccountTruth,
    OrderPreviewTruth,
    decimal_or_none,
    hash_account_id,
    mask_account_id,
    parse_balance_truth,
)
from joker.broker.interface import BrokerClient, BrokerError, BrokerSubmissionUnknown
from joker.broker.order_update import BrokerOrderUpdate
from joker.broker.position_intent import validate_position_intent
from joker.broker.webull_trade_api import (
    MockWebullTradeApi,
    WebullTradeApi,
    build_option_limit_order_payload,
    map_webull_order_status,
    new_client_order_id,
    position_from_webull_row,
)
from joker.config.settings import AppSettings, EnvSettings, LiveWebullCredentials
from joker.config.validation import redact_secrets
from joker.data.webull_errors import WebullApiError
from joker.data.webull_http import WebullHttpClient
from joker.persistence.broker_submission_journal import (
    SyncBrokerSubmissionJournal,
)
from joker.persistence.session_pnl_baseline import (
    SessionPnlBaseline,
    SessionPnlBaselineStore,
)
from joker.runtime.live_activation import LiveActivation
from joker.schemas.domain import BrokerOrder, OptionContract, OrderIntent, Position

logger = logging.getLogger(__name__)


class WebullLiveConfigError(BrokerError):
    """Safe live-broker configuration error (secrets redacted)."""


class HttpWebullLiveTradeApi(WebullTradeApi):
    """HTTP trade API bound exclusively to WEBULL_LIVE_* credentials."""

    def __init__(
        self,
        credentials: LiveWebullCredentials,
        *,
        http: WebullHttpClient | None = None,
    ) -> None:
        if credentials.api_env != "prod":
            raise WebullLiveConfigError(
                "WEBULL_LIVE_API_ENV must be 'prod' for WebullLiveClient"
            )
        if not credentials.live_trading_enabled:
            raise WebullLiveConfigError("WEBULL_LIVE_TRADING_ENABLED must be true")
        missing = credentials.missing_fields()
        if missing:
            raise WebullLiveConfigError(
                "Live credentials required: " + ", ".join(missing)
            )
        self._credentials = credentials
        self._account_id = str(credentials.account_id)
        # Build a minimal EnvSettings-like view for WebullHttpClient without paper fields.
        live_env = EnvSettings(  # type: ignore[call-arg]
            OPENAI_API_KEY="unused-live-http",
            WEBULL_APP_KEY=credentials.app_key,
            WEBULL_APP_SECRET=credentials.app_secret,
            WEBULL_ACCESS_TOKEN=credentials.access_token,
            WEBULL_API_ENV="prod",
            WEBULL_LIVE_TRADING_ENABLED=True,
        )
        self._http = http or WebullHttpClient(live_env)
        if credentials.access_token:
            self._http.set_access_token(credentials.access_token)

    def close(self) -> None:
        close = getattr(self._http, "close", None)
        if callable(close):
            close()

    def _assert_account(self, account_id: str) -> None:
        if account_id != self._account_id:
            raise WebullLiveConfigError(
                "Refusing live broker call: account_id does not match "
                f"configured production account ({mask_account_id(self._account_id)})"
            )

    def list_accounts(self) -> list[dict[str, Any]]:
        payload = self._http.request_json("broker_account_list")
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("accounts", "data", "result"):
                nested = payload.get(key)
                if isinstance(nested, list):
                    return [row for row in nested if isinstance(row, dict)]
        raise WebullApiError("Unexpected account list response shape")

    def get_balance(self, account_id: str) -> dict[str, Any]:
        self._assert_account(account_id)
        payload = self._http.request_json(
            "broker_account_balance", params={"account_id": account_id}
        )
        if isinstance(payload, dict):
            return payload
        raise WebullApiError("Unexpected balance response shape")

    def get_positions(self, account_id: str) -> list[dict[str, Any]]:
        self._assert_account(account_id)
        payload = self._http.request_json(
            "broker_account_positions", params={"account_id": account_id}
        )
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            nested = payload.get("positions") or payload.get("data") or []
            if isinstance(nested, list):
                return [row for row in nested if isinstance(row, dict)]
        raise WebullApiError("Unexpected positions response shape")

    def preview_order(
        self, account_id: str, new_orders: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self._assert_account(account_id)
        body = {"account_id": account_id, "new_orders": new_orders}
        payload = self._http.request_json("broker_order_preview", json_body=body)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
        raise WebullApiError("Unexpected preview-order response shape")

    def place_order(self, account_id: str, new_orders: list[dict[str, Any]]) -> dict[str, Any]:
        self._assert_account(account_id)
        body = {"account_id": account_id, "new_orders": new_orders}
        payload = self._http.request_json("broker_order_place", json_body=body)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
        raise WebullApiError("Unexpected place-order response shape")

    def cancel_order(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        self._assert_account(account_id)
        body = {"account_id": account_id, "client_order_id": client_order_id}
        payload = self._http.request_json("broker_order_cancel", json_body=body)
        if isinstance(payload, dict):
            return payload
        return {"client_order_id": client_order_id, "raw": payload}

    def get_order_detail(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        self._assert_account(account_id)
        payload = self._http.request_json(
            "broker_order_detail",
            params={"account_id": account_id, "client_order_id": client_order_id},
        )
        if isinstance(payload, dict):
            return payload
        raise WebullApiError("Unexpected order detail response shape")

    def list_open_orders(
        self, account_id: str, *, page_size: int = 20
    ) -> list[dict[str, Any]]:
        self._assert_account(account_id)
        payload = self._http.request_json(
            "broker_open_orders",
            params={"account_id": account_id, "page_size": str(page_size)},
        )
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("orders", "data", "result", "open_orders"):
                nested = payload.get(key)
                if isinstance(nested, list):
                    return [row for row in nested if isinstance(row, dict)]
        return []


def validate_live_broker_startup(
    *,
    app_settings: AppSettings,
    env: EnvSettings,
    trade_api: WebullTradeApi | None = None,
) -> LiveWebullCredentials:
    """Fail closed unless LIVE_GATED + live flags + exact account match."""
    if app_settings.mode is not SafetyMode.LIVE_GATED:
        raise WebullLiveConfigError(
            "WebullLiveClient requires mode LIVE_GATED"
        )
    if not app_settings.live_trading_enabled:
        raise WebullLiveConfigError(
            "WebullLiveClient requires app_settings.live_trading_enabled=true"
        )
    credentials = env.live_credentials_env()
    if not credentials.live_trading_enabled:
        raise WebullLiveConfigError("WEBULL_LIVE_TRADING_ENABLED must be true")
    if credentials.api_env != "prod":
        raise WebullLiveConfigError(
            "WEBULL_LIVE_API_ENV must be 'prod' (sandbox/uat blocked for live client)"
        )
    missing = credentials.missing_fields()
    if missing:
        raise WebullLiveConfigError(
            "Live credentials required: " + ", ".join(missing)
        )
    # Refuse paper account id reuse.
    paper_id = (env.webull_paper_account_id or "").strip()
    live_id = str(credentials.account_id)
    if paper_id and paper_id == live_id:
        raise WebullLiveConfigError(
            "WEBULL_LIVE_ACCOUNT_ID must not equal WEBULL_PAPER_ACCOUNT_ID"
        )
    api = trade_api or HttpWebullLiveTradeApi(credentials)
    try:
        accounts = api.list_accounts()
    except Exception as exc:
        raise WebullLiveConfigError(
            redact_secrets(f"Live account list failed: {exc}", env=env)
        ) from exc
    matched = any(str(row.get("account_id", "")) == live_id for row in accounts)
    if not matched:
        raise WebullLiveConfigError(
            "Configured WEBULL_LIVE_ACCOUNT_ID not returned by account-list API "
            f"({mask_account_id(live_id)}). Refusing automatic account selection."
        )
    return credentials


class WebullLiveClient(BrokerClient):
    """Production Webull broker adapter."""

    provider: Literal["webull_live"] = "webull_live"

    def __init__(
        self,
        env: EnvSettings,
        *,
        app_settings: AppSettings,
        activation: LiveActivation | None = None,
        journal: SyncBrokerSubmissionJournal | None = None,
        trade_api: WebullTradeApi | None = None,
        capture_only: bool = False,
        skip_account_list_check: bool = False,
        session_id: str | None = None,
        baseline_store: SessionPnlBaselineStore | None = None,
        objective_id: str | None = None,
    ) -> None:
        if app_settings.mode is not SafetyMode.LIVE_GATED:
            raise WebullLiveConfigError(
                "WebullLiveClient requires mode LIVE_GATED"
            )
        if not app_settings.live_trading_enabled:
            raise WebullLiveConfigError(
                "WebullLiveClient requires app_settings.live_trading_enabled=true"
            )
        credentials = env.live_credentials_env()
        if not capture_only:
            if activation is None:
                raise WebullLiveConfigError(
                    "WebullLiveClient requires LiveActivation for placement"
                )
            if journal is None:
                raise WebullLiveConfigError(
                    "WebullLiveClient requires durable submission journal for placement"
                )
            if not activation.is_active():
                raise WebullLiveConfigError("LiveActivation is expired or inactive")
        if not skip_account_list_check:
            validate_live_broker_startup(
                app_settings=app_settings, env=env, trade_api=trade_api
            )
        else:
            if credentials.api_env != "prod":
                raise WebullLiveConfigError("WEBULL_LIVE_API_ENV must be 'prod'")
            if not credentials.live_trading_enabled:
                raise WebullLiveConfigError(
                    "WEBULL_LIVE_TRADING_ENABLED must be true"
                )
            missing = credentials.missing_fields()
            if missing:
                raise WebullLiveConfigError(
                    "Live credentials required: " + ", ".join(missing)
                )
        self._env = env
        self._app = app_settings
        self._credentials = credentials
        self._account_id = str(credentials.account_id)
        self._account_id_hash = hash_account_id(self._account_id)
        if activation is not None:
            if activation.account_id_hash != self._account_id_hash:
                raise WebullLiveConfigError(
                    "LiveActivation account_id_hash does not match production account"
                )
            if objective_id is not None and str(activation.objective_id) != str(
                objective_id
            ):
                raise WebullLiveConfigError(
                    "LiveActivation objective_id mismatch"
                )
        self._activation = activation
        self._api = trade_api or HttpWebullLiveTradeApi(credentials)
        self._journal = journal
        self._capture_only = capture_only
        self._captured_payloads: list[dict[str, Any]] = []
        self._orders: dict[str, BrokerOrder] = {}
        self._session_id = session_id or "live-session"
        self._baseline_store = baseline_store
        self._truth_unavailable = False

    @property
    def account_id_hash(self) -> str:
        return self._account_id_hash

    @property
    def journal(self) -> SyncBrokerSubmissionJournal | None:
        return self._journal

    @property
    def activation(self) -> LiveActivation | None:
        return self._activation

    @property
    def captured_payloads(self) -> list[dict[str, Any]]:
        return list(self._captured_payloads)

    def close(self) -> None:
        close = getattr(self._api, "close", None)
        if callable(close):
            close()

    def _assert_armed(self) -> None:
        if self._credentials.api_env != "prod":
            raise WebullLiveConfigError("Live API environment must remain prod")
        if not self._app.live_trading_enabled or not self._env.webull_live_trading_enabled:
            raise WebullLiveConfigError("Live trading flags must remain enabled")
        if not self._capture_only:
            if self._activation is None or not self._activation.is_active():
                raise WebullLiveConfigError(
                    "LiveActivation required and must be active for placement"
                )
            if self._journal is None:
                raise WebullLiveConfigError(
                    "Durable journal required for live placement"
                )

    def build_payload(
        self,
        intent: OrderIntent,
        *,
        client_order_id: str | None = None,
        open_positions: tuple[Position, ...] | list[Position] = (),
    ) -> dict[str, Any]:
        cid = client_order_id or intent.intent_id or new_client_order_id()
        if len(cid) > 32:
            cid = hashlib.sha256(cid.encode("utf-8")).hexdigest()[:32]
        contract_id = (
            f"{intent.contract.symbol}:{intent.contract.expiration.isoformat()}:"
            f"{intent.contract.strike}:{intent.contract.option_type}"
        )
        validated = validate_position_intent(
            intent.position_intent,
            side=intent.side,
            open_positions=open_positions,
            contract_id=contract_id,
        )
        # Ensure intent carries validated value for payload builder.
        if intent.position_intent != validated:
            intent = intent.model_copy(update={"position_intent": validated})
        return build_option_limit_order_payload(
            intent, client_order_id=cid, account_id=self._account_id
        )

    def preview_order(
        self,
        intent: OrderIntent,
        *,
        client_order_id: str | None = None,
        open_positions: tuple[Position, ...] | list[Position] = (),
        expected_notional_usd: Decimal | None = None,
        max_notional_mismatch_pct: Decimal = Decimal("5"),
    ) -> OrderPreviewTruth:
        self._assert_armed()
        payload = self.build_payload(
            intent, client_order_id=client_order_id, open_positions=open_positions
        )
        try:
            raw = self._api.preview_order(self._account_id, [payload])
        except Exception as exc:
            raise BrokerError(
                redact_secrets(f"Webull live preview failed: {exc}", env=self._env)
            ) from exc
        raw_hash = hashlib.sha256(
            repr(sorted(raw.items())).encode("utf-8")
        ).hexdigest()
        accepted = bool(raw.get("accepted", True))
        reject_code = (
            str(raw.get("reject_code") or raw.get("code") or "") or None
        )
        reject_msg = (
            str(raw.get("reject_message") or raw.get("message") or "") or None
        )
        if reject_code and str(reject_code).upper() not in {"0", "OK", "SUCCESS"}:
            accepted = False
        if raw.get("accepted") is False:
            accepted = False
        cost = decimal_or_none(
            raw.get("estimated_cost")
            or raw.get("estimated_cost_usd")
            or raw.get("cost")
        )
        fees = decimal_or_none(
            raw.get("estimated_fees") or raw.get("fees") or raw.get("commission")
        )
        bp_effect = decimal_or_none(
            raw.get("buying_power_effect") or raw.get("bp_effect")
        )
        if (
            accepted
            and expected_notional_usd is not None
            and cost is not None
            and expected_notional_usd > 0
        ):
            delta = abs(cost - expected_notional_usd)
            pct = (delta / expected_notional_usd) * Decimal("100")
            if pct > max_notional_mismatch_pct:
                accepted = False
                reject_code = reject_code or "preview_cost_mismatch"
                reject_msg = reject_msg or "material unexplained notional mismatch"
        return OrderPreviewTruth(
            accepted=accepted,
            estimated_cost_usd=cost,
            estimated_fees_usd=fees,
            buying_power_effect_usd=bp_effect,
            rejection_code=None if accepted else (reject_code or "preview_rejected"),
            rejection_message=None if accepted else (reject_msg or "preview rejected"),
            raw_response_hash=raw_hash,
        )

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        self._assert_armed()
        positions = self.list_positions()
        payload = self.build_payload(intent, open_positions=positions)
        cid = str(payload["client_order_id"])
        self._captured_payloads.append(dict(payload))
        if self._capture_only:
            order = BrokerOrder(
                order_id=cid,
                intent_id=intent.intent_id,
                status="pending",
                contract=intent.contract,
                side=intent.side,
                quantity=intent.quantity,
                limit_price=intent.limit_price,
                position_intent=intent.position_intent,
                remaining_quantity=intent.quantity,
            )
            self._orders[cid] = order
            return order

        # Journal prepare is owned by gateway before preview; require existing row.
        assert self._journal is not None
        existing = self._journal.get(self._account_id_hash, cid)
        if existing is None:
            raise BrokerError(
                f"missing journal row before placement for {cid} — fail closed"
            )
        self._journal_transition(cid, "submission_started")
        try:
            response = self._api.place_order(self._account_id, [payload])
        except TimeoutError as exc:
            self._journal_transition(
                cid, "submission_unknown", last_error_code="timeout"
            )
            reconciled = self._reconcile_unknown(cid, intent)
            if reconciled is not None:
                return reconciled
            raise BrokerSubmissionUnknown(cid) from exc
        except Exception as exc:
            if _looks_ambiguous(exc):
                self._journal_transition(
                    cid,
                    "submission_unknown",
                    last_error_code=type(exc).__name__,
                )
                reconciled = self._reconcile_unknown(cid, intent)
                if reconciled is not None:
                    return reconciled
                raise BrokerSubmissionUnknown(
                    cid,
                    redact_secrets(f"Live submission unknown: {exc}", env=self._env),
                ) from exc
            self._journal_transition(
                cid, "rejected", last_error_code=type(exc).__name__
            )
            raise BrokerError(
                redact_secrets(f"Webull live place failed: {exc}", env=self._env)
            ) from exc

        status = map_webull_order_status(
            str(response.get("status") or response.get("order_status") or "SUBMITTED")
        )
        broker_order_id = str(
            response.get("order_id") or response.get("broker_order_id") or cid
        )
        journal_status = _journal_status_for(status)
        self._journal_transition(
            cid, journal_status, broker_order_id=broker_order_id
        )
        order = self._order_from_detail_or_intent(
            cid, response, intent=intent, status=status, broker_order_id=broker_order_id
        )
        self._orders[cid] = order
        return order

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self._assert_armed()
        self._journal_transition(order_id, "cancel_pending")
        try:
            self._api.cancel_order(self._account_id, order_id)
        except Exception as exc:
            raise BrokerError(
                redact_secrets(f"Webull live cancel failed: {exc}", env=self._env)
            ) from exc
        self._journal_transition(order_id, "cancelled")
        order = self.get_order(order_id)
        if order is None:
            raise BrokerError(f"Unknown order_id after cancel: {order_id}")
        cancelled = order.model_copy(update={"status": "cancelled"})
        self._orders[order_id] = cancelled
        return cancelled

    def get_order(self, order_id: str) -> BrokerOrder | None:
        """Reconstruct from broker detail + journal — never silent local-only truth."""
        try:
            detail = self._api.get_order_detail(self._account_id, order_id)
        except Exception:
            self._truth_unavailable = True
            return None
        self._truth_unavailable = False
        return self._reconstruct_order(order_id, detail)

    def list_open_orders(self) -> list[BrokerOrder]:
        try:
            rows = self._api.list_open_orders(self._account_id)
        except Exception as exc:
            self._truth_unavailable = True
            raise BrokerError(
                redact_secrets(
                    f"Live open-orders unavailable: {exc}", env=self._env
                )
            ) from exc
        self._truth_unavailable = False
        result: list[BrokerOrder] = []
        for row in rows:
            cid = str(row.get("client_order_id") or "")
            if not cid:
                continue
            order = self._reconstruct_order(cid, row)
            if order is not None and order.status in {
                "open",
                "pending",
                "partially_filled",
            }:
                result.append(order)
        return result

    def to_order_update(self, order: BrokerOrder) -> BrokerOrderUpdate:
        filled = int(order.filled_quantity or 0)
        remaining = (
            order.remaining_quantity
            if order.remaining_quantity is not None
            else max(0, order.quantity - filled)
        )
        return BrokerOrderUpdate(
            client_order_id=order.order_id,
            broker_order_id=order.order_id,
            status=order.status,  # type: ignore[arg-type]
            quantity=order.quantity,
            cumulative_filled_quantity=filled,
            remaining_quantity=int(remaining),
            average_fill_price=(
                Decimal(str(order.average_fill_price))
                if order.average_fill_price is not None
                else None
            ),
            limit_price=(
                Decimal(str(order.limit_price))
                if order.limit_price is not None
                else None
            ),
            side=order.side,
            position_intent=order.position_intent,
        )

    def list_positions(self) -> list[Position]:
        try:
            rows = self._api.get_positions(self._account_id)
        except Exception as exc:
            raise BrokerError(
                redact_secrets(f"Webull live positions failed: {exc}", env=self._env)
            ) from exc
        positions: list[Position] = []
        for row in rows:
            parsed = position_from_webull_row(row)
            if parsed is None:
                continue
            contract, qty, avg = parsed
            positions.append(
                Position(
                    position_id=str(row.get("position_id") or new_client_order_id()),
                    contract=contract,
                    quantity=qty,
                    avg_entry_price=avg,
                )
            )
        return positions

    def get_account_balance(self) -> float:
        truth = self.get_account_truth()
        if truth.cash_usd is None:
            raise BrokerError("Live cash balance unavailable — refusing fabricated zero")
        return float(truth.cash_usd)

    def get_daily_pnl(self) -> float:
        available, value = self.get_daily_pnl_available()
        if not available or value is None:
            raise BrokerError("Live session PnL unavailable — refusing fabricated zero")
        return float(value)

    def get_daily_pnl_available(self) -> tuple[bool, float | None]:
        truth = self.get_account_truth()
        if not truth.session_pnl_available or truth.session_pnl_usd is None:
            return False, None
        return True, float(truth.session_pnl_usd)

    def get_account_truth(self) -> BrokerAccountTruth:
        try:
            balance = self._api.get_balance(self._account_id)
        except Exception as exc:
            raise BrokerError(
                redact_secrets(f"Webull live balance failed: {exc}", env=self._env)
            ) from exc
        parsed = parse_balance_truth(balance)
        nlv = parsed["net_liquidation_value_usd"]
        cash = parsed["cash_usd"]
        session_pnl: Decimal | None = None
        session_available = False
        explicit = decimal_or_none(
            balance.get("day_pnl")
            or balance.get("today_pnl")
            or balance.get("session_pnl")
        )
        if explicit is not None:
            session_pnl = explicit
            session_available = True
        else:
            baseline = self._load_or_create_baseline(nlv=nlv, cash=cash)
            if baseline is not None and baseline.starting_nlv is not None and nlv is not None:
                session_pnl = nlv - baseline.starting_nlv
                if baseline.external_cash_adjustment is not None:
                    session_pnl -= baseline.external_cash_adjustment
                session_available = True
            else:
                # Unavailable — never fabricate zero.
                session_pnl = None
                session_available = False
        return BrokerAccountTruth(
            account_id_hash=self._account_id_hash,
            cash_usd=cash,
            buying_power_usd=parsed["buying_power_usd"],
            net_liquidation_value_usd=nlv,
            session_pnl_usd=session_pnl,
            session_pnl_available=session_available,
            positions=tuple(self.list_positions()),
            working_orders=tuple(self.list_open_orders()),
            captured_at=datetime.now(timezone.utc),
        )

    def _load_or_create_baseline(
        self, *, nlv: Decimal | None, cash: Decimal | None
    ) -> SessionPnlBaseline | None:
        if self._baseline_store is None or nlv is None:
            return None
        from joker.runtime.cognitive_session import exchange_trading_date

        trading_date = exchange_trading_date().isoformat()
        existing = self._baseline_store.get(
            account_id_hash=self._account_id_hash,
            trading_date=trading_date,
            session_id=self._session_id,
        )
        if existing is not None:
            return existing
        baseline = SessionPnlBaseline(
            account_id_hash=self._account_id_hash,
            trading_date=trading_date,
            session_id=self._session_id,
            starting_nlv=nlv,
            starting_cash=cash,
            captured_at=datetime.now(timezone.utc),
        )
        self._baseline_store.put(baseline)
        return baseline

    def _reconcile_unknown(
        self, client_order_id: str, intent: OrderIntent
    ) -> BrokerOrder | None:
        try:
            detail = self._api.get_order_detail(self._account_id, client_order_id)
        except Exception:
            return None
        status = map_webull_order_status(
            str(detail.get("status") or detail.get("order_status") or "")
        )
        broker_order_id = str(detail.get("order_id") or client_order_id)
        self._journal_transition(
            client_order_id,
            _journal_status_for(status),
            broker_order_id=broker_order_id,
        )
        order = self._order_from_detail_or_intent(
            client_order_id,
            detail,
            intent=intent,
            status=status,
            broker_order_id=broker_order_id,
        )
        self._orders[client_order_id] = order
        return order

    def _reconstruct_order(
        self, client_order_id: str, detail: dict[str, Any]
    ) -> BrokerOrder | None:
        journal_row = None
        if self._journal is not None:
            journal_row = self._journal.get(self._account_id_hash, client_order_id)
        status = map_webull_order_status(
            str(detail.get("status") or detail.get("order_status") or "")
        )
        contract = _contract_from_detail(detail)
        if contract is None and journal_row is not None and journal_row.contract_id:
            contract = _contract_from_contract_id(journal_row.contract_id)
        if contract is None:
            return None
        side_raw = str(
            detail.get("side")
            or (journal_row.side if journal_row else "")
            or "buy"
        ).lower()
        side = "sell" if side_raw.startswith("sell") else "buy"
        qty = int(
            float(
                detail.get("quantity")
                or (journal_row.quantity if journal_row else 0)
                or 0
            )
        )
        raw_filled = detail.get("filled_quantity")
        if raw_filled is None:
            raw_filled = detail.get("cumulative_filled_quantity")
        if raw_filled is None and status == "filled":
            raw_filled = qty
        if status == "partially_filled" and raw_filled is None:
            # Never invent fill quantity — leave fill truth unavailable.
            self._truth_unavailable = True
            logger.error(
                "partial_fill_quantity_unavailable",
                extra={"client_order_id": client_order_id},
            )
            return None
        filled = int(float(raw_filled or 0))
        if status == "partially_filled" and filled <= 0:
            self._truth_unavailable = True
            logger.error(
                "partial_fill_quantity_non_positive",
                extra={"client_order_id": client_order_id, "filled": filled},
            )
            return None
        remaining = max(0, qty - filled)
        limit = detail.get("limit_price")
        if limit is None and journal_row is not None:
            limit = journal_row.limit_price
        avg = detail.get("avg_filled_price") or detail.get("average_fill_price")
        position_intent = None
        if journal_row is not None:
            position_intent = journal_row.position_intent
        position_intent = position_intent or detail.get("position_intent")
        order = BrokerOrder(
            order_id=client_order_id,
            intent_id=client_order_id,
            status=status,  # type: ignore[arg-type]
            contract=contract,
            side=side,  # type: ignore[arg-type]
            quantity=qty,
            limit_price=float(limit) if limit is not None else None,
            filled_quantity=filled,
            remaining_quantity=remaining,
            average_fill_price=float(avg) if avg is not None else None,
            position_intent=position_intent,  # type: ignore[arg-type]
        )
        self._orders[client_order_id] = order
        return order

    def _order_from_detail_or_intent(
        self,
        cid: str,
        detail: dict[str, Any],
        *,
        intent: OrderIntent,
        status: str,
        broker_order_id: str,
    ) -> BrokerOrder:
        filled = int(
            float(
                detail.get("filled_quantity")
                or detail.get("cumulative_filled_quantity")
                or (intent.quantity if status == "filled" else 0)
            )
        )
        return BrokerOrder(
            order_id=cid,
            intent_id=intent.intent_id,
            status=status,  # type: ignore[arg-type]
            contract=intent.contract,
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
            filled_quantity=filled,
            remaining_quantity=max(0, intent.quantity - filled),
            average_fill_price=(
                float(detail["avg_filled_price"])
                if detail.get("avg_filled_price") is not None
                else None
            ),
            position_intent=intent.position_intent,
        )

    def _journal_transition(
        self,
        cid: str,
        status: str,
        *,
        broker_order_id: str | None = None,
        last_error_code: str | None = None,
        preview_hash: str | None = None,
    ) -> None:
        if self._journal is None:
            if self._capture_only:
                return
            raise BrokerError("journal required for live transition — fail closed")
        try:
            self._journal.transition(
                account_id_hash=self._account_id_hash,
                client_order_id=cid,
                status=status,  # type: ignore[arg-type]
                broker_order_id=broker_order_id,
                last_error_code=last_error_code,
                preview_hash=preview_hash,
            )
        except KeyError as exc:
            raise BrokerError(
                f"missing journal row for transition {cid}/{status} — fail closed"
            ) from exc


def _journal_status_for(status: str) -> str:
    if status == "filled":
        return "filled"
    if status == "partially_filled":
        return "partially_filled"
    if status == "rejected":
        return "rejected"
    if status == "cancelled":
        return "cancelled"
    return "accepted"


def _contract_from_detail(detail: dict[str, Any]) -> OptionContract | None:
    legs = detail.get("legs") if isinstance(detail.get("legs"), list) else []
    leg = legs[0] if legs and isinstance(legs[0], dict) else detail
    try:
        from datetime import date as date_cls

        symbol = str(leg.get("symbol") or detail.get("symbol") or "").upper()
        expire = leg.get("option_expire_date") or detail.get("option_expire_date")
        strike = leg.get("strike_price") or leg.get("option_exercise_price")
        opt = str(leg.get("option_type") or "").lower()
        if not symbol or not expire or strike is None or opt not in {"call", "put"}:
            return None
        expiration = date_cls.fromisoformat(str(expire)[:10])
        return OptionContract(
            symbol=symbol,
            expiration=expiration,
            strike=float(strike),
            option_type=opt,  # type: ignore[arg-type]
            is_0dte=expiration == date_cls.today(),
        )
    except Exception:
        return None


def _contract_from_contract_id(contract_id: str) -> OptionContract | None:
    parts = contract_id.split(":")
    if len(parts) != 4:
        return None
    try:
        from datetime import date as date_cls

        expiration = date_cls.fromisoformat(parts[1])
        return OptionContract(
            symbol=parts[0],
            expiration=expiration,
            strike=float(parts[2]),
            option_type=parts[3],  # type: ignore[arg-type]
            is_0dte=expiration == date_cls.today(),
        )
    except Exception:
        return None


def _looks_ambiguous(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return any(
        token in name or token in msg
        for token in ("timeout", "timed out", "connection", "reset", "unavailable")
    )


def create_mock_live_trade_api(account_id: str = "LIVE_ACCT_TEST") -> MockWebullTradeApi:
    """Test double labeled as individual brokerage (production-like)."""
    return MockWebullTradeApi(
        account_id=account_id,
        account_label="Individual Cash",
        account_class="INDIVIDUAL_CASH",
    )
