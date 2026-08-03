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
from joker.broker.interface import BrokerClient, BrokerError
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
    BrokerSubmissionRecord,
    DuplicateSubmissionError,
    SyncBrokerSubmissionJournal,
    payload_hash,
)
from joker.schemas.domain import BrokerOrder, OrderIntent, Position

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
        trade_api: WebullTradeApi | None = None,
        journal: SyncBrokerSubmissionJournal | None = None,
        capture_only: bool = False,
        skip_account_list_check: bool = False,
        process_armed: bool = True,
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
        if not process_armed:
            raise WebullLiveConfigError(
                "Live broker process is not armed for the current process"
            )
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
        self._api = trade_api or HttpWebullLiveTradeApi(credentials)
        self._journal = journal
        self._capture_only = capture_only
        self._captured_payloads: list[dict[str, Any]] = []
        self._orders: dict[str, BrokerOrder] = {}
        self._intent_by_order: dict[str, OrderIntent] = {}
        self._session_baseline_nlv: Decimal | None = None
        self._armed = process_armed

    @property
    def account_id_hash(self) -> str:
        return self._account_id_hash

    @property
    def captured_payloads(self) -> list[dict[str, Any]]:
        return list(self._captured_payloads)

    def close(self) -> None:
        close = getattr(self._api, "close", None)
        if callable(close):
            close()

    def _assert_armed(self) -> None:
        if not self._armed:
            raise WebullLiveConfigError("Live mode is not armed for this process")
        if self._credentials.api_env != "prod":
            raise WebullLiveConfigError("Live API environment must remain prod")
        if not self._app.live_trading_enabled or not self._env.webull_live_trading_enabled:
            raise WebullLiveConfigError("Live trading flags must remain enabled")

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
            )
            self._orders[cid] = order
            self._intent_by_order[cid] = intent
            return order

        self._journal_prepare(intent, payload, cid)
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
            raise BrokerError(
                f"Live submission unknown after timeout for {cid}"
            ) from exc
        except Exception as exc:
            # Ambiguous network failures → unknown; clear rejects stay rejected.
            if _looks_ambiguous(exc):
                self._journal_transition(
                    cid,
                    "submission_unknown",
                    last_error_code=type(exc).__name__,
                )
                reconciled = self._reconcile_unknown(cid, intent)
                if reconciled is not None:
                    return reconciled
                raise BrokerError(
                    redact_secrets(
                        f"Live submission unknown: {exc}", env=self._env
                    )
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
        journal_status = "filled" if status == "filled" else "accepted"
        if status == "rejected":
            journal_status = "rejected"
        self._journal_transition(
            cid, journal_status, broker_order_id=broker_order_id  # type: ignore[arg-type]
        )
        order = BrokerOrder(
            order_id=cid,
            intent_id=intent.intent_id,
            status=status,  # type: ignore[arg-type]
            contract=intent.contract,
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
        )
        self._orders[cid] = order
        self._intent_by_order[cid] = intent
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
        order = self._orders.get(order_id)
        if order is None:
            intent = self._intent_by_order.get(order_id)
            if intent is None:
                raise BrokerError(f"Unknown local order_id: {order_id}")
            order = BrokerOrder(
                order_id=order_id,
                intent_id=intent.intent_id,
                status="cancelled",
                contract=intent.contract,
                side=intent.side,
                quantity=intent.quantity,
                limit_price=intent.limit_price,
            )
        else:
            order.status = "cancelled"
        self._orders[order_id] = order
        return order

    def get_order(self, order_id: str) -> BrokerOrder | None:
        try:
            detail = self._api.get_order_detail(self._account_id, order_id)
        except Exception:
            return self._orders.get(order_id)
        status = map_webull_order_status(
            str(detail.get("status") or detail.get("order_status") or "")
        )
        local = self._orders.get(order_id)
        intent = self._intent_by_order.get(order_id)
        if local is not None:
            local.status = status  # type: ignore[assignment]
            return local
        if intent is None:
            return None
        order = BrokerOrder(
            order_id=order_id,
            intent_id=intent.intent_id,
            status=status,  # type: ignore[arg-type]
            contract=intent.contract,
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
        )
        self._orders[order_id] = order
        return order

    def list_open_orders(self) -> list[BrokerOrder]:
        try:
            rows = self._api.list_open_orders(self._account_id)
        except Exception:
            return [o for o in self._orders.values() if o.status in {"open", "pending"}]
        result: list[BrokerOrder] = []
        for row in rows:
            cid = str(row.get("client_order_id") or "")
            if not cid:
                continue
            local = self.get_order(cid)
            if local is not None and local.status in {"open", "pending"}:
                result.append(local)
        return result

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
        session_pnl: Decimal | None = None
        session_available = False
        # Prefer explicit day PnL from broker when present.
        explicit = decimal_or_none(
            balance.get("day_pnl")
            or balance.get("today_pnl")
            or balance.get("session_pnl")
        )
        if explicit is not None:
            session_pnl = explicit
            session_available = True
        elif nlv is not None:
            if self._session_baseline_nlv is None:
                self._session_baseline_nlv = nlv
                session_pnl = Decimal("0")
                session_available = True
            else:
                session_pnl = nlv - self._session_baseline_nlv
                session_available = True
        return BrokerAccountTruth(
            account_id_hash=self._account_id_hash,
            cash_usd=parsed["cash_usd"],
            buying_power_usd=parsed["buying_power_usd"],
            net_liquidation_value_usd=nlv,
            session_pnl_usd=session_pnl,
            session_pnl_available=session_available,
            positions=tuple(self.list_positions()),
            working_orders=tuple(self.list_open_orders()),
            captured_at=datetime.now(timezone.utc),
        )

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
        journal_status: Literal[
            "accepted", "filled", "rejected", "cancelled", "partially_filled", "reconciled"
        ]
        if status == "filled":
            journal_status = "filled"
        elif status == "rejected":
            journal_status = "rejected"
        elif status == "cancelled":
            journal_status = "cancelled"
        else:
            journal_status = "accepted"
        self._journal_transition(
            client_order_id, journal_status, broker_order_id=broker_order_id
        )
        order = BrokerOrder(
            order_id=client_order_id,
            intent_id=intent.intent_id,
            status=status,  # type: ignore[arg-type]
            contract=intent.contract,
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
        )
        self._orders[client_order_id] = order
        self._intent_by_order[client_order_id] = intent
        return order

    def _journal_prepare(
        self, intent: OrderIntent, payload: dict[str, Any], cid: str
    ) -> None:
        if self._journal is None:
            return
        record = BrokerSubmissionRecord(
            client_order_id=cid,
            broker_mode="webull_live",
            account_id_hash=self._account_id_hash,
            status="prepared",
            contract_id=(
                f"{intent.contract.symbol}:{intent.contract.expiration.isoformat()}:"
                f"{intent.contract.strike}:{intent.contract.option_type}"
            ),
            side=intent.side,
            position_intent=intent.position_intent,
            quantity=intent.quantity,
            limit_price=(
                f"{intent.limit_price:.2f}" if intent.limit_price is not None else None
            ),
            payload_hash=payload_hash(payload),
        )
        try:
            self._journal.prepare(record)
        except DuplicateSubmissionError as exc:
            raise BrokerError(str(exc)) from exc

    def _journal_transition(
        self,
        cid: str,
        status: str,
        *,
        broker_order_id: str | None = None,
        last_error_code: str | None = None,
    ) -> None:
        if self._journal is None:
            return
        try:
            self._journal.transition(
                account_id_hash=self._account_id_hash,
                client_order_id=cid,
                status=status,  # type: ignore[arg-type]
                broker_order_id=broker_order_id,
                last_error_code=last_error_code,
            )
        except KeyError:
            pass


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
