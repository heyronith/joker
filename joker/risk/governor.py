"""Deterministic risk governor — strict or agent_led hard-floor policy."""

from __future__ import annotations

from datetime import datetime, timezone

from joker.app.safety import SafetyMode
from joker.data.freshness import FreshnessConfig, evaluate_quote_freshness
from joker.schemas.domain import DailyState, RiskConfig, RiskDecision, TradeCandidate


class RiskReasonCode:
    MODE_BLOCKED = "MODE_BLOCKED"
    KILL_SWITCH = "KILL_SWITCH"
    WRONG_SYMBOL = "WRONG_SYMBOL"
    NOT_0DTE = "NOT_0DTE"
    SHORT_OPTION = "SHORT_OPTION"
    SPREAD_NOT_ALLOWED = "SPREAD_NOT_ALLOWED"
    STALE_QUOTE = "STALE_QUOTE"
    WIDE_SPREAD = "WIDE_SPREAD"
    MAX_PREMIUM = "MAX_PREMIUM"
    NO_STOP = "NO_STOP"
    NO_TAKE_PROFIT = "NO_TAKE_PROFIT"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    MAX_TRADES = "MAX_TRADES"
    MAX_OPEN_POSITIONS = "MAX_OPEN_POSITIONS"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    UNRESOLVED_ORDER = "UNRESOLVED_ORDER"
    INVALID_DIRECTION = "INVALID_DIRECTION"
    DELAYED_NOT_ALLOWED = "DELAYED_NOT_ALLOWED"
    FEED_SILENT = "FEED_SILENT"
    CAPITAL_EXCEEDED = "CAPITAL_EXCEEDED"
    GOAL_MET_PAUSE = "GOAL_MET_PAUSE"


class RiskGovernor:
    """
    Risk checks.

    - policy=strict: all soft caps enforce (legacy research default)
    - policy=agent_led: hard floors only — agent entry decisions execute unless
      catastrophic (kill switch, wrong symbol, silent feed, etc.)
    """

    def __init__(self, config: RiskConfig, mode: SafetyMode, live_enabled: bool = False) -> None:
        self.config = config
        self.mode = mode
        self.live_enabled = live_enabled
        self._seen_candidates: set[str] = set()

    @property
    def policy(self) -> str:
        return (self.config.policy or "strict").strip().lower()

    def evaluate(
        self,
        candidate: TradeCandidate,
        daily_state: DailyState,
        *,
        has_unresolved_order: bool = False,
        agent_override: bool = False,
        reference_time: datetime | None = None,
    ) -> RiskDecision:
        """Evaluate trade candidate. Soft caps ignored under agent_led."""
        _ = agent_override  # agents never mutate RiskConfig; policy is config-driven
        reasons: list[str] = []
        now = reference_time or datetime.now(timezone.utc)
        agent_led = self.policy == "agent_led"

        if self.config.kill_switch or daily_state.kill_switch:
            reasons.append(RiskReasonCode.KILL_SWITCH)

        if self.mode is SafetyMode.SHADOW:
            pass
        elif self.mode is SafetyMode.LIVE_GATED and not self.live_enabled:
            reasons.append(RiskReasonCode.MODE_BLOCKED)
        elif self.mode is SafetyMode.LIVE_GATED and self.live_enabled:
            pass
        elif self.mode is SafetyMode.PAPER:
            pass

        if candidate.contract.symbol != self.config.allowed_symbol:
            reasons.append(RiskReasonCode.WRONG_SYMBOL)
        if not candidate.contract.is_0dte:
            reasons.append(RiskReasonCode.NOT_0DTE)
        if candidate.direction not in ("long_call", "long_put"):
            reasons.append(RiskReasonCode.INVALID_DIRECTION)

        freshness = evaluate_quote_freshness(
            quote_timestamp=candidate.quote.timestamp,
            reference_time=now,
            delayed=candidate.quote.delayed,
            received_at=candidate.quote.received_at,
            config=FreshnessConfig(
                quote_max_age_seconds=self.config.quote_max_age_seconds,
                feed_max_silence_seconds=self.config.feed_max_silence_seconds,
                delayed_quote_max_age_seconds=self.config.delayed_quote_max_age_seconds,
                allow_delayed_quotes=self.config.allow_delayed_quotes,
            ),
        )
        # Hard floor: silent feed always blocks. Stale/delayed soft under agent_led.
        if not freshness.ok:
            if freshness.reason == "FEED_SILENT":
                reasons.append(RiskReasonCode.FEED_SILENT)
            elif not agent_led:
                if freshness.reason == "DELAYED_NOT_ALLOWED":
                    reasons.append(RiskReasonCode.DELAYED_NOT_ALLOWED)
                else:
                    reasons.append(RiskReasonCode.STALE_QUOTE)

        if candidate.stop_price <= 0:
            reasons.append(RiskReasonCode.NO_STOP)
        if candidate.take_profit_price <= 0:
            reasons.append(RiskReasonCode.NO_TAKE_PROFIT)

        if has_unresolved_order:
            reasons.append(RiskReasonCode.UNRESOLVED_ORDER)
        if candidate.candidate_id in self._seen_candidates:
            reasons.append(RiskReasonCode.DUPLICATE_ORDER)

        # Hard floor when a session capital budget is configured
        authorized = float(getattr(self.config, "authorized_capital_usd", 0.0) or 0.0)
        if authorized > 0:
            reserved = float(getattr(self.config, "reserved_capital_usd", 0.0) or 0.0)
            available = max(0.0, authorized - reserved)
            notional = candidate.entry_limit_price * 100.0 * max(1, int(candidate.quantity))
            if notional > available + 1e-6:
                reasons.append(RiskReasonCode.CAPITAL_EXCEEDED)

        if not agent_led:
            if candidate.quote.spread_pct > self.config.max_spread_pct:
                reasons.append(RiskReasonCode.WIDE_SPREAD)
            premium = candidate.entry_limit_price * 100
            if premium > self.config.max_premium_usd:
                reasons.append(RiskReasonCode.MAX_PREMIUM)
            if daily_state.daily_pnl_usd <= -self.config.max_daily_loss_usd:
                reasons.append(RiskReasonCode.MAX_DAILY_LOSS)
            if daily_state.trades_count >= self.config.max_trades_per_day:
                reasons.append(RiskReasonCode.MAX_TRADES)
            if daily_state.open_positions >= self.config.max_open_positions:
                reasons.append(RiskReasonCode.MAX_OPEN_POSITIONS)

        approved = len(reasons) == 0
        if approved:
            self._seen_candidates.add(candidate.candidate_id)

        return RiskDecision(
            candidate_id=candidate.candidate_id,
            approved=approved,
            reason_codes=reasons,
            message="approved" if approved else f"rejected: {', '.join(reasons)}",
            policy=self.policy,
        )
