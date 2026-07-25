"""Market event handler — connects replay/live feed to reactive engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from joker.app.safety import SafetyMode
from joker.broker.interface import BrokerClient, PaperBroker
from joker.data.provider import MarketDataProvider
from joker.execution.exit_manager import ExitManager, OpenTradeContext
from joker.execution.option_selector import OptionSelector, OptionSelectorConfig
from joker.features.engine import FeatureEngine
from joker.risk.capital import CapitalBudget
from joker.risk.governor import RiskGovernor
from joker.runtime.reactive_engine import ReactiveEngine, StateMachineError, TradingState
from joker.runtime.shadow import ShadowRuntime
from joker.schemas.domain import DailyState, OrderIntent, Playbook, PlaybookSetup, TradeCandidate
from joker.compliance.opra_sanitizer import exit_decision_safe_metadata
from joker.schemas.replay import (
    ExitDecision,
    MarketEvent,
    OptionQuoteEvent,
    ReplaySummary,
    SpyCandleEvent,
    SpyQuoteEvent,
)

if TYPE_CHECKING:
    from joker.data.webull_options_provider import WebullOptionsDataProvider


@dataclass
class PendingEntry:
    order_id: str
    candidate: TradeCandidate
    setup: PlaybookSetup
    source: str = "structured_signal"


@dataclass
class MarketHandlerState:
    features_updated: int = 0
    signals_detected: int = 0
    trades_entered: int = 0
    trades_exited: int = 0
    risk_rejections: int = 0
    option_selector_rejections: int = 0
    last_risk_decision: str | None = None
    last_exit_reason: str | None = None
    last_event_type: str | None = None
    latest_spy_price: float | None = None
    latest_option_mid: float | None = None
    open_trade: OpenTradeContext | None = None
    pending_entry: PendingEntry | None = None
    signaled_setups: set[str] = field(default_factory=set)
    setups_armed: int = 0
    exits_by_reason: dict[str, int] = field(default_factory=dict)
    exit_decisions: list[dict] = field(default_factory=list)
    trade_durations_minutes: list[float] = field(default_factory=list)
    max_adverse_excursion: float | None = None
    max_favorable_excursion: float | None = None
    missing_data_events: int = 0
    stale_quote_events: int = 0
    wide_spread_rejects: int = 0
    rule_violations: int = 0


class MarketEventHandler:
    """Process market events through features, signals, risk, and paper execution."""

    def __init__(
        self,
        provider: MarketDataProvider,
        reactive_engine: ReactiveEngine,
        risk_governor: RiskGovernor,
        broker: BrokerClient,
        feature_engine: FeatureEngine,
        option_selector: OptionSelector,
        exit_manager: ExitManager,
        *,
        mode: SafetyMode = SafetyMode.PAPER,
        run_id: str = "",
        daily_state: DailyState | None = None,
        on_log: Callable[[str, dict], None] | None = None,
        options_provider: WebullOptionsDataProvider | None = None,
        rules_auto_entry: bool = True,
        on_trade_outcome: Callable[[dict], None] | None = None,
        capital_budget: CapitalBudget | None = None,
    ) -> None:
        self.provider = provider
        self.engine = reactive_engine
        self.risk_governor = risk_governor
        self.broker = broker
        self.feature_engine = feature_engine
        self.option_selector = option_selector
        self.exit_manager = exit_manager
        self.mode = mode
        self.run_id = run_id
        self._options_provider = options_provider
        self.rules_auto_entry = rules_auto_entry
        self._on_trade_outcome = on_trade_outcome
        self.capital_budget = capital_budget
        self._pending_allocation: dict | None = None
        self._pause_when_goal_met = True
        self.daily_state = daily_state or DailyState(
            trading_day=provider.current_time.date(),
            run_id=run_id,
            mode=mode.value,
        )
        self.state = MarketHandlerState()
        self.shadow = ShadowRuntime(mode=mode) if mode is SafetyMode.SHADOW else None
        self._on_log = on_log

    def _log(self, event_type: str, payload: dict | None = None) -> None:
        if self._on_log:
            self._on_log(event_type, payload or {})

    def _attach_open_trade(
        self,
        *,
        position_id: str,
        entry_price: float,
        candidate: TradeCandidate,
        setup: PlaybookSetup,
    ) -> None:
        self.state.open_trade = OpenTradeContext(
            position_id=position_id,
            entry_price=entry_price,
            stop_price=candidate.stop_price,
            take_profit_price=candidate.take_profit_price,
            entry_time=self.provider.current_time,
            time_stop_minutes=setup.time_stop_minutes,
            quantity=max(1, int(candidate.quantity)),
            reserved_notional_usd=float(
                (self._pending_allocation or {}).get("notional_usd")
                or entry_price * 100.0 * max(1, int(candidate.quantity))
            ),
        )
        self._pending_allocation = None
        if self.engine.state == TradingState.OPEN_POSITION:
            self.engine.transition(TradingState.MANAGING_EXIT)
        self._log(
            "order.filled",
            {
                "position_id": position_id,
                "entry_price": entry_price,
                "quantity": candidate.quantity,
                "candidate_id": candidate.candidate_id,
                "setup_id": setup.setup_id,
            },
        )
    def _reconcile_pending_entry(self) -> None:
        """Promote working paper-account orders to open_trade once filled."""
        pending = self.state.pending_entry
        if pending is None or self.state.open_trade is not None:
            return

        positions = [p for p in self.broker.list_positions() if p.is_open]
        if positions:
            pos = positions[-1]
            self.state.pending_entry = None
            self._attach_open_trade(
                position_id=pos.position_id,
                entry_price=pos.avg_entry_price,
                candidate=pending.candidate,
                setup=pending.setup,
            )
            return

        order = self.broker.get_order(pending.order_id)
        if order is None:
            return
        if order.status == "filled":
            # Position list lag — synthesize from order limit until broker lists it.
            entry = order.limit_price or pending.candidate.entry_limit_price
            self.state.pending_entry = None
            self._attach_open_trade(
                position_id=order.order_id,
                entry_price=float(entry),
                candidate=pending.candidate,
                setup=pending.setup,
            )
            return
        if order.status in {"cancelled", "rejected"}:
            self._log(
                "order.failed",
                {
                    "order_id": order.order_id,
                    "status": order.status,
                    "candidate_id": pending.candidate.candidate_id,
                },
            )
            self.state.pending_entry = None
            if self.engine.state in (
                TradingState.OPEN_POSITION,
                TradingState.ENTERING,
                TradingState.MANAGING_EXIT,
            ):
                try:
                    self.engine.transition(TradingState.WATCHING, {"order_gone": order.status})
                except StateMachineError:
                    self.engine.state = TradingState.WATCHING
            return

    def _detect_signal(self, features, playbook: Playbook) -> PlaybookSetup | None:
        from joker.strategy.signal_rules import detect_setup_from_playbook

        return detect_setup_from_playbook(
            playbook,
            features,
            already_signaled=self.state.signaled_setups,
        )

    def _build_candidate(
        self,
        setup: PlaybookSetup,
        selected,
        *,
        stop_pct: float | None = None,
        take_profit_pct: float | None = None,
        quantity: int = 1,
    ) -> TradeCandidate:
        entry = selected.quote.ask
        sp = stop_pct if stop_pct is not None else setup.stop_pct
        tp = take_profit_pct if take_profit_pct is not None else setup.take_profit_pct
        stop = ExitManager.stop_from_entry(entry, sp)
        target = ExitManager.target_from_entry(entry, tp)
        return TradeCandidate(
            run_id=self.run_id,
            setup_id=setup.setup_id,
            contract=selected.contract,
            quote=selected.quote,
            direction=setup.direction,  # type: ignore[arg-type]
            entry_limit_price=entry,
            stop_price=stop,
            take_profit_price=target,
            quantity=max(1, int(quantity)),
            created_at=self.provider.current_time,
        )

    def _sync_risk_capital(self) -> None:
        if self.capital_budget is None:
            return
        cfg = self.risk_governor.config
        self.risk_governor.config = cfg.model_copy(
            update={
                "authorized_capital_usd": self.capital_budget.authorized_usd,
                "reserved_capital_usd": self.capital_budget.reserved_usd,
                "max_open_positions": self.capital_budget.plan.max_concurrent_positions,
            }
        )

    def _size_for_entry(
        self,
        premium: float,
        *,
        capital_fraction: float | None = None,
        target_contracts: int | None = None,
        confidence: float = 0.5,
        allocation_style: str = "auto",
        win_probability: float | None = None,
        expected_value_usd: float | None = None,
        expected_r: float | None = None,
        minutes_to_close: float | None = None,
    ) -> tuple[int, float]:
        """Return (quantity, notional). Falls back to 1 contract without a budget."""
        if self.capital_budget is None:
            return 1, premium * 100.0

        style = allocation_style if allocation_style in {
            "auto", "aggressive", "split", "conservative"
        } else "auto"
        result = self.capital_budget.allocate(
            premium_per_contract=premium,
            capital_fraction=capital_fraction,
            target_contracts=target_contracts,
            confidence=confidence,
            allocation_style=style,  # type: ignore[arg-type]
            win_probability=win_probability,
            expected_value_usd=expected_value_usd,
            expected_r=expected_r,
            minutes_to_close=minutes_to_close,
        )
        if result.quantity < 1:
            self._log(
                "capital.rejected",
                {
                    "reason": result.reason,
                    "available": self.capital_budget.available_usd,
                    "aggression_cap": result.aggression_cap,
                    "ev_gate": result.ev_gate,
                },
            )
            return 0, 0.0
        self._log(
            "capital.sized",
            {
                "quantity": result.quantity,
                "notional_usd": result.notional_usd,
                "fraction_used": result.capital_fraction_used,
                "aggression_cap": result.aggression_cap,
            },
        )
        return result.quantity, result.notional_usd

    def _try_exit(self, quote_event: OptionQuoteEvent) -> ExitDecision | None:
        if self.state.open_trade is None:
            return None
        decision = self.exit_manager.check_exit(
            self.state.open_trade,
            quote_event.mid,
            self.provider.current_time,
        )
        if decision is None:
            return None

        if self.mode is SafetyMode.SHADOW:
            label = None
            if self.shadow and self.shadow.records:
                rec = self.shadow.records[-1]
                self.shadow.simulate_outcome(rec, quote_event.mid)
                label = rec.shadow_result_label
            self._log(
                "exit.shadow",
                exit_decision_safe_metadata(
                    decision.reason.value,
                    shadow_result_label=label,
                ),
            )
        else:
            positions = [p for p in self.broker.list_positions() if p.is_open]
            if positions:
                pos = positions[0]
                qty = (
                    self.state.open_trade.quantity
                    if self.state.open_trade is not None
                    else pos.quantity
                )
                intent = OrderIntent(
                    candidate_id=str(uuid4()),
                    contract=pos.contract,
                    side="sell",
                    order_type="limit",
                    quantity=max(1, int(qty)),
                    limit_price=decision.exit_price,
                )
                exit_order = self.broker.submit_order(intent)
                self._log(
                    "order.submitted",
                    {
                        "side": "sell",
                        "order_id": exit_order.order_id,
                        "status": exit_order.status,
                        "limit_price": exit_order.limit_price,
                        "quantity": qty,
                        "reason": decision.reason.value,
                    },
                )
        entry_time = self.state.open_trade.entry_time if self.state.open_trade else None
        entry_price = self.state.open_trade.entry_price if self.state.open_trade else None
        qty = self.state.open_trade.quantity if self.state.open_trade else 1
        reserved = (
            self.state.open_trade.reserved_notional_usd if self.state.open_trade else 0.0
        )
        mae = self.state.max_adverse_excursion
        mfe = self.state.max_favorable_excursion
        # Per-trade PnL estimate from premium change (long options)
        trade_pnl = None
        if entry_price is not None and decision.exit_price is not None:
            trade_pnl = (decision.exit_price - entry_price) * 100.0 * max(1, int(qty))
        if self.capital_budget is not None and reserved > 0:
            self.capital_budget.release(
                reserved,
                realized_pnl_usd=float(trade_pnl or 0.0),
            )
            self._sync_risk_capital()
        self.state.open_trade = None
        self.state.trades_exited += 1
        self.state.last_exit_reason = decision.reason.value
        self.state.exits_by_reason[decision.reason.value] = (
            self.state.exits_by_reason.get(decision.reason.value, 0) + 1
        )
        self.state.exit_decisions.append(
            exit_decision_safe_metadata(decision.reason.value)
            if self.mode is SafetyMode.SHADOW
            else decision.model_dump(mode="json")
        )
        duration_minutes = None
        if entry_time:
            elapsed = (self.provider.current_time - entry_time).total_seconds() / 60.0
            self.state.trade_durations_minutes.append(elapsed)
            duration_minutes = elapsed
        if self.engine.state in (TradingState.OPEN_POSITION, TradingState.MANAGING_EXIT):
            self.engine.transition(TradingState.EXITED, {"reason": decision.reason.value})
            self.engine.transition(TradingState.COOLDOWN)
            self.engine.transition(TradingState.WATCHING)
        outcome_payload = {
            "exit_reason": decision.reason.value,
            "exit_price": decision.exit_price,
            "entry_price": entry_price,
            "quantity": qty,
            "mae": mae,
            "mfe": mfe,
            "duration_minutes": duration_minutes,
            "trade_pnl_usd": trade_pnl,
            "realized_pnl_usd": (
                self.capital_budget.realized_pnl_usd
                if self.capital_budget is not None
                else self.broker.get_daily_pnl()
            ),
            "capital": self.capital_budget.prompt_dict() if self.capital_budget else None,
        }
        self._log(
            "exit.executed",
            exit_decision_safe_metadata(decision.reason.value)
            if self.mode is SafetyMode.SHADOW
            else {**decision.model_dump(mode="json"), **outcome_payload},
        )
        if self._on_trade_outcome is not None:
            try:
                self._on_trade_outcome(outcome_payload)
            except Exception:
                pass
        self.state.max_adverse_excursion = None
        self.state.max_favorable_excursion = None
        return decision

    def handle_event(self, event: MarketEvent) -> None:
        self.state.last_event_type = event.event_type
        self._reconcile_pending_entry()
        self._log("market.event_received", {"event_type": event.event_type, "event_id": event.event_id})

        if isinstance(event, SpyQuoteEvent):
            self.state.latest_spy_price = event.price
        elif isinstance(event, SpyCandleEvent):
            pass

        snapshot = self.provider.get_latest_snapshot()
        if snapshot is None:
            self.state.missing_data_events += 1
            return

        features = self.feature_engine.compute(
            snapshot,
            reference_time=self.provider.current_time,
        )
        self.state.features_updated += 1
        self._log("feature.updated", {"trend": features.trend_label, "vwap_dist": features.distance_from_vwap_pct})

        if isinstance(event, OptionQuoteEvent):
            self.state.latest_option_mid = event.mid
            if self.state.open_trade is not None:
                entry = self.state.open_trade.entry_price
                adverse = entry - event.mid
                favorable = event.mid - entry
                if self.state.max_adverse_excursion is None or adverse > self.state.max_adverse_excursion:
                    self.state.max_adverse_excursion = adverse
                if self.state.max_favorable_excursion is None or favorable > self.state.max_favorable_excursion:
                    self.state.max_favorable_excursion = favorable
                self.state.open_trade = self.exit_manager.update_trailing(
                    self.state.open_trade, event.mid
                )
            exit_dec = self._try_exit(event)
            if exit_dec:
                return

        playbook = self.engine.active_playbook
        if playbook is None or self.engine.state not in (
            TradingState.WATCHING,
            TradingState.SETUP_ARMED,
            TradingState.COOLDOWN,
        ):
            return

        if self.state.open_trade is not None or self.state.pending_entry is not None:
            return

        if self.broker.list_open_orders() or any(p.is_open for p in self.broker.list_positions()):
            return

        if not self.rules_auto_entry:
            return

        setup = self._detect_signal(features, playbook)
        if setup is None:
            return

        self._attempt_entry(setup, snapshot)

    def try_enter_from_decision(
        self,
        decision: "IntradayDecision",
        *,
        min_confidence: float = 0.45,
    ) -> bool:
        """Execute an agent_led IntradayDecision (soft risk caps do not veto)."""
        from joker.schemas.domain import IntradayDecision, PlaybookSetup

        if not isinstance(decision, IntradayDecision):
            return False
        if decision.action not in ("enter", "confirm"):
            return False
        if decision.direction not in ("long_call", "long_put"):
            self._log("agent.decision_invalid", {"reason": "missing_direction"})
            return False
        if decision.confidence < min_confidence:
            self._log(
                "agent.decision_low_confidence",
                {"confidence": decision.confidence, "min": min_confidence},
            )
            return False

        playbook = self.engine.active_playbook
        setup: PlaybookSetup | None = None
        if playbook is not None and decision.setup_id:
            setup = next(
                (s for s in playbook.setups if s.setup_id == decision.setup_id),
                None,
            )
        if setup is None and playbook is not None:
            setup = next(
                (
                    s
                    for s in playbook.setups
                    if s.enabled and s.direction == decision.direction
                ),
                None,
            )
        if setup is None:
            setup = PlaybookSetup(
                setup_id=decision.setup_id or f"agent_{decision.direction}",
                name=f"agent_{decision.direction}",
                direction=decision.direction,
                enabled=True,
                stop_rule="agent",
                take_profit_rule="agent",
                stop_pct=decision.stop_pct,
                take_profit_pct=decision.take_profit_pct,
            )

        snapshot = self.provider.get_latest_snapshot()
        if snapshot is None:
            return False
        self._log(
            "agent.execute",
            {
                "direction": decision.direction,
                "confidence": decision.confidence,
                "rationale": (decision.rationale or decision.summary)[:200],
                "setup_id": setup.setup_id,
                "capital_fraction": decision.capital_fraction,
                "target_contracts": decision.target_contracts,
                "allocation_style": decision.allocation_style,
            },
        )
        return self._attempt_entry(
            setup,
            snapshot,
            stop_pct=decision.stop_pct,
            take_profit_pct=decision.take_profit_pct,
            source="agent_decision",
            capital_fraction=decision.capital_fraction,
            target_contracts=decision.target_contracts,
            confidence=decision.confidence,
            allocation_style=decision.allocation_style,
            win_probability=decision.win_probability,
            expected_value_usd=decision.expected_value_usd,
            expected_r=decision.expected_r,
        )

    def try_enter_from_proposal(
        self,
        proposal: "TradeProposal",
        *,
        min_confidence: float = 0.55,
    ) -> bool:
        """Attempt entry from an agent TradeProposal (risk policy applies)."""
        from joker.schemas.domain import TradeProposal

        if not isinstance(proposal, TradeProposal):
            return False
        if not proposal.propose_entry:
            return False
        if proposal.confidence < min_confidence:
            self._log(
                "proposal.rejected_low_confidence",
                {"confidence": proposal.confidence, "min": min_confidence},
            )
            return False
        playbook = self.engine.active_playbook
        if playbook is None:
            return False
        setup = next((s for s in playbook.setups if s.setup_id == proposal.setup_id), None)
        if setup is None or not setup.enabled:
            # Agent-led may invent setup ids — synthesize.
            if self.risk_governor.policy == "agent_led":
                from joker.schemas.domain import PlaybookSetup

                setup = PlaybookSetup(
                    setup_id=proposal.setup_id,
                    name=proposal.setup_id,
                    direction=proposal.direction,
                    enabled=True,
                    stop_rule="agent",
                    take_profit_rule="agent",
                    stop_pct=proposal.stop_pct,
                    take_profit_pct=proposal.take_profit_pct,
                )
            else:
                self._log("proposal.unknown_setup", {"setup_id": proposal.setup_id})
                return False
        elif setup.direction != proposal.direction:
            self._log(
                "proposal.direction_mismatch",
                {"setup": setup.direction, "proposal": proposal.direction},
            )
            return False
        snapshot = self.provider.get_latest_snapshot()
        if snapshot is None:
            return False
        return self._attempt_entry(
            setup,
            snapshot,
            stop_pct=proposal.stop_pct,
            take_profit_pct=proposal.take_profit_pct,
            source="agent_proposal",
        )
    def _attempt_entry(
        self,
        setup: PlaybookSetup,
        snapshot,
        *,
        stop_pct: float | None = None,
        take_profit_pct: float | None = None,
        source: str = "structured_signal",
        capital_fraction: float | None = None,
        target_contracts: int | None = None,
        confidence: float = 0.5,
        allocation_style: str = "auto",
        win_probability: float | None = None,
        expected_value_usd: float | None = None,
        expected_r: float | None = None,
    ) -> bool:
        if self.state.open_trade is not None or self.state.pending_entry is not None:
            return False
        if self.broker.list_open_orders() or any(p.is_open for p in self.broker.list_positions()):
            return False
        if setup.setup_id in self.state.signaled_setups and source == "structured_signal":
            return False
        if (
            self.capital_budget is not None
            and self.capital_budget.goal_met
            and self._pause_when_goal_met
        ):
            self._log("capital.goal_met_pause", self.capital_budget.prompt_dict())
            return False

        playbook = self.engine.active_playbook
        if playbook is None or self.engine.state not in (
            TradingState.WATCHING,
            TradingState.SETUP_ARMED,
            TradingState.COOLDOWN,
        ):
            return False

        quotes: list[OptionQuoteEvent] = []
        if hasattr(self.provider, "list_option_quote_events"):
            quotes = self.provider.list_option_quote_events()

        if not quotes and self._options_provider is not None and self._options_provider.is_available():
            try:
                call_snap, put_snap = self._options_provider.fetch_atm_snapshots(
                    snapshot.price,
                    expiration=self._options_provider.market_today(),
                )
                snaps = [s for s in (call_snap, put_snap) if s is not None]
                if snaps:
                    quotes = self._options_provider.to_quote_events(
                        snaps,
                        reference_time=self.provider.current_time,
                        allow_delayed_quotes=getattr(
                            self.option_selector.config, "allow_delayed_quotes", True
                        ),
                        feed_max_silence_seconds=getattr(
                            self.option_selector.config, "feed_max_silence_seconds", 60
                        ),
                        delayed_quote_max_age_seconds=getattr(
                            self.option_selector.config,
                            "delayed_quote_max_age_seconds",
                            900,
                        ),
                    )
                    self._log("options.quotes_loaded", {"count": len(quotes), "source": "webull"})
            except Exception as exc:
                self._log("options.unavailable", {"reason": str(exc)})
                self.state.missing_data_events += 1
                return False

        if not quotes:
            return False

        try:
            selected = self.option_selector.select_from_events(
                quotes,
                setup.direction,
                snapshot.price,
                self.provider.current_time,
            )
        except Exception as exc:
            reason = str(exc)
            if "STALE" in reason or "FEED_SILENT" in reason:
                self.state.stale_quote_events += 1
            elif "WIDE" in reason:
                self.state.wide_spread_rejects += 1
            elif "MISSING" in reason:
                self.state.missing_data_events += 1
            self.state.option_selector_rejections += 1
            self._log("option.rejected", {"reason": reason, "setup_id": setup.setup_id})
            return False

        premium = float(selected.quote.ask)
        minutes_to_close = None
        try:
            feat = self.feature_engine.compute(
                snapshot,
                prior_day_candles=getattr(self, "_prior_day_candles", None),
                premarket_candles=getattr(self, "_premarket_candles", None),
                reference_time=self.provider.current_time,
            )
            minutes_to_close = feat.minutes_to_close
        except Exception:
            minutes_to_close = None
        qty, notional = self._size_for_entry(
            premium,
            capital_fraction=capital_fraction,
            target_contracts=target_contracts,
            confidence=confidence,
            allocation_style=allocation_style,
            win_probability=win_probability,
            expected_value_usd=expected_value_usd,
            expected_r=expected_r,
            minutes_to_close=minutes_to_close,
        )
        if qty < 1:
            return False

        if self.option_selector.last_advisories:
            self._log(
                "option.advisory",
                {"advisories": list(self.option_selector.last_advisories)},
            )

        self._log(
            "option.selected",
            {
                "contract_id": selected.contract_id,
                "setup_id": setup.setup_id,
                "source": source,
                "quantity": qty,
                "notional_usd": notional,
            },
        )
        candidate = self._build_candidate(
            setup,
            selected,
            stop_pct=stop_pct,
            take_profit_pct=take_profit_pct,
            quantity=qty,
        )
        self._sync_risk_capital()
        self.state.signals_detected += 1
        self._log(
            "signal.detected",
            {
                "setup_id": setup.setup_id,
                "candidate_id": candidate.candidate_id,
                "source": source,
                "direction": setup.direction,
                "limit_price": candidate.entry_limit_price,
                "quantity": qty,
            },
        )

        if self.mode is SafetyMode.SHADOW and self.shadow:
            decision = self.risk_governor.evaluate(candidate, self.daily_state)
            record = self.shadow.record_candidate(candidate, decision, self.broker)
            self.state.last_risk_decision = decision.message
            if not decision.approved:
                self.state.risk_rejections += 1
            self._log("risk.decision", record.persist_metadata())
            self._log(
                "shadow.candidate",
                {**record.persist_metadata(), "approved": decision.approved},
            )
            self.state.signaled_setups.add(setup.setup_id)
            return decision.approved

        decision = None
        if self.capital_budget is not None:
            # Reserve only after risk approve; evaluate first.
            pass

        decision = self.engine.evaluate_signal(candidate, self.daily_state)
        self.state.last_risk_decision = decision.message
        self._log("risk.decision", decision.model_dump(mode="json"))

        if not decision.approved:
            self.state.risk_rejections += 1
            self.state.signaled_setups.add(setup.setup_id)
            return False

        if self.capital_budget is not None:
            try:
                self.capital_budget.reserve(notional)
                self._pending_allocation = {"notional_usd": notional, "quantity": qty}
                self._sync_risk_capital()
                self._log(
                    "capital.reserved",
                    {
                        "notional_usd": notional,
                        "quantity": qty,
                        "available_usd": self.capital_budget.available_usd,
                    },
                )
            except Exception as exc:
                self._log("capital.reserve_failed", {"reason": str(exc)})
                self.state.risk_rejections += 1
                return False

        order = self.engine.submit_entry(candidate)
        if order is not None:
            self._log(
                "order.submitted",
                {
                    "side": "buy",
                    "order_id": order.order_id,
                    "status": order.status,
                    "limit_price": order.limit_price,
                    "quantity": qty,
                    "candidate_id": candidate.candidate_id,
                    "setup_id": setup.setup_id,
                    "source": source,
                },
            )
        if order is not None and order.status == "rejected":
            if self.capital_budget is not None:
                self.capital_budget.release(notional, realized_pnl_usd=0.0)
                self._pending_allocation = None
                self._sync_risk_capital()
            self.state.risk_rejections += 1
            self.state.signaled_setups.add(setup.setup_id)
            return False

        self.state.trades_entered += 1
        positions = [p for p in self.broker.list_positions() if p.is_open]
        if positions:
            pos = positions[-1]
            self._attach_open_trade(
                position_id=pos.position_id,
                entry_price=pos.avg_entry_price,
                candidate=candidate,
                setup=setup,
            )
        elif order is not None and order.status in {"open", "pending", "filled"}:
            self.state.pending_entry = PendingEntry(
                order_id=order.order_id,
                candidate=candidate,
                setup=setup,
                source=source,
            )
            self._log(
                "order.pending_fill",
                {"order_id": order.order_id, "status": order.status},
            )
            if order.status == "filled":
                self._reconcile_pending_entry()
        self.state.signaled_setups.add(setup.setup_id)
        return True

    def build_summary(self, *, mock_agents: bool = True, is_synthetic: bool = False) -> ReplaySummary:
        return ReplaySummary(
            run_id=self.run_id,
            session_name="live_paper" if not is_synthetic else "replay",
            is_synthetic=is_synthetic,
            mock_agents=mock_agents,
            events_processed=self.state.features_updated,
            signals_detected=self.state.signals_detected,
            trades_entered=self.state.trades_entered,
            trades_exited=self.state.trades_exited,
            final_pnl_usd=self.broker.get_daily_pnl(),
            risk_rejections=self.state.risk_rejections,
            option_selector_rejections=self.state.option_selector_rejections,
        )
