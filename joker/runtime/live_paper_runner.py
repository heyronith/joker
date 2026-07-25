"""Live paper session — real Webull market data + auto paper execution.

Compatibility façade (Task 1):
- Prefer MarketRuntime for observation/bar/snapshot truth.
- Prefer ExecutionRuntime + ledger for order/fill/position accounting.
- This module remains the CLI entry for `joker paper run` and gradually delegates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from joker.agents.council import create_agent_council
from joker.agents.council_analysis import CouncilAnalysis, analyze_council
from joker.agents.llm_client import LLMClientError
from joker.agents.openai_agents import AgentError
from joker.agents.session_memory import SessionMicroMemory
from joker.app.safety import SafetyMode
from joker.broker.factory import BrokerFactoryError, resolve_live_paper_broker
from joker.broker.interface import BrokerClient
from joker.config.settings import AppSettings, EnvSettings
from joker.data.provider_factory import ProviderKind
from joker.data.webull_capability import capability_usable_for_shadow
from joker.data.webull_market_provider import WebullMarketDataProvider
from joker.data.webull_options_provider import (
    WebullOptionsDataProvider,
    create_webull_options_provider,
)
from joker.execution.exit_manager import ExitManager
from joker.execution.option_selector import OptionSelector, OptionSelectorConfig
from joker.features.engine import FeatureEngine
from joker.logging.event_log import EventLogWriter
from joker.reporting.metrics import StrategyQualityMetrics, compute_quality_metrics
from joker.reporting.replay_report import ReplayReportContext, ReplayReportGenerator
from joker.risk.capital import CapitalBudget, CapitalPlan
from joker.risk.governor import RiskGovernor
from joker.runtime.live_cli import should_stream_event
from joker.runtime.market_handler import MarketEventHandler
from joker.runtime.premarket import PremarketWorkflow
from joker.runtime.reactive_engine import ReactiveEngine, StateMachineError
from joker.runtime.run_manager import RunManager
from joker.schemas.domain import DailyState, Playbook, RiskConfig
from joker.schemas.replay import ReplaySummary, SpyQuoteEvent
from joker.storage.database import Database, ensure_database
from joker.storage.models import (
    AgentDecisionRecord,
    RiskDecisionRecord,
    SystemEventRecord,
    TradeCandidateRecord,
)
from joker.strategy.playbook_quality import PlaybookQualityValidator, PlaybookValidationResult, trim_playbook_enabled_setups


class LivePaperError(RuntimeError):
    """Fail-closed error for live paper session setup."""


@dataclass
class LivePaperRunConfig:
    symbol: str = "SPY"
    duration_seconds: float = 1800.0
    mock_agents: bool = False
    llm_client: Any | None = None
    webull_api: Any | None = None
    webull_options_api: Any | None = None
    # Test-only: inject a pre-approved playbook (never used for production CLI).
    approved_playbook: Playbook | None = None
    require_options: bool = True
    # Test/DI: inject broker or trade API for Webull paper path.
    broker: BrokerClient | None = None
    trade_api: Any | None = None
    # Session capital budget (required for agent sizing); tests may inject
    capital_budget: CapitalBudget | None = None


@dataclass
class LivePaperRunResult:
    run_id: str
    summary: ReplaySummary | None = None
    report_path: Path | None = None
    failures: list[str] = field(default_factory=list)
    playbook: Playbook | None = None
    council_analysis: CouncilAnalysis | None = None
    playbook_validation: PlaybookValidationResult | None = None
    events_processed: int = 0
    feed_health: str = "OK"
    options_available: bool = False
    paper_pnl_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    broker_kind: str = "local_paper"
    broker_label: str = "local PaperBroker"


def _risk_config_from_settings(
    settings: AppSettings, capital: CapitalBudget | None = None
) -> RiskConfig:
    r = settings.risk
    policy = (r.policy or "strict").strip().lower()
    if policy not in {"strict", "agent_led"}:
        policy = "strict"
    max_open = r.max_open_positions
    authorized = 0.0
    reserved = 0.0
    if capital is not None:
        max_open = capital.plan.max_concurrent_positions
        authorized = capital.authorized_usd
        reserved = capital.reserved_usd
    return RiskConfig(
        max_daily_loss_usd=r.max_daily_loss_usd,
        max_trades_per_day=r.max_trades_per_day,
        max_open_positions=max_open,
        max_premium_usd=r.max_premium_usd,
        max_spread_pct=r.max_spread_pct,
        quote_max_age_seconds=r.quote_max_age_seconds,
        allowed_symbol=r.allowed_symbol,
        kill_switch=r.kill_switch,
        allow_delayed_quotes=r.allow_delayed_quotes,
        feed_max_silence_seconds=r.feed_max_silence_seconds,
        delayed_quote_max_age_seconds=r.delayed_quote_max_age_seconds,
        policy=policy,  # type: ignore[arg-type]
        authorized_capital_usd=authorized,
        reserved_capital_usd=reserved,
    )


def _default_capital_budget(settings: AppSettings) -> CapitalBudget:
    c = settings.capital
    return CapitalBudget(
        plan=CapitalPlan(
            authorized_usd=float(c.authorized_usd),
            target_profit_pct=float(c.target_profit_pct),
            max_concurrent_positions=int(c.max_concurrent_positions),
            max_contracts_per_trade=int(c.max_contracts_per_trade),
            min_contracts_per_trade=int(c.min_contracts_per_trade),
            aggression_mode=str(c.aggression_mode),
            max_kelly_fraction=float(c.max_kelly_fraction),
            min_win_probability=float(c.min_win_probability),
            behind_goal_boost=float(c.behind_goal_boost),
            ahead_goal_dampen=float(c.ahead_goal_dampen),
        )
    )


class LivePaperRunner:
    """
    Real Webull SPY + 0DTE options → agents → risk → auto paper orders.

    Execution is either local PaperBroker or Webull paper account.
    Real-money live orders are never submitted.
    """

    def __init__(
        self,
        app_settings: AppSettings,
        env_settings: EnvSettings,
        db: Database | None = None,
        event_log: EventLogWriter | None = None,
    ) -> None:
        self.app_settings = app_settings
        self.env_settings = env_settings
        self.db = db or ensure_database(app_settings.db_path)
        self.event_log = event_log or EventLogWriter(
            app_settings.event_log_dir,
            redact_keys=app_settings.logging.redact_env_keys,
        )

    def _log(
        self,
        run_id: str,
        event_type: str,
        payload: dict,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.event_log.append(
            run_id=run_id,
            mode=SafetyMode.PAPER.value,
            source="live_paper",
            event_type=event_type,
            payload=payload,
        )
        if on_event is not None and should_stream_event(event_type):
            on_event(event_type, payload)

    def _assert_safe_mode(self) -> None:
        if self.app_settings.live_trading_enabled:
            raise LivePaperError(
                "live_trading_enabled must be false for paper sessions"
            )
        if self.env_settings.webull_live_trading_enabled:
            raise LivePaperError("WEBULL_LIVE_TRADING_ENABLED must remain false")
        if self.app_settings.mode is SafetyMode.LIVE_GATED:
            raise LivePaperError(
                "Live paper requires mode PAPER (not LIVE_GATED). "
                "Use config/paper.yaml."
            )
        if not self.env_settings.webull_market_data_enabled:
            raise LivePaperError(
                "WEBULL_MARKET_DATA_ENABLED must be true for live paper"
            )

    def run(
        self,
        config: LivePaperRunConfig,
        *,
        on_state: Callable[[dict[str, Any]], None] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> LivePaperRunResult:
        if config.symbol.upper() != "SPY":
            raise LivePaperError("Only SPY is supported")

        self._assert_safe_mode()

        # Force PAPER mode for this session regardless of display toggles.
        mode = SafetyMode.PAPER
        trading_day = date.today()
        run_manager = RunManager(self.db, self.event_log, self.app_settings)
        run_id = run_manager.start_run(trading_day=trading_day)
        result = LivePaperRunResult(run_id=run_id)
        failures: list[str] = []

        try:
            selection = resolve_live_paper_broker(
                self.app_settings,
                self.env_settings,
                trade_api=config.trade_api,
                broker=config.broker,
            )
        except BrokerFactoryError as exc:
            raise LivePaperError(str(exc)) from exc

        broker = selection.client
        result.broker_kind = selection.kind
        result.broker_label = selection.label

        def log(event_type: str, payload: dict) -> None:
            self._log(run_id, event_type, payload, on_event=on_event)
            if event_type == "risk.decision" and not payload.get("approved"):
                codes = payload.get("reason_codes") or []
                session_memory.record_risk_note(
                    ",".join(codes) if codes else str(payload.get("message", "rejected"))
                )
            if event_type == "order.filled":
                session_memory.note_entry(
                    direction=session_memory.last_entry_direction,
                    entry_price=payload.get("entry_price"),
                )

        session_memory = SessionMicroMemory()
        capital_budget = config.capital_budget or _default_capital_budget(self.app_settings)

        def on_trade_outcome(payload: dict) -> None:
            rec = session_memory.record_outcome(
                exit_reason=str(payload.get("exit_reason") or "unknown"),
                exit_price=payload.get("exit_price"),
                mae=payload.get("mae"),
                mfe=payload.get("mfe"),
                duration_minutes=payload.get("duration_minutes"),
                realized_pnl_usd=payload.get("realized_pnl_usd"),
            )
            log(
                "agent.outcome",
                {
                    "quality_note": rec.quality_note,
                    "exit_reason": rec.exit_reason,
                    "duration_minutes": rec.duration_minutes,
                },
            )

        log(
            "live_paper.started",
            {
                "symbol": config.symbol,
                "duration_seconds": config.duration_seconds,
                "mock_agents": config.mock_agents,
                "broker": selection.kind,
                "broker_label": selection.label,
                "auto_orders": selection.auto_orders,
                "live_money_orders": False,
                "is_synthetic": False,
                "capital": capital_budget.prompt_dict(),
            },
        )

        provider = WebullMarketDataProvider(
            self.env_settings,
            api=config.webull_api,
            quote_max_age_seconds=self.app_settings.risk.quote_max_age_seconds,
            feed_max_silence_seconds=self.app_settings.risk.feed_max_silence_seconds,
            allow_delayed_quotes=self.app_settings.risk.allow_delayed_quotes,
            poll_interval_seconds=self.app_settings.data.quote_poll_interval_seconds,
        )

        try:
            ok = provider.authenticate()
            log("webull.auth.result", {"success": ok})
            if not ok:
                raise LivePaperError("Webull market-data authentication failed")
        except Exception as exc:
            msg = str(exc)
            result.errors.append(msg)
            result.feed_health = "ERROR"
            log("provider.error", {"error": msg})
            result.summary = ReplaySummary(
                run_id=run_id,
                session_name="live_paper",
                is_synthetic=False,
                mock_agents=config.mock_agents,
                events_processed=0,
                signals_detected=0,
                trades_entered=0,
                trades_exited=0,
                final_pnl_usd=0.0,
                risk_rejections=0,
                failures=[msg],
            )
            run_manager.end_run(run_id)
            return result

        options_provider: WebullOptionsDataProvider | None = None
        try:
            options_provider = create_webull_options_provider(
                self.env_settings,
                api=config.webull_options_api,
                app_settings=self.app_settings,
            )
            options_provider.authenticate()
            if capability_usable_for_shadow():
                options_provider.verified = True
                result.options_available = True
            log("options.capability",
                {
                    "usable": capability_usable_for_shadow(),
                    "verified": options_provider.verified,
                },
            )
        except Exception as exc:
            msg = f"options_provider_failed: {exc}"
            log("options.unavailable", {"reason": str(exc)})
            if config.require_options:
                result.errors.append(msg)
                result.failures.append(msg)
                run_manager.end_run(run_id)
                return result

        # Warm snapshot from real Webull. Candles are best-effort (stock_bars may be unverified).
        try:
            try:
                candle_events = provider.fetch_candle_events("1m")
                log("market.candles_loaded",
                    {"count": len(candle_events), "source": "webull_stock"},
                )
            except Exception as candle_exc:
                log("market.candles_unavailable",
                    {
                        "reason": str(candle_exc),
                        "fallback": "quote_derived_candles",
                    },
                )

            snapshot_event = provider.fetch_snapshot_event()
            # Seed at least one candle from the live quote so features can evolve.
            snap0 = provider.get_latest_snapshot()
            if snap0 is not None and not snap0.candles:
                provider.append_quote_as_candle(snapshot_event)

            snapshot = provider.get_latest_snapshot()
            if snapshot is None:
                raise LivePaperError("No SPY snapshot from Webull")
            log("market.warmup",
                {
                    "candles": len(snapshot.candles),
                    "price": snapshot.price,
                    "delayed": provider.last_quote_delayed,
                    "feed_health": provider.feed_health,
                    "candle_source": getattr(provider, "candle_source", "unknown"),
                    "has_volume_bars": bool(getattr(provider, "has_volume_bars", False)),
                    "snapshot_event_id": snapshot_event.event_id,
                },
            )
        except Exception as exc:
            msg = f"warmup_failed: {exc}"
            result.errors.append(msg)
            result.failures.append(msg)
            log("provider.error", {"error": msg})
            result.summary = ReplaySummary(
                run_id=run_id,
                session_name="live_paper",
                is_synthetic=False,
                mock_agents=config.mock_agents,
                events_processed=0,
                signals_detected=0,
                trades_entered=0,
                trades_exited=0,
                final_pnl_usd=0.0,
                risk_rejections=0,
                failures=list(result.failures),
            )
            run_manager.end_run(run_id)
            return result

        risk_config = _risk_config_from_settings(self.app_settings, capital_budget)
        validator = PlaybookQualityValidator(risk_config)
        playbook: Playbook | None = None
        council_analysis: CouncilAnalysis | None = None
        playbook_validation: PlaybookValidationResult | None = None
        armed = False

        features = FeatureEngine(
            max_age_seconds=self.app_settings.risk.feed_max_silence_seconds
        ).compute(snapshot, reference_time=provider.current_time)

        from joker.memory import build_day_memory, save_session_lesson

        day_memory = build_day_memory(
            data_dir=self.app_settings.data_dir,
            db=self.db,
            as_of=trading_day,
            lookback_days=self.app_settings.agents.memory_lookback_days,
        )
        log("memory.loaded",
            {
                "available": day_memory.memory_available,
                "lessons": len(day_memory.prior_lessons),
                "recent_pnl_usd": day_memory.recent_pnl_usd,
            },
        )

        if config.approved_playbook is not None:
            # Test/injection path only — production CLI never sets this.
            playbook = config.approved_playbook.model_copy(update={"approved": True})
            playbook = trim_playbook_enabled_setups(playbook, risk_config)
            playbook_validation = validator.validate(playbook)
            if not playbook_validation.approved:
                failures.append(
                    f"injected_playbook_invalid: {playbook_validation.reason_codes}"
                )
        else:
            council_settings = self.app_settings.model_copy(
                update={
                    "agents": self.app_settings.agents.model_copy(
                        update={"mock_agents": config.mock_agents}
                    ),
                    "mode": mode,
                }
            )
            council = create_agent_council(
                council_settings,
                self.env_settings,
                llm_client=config.llm_client,
            )
            premarket = PremarketWorkflow(
                self.db, self.event_log, council_settings, council=council
            )
            try:
                pb = premarket.run(
                    run_id,
                    trading_day,
                    features,
                    env_settings=self.env_settings,
                    memory=day_memory,
                )
                records = self.db.list_by_run(AgentDecisionRecord, run_id)
                if records:
                    from joker.schemas.domain import AgentCouncilDecision

                    council_decision = AgentCouncilDecision.model_validate(
                        records[0].payload
                    )
                    council_analysis = analyze_council(council_decision)
                playbook = trim_playbook_enabled_setups(pb, risk_config)
                if playbook != pb:
                    log("playbook.trimmed",
                        {
                            "enabled_before": sum(1 for s in pb.setups if s.enabled),
                            "enabled_after": sum(1 for s in playbook.setups if s.enabled),
                        },
                    )
                playbook_validation = validator.validate(
                    playbook,
                    critic_blocked=(
                        council_analysis.council_blocked if council_analysis else False
                    ),
                )
                log("playbook.validation",
                    playbook_validation.model_dump(mode="json"),
                )
                if not playbook_validation.approved:
                    failures.append(
                        f"playbook_validation_failed: {playbook_validation.reason_codes}"
                    )
                elif council_analysis and council_analysis.council_blocked:
                    failures.append("council_blocked: critic flagged weak plan")
                else:
                    playbook = premarket.approve_playbook(run_id, playbook)
            except (AgentError, LLMClientError) as exc:
                failures.append(f"openai_council_failed: {exc}")
                log("live_paper.failure", {"error": failures[-1]})

        reactive = ReactiveEngine(
            RiskGovernor(risk_config, mode, live_enabled=False),
            broker,
        )

        if playbook and playbook.approved and (
            playbook_validation is None or playbook_validation.approved
        ):
            try:
                reactive.arm_playbook(playbook)
                armed = True
                log(
                    "playbook.approved",
                    {
                        "playbook_id": playbook.playbook_id,
                        "enabled_setups": sum(1 for s in playbook.setups if s.enabled),
                        "broker": selection.kind,
                    },
                )
            except StateMachineError as exc:
                failures.append(f"playbook_arm_failed: {exc}")
        else:
            if not failures:
                failures.append("no_active_playbook")
            log("live_paper.failure", {"error": "no_active_playbook"})

        daily_state = DailyState(
            trading_day=trading_day,
            run_id=run_id,
            mode=mode.value,
            playbook_approved=armed,
        )

        def on_log(event_type: str, payload: dict) -> None:
            log(event_type, payload)
            if event_type == "risk.decision":
                self.db.save(
                    RiskDecisionRecord(
                        run_id=run_id,
                        candidate_id=payload.get("candidate_id", "unknown"),
                        approved=payload.get("approved", False),
                        reason_codes=payload.get("reason_codes", []),
                        payload=payload,
                    )
                )
            if event_type == "signal.detected":
                self.db.save(
                    TradeCandidateRecord(
                        run_id=run_id,
                        candidate_id=payload.get("candidate_id", "unknown"),
                        payload=payload,
                    )
                )

        handler = MarketEventHandler(
            provider=provider,
            reactive_engine=reactive,
            risk_governor=reactive.risk_governor,
            broker=broker,
            feature_engine=FeatureEngine(
                max_age_seconds=self.app_settings.risk.feed_max_silence_seconds
            ),
            option_selector=OptionSelector(
                OptionSelectorConfig(
                    max_spread_pct=self.app_settings.risk.max_spread_pct,
                    max_premium_usd=self.app_settings.risk.max_premium_usd,
                    quote_max_age_seconds=self.app_settings.risk.quote_max_age_seconds,
                    allow_delayed_quotes=self.app_settings.risk.allow_delayed_quotes,
                    feed_max_silence_seconds=self.app_settings.risk.feed_max_silence_seconds,
                    delayed_quote_max_age_seconds=(
                        self.app_settings.risk.delayed_quote_max_age_seconds
                    ),
                    soft_liquidity_advisory=(
                        (self.app_settings.agents.execution_mode or "").strip().lower()
                        == "agent_led"
                        or (self.app_settings.risk.policy or "").strip().lower()
                        == "agent_led"
                    ),
                )
            ),
            exit_manager=ExitManager(),
            mode=mode,
            run_id=run_id,
            daily_state=daily_state,
            on_log=on_log,
            options_provider=options_provider,
            rules_auto_entry=(
                (self.app_settings.agents.execution_mode or "rules_hybrid").strip().lower()
                != "agent_led"
            ),
            on_trade_outcome=on_trade_outcome,
            capital_budget=capital_budget,
        )
        handler._pause_when_goal_met = bool(
            self.app_settings.capital.pause_entries_when_goal_met
        )
        # Prior / premarket levels from warmed candle buffer when available
        try:
            from joker.features.engine import split_session_candles

            snap_levels = provider.get_latest_snapshot()
            if snap_levels is not None and snap_levels.candles:
                prior, pm, _rth = split_session_candles(snap_levels.candles)
                handler._prior_day_candles = prior
                handler._premarket_candles = pm
        except Exception:
            pass
        if armed and playbook:
            handler.state.setups_armed = len([s for s in playbook.setups if s.enabled])

        events_processed = 0
        intraday_calls = 0
        decision_calls = 0
        proposals_acted = 0
        agent_cfg = self.app_settings.agents
        execution_mode = (agent_cfg.execution_mode or "rules_hybrid").strip().lower()
        agent_led = execution_mode == "agent_led"
        log(
            "execution.mode",
            {
                "execution_mode": execution_mode,
                "risk_policy": risk_config.policy,
                "rules_auto_entry": not agent_led,
            },
        )

        if armed:
            try:
                import time as _time

                from joker.agents.decision import (
                    DecisionAgentError,
                    mock_decision,
                    run_decision_agent,
                )
                from joker.agents.intraday import (
                    IntradayAgentError,
                    mock_intraday_result,
                    run_intraday_council,
                )
                from joker.agents.llm_client import OpenAILLMClient
                from joker.strategy.playbook_patch import PatchError, apply_patch

                poll = max(0.5, self.app_settings.data.quote_poll_interval_seconds)
                deadline = _time.monotonic() + max(config.duration_seconds, poll)
                last_intraday_at = 0.0
                last_decision_at = 0.0
                decision_interval = float(
                    getattr(agent_cfg, "decision_interval_seconds", 45.0) or 45.0
                )
                max_decision_calls = int(
                    getattr(agent_cfg, "max_decision_calls_per_session", 40) or 40
                )

                while _time.monotonic() < deadline:
                    try:
                        event = provider.fetch_snapshot_event()
                    except Exception as exc:
                        log("provider.poll_error", {"reason": str(exc)})
                        _time.sleep(poll)
                        continue

                    events_processed += 1
                    if isinstance(event, SpyQuoteEvent):
                        provider.append_quote_as_candle(event)
                    handler.handle_event(event)

                    latest = provider.get_latest_snapshot()
                    if (
                        handler.state.open_trade is not None
                        and options_provider is not None
                        and latest is not None
                    ):
                        try:
                            call_snap, put_snap = options_provider.fetch_atm_snapshots(
                                latest.price
                            )
                            snaps = [s for s in (call_snap, put_snap) if s is not None]
                            if snaps:
                                opt_events = options_provider.to_quote_events(
                                    snaps,
                                    reference_time=provider.current_time,
                                    allow_delayed_quotes=(
                                        self.app_settings.risk.allow_delayed_quotes
                                    ),
                                    feed_max_silence_seconds=(
                                        self.app_settings.risk.feed_max_silence_seconds
                                    ),
                                    delayed_quote_max_age_seconds=(
                                        self.app_settings.risk.delayed_quote_max_age_seconds
                                    ),
                                )
                                for oe in opt_events:
                                    handler.handle_event(oe)
                        except Exception as exc:
                            log(
                                "options.exit_refresh_failed",
                                {"reason": str(exc)},
                            )

                    now_mono = _time.monotonic()

                    # --- agent_led: primary AI decision loop (propose → confirm) ---
                    has_pending = session_memory.pending is not None
                    fast_confirm_min = float(
                        getattr(agent_cfg, "fast_confirm_min_seconds", 8.0) or 8.0
                    )
                    use_prefilter = bool(getattr(agent_cfg, "use_edge_prefilter", True))
                    interval_due = (now_mono - last_decision_at) >= (
                        fast_confirm_min if has_pending else decision_interval
                    )
                    if (
                        agent_led
                        and agent_cfg.intraday_enabled
                        and decision_calls < max_decision_calls
                        and proposals_acted < agent_cfg.max_proposals_per_session
                        and interval_due
                        and handler.state.open_trade is None
                        and handler.state.pending_entry is None
                    ):
                        snap_now = provider.get_latest_snapshot()
                        if snap_now is not None:
                            from joker.features.engine import split_session_candles

                            prior_c, pm_c, _ = split_session_candles(snap_now.candles or [])
                            if prior_c:
                                handler._prior_day_candles = prior_c
                            if pm_c:
                                handler._premarket_candles = pm_c
                            feat_now = FeatureEngine(
                                max_age_seconds=self.app_settings.risk.feed_max_silence_seconds
                            ).compute(
                                snap_now,
                                prior_day_candles=getattr(handler, "_prior_day_candles", None),
                                premarket_candles=getattr(handler, "_premarket_candles", None),
                                reference_time=provider.current_time,
                            )

                            skip_llm = False
                            if (
                                use_prefilter
                                and not has_pending
                                and not config.mock_agents
                            ):
                                from joker.strategy.edge_prefilter import edge_prefilter

                                pre = edge_prefilter(
                                    feat_now, goal_met=capital_budget.goal_met
                                )
                                if not pre.candidate:
                                    log(
                                        "agent.prefilter_skip",
                                        {"reason": pre.reason},
                                    )
                                    last_decision_at = now_mono
                                    skip_llm = True

                            if not skip_llm:
                                last_decision_at = now_mono
                                option_context: dict = {}
                                if (
                                    options_provider is not None
                                    and options_provider.is_available()
                                ):
                                    try:
                                        call_snap, put_snap = (
                                            options_provider.fetch_atm_snapshots(
                                                snap_now.price
                                            )
                                        )
                                        if call_snap is not None:
                                            option_context["atm_call"] = {
                                                "strike": call_snap.contract.strike,
                                                "bid": call_snap.bid,
                                                "ask": call_snap.ask,
                                                "mid": call_snap.mid,
                                                "spread_pct": call_snap.spread_pct,
                                            }
                                        if put_snap is not None:
                                            option_context["atm_put"] = {
                                                "strike": put_snap.contract.strike,
                                                "bid": put_snap.bid,
                                                "ask": put_snap.ask,
                                                "mid": put_snap.mid,
                                                "spread_pct": put_snap.spread_pct,
                                            }
                                    except Exception as exc:
                                        option_context["error"] = str(exc)
                                try:
                                    from joker.agents.decision import (
                                        confirm_gate,
                                        decision_from_pending,
                                        ev_entry_allowed,
                                        pending_from_decision,
                                    )

                                    require_propose = bool(
                                        getattr(
                                            agent_cfg,
                                            "require_propose_before_enter",
                                            True,
                                        )
                                    )
                                    # Expire stale proposals before asking the agent
                                    if session_memory.pending is not None:
                                        ok, reason = confirm_gate(
                                            session_memory.pending,
                                            spy_price=float(snap_now.price),
                                            option_context=option_context,
                                            ttl_seconds=float(
                                                getattr(
                                                    agent_cfg, "confirm_ttl_seconds", 120.0
                                                )
                                            ),
                                            max_spy_drift_pct=float(
                                                getattr(
                                                    agent_cfg,
                                                    "max_confirm_spy_drift_pct",
                                                    0.20,
                                                )
                                            ),
                                            max_option_mid_worsen_pct=float(
                                                getattr(
                                                    agent_cfg,
                                                    "max_confirm_option_mid_worsen_pct",
                                                    15.0,
                                                )
                                            ),
                                        )
                                        if not ok and reason.startswith("proposal_expired"):
                                            log(
                                                "agent.propose_expired",
                                                {"reason": reason},
                                            )
                                            session_memory.clear_pending()

                                    if config.mock_agents:
                                        decision = mock_decision(
                                            feat_now,
                                            playbook,
                                            session_memory=session_memory,
                                        )
                                    else:
                                        llm = config.llm_client or OpenAILLMClient(
                                            api_key=self.env_settings.openai_api_key,
                                            model=self.env_settings.openai_model,
                                            max_retries=agent_cfg.max_retries,
                                            default_timeout_seconds=float(
                                                agent_cfg.council_timeout_seconds
                                            ),
                                        )
                                        decision = run_decision_agent(
                                            llm,
                                            agent_cfg,
                                            run_id=run_id,
                                            playbook=playbook,
                                            features=feat_now,
                                            risk=risk_config,
                                            memory=day_memory,
                                            session_memory=session_memory,
                                            capital_budget=capital_budget,
                                            open_position=False,
                                            trades_entered=handler.state.trades_entered,
                                            spy_price=snap_now.price,
                                            option_context=option_context,
                                        )
                                    decision_calls += 1
                                    session_memory.record_decision(
                                        action=decision.action,
                                        direction=decision.direction,
                                        confidence=decision.confidence,
                                        summary=decision.summary or decision.rationale,
                                        spy_price=snap_now.price,
                                    )
                                    session_memory.update_option_mids(option_context)
                                    log(
                                        "agent.decision",
                                        {
                                            "action": decision.action,
                                            "direction": decision.direction,
                                            "confidence": decision.confidence,
                                            "win_probability": decision.win_probability,
                                            "expected_r": decision.expected_r,
                                            "expected_value_usd": decision.expected_value_usd,
                                            "summary": (
                                                decision.summary or decision.rationale
                                            )[:160],
                                            "pending": bool(session_memory.pending),
                                            "goal_gap_pct": capital_budget.goal_gap_pct,
                                            "aggression_cap": capital_budget.aggression_cap(
                                                minutes_to_close=feat_now.minutes_to_close
                                            ),
                                        },
                                    )
                                    if decision.patch is not None and playbook is not None:
                                        try:
                                            playbook = apply_patch(playbook, decision.patch)
                                            reactive.active_playbook = playbook
                                            log(
                                                "playbook.patched",
                                                {
                                                    "disable": decision.patch.disable_setup_ids,
                                                    "enable": decision.patch.enable_setup_ids,
                                                },
                                            )
                                        except PatchError as exc:
                                            log(
                                                "playbook.patch_rejected",
                                                {"reason": str(exc)},
                                            )

                                    action = decision.action
                                    if action == "abandon":
                                        session_memory.clear_pending()
                                        log("agent.propose_abandoned", {})
                                    elif action == "propose":
                                        if decision.direction in ("long_call", "long_put"):
                                            session_memory.set_pending(
                                                pending_from_decision(
                                                    decision,
                                                    spy_price=float(snap_now.price),
                                                    option_context=option_context,
                                                )
                                            )
                                            log(
                                                "agent.propose",
                                                {
                                                    "direction": decision.direction,
                                                    "confidence": decision.confidence,
                                                    "win_probability": decision.win_probability,
                                                    "expected_value_usd": decision.expected_value_usd,
                                                    "goal_gap_pct": capital_budget.goal_gap_pct,
                                                },
                                            )
                                    elif action in ("confirm", "enter"):
                                        pending = session_memory.pending
                                        pending_decision = None
                                        if require_propose and pending is None:
                                            # Treat bare enter as propose when two-step required
                                            if action == "enter" and decision.direction in (
                                                "long_call",
                                                "long_put",
                                            ):
                                                session_memory.set_pending(
                                                    pending_from_decision(
                                                        decision,
                                                        spy_price=float(snap_now.price),
                                                        option_context=option_context,
                                                    )
                                                )
                                                log(
                                                    "agent.propose",
                                                    {
                                                        "direction": decision.direction,
                                                        "confidence": decision.confidence,
                                                        "via": "enter_downgraded_to_propose",
                                                        "win_probability": decision.win_probability,
                                                        "expected_value_usd": decision.expected_value_usd,
                                                    },
                                                )
                                            else:
                                                log(
                                                    "agent.confirm_rejected",
                                                    {"reason": "no_pending_proposal"},
                                                )
                                        else:
                                            if pending is None:
                                                pending_decision = decision
                                            else:
                                                ok, reason = confirm_gate(
                                                    pending,
                                                    spy_price=float(snap_now.price),
                                                    option_context=option_context,
                                                    ttl_seconds=float(
                                                        getattr(
                                                            agent_cfg,
                                                            "confirm_ttl_seconds",
                                                            120.0,
                                                        )
                                                    ),
                                                    max_spy_drift_pct=float(
                                                        getattr(
                                                            agent_cfg,
                                                            "max_confirm_spy_drift_pct",
                                                            0.20,
                                                        )
                                                    ),
                                                    max_option_mid_worsen_pct=float(
                                                        getattr(
                                                            agent_cfg,
                                                            "max_confirm_option_mid_worsen_pct",
                                                            15.0,
                                                        )
                                                    ),
                                                )
                                                if not ok:
                                                    log(
                                                        "agent.confirm_rejected",
                                                        {"reason": reason},
                                                    )
                                                    if reason.startswith("proposal_expired"):
                                                        session_memory.clear_pending()
                                                    pending_decision = None
                                                else:
                                                    pending_decision = decision_from_pending(
                                                        pending,
                                                        confidence=decision.confidence
                                                        or pending.confidence,
                                                        capital_fraction=decision.capital_fraction,
                                                        target_contracts=decision.target_contracts,
                                                        allocation_style=decision.allocation_style,
                                                        win_probability=decision.win_probability,
                                                        expected_r=decision.expected_r,
                                                        expected_value_usd=decision.expected_value_usd,
                                                    )
                                                    if decision.rationale:
                                                        pending_decision = (
                                                            pending_decision.model_copy(
                                                                update={
                                                                    "rationale": decision.rationale,
                                                                    "summary": decision.summary
                                                                    or pending.summary,
                                                                }
                                                            )
                                                        )
                                        if pending_decision is not None:
                                            ok_ev, ev_reason = ev_entry_allowed(
                                                pending_decision,
                                                min_win_probability=float(
                                                    capital_budget.plan.min_win_probability
                                                ),
                                            )
                                            if not ok_ev:
                                                log(
                                                    "agent.confirm_rejected",
                                                    {"reason": ev_reason},
                                                )
                                            else:
                                                acted = handler.try_enter_from_decision(
                                                    pending_decision,
                                                    min_confidence=agent_cfg.min_proposal_confidence,
                                                )
                                                if acted:
                                                    proposals_acted += 1
                                                    session_memory.note_entry(
                                                        direction=pending_decision.direction,
                                                        entry_price=None,
                                                    )
                                                    session_memory.clear_pending()
                                                    log(
                                                        "agent.confirm_executed",
                                                        {
                                                            "direction": pending_decision.direction,
                                                            "confidence": pending_decision.confidence,
                                                            "win_probability": pending_decision.win_probability,
                                                            "expected_value_usd": pending_decision.expected_value_usd,
                                                            "goal_gap_pct": capital_budget.goal_gap_pct,
                                                        },
                                                    )
                                except (DecisionAgentError, Exception) as exc:
                                    log("agent.decision_failed", {"reason": str(exc)})

                    # --- rules_hybrid: legacy intraday council ---
                    elif (
                        not agent_led
                        and agent_cfg.intraday_enabled
                        and playbook is not None
                        and intraday_calls < agent_cfg.max_intraday_calls_per_session
                        and proposals_acted < agent_cfg.max_proposals_per_session
                        and (now_mono - last_intraday_at)
                        >= agent_cfg.intraday_interval_seconds
                        and handler.state.open_trade is None
                    ):
                        last_intraday_at = now_mono
                        snap_now = provider.get_latest_snapshot()
                        if snap_now is not None:
                            feat_now = FeatureEngine(
                                max_age_seconds=self.app_settings.risk.feed_max_silence_seconds
                            ).compute(snap_now, reference_time=provider.current_time)
                            try:
                                if config.mock_agents:
                                    intra = mock_intraday_result(
                                        playbook, feat_now, run_id=run_id
                                    )
                                else:
                                    llm = config.llm_client or OpenAILLMClient(
                                        api_key=self.env_settings.openai_api_key,
                                        model=self.env_settings.openai_model,
                                        max_retries=agent_cfg.max_retries,
                                        default_timeout_seconds=float(
                                            agent_cfg.council_timeout_seconds
                                        ),
                                    )
                                    intra = run_intraday_council(
                                        llm,
                                        agent_cfg,
                                        run_id=run_id,
                                        playbook=playbook,
                                        features=feat_now,
                                        memory=day_memory,
                                        open_position=False,
                                        trades_entered=handler.state.trades_entered,
                                        max_trades=self.app_settings.risk.max_trades_per_day,
                                    )
                                intraday_calls += 1
                                log(
                                    "intraday.council",
                                    {
                                        "summary": intra.summary,
                                        "has_patch": intra.patch is not None,
                                        "propose_entry": bool(
                                            intra.proposal and intra.proposal.propose_entry
                                        ),
                                    },
                                )
                                if intra.patch is not None:
                                    try:
                                        playbook = apply_patch(playbook, intra.patch)
                                        reactive.active_playbook = playbook
                                        log(
                                            "playbook.patched",
                                            {
                                                "disable": intra.patch.disable_setup_ids,
                                                "enable": intra.patch.enable_setup_ids,
                                            },
                                        )
                                    except PatchError as exc:
                                        log(
                                            "playbook.patch_rejected",
                                            {"reason": str(exc)},
                                        )
                                if intra.proposal is not None:
                                    acted = handler.try_enter_from_proposal(
                                        intra.proposal,
                                        min_confidence=agent_cfg.min_proposal_confidence,
                                    )
                                    if acted:
                                        proposals_acted += 1
                            except (IntradayAgentError, Exception) as exc:
                                log("intraday.failed", {"reason": str(exc)})

                    result.feed_health = provider.feed_health
                    result.options_available = bool(
                        options_provider and options_provider.is_available()
                    )
                    if on_state:
                        snap = provider.get_latest_snapshot()
                        on_state(
                            {
                                "run_id": run_id,
                                "provider": ProviderKind.WEBULL.value,
                                "market_price": snap.price if snap else None,
                                "feed_health": provider.feed_health,
                                "delayed": provider.last_quote_delayed,
                                "options_available": result.options_available,
                                "signals": handler.state.signals_detected,
                                "trades_entered": handler.state.trades_entered,
                                "trades_exited": handler.state.trades_exited,
                                "paper_pnl": broker.get_daily_pnl(),
                                "engine_state": reactive.state.value,
                                "intraday_calls": intraday_calls,
                                "decision_calls": decision_calls,
                                "proposals_acted": proposals_acted,
                                "execution_mode": execution_mode,
                                "capital_available": capital_budget.available_usd,
                                "capital_goal_pct": capital_budget.progress_to_goal_pct,
                                "broker": selection.kind,
                                "broker_label": selection.label,
                                "pending_order": bool(handler.state.pending_entry),
                                "open_trade": handler.state.open_trade is not None,
                                "auto_orders": selection.auto_orders,
                                "live_money_orders": False,
                            }
                        )
                    _time.sleep(min(poll, max(0.0, deadline - _time.monotonic())))
            except Exception as exc:
                msg = str(exc)
                result.errors.append(msg)
                result.feed_health = "ERROR"
                failures.append(msg)
                log("provider.error", {"error": msg})

        # Postmarket learner → day memory
        if armed and self.app_settings.agents.postmarket_learner_enabled:
            try:
                from joker.agents.intraday import (
                    mock_session_lesson,
                    run_postmarket_learner,
                )
                from joker.agents.llm_client import OpenAILLMClient

                session_stats = {
                    "events_processed": events_processed,
                    "signals_detected": handler.state.signals_detected,
                    "trades_entered": handler.state.trades_entered,
                    "trades_exited": handler.state.trades_exited,
                    "final_pnl_usd": broker.get_daily_pnl(),
                    "risk_rejections": handler.state.risk_rejections,
                    "intraday_calls": intraday_calls,
                    "decision_calls": decision_calls,
                    "proposals_acted": proposals_acted,
                    "execution_mode": execution_mode,
                    "risk_reject_codes": [],
                }
                if config.mock_agents:
                    lesson = mock_session_lesson(trading_day, session_stats)
                else:
                    llm = config.llm_client or OpenAILLMClient(
                        api_key=self.env_settings.openai_api_key,
                        model=self.env_settings.openai_model,
                        max_retries=agent_cfg.max_retries,
                        default_timeout_seconds=float(agent_cfg.council_timeout_seconds),
                    )
                    lesson = run_postmarket_learner(
                        llm,
                        agent_cfg,
                        trading_day=trading_day,
                        session_stats=session_stats,
                        memory=day_memory,
                    )
                save_session_lesson(self.app_settings.data_dir, lesson)
                log("memory.lesson_saved",
                    {
                        "trading_day": lesson.trading_day.isoformat(),
                        "summary": lesson.summary[:120],
                    },
                )
            except Exception as exc:
                log("memory.lesson_failed", {"reason": str(exc)})

        quality_metrics = (
            compute_quality_metrics(handler.state) if armed else StrategyQualityMetrics()
        )
        summary = (
            handler.build_summary(mock_agents=config.mock_agents, is_synthetic=False)
            if armed
            else ReplaySummary(
                run_id=run_id,
                session_name="live_paper",
                is_synthetic=False,
                mock_agents=config.mock_agents,
                events_processed=0,
                signals_detected=0,
                trades_entered=0,
                trades_exited=0,
                final_pnl_usd=0.0,
                risk_rejections=0,
                failures=failures,
            )
        )
        summary = summary.model_copy(
            update={
                "run_id": run_id,
                "events_processed": events_processed,
                "playbook_validation_approved": (
                    playbook_validation.approved if playbook_validation else False
                ),
                "council_blocked": (
                    council_analysis.council_blocked if council_analysis else False
                ),
                "failures": failures,
                "final_pnl_usd": broker.get_daily_pnl(),
            }
        )

        report_ctx = ReplayReportContext(
            run_id=run_id,
            trading_day=trading_day,
            is_synthetic=False,
            mock_agents=config.mock_agents,
            playbook=playbook,
            playbook_validation=playbook_validation,
            council_analysis=council_analysis,
            quality_metrics=quality_metrics,
            exit_decisions=handler.state.exit_decisions if armed else [],
            failures=failures,
            replay_summary=summary.model_dump(mode="json"),
        )
        report_path = ReplayReportGenerator(
            self.db, self.app_settings.reports_dir
        ).generate_replay_postmarket(report_ctx)

        self.db.save(
            SystemEventRecord(
                run_id=run_id,
                event_type="live_paper.summary",
                source="live_paper",
                payload={
                    **summary.model_dump(mode="json"),
                    "quality_metrics": quality_metrics.to_dict(),
                    "feed_health": result.feed_health,
                },
            )
        )
        log("live_paper.completed", summary.model_dump(mode="json"))
        run_manager.end_run(run_id)

        result.summary = summary
        result.report_path = report_path
        result.failures = failures
        result.playbook = playbook
        result.council_analysis = council_analysis
        result.playbook_validation = playbook_validation
        result.events_processed = events_processed
        result.paper_pnl_usd = broker.get_daily_pnl()
        return result
