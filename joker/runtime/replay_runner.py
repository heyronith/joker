"""Full replay session orchestration with OpenAI council and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from joker.agents.council import create_agent_council
from joker.agents.council_analysis import CouncilAnalysis, analyze_council
from joker.agents.llm_client import LLMClientError
from joker.agents.openai_agents import AgentError
from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.config.settings import AppSettings, EnvSettings
from joker.data.replay_loader import ReplayLoadError, load_replay_file
from joker.data.replay_provider import ReplayMarketDataProvider
from joker.execution.exit_manager import ExitManager
from joker.execution.option_selector import OptionSelector, OptionSelectorConfig
from joker.features.engine import FeatureEngine
from joker.logging.event_log import EventLogWriter
from joker.reporting.metrics import StrategyQualityMetrics, compute_quality_metrics
from joker.reporting.replay_report import ReplayReportContext, ReplayReportGenerator
from joker.risk.governor import RiskGovernor
from joker.runtime.market_handler import MarketEventHandler
from joker.runtime.premarket import PremarketWorkflow
from joker.runtime.reactive_engine import ReactiveEngine, StateMachineError
from joker.runtime.replay_clock import ReplayClockController
from joker.runtime.replay_errors import EmptyReplayFailure, ReplayLoadFailure
from joker.runtime.run_manager import RunManager
from joker.schemas.domain import AgentCouncilDecision, DailyState, Playbook, PlaybookSetup, RiskConfig
from joker.schemas.replay import OptionQuoteEvent, ReplaySpeedMode, ReplaySummary
from joker.storage.database import Database, ensure_database
from joker.storage.models import AgentDecisionRecord, RiskDecisionRecord, SystemEventRecord, TradeCandidateRecord
from joker.strategy.playbook_quality import PlaybookQualityValidator, PlaybookValidationResult, trim_playbook_enabled_setups


@dataclass
class ReplayRunConfig:
    replay_path: Path
    deterministic: bool = True
    speed: float = 1.0
    mock_agents: bool = True
    skip_premarket: bool = False
    llm_client: Any | None = None


@dataclass
class ReplayRunResult:
    summary: ReplaySummary
    report_path: Path | None = None
    failures: list[str] = field(default_factory=list)
    playbook: Playbook | None = None
    council_analysis: CouncilAnalysis | None = None
    playbook_validation: PlaybookValidationResult | None = None


class ReplayRunner:
    """Run a full replay session end-to-end with validation and reporting."""

    def __init__(
        self,
        app_settings: AppSettings,
        env_settings: EnvSettings | None = None,
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

    def _log(self, run_id: str, event_type: str, payload: dict) -> None:
        self.event_log.append(
            run_id=run_id,
            mode=self.app_settings.mode.value,
            source="replay",
            event_type=event_type,
            payload=payload,
        )

    def run(self, config: ReplayRunConfig) -> ReplayRunResult:
        failures: list[str] = []
        playbook: Playbook | None = None
        council_analysis: CouncilAnalysis | None = None
        playbook_validation: PlaybookValidationResult | None = None

        try:
            session = load_replay_file(config.replay_path)
        except ReplayLoadError as exc:
            raise ReplayLoadFailure(str(exc)) from exc

        if not session.events:
            raise EmptyReplayFailure("Replay file contains no events")

        run_manager = RunManager(self.db, self.event_log, self.app_settings)
        trading_day = session.metadata.trading_day
        run_id = run_manager.start_run(trading_day=trading_day)

        self._log(
            run_id,
            "replay.started",
            {
                "file": str(config.replay_path),
                "is_synthetic": session.metadata.is_synthetic,
                "deterministic": config.deterministic,
                "mock_agents": config.mock_agents,
            },
        )

        risk_config = RiskConfig(
            max_daily_loss_usd=self.app_settings.risk.max_daily_loss_usd,
            max_trades_per_day=self.app_settings.risk.max_trades_per_day,
            max_open_positions=self.app_settings.risk.max_open_positions,
            max_premium_usd=self.app_settings.risk.max_premium_usd,
            max_spread_pct=self.app_settings.risk.max_spread_pct,
            quote_max_age_seconds=self.app_settings.risk.quote_max_age_seconds,
        )
        validator = PlaybookQualityValidator(risk_config)
        armed = False

        if not config.skip_premarket:
            pre_provider = ReplayMarketDataProvider(session)
            snapshot = None
            for _ in pre_provider.stream_events():
                snapshot = pre_provider.get_latest_snapshot()
                if snapshot and len(snapshot.candles) >= 3:
                    break
            if snapshot is None:
                failures.append("missing_snapshot: using fallback mock snapshot")
                from joker.data.mock_provider import mock_spy_snapshot

                snapshot = mock_spy_snapshot()

            features = FeatureEngine(max_age_seconds=999999).compute(
                snapshot,
                reference_time=snapshot.timestamp,
            )
            council_settings = self.app_settings.model_copy(
                update={
                    "agents": self.app_settings.agents.model_copy(
                        update={"mock_agents": config.mock_agents}
                    )
                }
            )
            env_settings = self.env_settings
            if not config.mock_agents and env_settings is None:
                env_settings = EnvSettings()
            council = create_agent_council(
                council_settings,
                env_settings,
                llm_client=config.llm_client,
            )
            premarket = PremarketWorkflow(self.db, self.event_log, council_settings, council=council)
            try:
                pb = premarket.run(run_id, trading_day, features, env_settings=env_settings)
                records = self.db.list_by_run(AgentDecisionRecord, run_id)
                if records:
                    council_decision = AgentCouncilDecision.model_validate(records[0].payload)
                    council_analysis = analyze_council(council_decision)
                playbook = trim_playbook_enabled_setups(pb, risk_config)
                playbook_validation = validator.validate(
                    playbook,
                    critic_blocked=council_analysis.council_blocked if council_analysis else False,
                )
                self._log(run_id, "playbook.validation", playbook_validation.model_dump(mode="json"))
                if not playbook_validation.approved:
                    failures.append(
                        f"playbook_validation_failed: {playbook_validation.reason_codes}"
                    )
                    self._log(run_id, "replay.failure", {"error": failures[-1]})
                elif council_analysis and council_analysis.council_blocked:
                    failures.append("council_blocked: critic flagged weak plan")
                    self._log(run_id, "replay.failure", {"error": failures[-1]})
                else:
                    playbook = premarket.approve_playbook(run_id, playbook)
            except (AgentError, LLMClientError) as exc:
                failures.append(f"openai_council_failed: {exc}")
                self._log(run_id, "replay.failure", {"error": failures[-1]})
        else:
            playbook = Playbook(
                trading_day=trading_day,
                title="Replay test playbook",
                summary="test",
                setups=[
                    PlaybookSetup(
                        name="Trend call",
                        direction="long_call",
                        entry_conditions=["VWAP reclaim"],
                        stop_rule="50% premium stop",
                        take_profit_rule="100% premium target",
                    ),
                    PlaybookSetup(
                        name="Breakdown put",
                        direction="long_put",
                        entry_conditions=["VWAP rejection"],
                        stop_rule="50% premium stop",
                        take_profit_rule="100% premium target",
                    ),
                ],
                approved=True,
            )
            playbook_validation = validator.validate(playbook)

        provider = ReplayMarketDataProvider(session)
        if config.deterministic:
            controller = ReplayClockController.deterministic(provider.clock)
        else:
            provider.clock.mode = ReplaySpeedMode.ACCELERATED
            controller = ReplayClockController.accelerated(provider.clock, config.speed)

        broker = PaperBroker(
            initial_balance=self.app_settings.paper.initial_balance_usd,
            slippage_pct=0.0,
        )
        mode = self.app_settings.mode
        reactive = ReactiveEngine(
            RiskGovernor(risk_config, mode, self.app_settings.live_trading_enabled),
            broker,
        )

        if playbook and playbook.approved and (
            playbook_validation is None or playbook_validation.approved
        ):
            try:
                reactive.arm_playbook(playbook)
                armed = True
            except StateMachineError as exc:
                failures.append(f"playbook_arm_failed: {exc}")
                self._log(run_id, "replay.failure", {"error": failures[-1]})
        else:
            if not failures:
                failures.append("no_active_playbook")
            self._log(run_id, "replay.failure", {"error": "no_active_playbook"})

        daily_state = DailyState(
            trading_day=trading_day,
            run_id=run_id,
            mode=mode.value,
            playbook_approved=armed,
        )

        def on_log(event_type: str, payload: dict) -> None:
            self._log(run_id, event_type, payload)
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
            feature_engine=FeatureEngine(max_age_seconds=self.app_settings.risk.quote_max_age_seconds),
            option_selector=OptionSelector(
                OptionSelectorConfig(
                    max_spread_pct=self.app_settings.risk.max_spread_pct,
                    max_premium_usd=self.app_settings.risk.max_premium_usd,
                    quote_max_age_seconds=self.app_settings.risk.quote_max_age_seconds,
                )
            ),
            exit_manager=ExitManager(),
            mode=mode,
            run_id=run_id,
            daily_state=daily_state,
            on_log=on_log,
        )
        if armed and playbook:
            handler.state.setups_armed = len([s for s in playbook.setups if s.enabled])

        events_processed = 0
        prev_ts = provider.clock.current_time
        if armed:
            for event in provider.stream_events():
                controller.wait_until_next(event.timestamp)
                handler.handle_event(event)
                events_processed += 1
                prev_ts = event.timestamp
            if handler.state.open_trade and handler.state.latest_option_mid is not None:
                handler._try_exit(
                    OptionQuoteEvent(
                        timestamp=prev_ts,
                        symbol="SPY",
                        source="synthetic_option",
                        contract_id="eod",
                        expiration=trading_day,
                        strike=545.0,
                        option_type="put",
                        bid=handler.state.latest_option_mid,
                        ask=handler.state.latest_option_mid,
                        mid=handler.state.latest_option_mid,
                        spread_pct=1.0,
                        quote_timestamp=prev_ts,
                        is_synthetic=True,
                    )
                )
        quality_metrics = (
            compute_quality_metrics(handler.state) if armed else StrategyQualityMetrics()
        )

        summary = (
            handler.build_summary(mock_agents=config.mock_agents)
            if armed
            else ReplaySummary(
                run_id=run_id,
                session_name=session.metadata.name,
                is_synthetic=session.metadata.is_synthetic,
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
                "session_name": session.metadata.name,
                "is_synthetic": session.metadata.is_synthetic,
                "events_processed": events_processed,
                "playbook_validation_approved": (
                    playbook_validation.approved if playbook_validation else False
                ),
                "council_blocked": (
                    council_analysis.council_blocked if council_analysis else False
                ),
                "failures": failures,
            }
        )

        report_ctx = ReplayReportContext(
            run_id=run_id,
            trading_day=trading_day,
            is_synthetic=session.metadata.is_synthetic,
            mock_agents=config.mock_agents,
            playbook=playbook,
            playbook_validation=playbook_validation,
            council_analysis=council_analysis,
            quality_metrics=quality_metrics,
            exit_decisions=handler.state.exit_decisions if armed else [],
            failures=failures,
            replay_summary=summary.model_dump(mode="json"),
        )
        report_path = ReplayReportGenerator(self.db, self.app_settings.reports_dir).generate_replay_postmarket(
            report_ctx
        )

        self.db.save(
            SystemEventRecord(
                run_id=run_id,
                event_type="replay.summary",
                source="replay",
                payload={
                    **summary.model_dump(mode="json"),
                    "quality_metrics": quality_metrics.to_dict(),
                    "council_analysis": (
                        council_analysis.model_dump(mode="json") if council_analysis else None
                    ),
                },
            )
        )
        self._log(run_id, "replay.completed", summary.model_dump(mode="json"))
        run_manager.end_run(run_id)

        return ReplayRunResult(
            summary=summary,
            report_path=report_path,
            failures=failures,
            playbook=playbook,
            council_analysis=council_analysis,
            playbook_validation=playbook_validation,
        )
