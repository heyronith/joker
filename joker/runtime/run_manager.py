"""Run lifecycle management."""

from __future__ import annotations

from datetime import date

from joker.config.settings import AppSettings
from joker.logging.event_log import EventLogWriter
from joker.storage.database import Database
from joker.storage.models import RunStatus, TradingDayStateRecord, utc_now


class RunManager:
    """Coordinates run creation, event logging, and trading day state."""

    def __init__(
        self,
        db: Database,
        event_log: EventLogWriter,
        app_settings: AppSettings,
    ) -> None:
        self.db = db
        self.event_log = event_log
        self.app_settings = app_settings

    def start_run(self, trading_day: date | None = None) -> str:
        day = trading_day or date.today()
        run = self.db.create_run(
            mode=self.app_settings.mode.value,
            trading_day=day,
            config_snapshot={
                "mode": self.app_settings.mode.value,
                "live_trading_enabled": self.app_settings.live_trading_enabled,
            },
        )
        self.event_log.append(
            run_id=run.run_id,
            mode=run.mode,
            source="runtime",
            event_type="run.started",
            payload={"trading_day": day.isoformat()},
        )
        self.db.upsert_trading_day_state(
            TradingDayStateRecord(
                trading_day=day,
                run_id=run.run_id,
                mode=run.mode,
            )
        )
        return run.run_id

    def end_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETED) -> None:
        run = self.db.get_run(run_id)
        if run is None:
            return
        run.status = status.value
        run.ended_at = utc_now()
        self.db.save(run)
        self.event_log.append(
            run_id=run_id,
            mode=run.mode,
            source="runtime",
            event_type="run.ended",
            payload={"status": status.value},
        )
