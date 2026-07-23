"""Premarket workflow."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from joker.agents.council import create_agent_council
from joker.agents.mock_agents import AgentCouncilProtocol
from joker.config.settings import AppSettings, EnvSettings
from joker.logging.event_log import EventLogWriter
from joker.schemas.domain import Playbook, TechnicalFeatures
from joker.storage.database import Database
from joker.storage.models import AgentDecisionRecord


def write_premarket_report(path: Path, playbook: Playbook) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Premarket Report — {playbook.trading_day.isoformat()}

## {playbook.title}

{playbook.summary}

### Setups

"""
    for setup in playbook.setups:
        content += f"- **{setup.name}** ({setup.direction}) — stop: {setup.stop_rule}, TP: {setup.take_profit_rule}\n"
    path.write_text(content, encoding="utf-8")
    return path


class PremarketWorkflow:
    def __init__(
        self,
        db: Database,
        event_log: EventLogWriter,
        settings: AppSettings,
        council: AgentCouncilProtocol | None = None,
    ) -> None:
        self.db = db
        self.event_log = event_log
        self.settings = settings
        self.council = council

    def run(
        self,
        run_id: str,
        trading_day: date,
        features: TechnicalFeatures,
        env_settings: EnvSettings | None = None,
        memory: object | None = None,
    ) -> Playbook:
        council = self.council or create_agent_council(self.settings, env_settings)
        decision, playbook = council.run_premarket(
            run_id=run_id,
            trading_day=trading_day,
            features=features,
            max_loss=self.settings.risk.max_daily_loss_usd,
            max_trades=self.settings.risk.max_trades_per_day,
            memory=memory,
        )
        self.db.save(
            AgentDecisionRecord(
                run_id=run_id,
                agent_name="AgentCouncil",
                decision_type="premarket",
                payload=decision.model_dump(mode="json"),
            )
        )
        self.event_log.append(
            run_id=run_id,
            mode=self.settings.mode.value,
            source="premarket",
            event_type="playbook.created",
            payload={"playbook_id": playbook.playbook_id},
        )
        report_path = Path(self.settings.reports_dir) / "premarket" / f"{trading_day.isoformat()}.md"
        write_premarket_report(report_path, playbook)
        return playbook

    def approve_playbook(self, run_id: str, playbook: Playbook) -> Playbook:
        approved = playbook.model_copy(update={"approved": True})
        self.event_log.append(
            run_id=run_id,
            mode=self.settings.mode.value,
            source="premarket",
            event_type="playbook.approved",
            payload={"playbook_id": approved.playbook_id},
        )
        return approved
