"""Postmarket report generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from joker.storage.database import Database
from joker.storage.models import (
    AgentDecisionRecord,
    RiskDecisionRecord,
    TradeCandidateRecord,
)


@dataclass
class PerformanceMetrics:
    daily_pnl_usd: float = 0.0
    max_drawdown_usd: float = 0.0
    candidate_count: int = 0
    trade_count: int = 0
    skipped_trades: int = 0
    rejection_reasons: dict[str, int] | None = None
    rule_violations: int = 0
    slippage_assumption_pct: float = 2.0


def compute_metrics(
    candidates: list,
    risk_decisions: list,
    *,
    daily_pnl: float = 0.0,
    slippage_pct: float = 2.0,
) -> PerformanceMetrics:
    rejection_reasons: dict[str, int] = {}
    approved = 0
    rejected = 0
    for rd in risk_decisions:
        approved_flag = getattr(rd, "approved", False)
        reasons = getattr(rd, "reason_codes", None) or []
        if isinstance(rd, dict):
            approved_flag = rd.get("approved", False)
            reasons = rd.get("reason_codes", [])
        if approved_flag:
            approved += 1
        else:
            rejected += 1
            for code in reasons:
                rejection_reasons[code] = rejection_reasons.get(code, 0) + 1
    return PerformanceMetrics(
        daily_pnl_usd=daily_pnl,
        candidate_count=len(candidates),
        trade_count=approved,
        skipped_trades=rejected,
        rejection_reasons=rejection_reasons,
        slippage_assumption_pct=slippage_pct,
    )


class ReportGenerator:
    def __init__(self, db: Database, reports_dir: Path) -> None:
        self.db = db
        self.reports_dir = Path(reports_dir)

    def generate_postmarket(self, run_id: str, trading_day: date) -> Path:
        candidates = self.db.list_by_run(TradeCandidateRecord, run_id)
        risk_decisions = self.db.list_by_run(RiskDecisionRecord, run_id)
        agent_decisions = self.db.list_by_run(AgentDecisionRecord, run_id)

        metrics = compute_metrics(candidates, risk_decisions)

        path = self.reports_dir / "postmarket" / f"{trading_day.isoformat()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Postmarket Report — {trading_day.isoformat()}",
            "",
            f"Run ID: `{run_id}`",
            "",
            "## Performance",
            f"- P&L: ${metrics.daily_pnl_usd:,.2f}",
            f"- Candidates: {metrics.candidate_count}",
            f"- Trades: {metrics.trade_count}",
            f"- Skipped: {metrics.skipped_trades}",
            f"- Slippage assumption: {metrics.slippage_assumption_pct}%",
            "",
            "## Risk Rejections",
        ]
        if metrics.rejection_reasons:
            for code, count in sorted(metrics.rejection_reasons.items()):
                lines.append(f"- {code}: {count}")
        else:
            lines.append("- None")

        lines.extend(["", "## Agent Decisions", f"- Total: {len(agent_decisions)}"])

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
