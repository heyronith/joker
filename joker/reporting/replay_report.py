"""Enhanced postmarket reports for replay sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from joker.agents.council_analysis import CouncilAnalysis
from joker.compliance.opra_sanitizer import sanitize_for_report
from joker.reporting.generator import ReportGenerator, compute_metrics
from joker.reporting.metrics import StrategyQualityMetrics
from joker.schemas.domain import Playbook
from joker.storage.database import Database
from joker.storage.models import (
    AgentDecisionRecord,
    FillRecord,
    OrderRecord,
    RiskDecisionRecord,
    SystemEventRecord,
    TradeCandidateRecord,
)
from joker.strategy.playbook_quality import PlaybookValidationResult

SYNTHETIC_WARNING = (
    "> **Warning:** Synthetic replay results are not real performance and must not "
    "be interpreted as live trading performance."
)


@dataclass
class ReplayReportContext:
    run_id: str
    trading_day: date
    is_synthetic: bool
    mock_agents: bool
    playbook: Playbook | None = None
    playbook_validation: PlaybookValidationResult | None = None
    council_analysis: CouncilAnalysis | None = None
    quality_metrics: StrategyQualityMetrics | None = None
    exit_decisions: list[dict[str, Any]] | None = None
    failures: list[str] | None = None
    replay_summary: dict[str, Any] | None = None


class ReplayReportGenerator(ReportGenerator):
    """Generate auditable replay postmarket reports."""

    def generate_replay_postmarket(self, ctx: ReplayReportContext) -> Path:
        db = self.db
        run_id = ctx.run_id
        trading_day = ctx.trading_day

        candidates = db.list_by_run(TradeCandidateRecord, run_id)
        risk_decisions = db.list_by_run(RiskDecisionRecord, run_id)
        agent_decisions = db.list_by_run(AgentDecisionRecord, run_id)
        orders = db.list_by_run(OrderRecord, run_id)
        fills = db.list_by_run(FillRecord, run_id)
        system_events = db.list_by_run(SystemEventRecord, run_id)

        metrics = compute_metrics(candidates, risk_decisions)
        qm = ctx.quality_metrics

        agent_mode = "mock" if ctx.mock_agents else "openai"
        data_label = "synthetic replay" if ctx.is_synthetic else "replay"

        path = self.reports_dir / "postmarket" / f"{trading_day.isoformat()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Postmarket Report — {trading_day.isoformat()}",
            "",
            SYNTHETIC_WARNING if ctx.is_synthetic else "",
            "",
            f"Run ID: `{run_id}`",
            f"- Data source: **{data_label}**",
            f"- Agent mode: **{agent_mode} agents**",
            "",
        ]

        if ctx.failures:
            lines.extend(["## Failures", ""])
            for f in ctx.failures:
                lines.append(f"- {f}")
            lines.append("")

        if ctx.playbook:
            lines.extend([
                "## Playbook Summary",
                f"- Title: {ctx.playbook.title}",
                f"- Approved: {ctx.playbook.approved}",
                f"- Setups: {len(ctx.playbook.setups)}",
                f"- Summary: {ctx.playbook.summary}",
                "",
            ])

        if ctx.playbook_validation:
            pv = ctx.playbook_validation
            lines.extend([
                "## Playbook Validation",
                f"- Approved: {pv.approved}",
                f"- Reason codes: {', '.join(pv.reason_codes) or 'none'}",
                f"- Warnings: {', '.join(pv.warnings) or 'none'}",
                "",
            ])

        if ctx.council_analysis:
            ca = ctx.council_analysis
            lines.extend([
                "## Agent Council",
                f"- Synthesizer rationale: {ca.synthesizer_rationale}",
                f"- Council blocked: {ca.council_blocked}",
                "",
                "### Confidence by Agent",
            ])
            for agent, conf in sorted(ca.agent_confidence.items()):
                flag = " (low)" if agent in ca.low_confidence_agents else ""
                lines.append(f"- {agent}: {conf:.2f}{flag}")
            if ca.conflicting_regimes:
                lines.append(f"- Conflicting regimes: {', '.join(ca.regime_values)}")
            if ca.critic_warnings:
                lines.extend(["", "### Critic Warnings"])
                for w in ca.critic_warnings:
                    lines.append(f"- {w}")
            lines.extend(["", "### Agent Opinions"])
            for rec in agent_decisions:
                payload = rec.payload if hasattr(rec, "payload") else {}
                opinions = payload.get("opinions", []) if isinstance(payload, dict) else []
                for op in opinions:
                    if isinstance(op, dict):
                        lines.append(
                            f"- **{op.get('agent_name', '?')}** ({op.get('confidence', '?')}): "
                            f"{op.get('summary', '')}"
                        )
            lines.append("")

        lines.extend([
            "## Strategy Quality Metrics",
        ])
        if qm:
            for key, val in qm.to_dict().items():
                lines.append(f"- {key}: {val}")
        else:
            lines.append("- No quality metrics recorded")
        lines.append("")

        lines.extend([
            "## Risk Governor Decisions",
            f"- Total candidates: {metrics.candidate_count}",
            f"- Approved trades: {metrics.trade_count}",
            f"- Rejected/skipped: {metrics.skipped_trades}",
        ])
        if metrics.rejection_reasons:
            for code, count in sorted(metrics.rejection_reasons.items()):
                lines.append(f"  - {code}: {count}")
        lines.append("")

        lines.extend([
            "## Paper Orders & Fills",
            f"- Orders: {len(orders)}",
            f"- Fills: {len(fills)}",
            "",
        ])

        if ctx.exit_decisions:
            lines.extend(["## Exit Decisions", ""])
            safe_exits = sanitize_for_report(ctx.exit_decisions)
            for ex in safe_exits if isinstance(safe_exits, list) else [safe_exits]:
                if isinstance(ex, dict):
                    lines.append(
                        f"- exit_reason: {ex.get('exit_reason', ex.get('reason', '?'))}"
                    )
                else:
                    lines.append(f"- {ex}")
            lines.append("")

        lines.extend([
            "## P&L (secondary metric)",
            f"- Daily P&L: ${metrics.daily_pnl_usd:,.2f}",
            f"- Slippage assumption: {metrics.slippage_assumption_pct}%",
            "",
            "## Data Quality",
        ])
        if qm:
            lines.append(f"- Missing data events: {qm.missing_data_events}")
            lines.append(f"- Stale quote events: {qm.stale_quote_events}")
            lines.append(f"- Wide spread rejects: {qm.wide_spread_rejects}")
        lines.append("")

        path.write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")
        return path
