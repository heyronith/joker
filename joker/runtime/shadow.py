"""Shadow mode runtime — records would-trade without broker submit."""

from __future__ import annotations

from dataclasses import dataclass, field

from joker.app.safety import SafetyMode
from joker.broker.interface import BrokerClient
from joker.compliance.opra_sanitizer import shadow_safe_metadata
from joker.schemas.domain import RiskDecision, TradeCandidate


@dataclass
class ShadowRecord:
    candidate: TradeCandidate
    decision: RiskDecision
    simulated_entry: float
    simulated_exit: float | None = None
    simulated_pnl: float | None = None
    shadow_result_label: str | None = None
    risk_multiple_bucket: str | None = None

    def persist_metadata(self) -> dict:
        """Non-price metadata safe for SQLite/JSONL persistence."""
        return shadow_safe_metadata(
            setup_id=self.candidate.setup_id,
            direction=self.candidate.direction,
            candidate_created=True,
            would_trade_created=self.decision.approved,
            risk_reason_codes=list(self.decision.reason_codes),
            shadow_result_label=self.shadow_result_label,
            exit_reason=None,
            risk_multiple_bucket=self.risk_multiple_bucket,
        )


@dataclass
class ShadowRuntime:
    mode: SafetyMode
    records: list[ShadowRecord] = field(default_factory=list)

    def record_candidate(
        self,
        candidate: TradeCandidate,
        decision: RiskDecision,
        broker: BrokerClient,
    ) -> ShadowRecord:
        if self.mode is not SafetyMode.SHADOW:
            raise RuntimeError("ShadowRuntime requires SHADOW mode")
        if broker.list_open_orders():
            pass  # shadow never submits
        record = ShadowRecord(
            candidate=candidate,
            decision=decision,
            simulated_entry=candidate.entry_limit_price,
        )
        self.records.append(record)
        return record

    def simulate_outcome(self, record: ShadowRecord, exit_price: float) -> ShadowRecord:
        record.simulated_exit = exit_price
        pnl = (exit_price - record.simulated_entry) * 100
        record.simulated_pnl = pnl
        if pnl > 0:
            record.shadow_result_label = "WIN"
            record.risk_multiple_bucket = ">1R" if pnl > record.simulated_entry * 100 else "0-1R"
        elif pnl < 0:
            record.shadow_result_label = "LOSS"
            record.risk_multiple_bucket = "<0R"
        else:
            record.shadow_result_label = "FLAT"
            record.risk_multiple_bucket = "0-1R"
        return record
