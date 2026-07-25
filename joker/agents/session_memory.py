"""In-session micro-memory for DecisionAgent (not persisted long-term)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


DecisionAction = Literal["hold", "propose", "confirm", "abandon"]


@dataclass
class PendingProposal:
    direction: Literal["long_call", "long_put"]
    setup_id: str | None
    confidence: float
    stop_pct: float
    take_profit_pct: float
    rationale: str
    spy_price: float
    proposed_at: datetime
    atm_call_mid: float | None = None
    atm_put_mid: float | None = None
    summary: str = ""
    capital_fraction: float | None = None
    target_contracts: int | None = None
    allocation_style: str = "auto"
    win_probability: float | None = None
    expected_r: float | None = None
    expected_value_usd: float | None = None


@dataclass
class DecisionRecord:
    at: datetime
    action: str
    direction: str | None
    confidence: float
    summary: str
    spy_price: float | None = None


@dataclass
class TradeOutcomeRecord:
    at: datetime
    direction: str | None
    exit_reason: str
    entry_price: float | None
    exit_price: float | None
    realized_pnl_usd: float | None
    mae: float | None
    mfe: float | None
    duration_minutes: float | None
    quality_note: str = ""


def score_trade_quality(
    *,
    entry_price: float | None,
    exit_price: float | None,
    mae: float | None,
    mfe: float | None,
    exit_reason: str,
) -> str:
    """Compact in-session quality label for the next prompt (not a strategy claim)."""
    if entry_price is None or exit_price is None or entry_price <= 0:
        return f"incomplete_data reason={exit_reason}"
    pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
    mfe_pct = ((mfe or 0.0) / entry_price) * 100.0 if mfe is not None else None
    mae_pct = ((mae or 0.0) / entry_price) * 100.0 if mae is not None else None
    if pnl_pct > 5 and (mfe_pct is None or mfe_pct - pnl_pct < 15):
        grade = "good_capture"
    elif pnl_pct > 0:
        grade = "small_win"
    elif mfe_pct is not None and mfe_pct > 10 and pnl_pct < 0:
        grade = "gave_back_edge"
    elif mae_pct is not None and mae_pct > 25:
        grade = "poor_timing_or_stop"
    else:
        grade = "loss"
    return (
        f"{grade} pnl_pct={pnl_pct:.1f} "
        f"mfe_pct={mfe_pct if mfe_pct is not None else 'n/a'} "
        f"mae_pct={mae_pct if mae_pct is not None else 'n/a'} "
        f"exit={exit_reason}"
    )


@dataclass
class SessionMicroMemory:
    """Rolling context for one live paper session."""

    max_decisions: int = 10
    max_outcomes: int = 8
    pending: PendingProposal | None = None
    decisions: list[DecisionRecord] = field(default_factory=list)
    outcomes: list[TradeOutcomeRecord] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    last_option_mids: dict[str, float | None] = field(default_factory=dict)
    last_entry_direction: str | None = None
    last_entry_price: float | None = None

    def record_decision(
        self,
        *,
        action: str,
        direction: str | None,
        confidence: float,
        summary: str,
        spy_price: float | None = None,
    ) -> None:
        self.decisions.append(
            DecisionRecord(
                at=datetime.now(timezone.utc),
                action=action,
                direction=direction,
                confidence=confidence,
                summary=(summary or "")[:160],
                spy_price=spy_price,
            )
        )
        if len(self.decisions) > self.max_decisions:
            self.decisions = self.decisions[-self.max_decisions :]

    def record_risk_note(self, note: str) -> None:
        text = (note or "").strip()
        if not text:
            return
        self.risk_notes.append(text[:120])
        if len(self.risk_notes) > 12:
            self.risk_notes = self.risk_notes[-12:]

    def set_pending(self, proposal: PendingProposal) -> None:
        self.pending = proposal

    def clear_pending(self) -> None:
        self.pending = None

    def note_entry(self, *, direction: str | None, entry_price: float | None) -> None:
        self.last_entry_direction = direction
        self.last_entry_price = entry_price

    def record_outcome(
        self,
        *,
        exit_reason: str,
        exit_price: float | None,
        mae: float | None,
        mfe: float | None,
        duration_minutes: float | None,
        realized_pnl_usd: float | None = None,
        direction: str | None = None,
    ) -> TradeOutcomeRecord:
        direction = direction or self.last_entry_direction
        entry_price = self.last_entry_price
        quality = score_trade_quality(
            entry_price=entry_price,
            exit_price=exit_price,
            mae=mae,
            mfe=mfe,
            exit_reason=exit_reason,
        )
        rec = TradeOutcomeRecord(
            at=datetime.now(timezone.utc),
            direction=direction,
            exit_reason=exit_reason,
            entry_price=entry_price,
            exit_price=exit_price,
            realized_pnl_usd=realized_pnl_usd,
            mae=mae,
            mfe=mfe,
            duration_minutes=duration_minutes,
            quality_note=quality,
        )
        self.outcomes.append(rec)
        if len(self.outcomes) > self.max_outcomes:
            self.outcomes = self.outcomes[-self.max_outcomes :]
        self.last_entry_direction = None
        self.last_entry_price = None
        return rec

    def update_option_mids(self, option_context: dict[str, Any]) -> None:
        call = option_context.get("atm_call") or {}
        put = option_context.get("atm_put") or {}
        if isinstance(call, dict) and call.get("mid") is not None:
            self.last_option_mids["atm_call"] = float(call["mid"])
        if isinstance(put, dict) and put.get("mid") is not None:
            self.last_option_mids["atm_put"] = float(put["mid"])

    def prompt_dict(self) -> dict[str, Any]:
        pending = None
        if self.pending is not None:
            p = self.pending
            pending = {
                "direction": p.direction,
                "setup_id": p.setup_id,
                "confidence": p.confidence,
                "stop_pct": p.stop_pct,
                "take_profit_pct": p.take_profit_pct,
                "spy_price_at_propose": p.spy_price,
                "proposed_at": p.proposed_at.isoformat(),
                "atm_call_mid": p.atm_call_mid,
                "atm_put_mid": p.atm_put_mid,
                "summary": p.summary,
                "rationale": p.rationale[:200],
                "capital_fraction": p.capital_fraction,
                "target_contracts": p.target_contracts,
                "allocation_style": p.allocation_style,
                "win_probability": p.win_probability,
                "expected_r": p.expected_r,
                "expected_value_usd": p.expected_value_usd,
            }
        stats = self.expectancy_stats()
        return {
            "pending_proposal": pending,
            "session_expectancy": stats,
            "recent_decisions": [
                {
                    "at": d.at.isoformat(),
                    "action": d.action,
                    "direction": d.direction,
                    "confidence": d.confidence,
                    "summary": d.summary,
                    "spy_price": d.spy_price,
                }
                for d in self.decisions[-10:]
            ],
            "recent_outcomes": [
                {
                    "at": o.at.isoformat(),
                    "direction": o.direction,
                    "exit_reason": o.exit_reason,
                    "quality_note": o.quality_note,
                    "realized_pnl_usd": o.realized_pnl_usd,
                    "duration_minutes": o.duration_minutes,
                }
                for o in self.outcomes[-8:]
            ],
            "recent_risk_notes": list(self.risk_notes[-8:]),
            "prior_option_mids": dict(self.last_option_mids),
        }

    def expectancy_stats(self) -> dict[str, Any]:
        """Rolling win rate / avg R / expectancy from in-session outcomes."""
        rows = [o for o in self.outcomes if o.entry_price and o.exit_price and o.entry_price > 0]
        if not rows:
            return {
                "n": 0,
                "win_rate": None,
                "avg_r": None,
                "expectancy_r": None,
                "avg_pnl_usd": None,
            }
        rs: list[float] = []
        wins = 0
        pnls: list[float] = []
        for o in rows:
            assert o.entry_price and o.exit_price
            r = (o.exit_price - o.entry_price) / o.entry_price
            rs.append(r)
            if r > 0:
                wins += 1
            if o.realized_pnl_usd is not None:
                pnls.append(float(o.realized_pnl_usd))
            else:
                # Approximate 1-contract R in premium terms only
                pnls.append(r * o.entry_price * 100.0)
        n = len(rs)
        avg_r = sum(rs) / n
        return {
            "n": n,
            "win_rate": round(wins / n, 3),
            "avg_r": round(avg_r, 3),
            "expectancy_r": round(avg_r, 3),
            "avg_pnl_usd": round(sum(pnls) / len(pnls), 2) if pnls else None,
        }
