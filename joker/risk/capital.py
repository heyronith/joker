"""Daily authorized capital budget, goal tracking, and deterministic sizing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


class CapitalError(Exception):
    pass


@dataclass
class CapitalPlan:
    """User-confirmed session capital + goal (paper/sandbox risk budget)."""

    authorized_usd: float
    target_profit_pct: float = 20.0
    max_concurrent_positions: int = 1
    max_contracts_per_trade: int = 20
    min_contracts_per_trade: int = 1
    goal_label: str = "daily_profit_target"
    confirmed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    aggression_mode: str = "goal_adaptive"
    max_kelly_fraction: float = 0.35
    min_win_probability: float = 0.45
    behind_goal_boost: float = 0.15
    ahead_goal_dampen: float = 0.15

    def __post_init__(self) -> None:
        if self.authorized_usd <= 0:
            raise CapitalError("authorized_usd must be > 0")
        if self.target_profit_pct < 0:
            raise CapitalError("target_profit_pct must be >= 0")
        if self.max_concurrent_positions < 1:
            raise CapitalError("max_concurrent_positions must be >= 1")
        if self.max_contracts_per_trade < 1:
            raise CapitalError("max_contracts_per_trade must be >= 1")
        if self.min_contracts_per_trade < 1:
            raise CapitalError("min_contracts_per_trade must be >= 1")
        if self.min_contracts_per_trade > self.max_contracts_per_trade:
            raise CapitalError("min_contracts_per_trade > max_contracts_per_trade")

    @property
    def target_profit_usd(self) -> float:
        return self.authorized_usd * (self.target_profit_pct / 100.0)


@dataclass
class AllocationResult:
    quantity: int
    notional_usd: float
    remaining_after_usd: float
    reason: str
    capital_fraction_used: float
    aggression_cap: float = 0.0
    ev_gate: str = ""


@dataclass
class CapitalBudget:
    """
    Ledger for one session.

    authorized_usd = max premium capital the user is willing to put at risk today
    (debit paid for long options), not broker buying power / live money.
    """

    plan: CapitalPlan
    reserved_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    trades_opened: int = 0
    open_positions: int = 0

    @property
    def authorized_usd(self) -> float:
        return self.plan.authorized_usd

    @property
    def available_usd(self) -> float:
        return max(0.0, self.plan.authorized_usd - self.reserved_usd)

    @property
    def target_profit_usd(self) -> float:
        return self.plan.target_profit_usd

    @property
    def progress_to_goal_pct(self) -> float:
        target = self.target_profit_usd
        if target <= 0:
            return 100.0 if self.realized_pnl_usd >= 0 else 0.0
        return max(0.0, min(200.0, (self.realized_pnl_usd / target) * 100.0))

    @property
    def goal_met(self) -> bool:
        return self.realized_pnl_usd >= self.target_profit_usd

    @property
    def goal_gap_pct(self) -> float:
        """How far behind (positive) or ahead (negative) of the goal, in % of target."""
        return 100.0 - self.progress_to_goal_pct

    def aggression_cap(self, *, minutes_to_close: float | None = None) -> float:
        """
        Max capital fraction allowed given goal progress and session clock.

        Behind goal with time left → boost. Ahead / late day → dampen.
        ``target_attainment`` mode never applies late-session dampening;
        urgency emerges from the target-hit probability policy instead.
        """
        base = self.plan.max_kelly_fraction
        if self.plan.aggression_mode == "target_attainment":
            return max(0.05, min(1.0, float(base) if base > 0 else 1.0))
        if self.plan.aggression_mode != "goal_adaptive":
            return base

        progress = self.progress_to_goal_pct
        cap = base
        if progress < 50.0:
            # Behind: boost up to behind_goal_boost
            deficit = (50.0 - progress) / 50.0
            cap = min(0.85, base + self.plan.behind_goal_boost * deficit)
        elif progress >= 100.0:
            cap = min(base, 0.15)
        elif progress >= 80.0:
            cap = max(0.15, base - self.plan.ahead_goal_dampen)

        if minutes_to_close is not None:
            if minutes_to_close < 30:
                cap = min(cap, 0.25)
            elif minutes_to_close < 60:
                cap = min(cap, max(0.2, base))
        return max(0.05, min(0.85, cap))

    def prompt_dict(self, *, minutes_to_close: float | None = None) -> dict[str, Any]:
        agg = self.aggression_cap(minutes_to_close=minutes_to_close)
        stance = "hunt"
        if self.goal_met:
            stance = "defend_pause"
        elif self.progress_to_goal_pct >= 80:
            stance = "defend"
        elif self.progress_to_goal_pct < 40:
            stance = "hunt_aggressive"
        return {
            "authorized_usd": round(self.plan.authorized_usd, 2),
            "available_usd": round(self.available_usd, 2),
            "reserved_usd": round(self.reserved_usd, 2),
            "realized_pnl_usd": round(self.realized_pnl_usd, 2),
            "target_profit_pct": self.plan.target_profit_pct,
            "target_profit_usd": round(self.target_profit_usd, 2),
            "progress_to_goal_pct": round(self.progress_to_goal_pct, 1),
            "goal_gap_pct": round(self.goal_gap_pct, 1),
            "goal_met": self.goal_met,
            "aggression_cap": round(agg, 3),
            "stance": stance,
            "open_positions": self.open_positions,
            "max_concurrent_positions": self.plan.max_concurrent_positions,
            "max_contracts_per_trade": self.plan.max_contracts_per_trade,
            "trades_opened": self.trades_opened,
            "min_win_probability": self.plan.min_win_probability,
            "note": (
                "Allocate within available_usd and aggression_cap. "
                "Only enter when expected_value_usd > 0 and win_probability >= min. "
                "Goal is a session objective — not a guarantee. Never exceed authorized capital."
            ),
        }

    def can_open_position(self) -> bool:
        return self.open_positions < self.plan.max_concurrent_positions

    def premium_cost_usd(self, premium_per_contract: float, quantity: int) -> float:
        return max(0.0, float(premium_per_contract)) * 100.0 * max(0, int(quantity))

    def max_affordable_contracts(self, premium_per_contract: float) -> int:
        if premium_per_contract <= 0:
            return 0
        per = premium_per_contract * 100.0
        if per <= 0:
            return 0
        raw = int(self.available_usd // per)
        return max(0, min(raw, self.plan.max_contracts_per_trade))

    def allocate(
        self,
        *,
        premium_per_contract: float,
        capital_fraction: float | None = None,
        target_contracts: int | None = None,
        confidence: float = 0.5,
        allocation_style: Literal["auto", "aggressive", "split", "conservative"] = "auto",
        win_probability: float | None = None,
        expected_value_usd: float | None = None,
        expected_r: float | None = None,
        minutes_to_close: float | None = None,
    ) -> AllocationResult:
        """
        Deterministic size under capital ceiling + EV / aggression gates.

        Agent hints are soft; EV ≤ 0 or low p(win) rejects. Fraction is capped by
        aggression_cap (goal/time adaptive).
        """
        agg = self.aggression_cap(minutes_to_close=minutes_to_close)
        if not self.can_open_position():
            return AllocationResult(0, 0.0, self.available_usd, "max_concurrent_positions", 0.0, agg)
        if premium_per_contract <= 0:
            return AllocationResult(0, 0.0, self.available_usd, "invalid_premium", 0.0, agg)

        if expected_value_usd is not None and expected_value_usd <= 0:
            return AllocationResult(
                0, 0.0, self.available_usd, "ev_non_positive", 0.0, agg, "ev<=0"
            )
        if win_probability is not None and win_probability < self.plan.min_win_probability:
            return AllocationResult(
                0,
                0.0,
                self.available_usd,
                "win_probability_low",
                0.0,
                agg,
                f"p_win<{self.plan.min_win_probability}",
            )

        max_qty = self.max_affordable_contracts(premium_per_contract)
        if max_qty < self.plan.min_contracts_per_trade:
            return AllocationResult(0, 0.0, self.available_usd, "insufficient_capital", 0.0, agg)

        if capital_fraction is not None:
            frac = max(0.0, min(1.0, float(capital_fraction)))
        elif allocation_style == "aggressive":
            frac = min(1.0, 0.7 + 0.3 * confidence)
        elif allocation_style == "conservative":
            frac = min(0.35, 0.15 + 0.25 * confidence)
        elif allocation_style == "split":
            frac = min(0.55, 0.25 + 0.4 * confidence)
        else:
            frac = min(0.85, 0.2 + 0.7 * confidence)

        # Scale by EV quality when provided
        if expected_r is not None and expected_r > 0:
            frac = min(1.0, frac * min(1.5, 0.5 + expected_r / 2.0))
        if win_probability is not None:
            # Kelly-ish soft scale: p - (1-p)/b approximated via expected_r when present
            if expected_r is not None and expected_r > 0:
                kelly = max(0.0, win_probability - (1.0 - win_probability) / max(expected_r, 0.1))
                frac = min(frac, max(0.05, kelly))
            else:
                frac = frac * (0.5 + 0.5 * win_probability)

        frac = min(frac, agg)

        if target_contracts is not None:
            desired = int(target_contracts)
        else:
            budget_slice = self.available_usd * frac
            per = premium_per_contract * 100.0
            desired = int(budget_slice // per) if per > 0 else 0

        # Cap target_contracts by aggression as well
        qty = max(0, min(desired, max_qty))
        if qty < self.plan.min_contracts_per_trade:
            if frac > 0 and max_qty >= self.plan.min_contracts_per_trade:
                qty = self.plan.min_contracts_per_trade
            else:
                return AllocationResult(0, 0.0, self.available_usd, "qty_below_min", 0.0, agg)

        notional = self.premium_cost_usd(premium_per_contract, qty)
        if notional > self.available_usd + 1e-6:
            return AllocationResult(0, 0.0, self.available_usd, "capital_exceeded", 0.0, agg)

        used_frac = notional / self.available_usd if self.available_usd > 0 else 0.0
        return AllocationResult(
            quantity=qty,
            notional_usd=notional,
            remaining_after_usd=self.available_usd - notional,
            reason="ok",
            capital_fraction_used=used_frac,
            aggression_cap=agg,
            ev_gate="ok",
        )

    def reserve(self, notional_usd: float) -> None:
        if notional_usd < 0:
            raise CapitalError("cannot reserve negative notional")
        if notional_usd > self.available_usd + 1e-6:
            raise CapitalError(
                f"reserve {notional_usd:.2f} exceeds available {self.available_usd:.2f}"
            )
        self.reserved_usd += notional_usd
        self.open_positions += 1
        self.trades_opened += 1

    def release(self, notional_usd: float, *, realized_pnl_usd: float = 0.0) -> None:
        self.reserved_usd = max(0.0, self.reserved_usd - max(0.0, notional_usd))
        self.open_positions = max(0, self.open_positions - 1)
        self.realized_pnl_usd += realized_pnl_usd
