"""Application-level safety and mode definitions."""

from enum import Enum


class SafetyMode(str, Enum):
    """Trading safety mode. LIVE_GATED requires explicit config opt-in."""

    PAPER = "PAPER"
    SHADOW = "SHADOW"
    LIVE_GATED = "LIVE_GATED"

    @classmethod
    def from_string(cls, value: str) -> "SafetyMode":
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"Invalid safety mode '{value}'. Must be one of: {valid}."
            ) from exc

    def allows_broker_submit(self, live_trading_enabled: bool) -> bool:
        """Return True only when live broker order submission is permitted."""
        if self is SafetyMode.LIVE_GATED and live_trading_enabled:
            return True
        return False

    def allows_paper_execution(self) -> bool:
        return self is SafetyMode.PAPER

    def records_shadow_candidates(self) -> bool:
        return self is SafetyMode.SHADOW
