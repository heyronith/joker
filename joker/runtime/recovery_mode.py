"""Explicit recovery-mode semantics shared across runtime and CLI paths."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class RecoveryMode(StrEnum):
    NORMAL = "normal"
    RECONCILIATION_ONLY = "reconciliation_only"
    BROKER_ONLY = "broker_only"


def coerce_recovery_mode(value: Any) -> RecoveryMode:
    """Return a validated recovery mode, defaulting to normal."""
    if isinstance(value, RecoveryMode):
        return value
    raw = str(value or RecoveryMode.NORMAL.value).strip().lower()
    try:
        return RecoveryMode(raw)
    except ValueError as exc:  # pragma: no cover - defensive fail-closed
        raise ValueError(f"invalid recovery mode: {value!r}") from exc


def recovery_mode_value(source: Any) -> RecoveryMode:
    """Extract the explicit recovery mode, falling back to the legacy boolean."""
    value = getattr(source, "recovery_mode", None)
    if value is not None:
        mode = coerce_recovery_mode(value)
        if mode is RecoveryMode.NORMAL and bool(
            getattr(source, "reconciliation_only_recovery", False)
        ):
            return RecoveryMode.RECONCILIATION_ONLY
        return mode
    if bool(getattr(source, "reconciliation_only_recovery", False)):
        return RecoveryMode.RECONCILIATION_ONLY
    return RecoveryMode.NORMAL


def is_recovery_only_mode(source: Any) -> bool:
    """True when the runtime should skip the full cognitive entry stack."""
    return recovery_mode_value(source) in {
        RecoveryMode.RECONCILIATION_ONLY,
        RecoveryMode.BROKER_ONLY,
    }
