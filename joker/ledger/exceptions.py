"""Ledger domain exceptions."""

from __future__ import annotations


class LedgerError(Exception):
    """Base error for ledger operations."""


class IdempotencyConflict(LedgerError):
    """Raised when an append conflicts on a unique idempotency key with different payload."""


class ReconciliationError(LedgerError):
    """Raised when reconciliation cannot produce a safe report."""
