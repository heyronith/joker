"""Recover durable session objectives from Task 1 truth on restart."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


async def recover_session_objective(
    objective_service: Any,
    *,
    session_id: str,
    execution_runtime: Any | None = None,
    unresolved_reconciliation: bool = False,
) -> Any | None:
    """Load or recompute objective state before Task 2 starts.

    Merges open reservations with ledger projection (realised PnL + open
    position count). Never invents capital. Marks reconciliation unresolved
    so ENTRY/PROBE/ADD stay fail-closed until consistency is restored.
    """
    objective_service.mark_reconciliation_unresolved(bool(unresolved_reconciliation))
    realised = Decimal("0.00")
    open_count = 0
    if execution_runtime is not None:
        try:
            projection = await execution_runtime.project_session()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "objective_recovery_projection_failed",
                extra={"session_id": session_id, "error": str(exc)},
            )
            projection = None
        if projection is not None:
            positions = getattr(projection, "positions", {}) or {}
            for pos in positions.values():
                qty = getattr(pos, "quantity", None)
                if qty is None and isinstance(pos, dict):
                    qty = pos.get("quantity") or pos.get("net_quantity")
                try:
                    q = Decimal(str(qty or 0))
                except Exception:
                    q = Decimal("0")
                if q != 0:
                    open_count += 1
                pnl = getattr(pos, "realized_pnl", None)
                if pnl is None and isinstance(pos, dict):
                    pnl = pos.get("realized_pnl")
                if pnl is not None:
                    realised += Decimal(str(pnl))
    loaded = await objective_service.load_or_recover(session_id)
    if loaded is None:
        # Service may already hold an in-memory confirmed objective from CLI.
        try:
            return await objective_service.recompute_from_truth(
                realised_pnl_usd=realised,
                open_position_count=open_count,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "objective_recompute_skipped",
                extra={"session_id": session_id, "error": str(exc)},
            )
            return None
    return await objective_service.recompute_from_truth(
        realised_pnl_usd=realised,
        open_position_count=open_count,
    )
