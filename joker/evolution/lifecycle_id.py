"""Stable position-lifecycle identity helpers."""

from __future__ import annotations


def make_position_lifecycle_id(
    *,
    session_id: str,
    originating_entry_client_order_id: str,
    contract_id: str,
) -> str:
    """Authoritative lifecycle identity — never derived from broker-ID ordering."""
    return f"{session_id}:{originating_entry_client_order_id}:{contract_id}"
