"""Convert Webull option snapshots into Task 1 option-surface ingest rows."""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

from joker.runtime.execution_runtime import contract_id_for
from joker.schemas.domain import OptionContract
from joker.schemas.options_data import OptionSnapshot


def option_snapshot_to_surface_row(snapshot: OptionSnapshot) -> dict[str, Any]:
    """Build a Task 1 ``ingest_option_quotes`` row from a Webull OptionSnapshot."""
    meta = snapshot.contract
    contract = OptionContract(
        symbol=meta.underlying_symbol or "SPY",
        expiration=meta.expiration,
        strike=float(meta.strike),
        option_type=meta.option_type,
        is_0dte=True,
    )
    return {
        "contract_id": contract_id_for(contract),
        "symbol": contract.symbol,
        "expiry": meta.expiration,
        "strike": str(meta.strike),
        "option_type": meta.option_type,
        "bid": snapshot.bid,
        "ask": snapshot.ask,
        "last": snapshot.last,
        "volume": snapshot.volume,
        "open_interest": snapshot.open_interest,
        "implied_volatility": snapshot.implied_volatility,
        "delta": snapshot.delta,
        "gamma": snapshot.gamma,
        "theta": snapshot.theta,
        "vega": snapshot.vega,
        "quote_timestamp": snapshot.quote_timestamp,
        "source_contract_id": meta.contract_id,
    }


def option_snapshots_to_surface_rows(
    snapshots: Sequence[OptionSnapshot],
) -> list[dict[str, Any]]:
    """Convert many snapshots; skip entries that cannot form a valid 0DTE contract."""
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        try:
            rows.append(option_snapshot_to_surface_row(snap))
        except Exception:
            continue
    return rows


def filter_0dte_contracts(
    snapshots: Sequence[OptionSnapshot],
    *,
    trading_date: date,
) -> list[OptionSnapshot]:
    """Keep only contracts whose expiry equals the exchange trading date."""
    return [s for s in snapshots if s.contract.expiration == trading_date]
