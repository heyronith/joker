"""Convert Webull option snapshots into Task 1 option-surface ingest rows."""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

from joker.runtime.execution_runtime import contract_id_for
from joker.schemas.domain import OptionContract
from joker.schemas.options_data import OptionSnapshot


def option_snapshot_to_surface_row(
    snapshot: OptionSnapshot,
    *,
    trading_date: date | None = None,
) -> dict[str, Any]:
    """Build a Task 1 ``ingest_option_quotes`` row from a Webull OptionSnapshot.

    ``is_0dte`` is derived only after comparing contract expiry to the exchange
    trading date — never assumed true at conversion time.
    """
    meta = snapshot.contract
    expiry = meta.expiration
    is_0dte = trading_date is not None and expiry == trading_date
    contract = OptionContract(
        symbol=meta.underlying_symbol or "SPY",
        expiration=expiry,
        strike=float(meta.strike),
        option_type=meta.option_type,
        is_0dte=is_0dte,
    )
    return {
        "contract_id": contract_id_for(contract),
        "symbol": contract.symbol,
        "expiry": expiry,
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
        "is_0dte": is_0dte,
    }


def option_snapshots_to_surface_rows(
    snapshots: Sequence[OptionSnapshot],
    *,
    trading_date: date | None = None,
) -> list[dict[str, Any]]:
    """Convert snapshots whose expiry matches ``trading_date`` when provided."""
    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        if trading_date is not None and snap.contract.expiration != trading_date:
            continue
        if (snap.contract.underlying_symbol or "SPY").upper() != "SPY":
            continue
        try:
            rows.append(
                option_snapshot_to_surface_row(snap, trading_date=trading_date)
            )
        except Exception:
            continue
    return rows


def filter_0dte_contracts(
    snapshots: Sequence[OptionSnapshot],
    *,
    trading_date: date,
) -> list[OptionSnapshot]:
    """Keep only SPY contracts whose expiry equals the exchange trading date."""
    return [
        s
        for s in snapshots
        if (s.contract.underlying_symbol or "SPY").upper() == "SPY"
        and s.contract.expiration == trading_date
    ]
