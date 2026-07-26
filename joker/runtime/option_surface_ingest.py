"""Convert Webull option snapshots into Task 1 option-surface ingest rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

from joker.runtime.execution_runtime import contract_id_for
from joker.schemas.domain import OptionContract
from joker.schemas.options_data import OptionSnapshot


@dataclass
class SurfaceRowConversionResult:
    """Row conversion outcome with explicit conversion-failure accounting."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    input_count: int = 0
    converted_count: int = 0
    skipped_symbol_or_expiry: int = 0
    conversion_failures: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.conversion_failures and self.converted_count == self.input_count

    def to_data_quality_findings(self) -> list:
        if self.complete:
            return []
        from joker.market.quality import (
            DataQualityCode,
            DataQualityFinding,
            DataQualitySeverity,
        )

        return [
            DataQualityFinding(
                code=DataQualityCode.PARTIAL_OPTION_SURFACE,
                severity=DataQualitySeverity.ERROR,
                message=(
                    "option surface row conversion incomplete; persisted rows do not "
                    "cover every fetched snapshot"
                ),
                symbol="SPY",
                details={
                    "input_count": self.input_count,
                    "converted_count": self.converted_count,
                    "skipped_symbol_or_expiry": self.skipped_symbol_or_expiry,
                    "conversion_failure_count": len(self.conversion_failures),
                    "conversion_failures": "; ".join(self.conversion_failures)[:500],
                },
            )
        ]


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
    """Convert snapshots; prefer ``convert_option_snapshots_to_surface_rows``."""
    return convert_option_snapshots_to_surface_rows(
        snapshots, trading_date=trading_date
    ).rows


def convert_option_snapshots_to_surface_rows(
    snapshots: Sequence[OptionSnapshot],
    *,
    trading_date: date | None = None,
) -> SurfaceRowConversionResult:
    """Convert snapshots and record explicit conversion failures."""
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    skipped = 0
    for snap in snapshots:
        if trading_date is not None and snap.contract.expiration != trading_date:
            skipped += 1
            failures.append(
                f"skipped_expiry:{snap.contract.contract_id}:{snap.contract.expiration}"
            )
            continue
        if (snap.contract.underlying_symbol or "SPY").upper() != "SPY":
            skipped += 1
            failures.append(
                f"skipped_symbol:{snap.contract.contract_id}:"
                f"{snap.contract.underlying_symbol}"
            )
            continue
        try:
            rows.append(
                option_snapshot_to_surface_row(snap, trading_date=trading_date)
            )
        except Exception as exc:  # noqa: BLE001 — counted as conversion failure
            failures.append(
                f"conversion:{snap.contract.contract_id}:{type(exc).__name__}:{exc}"
            )
    return SurfaceRowConversionResult(
        rows=rows,
        input_count=len(snapshots),
        converted_count=len(rows),
        skipped_symbol_or_expiry=skipped,
        conversion_failures=tuple(failures),
    )


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
