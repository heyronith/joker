"""Options data schemas for Webull verification (Phase 19)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from joker.compliance.data_classification import DataClassification
from joker.schemas.domain import SCHEMA_VERSION, VersionedModel


class OptionFieldAvailability(VersionedModel):
    """Tracks which fields Webull actually returned."""

    bid: bool = False
    ask: bool = False
    mid: bool = False
    last: bool = False
    spread_pct: bool = False
    volume: bool = False
    open_interest: bool = False
    implied_volatility: bool = False
    delta: bool = False
    gamma: bool = False
    theta: bool = False
    vega: bool = False
    quote_timestamp: bool = False
    delayed: bool = False
    contract_id: bool = False
    instrument_id: bool = False

    def unavailable_fields(self) -> list[str]:
        return [
            name
            for name, available in self.model_dump().items()
            if name != "schema_version" and not available
        ]


class OptionContractMetadata(VersionedModel):
    underlying_symbol: str = "SPY"
    expiration: date
    strike: float
    option_type: Literal["call", "put"]
    contract_id: str | None = None
    instrument_id: str | None = None
    source: str = "webull_opra"


class OptionSnapshot(VersionedModel):
    contract: OptionContractMetadata
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    last: float | None = None
    spread_pct: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    quote_timestamp: datetime | None = None
    delayed: bool | None = None
    source: str = "webull_opra"
    field_availability: OptionFieldAvailability = Field(
        default_factory=OptionFieldAvailability
    )
    data_classification: str = DataClassification.RAW_OPRA.value
    persist_allowed: bool = False
    openai_allowed: bool = False
    is_synthetic: bool = False


class OptionDataQualityWarning(VersionedModel):
    code: str
    message: str


class OptionDataCapabilityReport(VersionedModel):
    """Summary of what Webull options endpoints provide."""

    provider: str = "webull"
    contract_discovery: bool = False
    same_day_expiration: bool = False
    snapshot_bid_ask: bool = False
    volume: bool = False
    open_interest: bool = False
    implied_volatility: bool = False
    greeks: bool = False
    historical_bars: Literal["yes", "no", "unknown"] = "unknown"
    ticks: Literal["yes", "no", "unknown"] = "unknown"
    delayed_data: bool | None = None
    verified: bool = False
    unavailable_fields: list[str] = Field(default_factory=list)


class OptionDataDiagnosticReport(VersionedModel):
    checked_at: datetime
    credentials_present: bool = False
    auth_pass: bool = False
    spy_snapshot_pass: bool = False
    contract_discovery_pass: bool = False
    same_day_expiration_found: bool = False
    atm_call_snapshot_pass: bool = False
    atm_put_snapshot_pass: bool = False
    bid_ask_available: bool = False
    volume_available: bool = False
    open_interest_available: bool = False
    iv_available: bool = False
    greeks_available: bool = False
    historical_bars: Literal["yes", "no", "unknown"] = "unknown"
    ticks: Literal["yes", "no", "unknown"] = "unknown"
    delayed_status: str | None = None
    likely_issue: str | None = None
    capability: OptionDataCapabilityReport | None = None
    endpoint_status: dict[str, str] = Field(default_factory=dict)
    warnings: list[OptionDataQualityWarning] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)

    def to_lines(self) -> list[str]:
        lines = [
            f"Options data diagnostics ({self.checked_at.isoformat()})",
            f"- credentials: {'present' if self.credentials_present else 'missing'}",
            f"- auth: {'pass' if self.auth_pass else 'fail'}",
            f"- SPY snapshot: {'pass' if self.spy_snapshot_pass else 'fail'}",
            f"- contract discovery: {'pass' if self.contract_discovery_pass else 'fail'}",
            f"- same-day expiration: {'yes' if self.same_day_expiration_found else 'no'}",
            f"- ATM call snapshot: {'pass' if self.atm_call_snapshot_pass else 'fail'}",
            f"- ATM put snapshot: {'pass' if self.atm_put_snapshot_pass else 'fail'}",
            f"- bid/ask available: {'yes' if self.bid_ask_available else 'no'}",
            f"- volume available: {'yes' if self.volume_available else 'no'}",
            f"- open interest available: {'yes' if self.open_interest_available else 'no'}",
            f"- IV available: {'yes' if self.iv_available else 'no'}",
            f"- Greeks available: {'yes' if self.greeks_available else 'no'}",
            f"- historical bars: {self.historical_bars}",
            f"- ticks: {self.ticks}",
        ]
        if self.delayed_status:
            lines.append(f"- delayed/realtime: {self.delayed_status}")
        if self.likely_issue:
            lines.append(f"- likely issue: {self.likely_issue}")
        if self.endpoint_status:
            lines.append("- endpoint verification:")
            for name, status in sorted(self.endpoint_status.items()):
                lines.append(f"  - {name}: {status}")
        if self.capability and self.capability.unavailable_fields:
            lines.append(
                f"- unavailable fields: {', '.join(self.capability.unavailable_fields)}"
            )
        return lines
