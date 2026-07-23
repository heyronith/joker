"""Persisted Webull options capability cache."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from joker.schemas.domain import SCHEMA_VERSION, VersionedModel


DEFAULT_CAPABILITY_PATH = Path("data/capabilities/webull_options_capability.json")


class WebullOptionsCapability(VersionedModel):
    checked_at: datetime
    provider: str = "webull"
    symbol: str = "SPY"
    auth_pass: bool = False
    contract_discovery_verified: bool = False
    contract_discovery_succeeded: bool = False
    snapshot_verified: bool = True
    snapshot_succeeded: bool = False
    bid_ask_available: bool = False
    timestamp_available: bool = False
    same_day_expiration_found: bool = False
    volume_available: bool = False
    open_interest_available: bool = False
    iv_available: bool = False
    greeks_available: bool = False
    delayed_status: str | None = None
    usable_for_shadow: bool = False
    usable_for_replay_capture: bool = False
    likely_issue: str | None = None
    expiration_tested: date | None = None
    endpoint_status: dict[str, str] = Field(default_factory=dict)
    missing_required_fields: list[str] = Field(default_factory=list)
    optional_missing_fields: list[str] = Field(default_factory=list)


def load_capability(path: Path | None = None) -> WebullOptionsCapability | None:
    p = path or DEFAULT_CAPABILITY_PATH
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return WebullOptionsCapability.model_validate(data)


def save_capability(cap: WebullOptionsCapability, path: Path | None = None) -> Path:
    p = path or DEFAULT_CAPABILITY_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(cap.model_dump_json(indent=2), encoding="utf-8")
    return p


def capability_usable_for_shadow(path: Path | None = None) -> bool:
    cap = load_capability(path)
    return cap is not None and cap.usable_for_shadow
