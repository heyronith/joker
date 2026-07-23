"""Redacted Webull response shape capture for contract verification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from joker.config.validation import redact_secrets
from joker.config.settings import EnvSettings


def _field_types(obj: Any, depth: int = 0) -> Any:
    if depth > 3:
        return "..."
    if isinstance(obj, dict):
        return {k: _field_types(v, depth + 1) for k, v in list(obj.items())[:40]}
    if isinstance(obj, list):
        if not obj:
            return []
        return [_field_types(obj[0], depth + 1)]
    return type(obj).__name__


def _presence_flags(data: dict[str, Any]) -> dict[str, bool]:
    keys = {k.lower() for k in data.keys()}
    return {
        "bid": "bid" in keys,
        "ask": "ask" in keys,
        "timestamp": any(k in keys for k in ("quote_time", "timestamp", "last_trade_time")),
        "volume": "volume" in keys,
        "open_interest": "open_interest" in keys,
        "iv": "imp_vol" in keys or "impliedvolatility" in keys,
        "greeks": any(k in keys for k in ("delta", "gamma", "theta", "vega")),
        "delayed": "delayed" in keys or "isdelayed" in keys,
    }


def summarize_response(
    *,
    endpoint_name: str,
    status_code: int,
    payload: Any,
    error: str | None = None,
    classification: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "endpoint": endpoint_name,
        "status_code": status_code,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "classification": classification,
        "top_level_type": type(payload).__name__,
    }
    if isinstance(payload, dict):
        summary["top_level_keys"] = list(payload.keys())[:30]
        summary["field_types"] = _field_types(payload)
        sample = payload
        if "data" in payload and isinstance(payload["data"], (dict, list)):
            sample = payload["data"][0] if isinstance(payload["data"], list) and payload["data"] else payload["data"]
        if isinstance(sample, dict):
            summary["presence"] = _presence_flags(sample)
    elif isinstance(payload, list) and payload:
        summary["top_level_keys"] = ["array"]
        summary["field_types"] = _field_types(payload[0])
        if isinstance(payload[0], dict):
            summary["presence"] = _presence_flags(payload[0])
    return summary


def write_contract_capture(
    summaries: list[dict[str, Any]],
    *,
    output_dir: Path | None = None,
    env: EnvSettings | None = None,
) -> Path:
    out = output_dir or Path("data/captures/webull_contract")
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"webull_contract_{ts}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in summaries:
            raw = json.dumps(item)
            if env is not None:
                raw = redact_secrets(raw, env=env)
            handle.write(raw + "\n")
    return path
