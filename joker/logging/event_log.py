"""Structured JSONL event logging."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from joker.compliance.opra_sanitizer import sanitize_for_persistence

logger = logging.getLogger(__name__)

SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[a-zA-Z0-9]{10,}|"
    r"(api[_-]?key|secret|token|password|pin)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EventLogEntry:
    timestamp: str
    run_id: str
    mode: str
    source: str
    event_type: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "mode": self.mode,
            "source": self.source,
            "event_type": self.event_type,
            "payload": self.payload,
        }


class EventLogWriter:
    """Append-only JSONL event log with secret redaction."""

    def __init__(self, log_dir: Path, redact_keys: list[str] | None = None) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.redact_keys = {k.lower() for k in (redact_keys or [])}

    def _log_path(self, run_id: str) -> Path:
        return self.log_dir / f"{run_id}.jsonl"

    def _redact_value(self, key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._redact_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_value(key, item) for item in value]
        if key.lower() in self.redact_keys:
            return "[REDACTED]"
        if isinstance(value, str):
            if SECRET_VALUE_PATTERN.search(value):
                return "[REDACTED]"
        return value

    def append(
        self,
        *,
        run_id: str,
        mode: str,
        source: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> EventLogEntry:
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        redacted = self._redact_value("", payload or {})
        safe_payload = sanitize_for_persistence(redacted)
        entry = EventLogEntry(
            timestamp=ts,
            run_id=run_id,
            mode=mode,
            source=source,
            event_type=event_type,
            payload=safe_payload,
        )
        line = json.dumps(entry.to_dict(), default=str)
        if SECRET_VALUE_PATTERN.search(line):
            raise ValueError("Refusing to write event log entry containing secret-like values")
        with self._log_path(run_id).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return entry

    def read_all(self, run_id: str) -> list[dict[str, Any]]:
        path = self._log_path(run_id)
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
