"""OpenAI input audit events for OPRA compliance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from joker.compliance.opra_sanitizer import _contains_raw_opra_values, sanitize_for_openai


@dataclass(frozen=True)
class OpenAIInputAudit:
    prompt_type: str
    raw_opra_detected: bool
    sanitized: bool
    allowed_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": "openai.input_audit",
            "prompt_type": self.prompt_type,
            "raw_opra_detected": self.raw_opra_detected,
            "sanitized": self.sanitized,
            "allowed_fields": self.allowed_fields,
        }


def audit_and_sanitize_openai_context(
    context: dict[str, Any],
    *,
    prompt_type: str,
) -> tuple[dict[str, Any], OpenAIInputAudit]:
    raw_detected = _contains_raw_opra_values(context)
    sanitized = sanitize_for_openai(context)
    allowed = sorted(k for k in sanitized.keys() if not str(k).startswith("_"))
    audit = OpenAIInputAudit(
        prompt_type=prompt_type,
        raw_opra_detected=raw_detected,
        sanitized=True,
        allowed_fields=allowed,
    )
    return sanitized, audit
