"""Playbook quality validation for OpenAI-generated plans."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from joker.schemas.domain import Playbook, RiskConfig, SCHEMA_VERSION, VersionedModel


def max_enabled_playbook_setups(risk_config: RiskConfig | None) -> int:
    """
    How many setups may be enabled on the playbook.

    V1 SPY 0DTE typically arms one call and one put; daily trade count is enforced
    separately by RiskGovernor (max_trades_per_day).
    """
    if risk_config is None:
        return 2
    return max(2, risk_config.max_trades_per_day)


def trim_playbook_enabled_setups(playbook: Playbook, risk_config: RiskConfig | None) -> Playbook:
    """Disable excess enabled setups (keep first N) so validation can pass fail-closed."""
    limit = max_enabled_playbook_setups(risk_config)
    enabled_indices = [i for i, s in enumerate(playbook.setups) if s.enabled]
    if len(enabled_indices) <= limit:
        return playbook
    disable = set(enabled_indices[limit:])
    setups = [
        s.model_copy(update={"enabled": False}) if idx in disable else s
        for idx, s in enumerate(playbook.setups)
    ]
    return playbook.model_copy(update={"setups": setups})


class PlaybookQualityReason:
    WRONG_SYMBOL = "WRONG_SYMBOL"
    INVALID_DIRECTION = "INVALID_DIRECTION"
    SHORT_OPTION_LANGUAGE = "SHORT_OPTION_LANGUAGE"
    SPREAD_LANGUAGE = "SPREAD_LANGUAGE"
    MISSING_STOP = "MISSING_STOP"
    MISSING_TAKE_PROFIT = "MISSING_TAKE_PROFIT"
    EMPTY_ENTRY_CONDITIONS = "EMPTY_ENTRY_CONDITIONS"
    GUARANTEED_PROFIT = "GUARANTEED_PROFIT"
    FORCED_TRADE = "FORCED_TRADE"
    RISK_MODIFICATION = "RISK_MODIFICATION"
    UNAVAILABLE_DATA = "UNAVAILABLE_DATA"
    TOO_MANY_SETUPS = "TOO_MANY_SETUPS"
    RISK_CONFIG_CONFLICT = "RISK_CONFIG_CONFLICT"
    NO_SETUPS = "NO_SETUPS"
    CRITIC_BLOCKED = "CRITIC_BLOCKED"


FORBIDDEN_LANGUAGE = (
    re.compile(r"\bshort\s+(call|put|option)", re.IGNORECASE),
    re.compile(r"\b(naked|uncovered)\s+(call|put)", re.IGNORECASE),
    re.compile(r"\b(credit|debit)\s+spread", re.IGNORECASE),
    re.compile(r"\biron\s+condor\b", re.IGNORECASE),
    re.compile(r"\bstraddle\b", re.IGNORECASE),
    re.compile(r"\bstrangle\b", re.IGNORECASE),
    re.compile(r"\bguaranteed\s+profit", re.IGNORECASE),
    re.compile(r"\b(can'?t|cannot)\s+lose", re.IGNORECASE),
    re.compile(r"\brisk\s*free", re.IGNORECASE),
    re.compile(r"\bmust\s+(buy|trade|enter)", re.IGNORECASE),
    re.compile(r"\bforce\s+trade", re.IGNORECASE),
    re.compile(r"\b(increase|raise|disable)\s+(max\s+)?(loss|trades|kill)", re.IGNORECASE),
    re.compile(r"\blive\s+price\s+is\s+\$?\d", re.IGNORECASE),
    re.compile(r"\bcurrent\s+spy\s+is\s+\$?\d", re.IGNORECASE),
)


class PlaybookValidationResult(VersionedModel):
    approved: bool
    reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlaybookQualityValidator:
    """Deterministic validation gate for agent-generated playbooks."""

    def __init__(self, risk_config: RiskConfig | None = None) -> None:
        self.risk_config = risk_config

    def _blob(self, playbook: Playbook) -> str:
        return playbook.model_dump_json().lower()

    def _check_forbidden_language(self, text: str) -> list[str]:
        reasons: list[str] = []
        for pattern in FORBIDDEN_LANGUAGE:
            if pattern.search(text):
                label = pattern.pattern
                if "short" in label or "naked" in label:
                    reasons.append(PlaybookQualityReason.SHORT_OPTION_LANGUAGE)
                elif "spread" in label or "condor" in label or "straddle" in label:
                    reasons.append(PlaybookQualityReason.SPREAD_LANGUAGE)
                elif "guaranteed" in label or "lose" in label or "risk" in label:
                    reasons.append(PlaybookQualityReason.GUARANTEED_PROFIT)
                elif "must" in label or "force" in label:
                    reasons.append(PlaybookQualityReason.FORCED_TRADE)
                elif "loss" in label or "kill" in label:
                    reasons.append(PlaybookQualityReason.RISK_MODIFICATION)
                elif "price" in label or "spy" in label:
                    reasons.append(PlaybookQualityReason.UNAVAILABLE_DATA)
        return list(dict.fromkeys(reasons))

    def validate(
        self,
        playbook: Playbook,
        *,
        critic_blocked: bool = False,
        max_setups: int | None = None,
    ) -> PlaybookValidationResult:
        reasons: list[str] = []
        warnings: list[str] = []

        if critic_blocked:
            reasons.append(PlaybookQualityReason.CRITIC_BLOCKED)

        if not playbook.setups:
            reasons.append(PlaybookQualityReason.NO_SETUPS)

        blob = self._blob(playbook)
        if "spy" not in blob and playbook.title:
            reasons.append(PlaybookQualityReason.WRONG_SYMBOL)

        max_trades = max_setups
        if self.risk_config:
            max_trades = max_enabled_playbook_setups(self.risk_config)
        enabled = [s for s in playbook.setups if s.enabled]
        if max_trades is not None and len(enabled) > max_trades:
            reasons.append(PlaybookQualityReason.TOO_MANY_SETUPS)

        for setup in playbook.setups:
            if setup.direction not in ("long_call", "long_put"):
                reasons.append(PlaybookQualityReason.INVALID_DIRECTION)
            if not setup.stop_rule.strip():
                reasons.append(PlaybookQualityReason.MISSING_STOP)
            if not setup.take_profit_rule.strip():
                reasons.append(PlaybookQualityReason.MISSING_TAKE_PROFIT)
            if not setup.entry_conditions:
                reasons.append(PlaybookQualityReason.EMPTY_ENTRY_CONDITIONS)

            setup_text = f"{setup.name} {setup.stop_rule} {setup.take_profit_rule} {' '.join(setup.entry_conditions)}"
            for code in self._check_forbidden_language(setup_text):
                reasons.append(code)

        for code in self._check_forbidden_language(playbook.summary + " " + playbook.title):
            reasons.append(code)

        if self.risk_config:
            for token in ("kill_switch", "max_daily_loss", "live_trading"):
                if token in blob:
                    reasons.append(PlaybookQualityReason.RISK_CONFIG_CONFLICT)

        reasons = list(dict.fromkeys(reasons))
        return PlaybookValidationResult(
            approved=len(reasons) == 0,
            reason_codes=reasons,
            warnings=warnings,
        )
