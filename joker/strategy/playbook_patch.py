"""Playbook patch application with validation."""

from __future__ import annotations

from datetime import datetime, timezone

from joker.schemas.domain import Playbook, PlaybookPatch


class PatchError(Exception):
    pass


def patch_loosens_risk(patch: PlaybookPatch) -> bool:
    """Patches cannot loosen hard risk limits in V1."""
    forbidden_keys = (
        "max_loss",
        "max_trades",
        "kill_switch",
        "live_trading",
        "max_daily_loss",
        "max_premium",
    )
    blob = patch.model_dump_json().lower()
    return any(k in blob for k in forbidden_keys)


def validate_patch_safe(patch: PlaybookPatch) -> None:
    """Reject patches that attempt to modify hard risk or kill switch."""
    if patch_loosens_risk(patch):
        raise PatchError("Patch cannot modify hard risk limits or kill switch")
    reason_lower = patch.reason.lower()
    if any(
        phrase in reason_lower
        for phrase in ("disable kill", "kill switch off", "bypass risk", "approve trade")
    ):
        raise PatchError("Patch reason indicates forbidden risk modification")


def apply_patch(playbook: Playbook, patch: PlaybookPatch) -> Playbook:
    validate_patch_safe(patch)
    if patch.playbook_id != playbook.playbook_id:
        raise PatchError("Patch playbook_id mismatch")
    if patch.expires_at and patch.expires_at.replace(tzinfo=timezone.utc) < datetime.now(
        timezone.utc
    ):
        raise PatchError("Patch expired")

    updated_setups = []
    for setup in playbook.setups:
        setup = setup.model_copy()
        if setup.setup_id in patch.disable_setup_ids:
            setup.enabled = False
        if setup.setup_id in patch.enable_setup_ids:
            setup.enabled = True
        updated_setups.append(setup)

    return playbook.model_copy(update={"setups": updated_setups})
