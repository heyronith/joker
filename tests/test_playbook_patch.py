"""Phase 10 intraday patch tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from joker.schemas.domain import Playbook, PlaybookPatch, PlaybookSetup
from joker.strategy.playbook_patch import PatchError, apply_patch, patch_loosens_risk, validate_patch_safe


def _playbook() -> Playbook:
    setup = PlaybookSetup(
        setup_id="s1",
        name="Primary",
        direction="long_call",
        stop_rule="50%",
        take_profit_rule="100%",
    )
    return Playbook(
        trading_day=date.today(),
        title="t",
        summary="s",
        setups=[setup],
        approved=True,
    )


def test_disable_setup_patch() -> None:
    pb = _playbook()
    patch = PlaybookPatch(
        playbook_id=pb.playbook_id,
        author_agent="CriticAgent",
        reason="conditions deteriorated",
        disable_setup_ids=["s1"],
    )
    updated = apply_patch(pb, patch)
    assert updated.setups[0].enabled is False


def test_expired_patch_rejected() -> None:
    pb = _playbook()
    patch = PlaybookPatch(
        playbook_id=pb.playbook_id,
        author_agent="CriticAgent",
        reason="late",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    with pytest.raises(PatchError, match="expired"):
        apply_patch(pb, patch)


def test_patch_cannot_loosen_risk() -> None:
    patch = PlaybookPatch(
        playbook_id="x",
        author_agent="BadAgent",
        reason="increase max_loss please",
    )
    assert patch_loosens_risk(patch) is True
    with pytest.raises(PatchError):
        validate_patch_safe(patch)


def test_kill_switch_patch_rejected() -> None:
    patch = PlaybookPatch(
        playbook_id="x",
        author_agent="BadAgent",
        reason="disable kill switch",
    )
    with pytest.raises(PatchError):
        validate_patch_safe(patch)
