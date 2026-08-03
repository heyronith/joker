"""Process-local live activation — intentional arming for a funded account/objective."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from secrets import token_urlsafe
from uuid import UUID


@dataclass(frozen=True)
class LiveActivation:
    account_id_hash: str
    objective_id: UUID
    authorized_capital_usd: Decimal
    activated_at: datetime
    expires_at: datetime
    activation_nonce: str

    def is_active(self, *, now: datetime | None = None) -> bool:
        current = now or datetime.now(timezone.utc)
        return self.activated_at <= current < self.expires_at


def create_live_activation(
    *,
    account_id_hash: str,
    objective_id: UUID,
    authorized_capital_usd: Decimal,
    ttl_seconds: int = 3600,
) -> LiveActivation:
    """Construct a non-reusable process-local activation (not stored in config)."""
    now = datetime.now(timezone.utc)
    return LiveActivation(
        account_id_hash=account_id_hash,
        objective_id=objective_id,
        authorized_capital_usd=authorized_capital_usd,
        activated_at=now,
        expires_at=now + timedelta(seconds=ttl_seconds),
        activation_nonce=token_urlsafe(24),
    )
