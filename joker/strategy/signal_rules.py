"""Strategy signal evaluation from structured playbook setups."""

from __future__ import annotations

from joker.schemas.domain import Playbook, PlaybookSetup, TechnicalFeatures


def setup_matches_features(setup: PlaybookSetup, features: TechnicalFeatures) -> bool:
    """Return True when structured entry rules are satisfied."""
    if not setup.enabled:
        return False

    trend = features.trend_label or "unknown"
    if setup.require_trend != "any" and trend != setup.require_trend:
        return False

    dist = features.distance_from_vwap_pct
    if dist is None:
        dist = 0.0

    if setup.vwap_side == "above" and dist < setup.min_vwap_distance_pct:
        return False
    if setup.vwap_side == "below" and dist > -abs(setup.min_vwap_distance_pct):
        return False
    if setup.vwap_side == "either" and setup.min_vwap_distance_pct > 0:
        if abs(dist) < setup.min_vwap_distance_pct:
            return False

    mom = features.momentum_5m
    if mom is None:
        mom = 0.0

    if setup.direction == "long_call":
        if mom < setup.min_momentum_pct:
            return False
        if setup.max_momentum_pct is not None and mom > setup.max_momentum_pct:
            return False
    elif setup.direction == "long_put":
        # For puts, min_momentum_pct is typically negative (e.g. -0.1)
        threshold = setup.min_momentum_pct
        if threshold >= 0:
            threshold = -abs(threshold) if threshold > 0 else -0.1
        if mom > threshold:
            return False
        if setup.max_momentum_pct is not None and mom < setup.max_momentum_pct:
            return False

    return True


def detect_setup_from_playbook(
    playbook: Playbook,
    features: TechnicalFeatures,
    *,
    already_signaled: set[str] | None = None,
) -> PlaybookSetup | None:
    """Pick first enabled setup whose structured rules match features."""
    signaled = already_signaled or set()
    for setup in playbook.setups:
        if setup.setup_id in signaled:
            continue
        if setup_matches_features(setup, features):
            return setup
    return None
