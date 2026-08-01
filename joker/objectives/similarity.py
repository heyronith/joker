"""Deterministic, configuration-driven similarity for historical analogues."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

from joker.objectives.historical_schemas import SimilarityPolicy

SIMILARITY_POLICY_VERSION = "1.0.0"


def _as_set(values: Sequence[Any] | None) -> set[str]:
    if not values:
        return set()
    return {str(v) for v in values if v is not None}


def _bucket_match(a: str | None, b: str | None) -> Decimal:
    # Unspecified query dimension does not constrain eligibility.
    if a is None or str(a) in {"", "unknown"}:
        return Decimal("1")
    if b is None or str(b) in {"", "unknown"}:
        return Decimal("0.5")
    return Decimal("1") if str(a) == str(b) else Decimal("0")


def _jaccard(a: set[str], b: set[str]) -> Decimal:
    if not a and not b:
        return Decimal("1")
    # Unspecified query dimension does not constrain eligibility.
    if not a:
        return Decimal("1")
    if not b:
        return Decimal("0.5")
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return Decimal("0")
    return (Decimal(inter) / Decimal(union)).quantize(Decimal("0.0001"))


def score_similarity(
    *,
    query_strategy_family: str | None,
    query_direction: str | None,
    query_pattern_ids: Sequence[UUID] | Sequence[str] = (),
    query_regime_labels: Sequence[str] = (),
    query_session_phase: str = "unknown",
    query_volatility_bucket: str | None = None,
    query_liquidity_bucket: str | None = None,
    query_premium_bucket: str | None = None,
    query_horizon_bucket: str | None = None,
    episode_strategy_family: str | None = None,
    episode_direction: str | None = None,
    episode_pattern_ids: Sequence[UUID] | Sequence[str] = (),
    episode_regime_labels: Sequence[str] = (),
    episode_session_phase: str = "unknown",
    episode_volatility_bucket: str | None = None,
    episode_liquidity_bucket: str | None = None,
    episode_premium_bucket: str | None = None,
    episode_horizon_bucket: str | None = None,
    policy: SimilarityPolicy | None = None,
) -> tuple[Decimal, dict[str, Decimal]]:
    """Return (final_score, components) in [0, 1]."""
    pol = policy or SimilarityPolicy(policy_version=SIMILARITY_POLICY_VERSION)
    # Strategy family is distinct from direction — never substitute one for the other.
    family_q = (query_strategy_family or "").lower()
    family_e = (episode_strategy_family or "").lower()
    if not family_q:
        strategy_family_match = Decimal("1")
    elif not family_e:
        strategy_family_match = Decimal("0.5")
    elif family_q == family_e:
        strategy_family_match = Decimal("1")
    else:
        strategy_family_match = Decimal("0")
    direction_q = (query_direction or "").lower()
    direction_e = (episode_direction or "").lower()
    if not direction_q:
        direction_match = Decimal("1")
    elif not direction_e:
        direction_match = Decimal("0.5")
    elif direction_q == direction_e:
        direction_match = Decimal("1")
    else:
        direction_match = Decimal("0")
    pattern_overlap = _jaccard(_as_set(query_pattern_ids), _as_set(episode_pattern_ids))
    regime_similarity = _jaccard(
        _as_set(query_regime_labels), _as_set(episode_regime_labels)
    )
    if str(query_session_phase) in {"", "unknown"}:
        session_phase_match = Decimal("1")
    elif str(episode_session_phase) in {"", "unknown"}:
        session_phase_match = Decimal("0.5")
    elif str(query_session_phase) == str(episode_session_phase):
        session_phase_match = Decimal("1")
    else:
        session_phase_match = Decimal("0")
    components = {
        "strategy_family_match": strategy_family_match,
        "direction_match": direction_match,
        "pattern_overlap": pattern_overlap,
        "regime_similarity": regime_similarity,
        "session_phase_match": session_phase_match,
        "volatility_similarity": _bucket_match(
            query_volatility_bucket, episode_volatility_bucket
        ),
        "liquidity_similarity": _bucket_match(
            query_liquidity_bucket, episode_liquidity_bucket
        ),
        "premium_similarity": _bucket_match(
            query_premium_bucket, episode_premium_bucket
        ),
        "horizon_similarity": _bucket_match(
            query_horizon_bucket, episode_horizon_bucket
        ),
    }
    weights = pol.weight_map()
    # Direction is scored but folded into family weight budget when absent from policy.
    scored_keys = [k for k in weights if k in components]
    weight_sum = sum((weights[k] for k in scored_keys), start=Decimal("0"))
    if weight_sum <= 0:
        return Decimal("0"), components
    # If policy has no direction weight, blend a small share from strategy_family.
    if "direction_match" not in weights:
        fam_w = weights.get("strategy_family_match", Decimal("0"))
        if fam_w > 0:
            weights = dict(weights)
            weights["strategy_family_match"] = (fam_w * Decimal("0.75")).quantize(
                Decimal("0.0001")
            )
            weights["direction_match"] = (fam_w * Decimal("0.25")).quantize(
                Decimal("0.0001")
            )
            scored_keys = [k for k in weights if k in components]
            weight_sum = sum((weights[k] for k in scored_keys), start=Decimal("0"))
    total = sum(
        (components[k] * weights[k] for k in scored_keys),
        start=Decimal("0"),
    )
    if weight_sum != Decimal("1"):
        total = (total / weight_sum).quantize(Decimal("0.0001"))
    else:
        total = total.quantize(Decimal("0.0001"))
    return total, components
