"""Aggregate factual model-call telemetry for evolution experiments."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Sequence
from uuid import UUID


def aggregate_model_call_telemetry(
    records: Sequence[Any],
    *,
    cost_per_1k_input: Decimal | None = None,
    cost_per_1k_output: Decimal | None = None,
) -> dict[str, Any]:
    """Build factual cost/latency aggregates from ModelCallRecord-like objects."""
    if not records:
        return {
            "model_calls": 0,
            "latency_ms": Decimal("0"),
            "cost_gbp": None,
            "cost_known": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "unknown_cost": True,
        }
    latency = Decimal("0")
    input_tokens = 0
    output_tokens = 0
    for rec in records:
        if getattr(rec, "latency_ms", None) is not None:
            latency += Decimal(str(rec.latency_ms))
        input_tokens += int(getattr(rec, "input_tokens", 0) or 0)
        output_tokens += int(getattr(rec, "output_tokens", 0) or 0)
    cost_known = cost_per_1k_input is not None and cost_per_1k_output is not None
    cost = None
    if cost_known:
        cost = (
            Decimal(input_tokens) * cost_per_1k_input / Decimal("1000")
            + Decimal(output_tokens) * cost_per_1k_output / Decimal("1000")
        )
    return {
        "model_calls": len(records),
        "latency_ms": latency,
        "cost_gbp": cost,
        "cost_known": cost_known,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "unknown_cost": not cost_known,
    }


def brier_score(pairs: Iterable[tuple[Decimal, int]]) -> Decimal | None:
    """pairs: (predicted_probability, binary_outcome)."""
    items = list(pairs)
    if not items:
        return None
    total = Decimal("0")
    for pred, outcome in items:
        total += (pred - Decimal(outcome)) ** 2
    return total / Decimal(len(items))


def expected_calibration_error(
    pairs: Iterable[tuple[Decimal, int]], *, buckets: int = 10
) -> Decimal | None:
    items = list(pairs)
    if not items:
        return None
    bucket_preds: dict[int, list[Decimal]] = {i: [] for i in range(buckets)}
    bucket_outs: dict[int, list[int]] = {i: [] for i in range(buckets)}
    for pred, outcome in items:
        idx = min(buckets - 1, int(pred * buckets))
        bucket_preds[idx].append(pred)
        bucket_outs[idx].append(outcome)
    ece = Decimal("0")
    n = Decimal(len(items))
    for i in range(buckets):
        if not bucket_preds[i]:
            continue
        mean_pred = sum(bucket_preds[i]) / Decimal(len(bucket_preds[i]))
        mean_out = Decimal(sum(bucket_outs[i])) / Decimal(len(bucket_outs[i]))
        ece += (Decimal(len(bucket_preds[i])) / n) * abs(mean_pred - mean_out)
    return ece


def extract_confidence_outcome_pairs(
    *,
    meta_confidence: Decimal | None,
    traded: bool,
    realised_pnl: Decimal | None,
) -> list[tuple[Decimal, int]]:
    """Simple entry-confidence vs profitable-outcome calibration pairs."""
    if meta_confidence is None:
        return []
    if not traded or realised_pnl is None:
        return [(meta_confidence, 0)]
    return [(meta_confidence, 1 if realised_pnl > 0 else 0)]
