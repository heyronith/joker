"""Factual telemetry — no invented cost, latency, or calibration."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from joker.evolution.telemetry import (
    aggregate_model_call_telemetry,
    extract_confidence_outcome_pairs,
)


def test_aggregate_missing_records_are_unknown() -> None:
    out = aggregate_model_call_telemetry(
        [],
        cost_per_1k_input=Decimal("0.001"),
        cost_per_1k_output=Decimal("0.002"),
        pricing_version="task3-fake-v1",
    )
    assert out["model_calls"] == 0
    assert out["cost_known"] is False
    assert out["cost_gbp"] is None
    assert out["latency_ms"] is None
    assert out["input_tokens"] is None
    assert out["output_tokens"] is None
    assert out["cost_source"] == "missing"
    assert out["pricing_version"] is None


def test_aggregate_persisted_records_are_factual() -> None:
    records = [
        SimpleNamespace(
            request_id=uuid4(),
            latency_ms=12,
            input_tokens=10,
            output_tokens=20,
        ),
        SimpleNamespace(
            request_id=uuid4(),
            latency_ms=8,
            input_tokens=5,
            output_tokens=15,
        ),
    ]
    out = aggregate_model_call_telemetry(
        records,
        cost_per_1k_input=Decimal("0.001"),
        cost_per_1k_output=Decimal("0.002"),
        pricing_version="task3-fake-v1",
    )
    assert out["model_calls"] == 2
    assert out["cost_known"] is True
    assert out["cost_source"] == "persisted_model_calls"
    assert out["pricing_version"] == "task3-fake-v1"
    assert out["input_tokens"] == 15
    assert out["output_tokens"] == 35
    assert out["latency_ms"] == Decimal("20")
    assert out["cost_gbp"] == (
        Decimal("15") * Decimal("0.001") / Decimal("1000")
        + Decimal("35") * Decimal("0.002") / Decimal("1000")
    )


def test_aggregate_without_pricing_version_is_unknown_cost() -> None:
    records = [
        SimpleNamespace(latency_ms=1, input_tokens=1, output_tokens=1),
    ]
    out = aggregate_model_call_telemetry(
        records,
        cost_per_1k_input=Decimal("0.001"),
        cost_per_1k_output=Decimal("0.002"),
        pricing_version=None,
    )
    assert out["cost_known"] is False
    assert out["cost_gbp"] is None
    assert out["cost_source"] == "persisted_model_calls"


def test_missing_confidence_yields_no_calibration_pairs() -> None:
    pairs = extract_confidence_outcome_pairs(
        meta_confidence=None,
        traded=True,
        realised_pnl=Decimal("10"),
    )
    assert pairs == []


def test_observed_confidence_yields_real_pair() -> None:
    pairs = extract_confidence_outcome_pairs(
        meta_confidence=Decimal("0.7"),
        traded=True,
        realised_pnl=Decimal("10"),
    )
    assert pairs == [(Decimal("0.7"), 1)]


def test_missing_outcome_creates_no_calibration_sample() -> None:
    assert (
        extract_confidence_outcome_pairs(
            meta_confidence=Decimal("0.8"),
            traded=False,
            realised_pnl=None,
        )
        == []
    )
    assert (
        extract_confidence_outcome_pairs(
            meta_confidence=Decimal("0.8"),
            traded=True,
            realised_pnl=None,
        )
        == []
    )
