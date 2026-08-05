from __future__ import annotations

import json

from joker.cli.graph_view import GraphView, render_graph_event


def _payload() -> dict:
    return {
        "goal": {
            "authorized_capital": "200",
            "available_capital": "150",
            "realized_pnl": "10",
            "remaining_goal_gap": "40",
            "target": "50",
            "deadline": "2026-08-05T15:30:00-04:00",
            "time_remaining_seconds": 900,
            "maximum_positions": 2,
        },
        "market": {
            "spy_price": "500",
            "market_direction": "bullish",
            "volatility_regime": "normal",
            "session_phase": "regular",
            "option_surface_size": 400,
            "eligible_contract_count": 120,
            "data_quality_state": "healthy",
        },
        "theses": [
            {
                "agent_role": "bullish_inventor",
                "strategy_name": "momentum",
                "strategy_family": "directional",
                "direction": "bullish",
                "confidence": 0.7,
                "thesis_summary": "continuation above VWAP",
                "expected_horizon_seconds": 300,
                "key_evidence": ["e1"],
            }
        ],
        "contracts": [
            {
                "rank": 1,
                "strategy": "momentum",
                "contract_id": "SPY:2026-08-05:500:call",
                "option_type": "call",
                "strike": "500",
                "probability_goal": "0.40",
                "probability_wait": "0.20",
                "probability_delta": "0.20",
                "selected": True,
            }
        ],
        "portfolios": [
            {
                "rank": 1,
                "component_contract_ids": ["SPY:2026-08-05:500:call"],
                "component_quantities": [1],
                "capital_deployed": "100",
                "maximum_loss": "100",
                "expected_pnl": "20",
                "probability_goal": "0.40",
                "probability_wait": "0.20",
                "probability_delta": "0.20",
                "selected": True,
            }
        ],
        "reviews": [
            {
                "reviewer_role": "falsifier",
                "reviewed_id": "strategy-1",
                "verdict": "support",
                "confidence": 0.6,
                "claims_summary": ["spread acceptable"],
                "failure_modes": ["false breakout"],
                "required_revisions": [],
            }
        ],
        "decision": {
            "action": "enter",
            "selected_probability_goal": "0.40",
            "wait_probability_goal": "0.20",
            "probability_delta": "0.20",
        },
        "execution": {
            "quote_revalidation": "passed",
            "capital_reservation": "passed",
            "order_submission": "not_started",
            "reconciliation": "not_started",
        },
    }


def test_compact_view_is_high_signal_and_short() -> None:
    rendered = render_graph_event(
        "target.portfolio.selected",
        {
            "action": "enter",
            "selected_probability_goal": "0.4",
            "wait_probability_goal": "0.2",
            "probability_delta": "0.2",
            "reason_codes": ["improves_wait"],
        },
        view=GraphView.COMPACT,
    )
    assert "TARGET ENTER" in rendered
    assert "p_goal=0.4" in rendered
    assert "\n" not in rendered


def test_verbose_view_contains_all_required_panels_and_probabilities() -> None:
    rendered = render_graph_event(
        "target.portfolio.selected", _payload(), view="verbose"
    )
    for panel in (
        "GOAL",
        "MARKET",
        "AGENT THESES",
        "CONTRACT GRID",
        "PORTFOLIO GRID",
        "DEBATE",
        "DECISION",
        "EXECUTION",
    ):
        assert panel in rendered
    assert "probability_goal=0.40" in rendered
    assert "probability_wait=0.20" in rendered
    assert "failure_modes=['false breakout']" in rendered


def test_json_mode_is_valid_and_redacts_sensitive_fields() -> None:
    payload = _payload()
    payload["api_key"] = "sk-secret-value"
    payload["account_id"] = "123456"
    payload["raw_prompt"] = "hidden reasoning prompt"
    payload["raw_surface"] = [{"contract": index} for index in range(200)]
    rendered = render_graph_event(
        "portfolio.grid.scored", payload, view="json"
    )
    decoded = json.loads(rendered)
    assert decoded["event_type"] == "portfolio.grid.scored"
    assert decoded["payload"]["api_key"] == "[REDACTED]"
    assert decoded["payload"]["account_id"] == "[REDACTED]"
    assert decoded["payload"]["raw_prompt"] == "[REDACTED]"
    assert decoded["payload"]["raw_surface"] == "[REDACTED]"
    assert "sk-secret-value" not in rendered
    assert "hidden reasoning prompt" not in rendered
