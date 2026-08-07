"""Safe terminal/JSON renderer for observable cognitive-graph evidence."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

SENSITIVE_KEY_PARTS = (
    "account",
    "api_key",
    "cookie",
    "credential",
    "password",
    "pin",
    "prompt",
    "raw_surface",
    "secret",
    "token",
)

GRAPH_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "graph.cycle.started",
        "strategy.thesis.generated",
        "chain.universe.built",
        "contract.outcome.estimated",
        "contract.grid.scored",
        "portfolio.grid.scored",
        "debate.review.completed",
        "target.portfolio.selected",
        "target.wait.selected",
        "execution.revalidation",
        "execution.reoptimization_required",
        "graph.cycle.completed",
    }
)
SECRET_VALUE_PATTERN = re.compile(
    r"(sk-[a-zA-Z0-9_-]{8,}|"
    r"(api[_-]?key|secret|token|password|pin)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


class GraphView(StrEnum):
    COMPACT = "compact"
    VERBOSE = "verbose"
    JSON = "json"


def sanitize_graph_evidence(value: Any, *, key: str = "") -> Any:
    """Redact secrets/account identifiers and cap unsafe raw collections."""
    normalized = key.lower()
    if any(part in normalized for part in SENSITIVE_KEY_PARTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(k): sanitize_graph_evidence(v, key=str(k))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_graph_evidence(item, key=key) for item in value[:100]]
    if isinstance(value, str):
        if SECRET_VALUE_PATTERN.search(value):
            return "[REDACTED]"
        if len(value) > 2_000:
            return value[:2_000] + "…"
    return value


def render_graph_event(
    event_type: str,
    payload: dict[str, Any] | None,
    *,
    view: GraphView | str = GraphView.COMPACT,
    top_contract_rows: int = 10,
    top_portfolio_rows: int = 10,
) -> str:
    """Render the same safe structured payload as compact, verbose, or JSON."""
    mode = GraphView(str(view))
    safe = sanitize_graph_evidence(payload or {})
    if mode is GraphView.JSON:
        return json.dumps(
            {"event_type": event_type, "payload": safe},
            sort_keys=True,
            default=str,
        )
    if mode is GraphView.COMPACT:
        return _compact(event_type, safe)
    return _verbose(
        event_type,
        safe,
        top_contract_rows=max(1, top_contract_rows),
        top_portfolio_rows=max(1, top_portfolio_rows),
    )


def _compact(event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "chain.universe.built":
        return (
            "CHAIN "
            f"surface={payload.get('source_contract_count', 0)} "
            f"eligible={payload.get('eligible_contract_count', 0)} "
            f"evaluated={payload.get('evaluated_contract_count', 0)}"
        )
    if event_type == "contract.grid.scored":
        return (
            f"CONTRACT GRID rows={payload.get('row_count', 0)} "
            f"best_p_goal={payload.get('best_probability_goal')} "
            f"p_wait={payload.get('wait_probability_goal')}"
        )
    if event_type == "portfolio.grid.scored":
        return (
            f"PORTFOLIO GRID rows={payload.get('row_count', 0)} "
            f"best_p_goal={payload.get('best_probability_goal')}"
        )
    if event_type in {"target.portfolio.selected", "target.wait.selected"}:
        return (
            f"TARGET {str(payload.get('action', '')).upper()} "
            f"p_goal={payload.get('selected_probability_goal')} "
            f"p_wait={payload.get('wait_probability_goal')} "
            f"delta={payload.get('probability_delta')} "
            f"reasons={','.join(payload.get('reason_codes') or [])}"
        )
    if event_type == "strategy.thesis.generated":
        return (
            f"THESIS {payload.get('strategy_name')} "
            f"family={payload.get('strategy_family')} "
            f"direction={payload.get('direction')} "
            f"confidence={payload.get('confidence')}"
        )
    if event_type == "debate.review.completed":
        return (
            f"REVIEW {payload.get('reviewer_role')} "
            f"verdict={payload.get('verdict')} "
            f"confidence={payload.get('confidence')}"
        )
    if event_type == "graph.cycle.completed":
        line = (
            f"CYCLE {payload.get('outcome')} "
            f"action={payload.get('decision_action')}"
        )
        errors = _structured_errors(payload)
        if errors:
            line += "\n" + "\n".join(_error_lines(errors, verbose=False))
        return line
    return f"{event_type} " + " ".join(
        f"{key}={payload.get(key)}" for key in list(payload)[:4]
    )


def _structured_errors(payload: dict[str, Any]) -> list[dict[str, Any]]:
    errors = payload.get("errors")
    if not isinstance(errors, list):
        return []
    return [error for error in errors if isinstance(error, dict)]


def _error_lines(errors: list[dict[str, Any]], *, verbose: bool) -> list[str]:
    """Render blocking-reason lines so operators never see a bare error code."""
    lines: list[str] = []
    for error in errors:
        message = str(error.get("message") or "")
        if not verbose and len(message) > 200:
            message = message[:200] + "…"
        lines.append(
            f"  OBJECTIVE ERROR code={error.get('code')} "
            f"node={error.get('node')} "
            f"recoverable={error.get('recoverable')} "
            f"message={message}"
        )
    return lines


def _verbose(
    event_type: str,
    payload: dict[str, Any],
    *,
    top_contract_rows: int,
    top_portfolio_rows: int,
) -> str:
    sections: list[str] = [f"── {event_type} ──"]
    if event_type == "graph.cycle.completed":
        sections.extend(
            [
                "CYCLE",
                _fields(
                    payload,
                    (
                        "cycle_id",
                        "outcome",
                        "decision_action",
                        "execution_command_ids",
                        "error_codes",
                    ),
                ),
            ]
        )
    goal = payload.get("goal")
    if isinstance(goal, dict):
        sections.extend(
            [
                "GOAL",
                _fields(
                    goal,
                    (
                        "authorized_capital",
                        "available_capital",
                        "realized_pnl",
                        "remaining_goal_gap",
                        "target",
                        "deadline",
                        "time_remaining_seconds",
                        "maximum_positions",
                    ),
                ),
            ]
        )
    market = payload.get("market")
    if isinstance(market, dict):
        sections.extend(
            [
                "MARKET",
                _fields(
                    market,
                    (
                        "spy_price",
                        "market_direction",
                        "volatility_regime",
                        "session_phase",
                        "option_surface_size",
                        "eligible_contract_count",
                        "data_quality_state",
                    ),
                ),
            ]
        )
    theses = payload.get("theses")
    if isinstance(theses, list):
        sections.append("AGENT THESES")
        sections.extend(
            _table_rows(
                theses,
                (
                    "agent_role",
                    "strategy_name",
                    "strategy_family",
                    "direction",
                    "confidence",
                    "thesis_summary",
                    "expected_horizon_seconds",
                    "key_evidence",
                ),
            )
        )
    contracts = payload.get("contracts")
    if isinstance(contracts, list):
        sections.append("CONTRACT GRID")
        sections.extend(
            _table_rows(
                contracts[:top_contract_rows],
                (
                    "rank",
                    "strategy",
                    "contract_id",
                    "option_type",
                    "strike",
                    "distance_from_spot",
                    "delta",
                    "bid",
                    "ask",
                    "relative_spread",
                    "quantity",
                    "capital_required",
                    "maximum_loss",
                    "useful_upside",
                    "probability_goal",
                    "probability_wait",
                    "probability_delta",
                    "estimate_type",
                    "reason_codes",
                    "selected",
                ),
            )
        )
    portfolios = payload.get("portfolios")
    if isinstance(portfolios, list):
        sections.append("PORTFOLIO GRID")
        sections.extend(
            _table_rows(
                portfolios[:top_portfolio_rows],
                (
                    "rank",
                    "component_contract_ids",
                    "component_quantities",
                    "capital_deployed",
                    "maximum_loss",
                    "expected_pnl",
                    "probability_goal",
                    "probability_wait",
                    "probability_delta",
                    "selected",
                ),
            )
        )
    reviews = payload.get("reviews")
    if isinstance(reviews, list):
        sections.append("DEBATE")
        sections.extend(
            _table_rows(
                reviews,
                (
                    "reviewer_role",
                    "reviewed_id",
                    "verdict",
                    "confidence",
                    "claims_summary",
                    "failure_modes",
                    "required_revisions",
                ),
            )
        )
    decision = payload.get("decision")
    if isinstance(decision, dict):
        sections.extend(["DECISION", _fields(decision, tuple(decision.keys()))])
    execution = payload.get("execution")
    if isinstance(execution, dict):
        sections.extend(["EXECUTION", _fields(execution, tuple(execution.keys()))])
    errors = _structured_errors(payload)
    if errors:
        sections.append("ERRORS")
        sections.extend(_error_lines(errors, verbose=True))
    if len(sections) == 1:
        sections.append(_fields(payload, tuple(payload.keys())))
    return "\n".join(section for section in sections if section)


def _fields(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    return " | ".join(
        f"{key}={payload.get(key)}" for key in keys if key in payload
    )


def _table_rows(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[str]:
    rendered: list[str] = []
    for index, row in enumerate(rows, start=1):
        rendered.append(
            f"{index:>2}. "
            + " | ".join(
                f"{key}={row.get(key)}" for key in keys if key in row
            )
        )
    return rendered or ["(none)"]
