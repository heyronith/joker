"""Resolve factual TradingEpisode similarity metadata from production provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from joker.objectives.historical_outcomes import session_phase_from_exchange_ts


DirectionLabel = Literal["bullish", "bearish", "neutral", "none"]
SequencePolicy = Literal["globally_contiguous", "monotonic_only", "unavailable"]


@dataclass
class EpisodeSimilarityMetadata:
    """Factual dimensions for historical EV similarity — never invented."""

    direction: DirectionLabel = "none"
    strategy_family: str | None = None
    pattern_ids: tuple[UUID, ...] = ()
    option_type: str | None = None
    session_phase: str | None = None
    volatility_bucket: str | None = None
    liquidity_bucket: str | None = None
    market_regime_tags: tuple[str, ...] = ()
    parent_strategy_id: UUID | None = None
    findings: list[str] = field(default_factory=list)

    # Pattern provenance missing reduces similarity but does not alone make the
    # episode lifecycle-incomplete or EV-ineligible (family/direction/type do).
    _CRITICAL_EV_GAPS = frozenset(
        {
            "historical_strategy_family_missing",
            "historical_direction_missing",
            "historical_option_type_missing",
        }
    )

    @property
    def historical_ev_eligible(self) -> bool:
        return not any(f in self._CRITICAL_EV_GAPS for f in self.findings)


def classify_long_option_direction(
    *,
    option_type: str | None,
    entry_side: str | None,
) -> DirectionLabel:
    """Map long single-leg option structure to direction; unresolved → none."""
    side = (entry_side or "").lower()
    otype = (option_type or "").lower()
    if side != "buy" or otype not in {"call", "put"}:
        return "none"
    if otype == "call":
        return "bullish"
    return "bearish"


def option_type_from_contract_id(contract_id: str | None) -> str | None:
    """Extract option type from SYMBOL:YYYY-MM-DD:strike:type without inventing 0DTE."""
    if not contract_id:
        return None
    parts = str(contract_id).split(":")
    if len(parts) != 4:
        return None
    otype = parts[3].lower()
    if otype not in {"call", "put"}:
        return None
    return otype


def _bucket_from_vol_state(state: object | None, *, summary: str | None = None) -> str | None:
    raw = f"{state or ''} {summary or ''}".lower()
    if not raw.strip():
        return None
    if "extreme" in raw:
        return "extreme"
    if "high" in raw or "elevated" in raw:
        return "high"
    if "low" in raw:
        return "low"
    if raw.strip() in {"low", "medium", "high", "extreme"}:
        return raw.strip()
    return "medium"


def _bucket_from_spread(spread: str | None) -> str | None:
    if not spread:
        return None
    raw = str(spread).lower()
    if "tight" in raw:
        return "tight"
    if "wide" in raw:
        return "wide"
    return "normal"


async def resolve_episode_similarity_metadata(
    *,
    contract_id: str | None,
    entry_orders: tuple[Any, ...],
    strategy_id: UUID | None,
    entry_cycle_id: str | None,
    session_id: str,
    strategy_repo: Any | None = None,
    world_model_repo: Any | None = None,
    cycle_registry: Any | None = None,
    exchange_timestamp: datetime | None = None,
    market_regime_tags: tuple[str, ...] = (),
) -> EpisodeSimilarityMetadata:
    """Resolve similarity fields from strategy / world-model provenance."""
    meta = EpisodeSimilarityMetadata(market_regime_tags=market_regime_tags)
    meta.option_type = option_type_from_contract_id(contract_id)
    if meta.option_type is None:
        meta.findings.append("historical_option_type_missing")

    entry_side = None
    if entry_orders:
        entry_side = getattr(entry_orders[0], "side", None)
    meta.direction = classify_long_option_direction(
        option_type=meta.option_type, entry_side=entry_side
    )
    if meta.direction == "none":
        meta.findings.append("historical_direction_missing")

    meta.parent_strategy_id = strategy_id
    strategy = None
    if strategy_id is not None and strategy_repo is not None:
        getter = getattr(strategy_repo, "get_by_id", None)
        if getter is not None:
            strategy = await getter(strategy_id)

    if strategy is None:
        meta.findings.append("historical_strategy_family_missing")
        meta.findings.append("historical_pattern_provenance_missing")
    else:
        family = getattr(strategy, "strategy_family", None)
        # Never invent family from agent role — explicit StrategyHypothesis only.
        if family:
            meta.strategy_family = str(family)
        else:
            meta.findings.append("historical_strategy_family_missing")

        pattern_raw = tuple(getattr(strategy, "source_hypothesis_ids", ()) or ())
        pattern_ids: list[UUID] = []
        for item in pattern_raw:
            try:
                pattern_ids.append(UUID(str(item)))
            except Exception:
                continue
        if pattern_ids:
            meta.pattern_ids = tuple(dict.fromkeys(pattern_ids))
        else:
            meta.findings.append("historical_pattern_provenance_missing")

        # Prefer strategy direction when structure classification is none but strategy known.
        if meta.direction == "none":
            sdir = getattr(strategy, "direction", None)
            sdir_val = getattr(sdir, "value", sdir)
            if str(sdir_val).lower() in {"bullish", "bearish", "neutral"}:
                meta.direction = str(sdir_val).lower()  # type: ignore[assignment]
                meta.findings = [
                    f
                    for f in meta.findings
                    if f != "historical_direction_missing"
                ]

    # World-model / session context from entry cycle when available.
    world_model = None
    if cycle_registry is not None and entry_cycle_id and world_model_repo is not None:
        try:
            record = await cycle_registry.get(
                session_id=session_id, graph_kind="decision", cycle_id=entry_cycle_id
            )
        except Exception:
            record = None
        if record is not None:
            payload = record.payload or {}
            wm_id = payload.get("world_model_id")
            if wm_id is not None:
                try:
                    world_model = await world_model_repo.get_by_id(UUID(str(wm_id)))
                except Exception:
                    world_model = None

    if world_model is not None:
        ms = getattr(world_model, "market_structure", None)
        tags: list[str] = list(meta.market_regime_tags)
        if ms is not None:
            primary = getattr(ms, "primary_direction", None)
            if primary is not None:
                tags.append(str(getattr(primary, "value", primary)))
            tags.append(
                "range_bound" if getattr(ms, "range_bound", False) else "trend"
            )
        meta.market_regime_tags = tuple(dict.fromkeys(t for t in tags if t))
        vol = getattr(world_model, "volatility_state", None)
        if vol is not None:
            meta.volatility_bucket = _bucket_from_vol_state(
                getattr(vol, "state", None),
                summary=str(getattr(vol, "summary", "") or ""),
            )
        opt = getattr(world_model, "options_state", None)
        if opt is not None:
            meta.liquidity_bucket = _bucket_from_spread(
                str(getattr(opt, "spread_conditions", "") or "")
            )
        temporal = getattr(world_model, "temporal_state", None)
        if temporal is not None:
            phase = str(getattr(temporal, "session_phase", "") or "")
            if phase and phase != "unknown":
                meta.session_phase = phase

    if meta.session_phase in {None, "", "unknown"} and exchange_timestamp is not None:
        meta.session_phase = session_phase_from_exchange_ts(exchange_timestamp)

    if not meta.historical_ev_eligible:
        meta.findings.append("historical_ev_eligible=false")
        # Deduplicate while preserving order.
        meta.findings = list(dict.fromkeys(meta.findings))

    return meta


def _horizon_fail(findings: list[str], *codes: str) -> tuple[bool, tuple[str, ...]]:
    findings.extend(codes)
    findings.extend(
        (
            "historical_ev_eligible=false",
            "promotion_eligible=false",
            "truth_degraded=true",
        )
    )
    return False, tuple(dict.fromkeys(findings))


def verify_event_horizon(
    horizon: Any | None,
    *,
    entry_ts: datetime | None,
    terminal_ts: datetime | None,
    entry_event_id: UUID | None = None,
    terminal_event_id: UUID | None = None,
    sequence_policy: SequencePolicy = "unavailable",
    legitimate_sequence_gaps: frozenset[int] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Return (complete, findings). Fail closed on gaps / anchors / duplicates."""
    findings: list[str] = []
    if horizon is None:
        return _horizon_fail(findings, "authoritative_horizon_incomplete")
    events = tuple(getattr(horizon, "events", ()) or ())
    market_ids = tuple(getattr(horizon, "market_event_ids", ()) or ())
    if not events or not market_ids:
        return _horizon_fail(findings, "authoritative_horizon_incomplete")

    if sequence_policy not in {
        "globally_contiguous",
        "monotonic_only",
        "unavailable",
    }:
        return _horizon_fail(findings, "authoritative_horizon_sequence_gap")

    event_ids: list[UUID] = []
    for ev in events:
        eid = getattr(ev, "event_id", None)
        if eid is None:
            return _horizon_fail(findings, "authoritative_horizon_incomplete")
        try:
            event_ids.append(eid if isinstance(eid, UUID) else UUID(str(eid)))
        except Exception:
            return _horizon_fail(findings, "authoritative_horizon_incomplete")

    if len(event_ids) != len(set(event_ids)):
        return _horizon_fail(findings, "authoritative_horizon_duplicate_event")

    id_set = set(event_ids)
    if entry_event_id is not None and entry_event_id not in id_set:
        return _horizon_fail(findings, "authoritative_horizon_entry_missing")
    if terminal_event_id is not None and terminal_event_id not in id_set:
        return _horizon_fail(findings, "authoritative_horizon_terminal_missing")

    timestamps: list[datetime] = []
    sequences: list[int | None] = []
    for ev in events:
        ts = getattr(ev, "exchange_timestamp", None)
        if ts is None:
            return _horizon_fail(findings, "authoritative_horizon_incomplete")
        timestamps.append(ts)
        seq = getattr(ev, "sequence", None)
        sequences.append(seq if isinstance(seq, int) else None)

    # Horizon list order is authoritative (sequence → timestamp → event_id).
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            return _horizon_fail(findings, "authoritative_horizon_non_monotonic")

    if entry_ts is not None and timestamps[0] > entry_ts:
        return _horizon_fail(findings, "authoritative_horizon_time_coverage_incomplete")
    if terminal_ts is not None and timestamps[-1] < terminal_ts:
        return _horizon_fail(findings, "authoritative_horizon_time_coverage_incomplete")
    if entry_ts is not None and terminal_ts is not None and terminal_ts < entry_ts:
        return _horizon_fail(findings, "authoritative_horizon_time_coverage_incomplete")

    present_seqs = [s for s in sequences if s is not None]
    if sequence_policy == "unavailable":
        return _horizon_fail(findings, "authoritative_horizon_sequence_gap")

    if len(present_seqs) != len(sequences):
        return _horizon_fail(findings, "authoritative_horizon_sequence_gap")

    for i in range(1, len(present_seqs)):
        if present_seqs[i] < present_seqs[i - 1]:
            return _horizon_fail(findings, "authoritative_horizon_non_monotonic")

    if sequence_policy == "globally_contiguous":
        lo, hi = min(present_seqs), max(present_seqs)
        expected = set(range(lo, hi + 1))
        if set(present_seqs) != expected or len(present_seqs) != (hi - lo + 1):
            return _horizon_fail(findings, "authoritative_horizon_sequence_gap")
    elif sequence_policy == "monotonic_only":
        for i in range(1, len(present_seqs)):
            gap = present_seqs[i] - present_seqs[i - 1]
            if gap <= 0:
                return _horizon_fail(findings, "authoritative_horizon_non_monotonic")
            if gap > 1:
                missing = set(range(present_seqs[i - 1] + 1, present_seqs[i]))
                allowed = legitimate_sequence_gaps or frozenset()
                if not missing.issubset(allowed):
                    return _horizon_fail(findings, "authoritative_horizon_sequence_gap")

    if findings:
        return _horizon_fail(findings)
    return True, ()
