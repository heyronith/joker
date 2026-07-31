"""Task-3 factual historical outcome retrieval for goal-driven EV.

Production live estimates use completed, leakage-safe paper episodes only.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from statistics import median
from typing import Any
from uuid import UUID, uuid4

from joker.objectives.config import HistoricalOutcomeSettings
from joker.objectives.historical_schemas import (
    ComparableOutcome,
    HistoricalLeakageReport,
    HistoricalOutcomeQuery,
    HistoricalOutcomeSummary,
    SimilarityPolicy,
)
from joker.objectives.similarity import SIMILARITY_POLICY_VERSION, score_similarity

EpisodeLoader = Callable[[], Awaitable[Sequence[Any]]]
EvaluationLoader = Callable[[UUID], Awaitable[Sequence[Any]]]


def independence_key(episode: Any) -> str:
    """One authoritative episode contributes at most one live-EV observation."""
    eid = str(getattr(episode, "episode_id", ""))
    entry = str(
        getattr(episode, "entry_decision_event_id", None)
        or getattr(episode, "position_lifecycle_id", None)
        or getattr(episode, "initial_snapshot_id", "")
    )
    terminal = str(
        getattr(episode, "terminal_event_id", None)
        or getattr(episode, "terminal_snapshot_id", "")
    )
    return f"{eid}|{entry}|{terminal}"


def _z_for_confidence(level: float) -> Decimal:
    # Normal approx; adequate for configured n>=20.
    if level >= 0.99:
        return Decimal("2.576")
    if level >= 0.95:
        return Decimal("1.960")
    if level >= 0.90:
        return Decimal("1.645")
    return Decimal("1.960")


def _premium_bucket(premium: Decimal | None) -> str | None:
    if premium is None:
        return None
    p = float(premium)
    if p < 0.50:
        return "lt_0.50"
    if p < 1.00:
        return "0.50_1.00"
    if p < 2.00:
        return "1.00_2.00"
    if p < 5.00:
        return "2.00_5.00"
    return "gte_5.00"


def _horizon_bucket(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 300:
        return "lt_5m"
    if seconds < 900:
        return "5_15m"
    if seconds < 3600:
        return "15_60m"
    return "gte_60m"


def _session_phase_from_ts(ts: datetime | None) -> str:
    if ts is None:
        return "unknown"
    local = ts.astimezone(timezone.utc)
    # Rough ET-proxy buckets using UTC offset-naive hour is avoided; use hour ET approx.
    hour = (local.hour - 4) % 24  # EDT approximation for bucketing only
    if 9 <= hour < 11:
        return "open"
    if 11 <= hour < 14:
        return "midday"
    if 14 <= hour < 16:
        return "close"
    return "other"


class HistoricalOutcomeService:
    """Load comparable Task-3 episodes and aggregate leakage-safe EV statistics."""

    def __init__(
        self,
        *,
        episode_loader: EpisodeLoader | None = None,
        evaluation_loader: EvaluationLoader | None = None,
        settings: HistoricalOutcomeSettings | None = None,
        similarity_policy: SimilarityPolicy | None = None,
        approved_evaluator_versions: frozenset[str] | None = None,
        repository: Any | None = None,
    ) -> None:
        self._episode_loader = episode_loader
        self._evaluation_loader = evaluation_loader
        self._settings = settings or HistoricalOutcomeSettings()
        self._policy = similarity_policy or SimilarityPolicy(
            policy_version=SIMILARITY_POLICY_VERSION
        )
        self._approved_evaluators = approved_evaluator_versions or frozenset(
            {"3.0.0", "3.1.0", "3.2.0"}
        )
        self._repo = repository
        self._seeded: list[Any] = []

    def seed_episodes_for_tests(self, episodes: Sequence[Any]) -> None:
        """Test helper — production uses episode_loader repositories."""
        self._seeded = list(episodes)

    async def query_comparable_outcomes(
        self,
        query: HistoricalOutcomeQuery,
    ) -> tuple[
        HistoricalOutcomeSummary,
        HistoricalLeakageReport,
        tuple[ComparableOutcome, ...],
    ]:
        settings = self._settings
        exclusion: dict[str, int] = {}
        future: list[UUID] = []
        current: list[UUID] = []
        duplicates: list[UUID] = []
        incomplete: list[UUID] = []
        degraded: list[UUID] = []
        dataset_overlap: list[UUID] = []

        episodes = await self._load_episodes()
        seen_keys: set[str] = set()
        candidates: list[ComparableOutcome] = []
        max_age = timedelta(days=int(settings.maximum_episode_age_days))
        as_of = query.as_of_timestamp

        for episode in episodes:
            eid = UUID(str(episode.episode_id))
            reasons = self._eligibility_exclusions(episode, query, as_of, max_age)
            if "future_terminal" in reasons:
                future.append(eid)
                exclusion["future_terminal"] = exclusion.get("future_terminal", 0) + 1
                continue
            if "current_episode" in reasons:
                current.append(eid)
                exclusion["current_episode"] = exclusion.get("current_episode", 0) + 1
                continue
            if "incomplete" in reasons:
                incomplete.append(eid)
                exclusion["incomplete"] = exclusion.get("incomplete", 0) + 1
                continue
            if "truth_degraded" in reasons:
                degraded.append(eid)
                exclusion["truth_degraded"] = exclusion.get("truth_degraded", 0) + 1
                continue
            if "synthetic_replay" in reasons:
                exclusion["synthetic_replay"] = exclusion.get("synthetic_replay", 0) + 1
                continue
            if "age" in reasons:
                exclusion["age"] = exclusion.get("age", 0) + 1
                continue
            if "missing_pnl" in reasons or "not_closed_trade" in reasons:
                exclusion["not_factual_closed"] = (
                    exclusion.get("not_factual_closed", 0) + 1
                )
                continue
            if reasons:
                for r in reasons:
                    exclusion[r] = exclusion.get(r, 0) + 1
                continue

            key = independence_key(episode)
            if key in seen_keys:
                duplicates.append(eid)
                exclusion["duplicate_truth"] = exclusion.get("duplicate_truth", 0) + 1
                continue
            seen_keys.add(key)

            evaluations = await self._load_evaluations(eid)
            evaluation = self._select_evaluation(evaluations)
            if evaluation is None:
                exclusion["no_approved_evaluation"] = (
                    exclusion.get("no_approved_evaluation", 0) + 1
                )
                continue
            if getattr(evaluation, "created_at", None) is not None:
                if evaluation.created_at > as_of:
                    future.append(eid)
                    exclusion["future_evaluation"] = (
                        exclusion.get("future_evaluation", 0) + 1
                    )
                    continue

            outcome = self._to_comparable(episode, evaluation, query)
            if outcome.similarity_score < query.minimum_similarity:
                exclusion["below_similarity"] = exclusion.get("below_similarity", 0) + 1
                continue
            if (
                settings.require_same_strategy_family
                and query.strategy_family
                and outcome.similarity_components.get("strategy_family_match", 0)
                < Decimal("1")
            ):
                exclusion["strategy_family_mismatch"] = (
                    exclusion.get("strategy_family_mismatch", 0) + 1
                )
                continue
            candidates.append(outcome)

        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        selected = candidates[: int(query.maximum_samples)]
        leakage_safe = not (future or current or dataset_overlap)
        # Duplicate exclusions are handled; they do not make leakage unsafe.
        report = HistoricalLeakageReport(
            query_id=query.query_id,
            excluded_future_episodes=tuple(future),
            excluded_current_episode=tuple(current),
            excluded_dataset_overlap=tuple(dataset_overlap),
            excluded_duplicate_truth=tuple(duplicates),
            excluded_incomplete=tuple(incomplete),
            excluded_truth_degraded=tuple(degraded),
            safe=leakage_safe,
            notes=()
            if leakage_safe
            else ("leakage_exclusions_present_summary_invalid_if_unresolved",),
        )
        summary = self._aggregate(query, selected, exclusion, report)
        if self._repo is not None:
            self._repo.save_historical_query(query)
            self._repo.save_historical_summary(summary)
            self._repo.save_leakage_report(report)
        return summary, report, tuple(selected)

    async def summarize_for_strategy(
        self,
        *,
        objective_id: UUID,
        strategy_id: UUID,
        snapshot_id: UUID,
        as_of_timestamp: datetime,
        direction: str | None = None,
        strategy_family: str | None = None,
        pattern_ids: Sequence[UUID] = (),
        regime_labels: Sequence[str] = (),
        session_phase: str = "unknown",
        option_type: str | None = None,
        premium_per_contract_usd: Decimal | None = None,
        expected_horizon_seconds: int | None = None,
        configuration_version_id: UUID | None = None,
        current_episode_id: UUID | None = None,
    ) -> HistoricalOutcomeSummary:
        query = HistoricalOutcomeQuery(
            objective_id=objective_id,
            strategy_id=strategy_id,
            snapshot_id=snapshot_id,
            configuration_version_id=configuration_version_id,
            pattern_ids=tuple(pattern_ids),
            strategy_family=strategy_family or direction,
            direction=direction,
            option_type=option_type,
            regime_labels=tuple(regime_labels),
            session_phase=session_phase,
            premium_bucket=_premium_bucket(premium_per_contract_usd),
            horizon_bucket=_horizon_bucket(expected_horizon_seconds),
            maximum_samples=self._settings.maximum_samples,
            minimum_similarity=Decimal(str(self._settings.minimum_similarity)),
            as_of_timestamp=as_of_timestamp,
            current_episode_id=current_episode_id,
            allow_synthetic_replay=False,
        )
        summary, _report, _outcomes = await self.query_comparable_outcomes(query)
        return summary

    def get_summary(self, summary_id: UUID | str) -> HistoricalOutcomeSummary | None:
        if self._repo is None:
            return None
        return self._repo.get_historical_summary(summary_id)

    async def _load_episodes(self) -> Sequence[Any]:
        if self._seeded:
            return list(self._seeded)
        if self._episode_loader is None:
            return ()
        return await self._episode_loader()

    async def _load_evaluations(self, episode_id: UUID) -> Sequence[Any]:
        ep = next(
            (
                e
                for e in self._seeded
                if str(getattr(e, "episode_id", "")) == str(episode_id)
            ),
            None,
        )
        if ep is not None and hasattr(ep, "evaluation"):
            return (ep.evaluation,)
        if self._evaluation_loader is None:
            return ()
        return await self._evaluation_loader(episode_id)

    def _select_evaluation(self, evaluations: Sequence[Any]) -> Any | None:
        valid = [
            e
            for e in evaluations
            if getattr(e, "valid", True)
            and str(getattr(e, "evaluator_version", "")) in self._approved_evaluators
        ]
        if not valid:
            return None
        return sorted(
            valid,
            key=lambda e: getattr(e, "created_at", datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )[0]

    def _eligibility_exclusions(
        self,
        episode: Any,
        query: HistoricalOutcomeQuery,
        as_of: datetime,
        max_age: timedelta,
    ) -> list[str]:
        reasons: list[str] = []
        eid = UUID(str(episode.episode_id))
        if query.current_episode_id is not None and eid == query.current_episode_id:
            reasons.append("current_episode")
        if not bool(getattr(episode, "completed", False)):
            reasons.append("incomplete")
        findings = tuple(getattr(episode, "completeness_findings", ()) or ())
        if any("truth_degraded" in str(f) for f in findings):
            reasons.append("truth_degraded")
        if any("historical_ev_eligible=false" in str(f) for f in findings):
            reasons.append("historical_ev_ineligible")
        if getattr(episode, "action_class", None) != "closed_trade":
            reasons.append("not_closed_trade")
        pnl = getattr(episode, "realised_pnl", None)
        if pnl is None:
            reasons.append("missing_pnl")
        terminal_ts = getattr(episode, "terminal_event_timestamp", None)
        if terminal_ts is not None and terminal_ts > as_of:
            reasons.append("future_terminal")
        entry_ts = getattr(episode, "entry_decision_timestamp", None)
        if entry_ts is not None and entry_ts > as_of:
            reasons.append("future_entry")
        if terminal_ts is not None and (as_of - terminal_ts) > max_age:
            reasons.append("age")
        if not query.allow_synthetic_replay:
            sample = getattr(episode, "sample", None)
            if sample is not None and int(sample) > 0:
                reasons.append("synthetic_replay")
            # Replay-derived markers in findings
            if any("replay" in str(f).lower() for f in findings):
                reasons.append("synthetic_replay")
        if getattr(episode, "terminal_event_id", None) is None:
            reasons.append("incomplete")
        if getattr(episode, "entry_decision_event_id", None) is None:
            reasons.append("incomplete")
        if getattr(episode, "initial_snapshot_id", None) is None:
            reasons.append("incomplete")
        return reasons

    def _to_comparable(
        self, episode: Any, evaluation: Any, query: HistoricalOutcomeQuery
    ) -> ComparableOutcome:
        entry_ts = episode.entry_decision_timestamp
        terminal_ts = episode.terminal_event_timestamp
        assert entry_ts is not None and terminal_ts is not None
        direction = str(getattr(episode, "direction", None) or "none")
        premium = getattr(episode, "entry_price", None)
        sim, components = score_similarity(
            query_strategy_family=query.strategy_family,
            query_direction=query.direction,
            query_pattern_ids=query.pattern_ids,
            query_regime_labels=query.regime_labels,
            query_session_phase=query.session_phase,
            query_volatility_bucket=query.volatility_bucket,
            query_liquidity_bucket=query.liquidity_bucket,
            query_premium_bucket=query.premium_bucket,
            query_horizon_bucket=query.horizon_bucket,
            episode_strategy_family=direction,
            episode_direction=direction,
            episode_regime_labels=tuple(getattr(episode, "market_regime_tags", ()) or ()),
            episode_session_phase=_session_phase_from_ts(entry_ts),
            episode_premium_bucket=_premium_bucket(
                Decimal(str(premium)) if premium is not None else None
            ),
            episode_horizon_bucket=_horizon_bucket(
                getattr(episode, "holding_seconds", None)
            ),
            policy=self._policy,
        )
        pnl = Decimal(str(episode.realised_pnl)).quantize(Decimal("0.01"))
        if pnl > 0:
            label = "profit"
        elif pnl < 0:
            label = "loss"
        else:
            label = "flat"
        evidence: list[UUID] = []
        for raw in getattr(episode, "cognitive_artifact_ids", ()) or ():
            try:
                evidence.append(UUID(str(raw)))
            except Exception:
                continue
        evidence.append(UUID(str(evaluation.evaluation_id)))
        return ComparableOutcome(
            episode_id=UUID(str(episode.episode_id)),
            evaluation_id=UUID(str(evaluation.evaluation_id)),
            strategy_id=getattr(episode, "parent_strategy_id", None),
            configuration_version_id=UUID(str(episode.configuration_version_id)),
            entry_snapshot_id=UUID(str(episode.initial_snapshot_id)),
            terminal_event_id=UUID(str(episode.terminal_event_id)),
            entry_timestamp=entry_ts,
            terminal_timestamp=terminal_ts,
            regime_labels=tuple(getattr(episode, "market_regime_tags", ()) or ()),
            session_phase=_session_phase_from_ts(entry_ts),
            option_type=None,
            entry_premium_usd=(
                Decimal(str(premium)).quantize(Decimal("0.01"))
                if premium is not None
                else None
            ),
            holding_seconds=getattr(episode, "holding_seconds", None),
            realized_pnl_usd=pnl,
            outcome_label=label,
            similarity_score=sim,
            similarity_components=components,
            evidence_ids=tuple(evidence),
            complete=True,
            independence_key=independence_key(episode),
            historical_ev_eligible=True,
        )

    def _aggregate(
        self,
        query: HistoricalOutcomeQuery,
        outcomes: Sequence[ComparableOutcome],
        exclusion: dict[str, int],
        report: HistoricalLeakageReport,
    ) -> HistoricalOutcomeSummary:
        settings = self._settings
        invalidation: list[str] = []
        if not report.safe:
            invalidation.append("leakage_safety_not_established")

        n = len(outcomes)
        if n == 0:
            return HistoricalOutcomeSummary(
                query_id=query.query_id,
                strategy_id=query.strategy_id,
                snapshot_id=query.snapshot_id,
                sample_count=0,
                profitable_count=0,
                losing_count=0,
                flat_count=0,
                minimum_similarity=query.minimum_similarity,
                exclusion_counts=dict(exclusion),
                valid_for_ev=False,
                invalidation_reasons=tuple(
                    invalidation or ("insufficient_calibrated_samples_for_ev",)
                ),
                similarity_policy_version=self._policy.policy_version,
            )

        pnls = [o.realized_pnl_usd for o in outcomes]
        weights = [o.similarity_score for o in outcomes]
        profitable = sum(1 for p in pnls if p > 0)
        losing = sum(1 for p in pnls if p < 0)
        flat = sum(1 for p in pnls if p == 0)

        if settings.use_similarity_weighting:
            w_sum = sum(weights, start=Decimal("0"))
            w2 = sum((w * w for w in weights), start=Decimal("0"))
            weighted_ev = (
                sum((p * w for p, w in zip(pnls, weights, strict=True)), start=Decimal("0"))
                / w_sum
            ).quantize(Decimal("0.01"))
            effective_n = (
                ((w_sum * w_sum) / w2).quantize(Decimal("0.01"))
                if w2 > 0
                else Decimal("0")
            )
            avg = weighted_ev
        else:
            avg = (sum(pnls, start=Decimal("0")) / Decimal(n)).quantize(Decimal("0.01"))
            effective_n = Decimal(n)

        med = Decimal(str(median([float(p) for p in pnls]))).quantize(Decimal("0.01"))
        mean_f = float(avg)
        var = sum((float(p) - mean_f) ** 2 for p in pnls) / max(n - 1, 1)
        std = Decimal(str(math.sqrt(var))).quantize(Decimal("0.01"))
        z = _z_for_confidence(float(settings.confidence_level))
        eff = float(effective_n) if effective_n > 0 else float(n)
        sem = float(std) / math.sqrt(eff) if eff > 0 else float(std)
        lcb = (avg - z * Decimal(str(sem))).quantize(Decimal("0.01"))

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        avg_win = (
            (sum(wins, start=Decimal("0")) / Decimal(len(wins))).quantize(Decimal("0.01"))
            if wins
            else None
        )
        avg_loss = (
            (sum(losses, start=Decimal("0")) / Decimal(len(losses))).quantize(
                Decimal("0.01")
            )
            if losses
            else None
        )
        payoff = None
        if avg_win is not None and avg_loss is not None and avg_loss != 0:
            payoff = (avg_win / abs(avg_loss)).quantize(Decimal("0.01"))
        hit = (Decimal(profitable) / Decimal(n)).quantize(Decimal("0.0001"))
        avg_sim = (
            sum(weights, start=Decimal("0")) / Decimal(n)
        ).quantize(Decimal("0.0001"))

        valid = (
            report.safe
            and n >= int(settings.minimum_samples_for_ev)
            and effective_n >= Decimal(str(settings.minimum_effective_sample_size))
            and lcb is not None
        )
        if settings.require_lower_confidence_bound_positive:
            valid = valid and lcb > 0
        else:
            valid = valid and avg > 0

        if n < int(settings.minimum_samples_for_ev):
            invalidation.append("insufficient_calibrated_samples_for_ev")
        if effective_n < Decimal(str(settings.minimum_effective_sample_size)):
            invalidation.append("insufficient_effective_sample_size")
        if settings.require_lower_confidence_bound_positive and lcb <= 0:
            invalidation.append("lower_confidence_bound_not_positive")
        if not report.safe:
            valid = False

        return HistoricalOutcomeSummary(
            query_id=query.query_id,
            strategy_id=query.strategy_id,
            snapshot_id=query.snapshot_id,
            sample_count=n,
            profitable_count=profitable,
            losing_count=losing,
            flat_count=flat,
            average_pnl_usd=avg,
            median_pnl_usd=med,
            pnl_standard_deviation_usd=std,
            hit_rate=hit,
            average_win_usd=avg_win,
            average_loss_usd=avg_loss,
            payoff_ratio=payoff,
            lower_confidence_bound_ev_usd=lcb,
            effective_sample_size=effective_n,
            minimum_similarity=query.minimum_similarity,
            average_similarity=avg_sim,
            comparable_episode_ids=tuple(o.episode_id for o in outcomes),
            evaluation_ids=tuple(o.evaluation_id for o in outcomes),
            evidence_ids=tuple(
                eid for o in outcomes for eid in o.evidence_ids
            ),
            exclusion_counts=dict(exclusion),
            valid_for_ev=bool(valid),
            invalidation_reasons=tuple(dict.fromkeys(invalidation)),
            similarity_policy_version=self._policy.policy_version,
        )


def build_historical_outcome_service_from_evolution_repos(
    *,
    episode_repo: Any,
    evaluation_repo: Any,
    settings: HistoricalOutcomeSettings | None = None,
    repository: Any | None = None,
) -> HistoricalOutcomeService:
    """Wire production Task-3 repositories into the historical outcome service."""

    async def _episodes() -> Sequence[Any]:
        return await episode_repo.list_completed(limit=2000)

    async def _evals(episode_id: UUID) -> Sequence[Any]:
        return await evaluation_repo.list_by_episode(episode_id)

    return HistoricalOutcomeService(
        episode_loader=_episodes,
        evaluation_loader=_evals,
        settings=settings,
        repository=repository,
    )
