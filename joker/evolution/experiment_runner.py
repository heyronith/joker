"""Isolated champion/challenger experiment runner with durable episode results."""

from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from joker.evolution.experiment_results_store import ExperimentEpisodeResultStore
from joker.evolution.hashing import hash_model
from joker.evolution.idempotency import experiment_episode_key
from joker.evolution.promotion_gate import PromotionEligibilityGate
from joker.evolution.repositories import ExperimentRepository
from joker.evolution.schemas import (
    ExperimentDefinition,
    ExperimentResult,
    ExperimentSliceResult,
    TradingEpisode,
)

ReplayFn = Callable[[TradingEpisode, UUID, int], Awaitable[dict[str, Any]]]


class ExperimentRunnerError(RuntimeError):
    pass


class ExperimentRunner:
    """Replay-only experiments; persists every episode/sample idempotency key."""

    def __init__(
        self,
        experiment_repo: ExperimentRepository,
        *,
        gate: PromotionEligibilityGate | None = None,
        repeated_samples: int = 3,
        db_path: str | Path | None = None,
        result_store: ExperimentEpisodeResultStore | None = None,
        replay_service: Any | None = None,
        bootstrap_seed: int = 42,
        bootstrap_samples: int = 200,
        confidence_level: Decimal = Decimal("0.95"),
    ) -> None:
        self._experiments = experiment_repo
        self._gate = gate or PromotionEligibilityGate()
        self._repeated_samples = repeated_samples
        self._replay_service = replay_service
        self._bootstrap_seed = bootstrap_seed
        self._bootstrap_samples = bootstrap_samples
        self._confidence_level = confidence_level
        if result_store is not None:
            self._results = result_store
        elif db_path is not None:
            self._results = ExperimentEpisodeResultStore(db_path)
        else:
            raise ExperimentRunnerError(
                "ExperimentRunner requires db_path or result_store for durable idempotency"
            )

    async def create(self, definition: ExperimentDefinition) -> ExperimentDefinition:
        await self._experiments.append_definition(definition)
        return definition

    def _resolve_replay_fn(self, replay_fn: ReplayFn | None) -> ReplayFn:
        if replay_fn is not None:
            return replay_fn
        if self._replay_service is None:
            raise ExperimentRunnerError(
                "replay_fn is required unless ExperimentRunner was constructed "
                "with a CognitiveReplayService"
            )
        return self._replay_service.replay_episode

    async def run(
        self,
        experiment_id: UUID | str,
        *,
        episodes: list[TradingEpisode],
        partition_map: dict[str, tuple[UUID, ...]],
        replay_fn: ReplayFn | None = None,
        adversarial_passed: bool = True,
    ) -> ExperimentResult:
        if self._results is None:
            raise ExperimentRunnerError("experiment result store is required")
        await self._results.initialize()
        resolved_replay = self._resolve_replay_fn(replay_fn)
        definition = await self._experiments.get_definition(experiment_id)
        if definition is None:
            raise ExperimentRunnerError(f"experiment not found: {experiment_id}")
        await self._experiments.mark_status(experiment_id, "running")

        by_id = {ep.episode_id: ep for ep in episodes}
        slice_results: list[ExperimentSliceResult] = []
        champ_vals: list[Decimal] = []
        chall_vals: list[Decimal] = []
        champ_latency: list[Decimal] = []
        chall_latency: list[Decimal] = []
        champ_cost = Decimal("0")
        chall_cost = Decimal("0")
        missing: list[UUID] = []
        model_calls = 0
        cost = Decimal("0")
        completed_keys = await self._results.list_keys(experiment_id)

        for slice_name, ids in partition_map.items():
            deltas: list[Decimal] = []
            missing_count = 0
            for eid in ids:
                ep = by_id.get(eid)
                if ep is None:
                    missing.append(eid)
                    missing_count += 1
                    continue
                for sample in range(1, self._repeated_samples + 1):
                    for cfg_id, bucket in (
                        (definition.champion_version_id, "champion"),
                        (definition.challenger_version_id, "challenger"),
                    ):
                        key = experiment_episode_key(
                            experiment_id, eid, cfg_id, sample
                        )
                        if key in completed_keys:
                            payload = await self._results.get_payload(key)
                            assert payload is not None
                        else:
                            payload = await resolved_replay(ep, cfg_id, sample)
                            if payload.get("broker_submit"):
                                raise ExperimentRunnerError(
                                    "experiment replay attempted broker submission"
                                )
                            await self._results.append(
                                idempotency_key=key,
                                experiment_id=experiment_id,
                                episode_id=eid,
                                configuration_version_id=cfg_id,
                                sample_number=sample,
                                payload=payload,
                            )
                            completed_keys.add(key)
                            await self._experiments.mark_status(
                                experiment_id, "running", recovery_cursor=key
                            )
                        pnl = Decimal(str(payload.get("realised_pnl", 0)))
                        calls = int(payload.get("model_calls", 1))
                        sample_cost_raw = payload.get("cost_gbp")
                        sample_cost = (
                            Decimal(str(sample_cost_raw))
                            if sample_cost_raw is not None
                            else Decimal("0")
                        )
                        latency_raw = payload.get("latency_ms")
                        latency = (
                            Decimal(str(latency_raw))
                            if latency_raw is not None
                            else Decimal("0")
                        )
                        model_calls += calls
                        cost += sample_cost
                        if bucket == "champion":
                            champ_vals.append(pnl)
                            champ_latency.append(latency)
                            champ_cost += sample_cost
                        else:
                            chall_vals.append(pnl)
                            chall_latency.append(latency)
                            chall_cost += sample_cost
                            if champ_vals:
                                deltas.append(chall_vals[-1] - champ_vals[-1])
                        if (
                            sample_cost_raw is not None
                            and cost > definition.maximum_cost_gbp
                        ):
                            await self._experiments.mark_status(experiment_id, "failed")
                            raise ExperimentRunnerError("maximum_cost_gbp exceeded")
            avg_delta = (
                sum(deltas) / Decimal(len(deltas)) if deltas else Decimal("0")
            )
            ci, ci_meta = self._bootstrap_ci(deltas)
            slice_results.append(
                ExperimentSliceResult(
                    slice_name=slice_name,
                    metrics={
                        "pnl_delta": avg_delta,
                        "samples": len(deltas),
                        "ci_method": ci_meta["method"],
                        "ci_seed": ci_meta["seed"],
                        "ci_sample_count": ci_meta["sample_count"],
                        "ci_confidence_level": str(ci_meta["confidence_level"]),
                    },
                    episode_count=len(ids) - missing_count,
                    missing_episode_count=missing_count,
                    confidence_intervals={"pnl_delta": ci},
                )
            )

        champ_mean = (
            sum(champ_vals) / Decimal(len(champ_vals)) if champ_vals else Decimal("0")
        )
        chall_mean = (
            sum(chall_vals) / Decimal(len(chall_vals)) if chall_vals else Decimal("0")
        )
        champ_tail = min(champ_vals) if champ_vals else Decimal("0")
        chall_tail = min(chall_vals) if chall_vals else Decimal("0")
        champ_lat = (
            sum(champ_latency) / Decimal(len(champ_latency))
            if champ_latency
            else Decimal("0")
        )
        chall_lat = (
            sum(chall_latency) / Decimal(len(chall_latency))
            if chall_latency
            else Decimal("0")
        )
        champ_cal = _mad(champ_vals, champ_mean)
        chall_cal = _mad(chall_vals, chall_mean)
        champ_brier, champ_ece, champ_cal_n = _calibration_from_pairs(
            await self._collect_calibration_pairs(
                experiment_id, definition.champion_version_id
            )
        )
        chall_brier, chall_ece, chall_cal_n = _calibration_from_pairs(
            await self._collect_calibration_pairs(
                experiment_id, definition.challenger_version_id
            )
        )
        champ_cost_known, chall_cost_known = await self._cost_known_flags(
            experiment_id,
            definition.champion_version_id,
            definition.challenger_version_id,
        )
        champ_cal_metric = champ_ece if champ_ece is not None else champ_cal
        chall_cal_metric = chall_ece if chall_ece is not None else chall_cal
        deltas_all = [b - a for a, b in zip(champ_vals, chall_vals)]
        ci_all, ci_meta = self._bootstrap_ci(deltas_all)
        required_missing: list[str] = []
        if not champ_vals or not chall_vals:
            required_missing.append("missing_replay_pnl_metrics")
        result = ExperimentResult(
            result_id=uuid4(),
            experiment_id=definition.experiment_id,
            per_slice_results=tuple(slice_results),
            aggregate_metrics={
                "champion_mean_pnl": champ_mean,
                "challenger_mean_pnl": chall_mean,
                "pnl_delta": chall_mean - champ_mean,
                "ci_method": ci_meta["method"],
                "ci_seed": ci_meta["seed"],
                "ci_sample_count": ci_meta["sample_count"],
                "ci_confidence_level": str(ci_meta["confidence_level"]),
            },
            confidence_intervals={"pnl_delta": ci_all},
            cost_gbp=cost,
            model_call_counts={"total": model_calls},
            missing_episodes=tuple(missing),
            champion_metrics={
                "mean_pnl": champ_mean,
                "tail_loss": champ_tail,
                "calibration_error": champ_cal_metric,
                "pnl_mean_absolute_deviation": champ_cal,
                **(
                    {"brier_score": champ_brier}
                    if champ_brier is not None
                    else {}
                ),
                **(
                    {"expected_calibration_error": champ_ece}
                    if champ_ece is not None
                    else {}
                ),
                "calibration_sample_count": Decimal(champ_cal_n),
                "latency_ms": champ_lat,
                "cost_gbp": champ_cost,
                "cost_known": champ_cost_known,
            },
            challenger_metrics={
                "mean_pnl": chall_mean,
                "tail_loss": chall_tail,
                "calibration_error": chall_cal_metric,
                "pnl_mean_absolute_deviation": chall_cal,
                **(
                    {"brier_score": chall_brier}
                    if chall_brier is not None
                    else {}
                ),
                **(
                    {"expected_calibration_error": chall_ece}
                    if chall_ece is not None
                    else {}
                ),
                "calibration_sample_count": Decimal(chall_cal_n),
                "latency_ms": chall_lat,
                "cost_gbp": chall_cost,
                "cost_known": chall_cost_known,
            },
            eligibility_outcome=False,
            gate_rejection_codes=tuple(required_missing),
            content_hash="",
        )
        holdout_count = len(partition_map.get("holdout", ()))
        eligibility = self._gate.evaluate(
            result=result,
            holdout_episode_count=holdout_count,
            completed_episode_count=len(champ_vals),
            adversarial_passed=adversarial_passed and not required_missing,
        )
        gate_codes = list(eligibility.gate_codes)
        gate_codes.extend(required_missing)
        result = result.model_copy(
            update={
                "eligibility_outcome": eligibility.eligible and not required_missing,
                "gate_rejection_codes": tuple(dict.fromkeys(gate_codes)),
            }
        )
        result = result.model_copy(
            update={"content_hash": hash_model(result, exclude={"created_at"})}
        )
        await self._experiments.append_result(result)
        await self._experiments.mark_status(
            experiment_id, "completed", recovery_cursor=None
        )
        return result

    async def resume(
        self,
        experiment_id: UUID | str,
        *,
        episodes: list[TradingEpisode],
        partition_map: dict[str, tuple[UUID, ...]],
        replay_fn: ReplayFn | None = None,
    ) -> ExperimentResult:
        existing = await self._experiments.get_result(experiment_id)
        if existing is not None:
            return existing
        return await self.run(
            experiment_id,
            episodes=episodes,
            partition_map=partition_map,
            replay_fn=replay_fn,
        )

    def _bootstrap_ci(
        self, values: list[Decimal]
    ) -> tuple[tuple[Decimal, Decimal], dict[str, Any]]:
        meta = {
            "method": "bootstrap_percentile",
            "seed": self._bootstrap_seed,
            "sample_count": self._bootstrap_samples,
            "confidence_level": self._confidence_level,
        }
        if not values:
            return (Decimal("0"), Decimal("0")), meta
        rng = random.Random(self._bootstrap_seed)
        n = len(values)
        means: list[Decimal] = []
        for _ in range(self._bootstrap_samples):
            sample = [values[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / Decimal(n))
        means.sort()
        alpha = (Decimal("1") - self._confidence_level) / Decimal("2")
        lo_i = int(alpha * Decimal(len(means)))
        hi_i = max(lo_i, int((Decimal("1") - alpha) * Decimal(len(means))) - 1)
        return (means[lo_i], means[hi_i]), meta

    async def _collect_calibration_pairs(
        self, experiment_id: UUID | str, configuration_version_id: UUID
    ) -> list[tuple[Decimal, int]]:
        keys = await self._results.list_keys(experiment_id)
        pairs: list[tuple[Decimal, int]] = []
        for key in keys:
            if str(configuration_version_id) not in key:
                continue
            payload = await self._results.get_payload(key)
            if not payload:
                continue
            for pred, outcome in payload.get("calibration_pairs") or []:
                pairs.append((Decimal(str(pred)), int(outcome)))
        return pairs

    async def _cost_known_flags(
        self,
        experiment_id: UUID | str,
        champion_id: UUID,
        challenger_id: UUID,
    ) -> tuple[bool, bool]:
        keys = await self._results.list_keys(experiment_id)
        champ = False
        chall = False
        for key in keys:
            payload = await self._results.get_payload(key)
            if not payload:
                continue
            known = bool(payload.get("cost_known", payload.get("cost_gbp") is not None))
            if str(champion_id) in key and known:
                champ = True
            if str(challenger_id) in key and known:
                chall = True
        return champ, chall


def _mad(values: list[Decimal], mean: Decimal) -> Decimal:
    if not values:
        return Decimal("0")
    return sum(abs(v - mean) for v in values) / Decimal(len(values))


def _calibration_from_pairs(
    pairs: list[tuple[Decimal, int]],
) -> tuple[Decimal | None, Decimal | None, int]:
    from joker.evolution.telemetry import brier_score, expected_calibration_error

    if not pairs:
        return None, None, 0
    return brier_score(pairs), expected_calibration_error(pairs), len(pairs)
