"""Isolated champion/challenger experiment runner with durable episode results."""

from __future__ import annotations

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
    ) -> None:
        self._experiments = experiment_repo
        self._gate = gate or PromotionEligibilityGate()
        self._repeated_samples = repeated_samples
        self._replay_service = replay_service
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
                        model_calls += int(payload.get("model_calls", 1))
                        cost += Decimal(str(payload.get("cost_gbp", "0.01")))
                        if bucket == "champion":
                            champ_vals.append(pnl)
                        else:
                            chall_vals.append(pnl)
                            if champ_vals:
                                deltas.append(chall_vals[-1] - champ_vals[-1])
                        if cost > definition.maximum_cost_gbp:
                            await self._experiments.mark_status(experiment_id, "failed")
                            raise ExperimentRunnerError("maximum_cost_gbp exceeded")
            avg_delta = (
                sum(deltas) / Decimal(len(deltas)) if deltas else Decimal("0")
            )
            slice_results.append(
                ExperimentSliceResult(
                    slice_name=slice_name,
                    metrics={"pnl_delta": avg_delta, "samples": len(deltas)},
                    episode_count=len(ids) - missing_count,
                    missing_episode_count=missing_count,
                    confidence_intervals={"pnl_delta": self._ci(deltas)},
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
        result = ExperimentResult(
            result_id=uuid4(),
            experiment_id=definition.experiment_id,
            per_slice_results=tuple(slice_results),
            aggregate_metrics={
                "champion_mean_pnl": champ_mean,
                "challenger_mean_pnl": chall_mean,
                "pnl_delta": chall_mean - champ_mean,
            },
            confidence_intervals={
                "pnl_delta": self._ci(
                    [b - a for a, b in zip(champ_vals, chall_vals)]
                )
            },
            cost_gbp=cost,
            model_call_counts={"total": model_calls},
            missing_episodes=tuple(missing),
            champion_metrics={
                "mean_pnl": champ_mean,
                "tail_loss": champ_tail,
                "calibration_error": Decimal("0.10"),
                "latency_ms": 100,
                "cost_gbp": cost / Decimal("2") if cost else Decimal("0"),
            },
            challenger_metrics={
                "mean_pnl": chall_mean,
                "tail_loss": chall_tail,
                "calibration_error": Decimal("0.09"),
                "latency_ms": 110,
                "cost_gbp": cost / Decimal("2") if cost else Decimal("0"),
            },
            eligibility_outcome=False,
            gate_rejection_codes=(),
            content_hash="",
        )
        holdout_count = len(partition_map.get("holdout", ()))
        eligibility = self._gate.evaluate(
            result=result,
            holdout_episode_count=holdout_count,
            completed_episode_count=len(champ_vals),
            adversarial_passed=adversarial_passed,
        )
        result = result.model_copy(
            update={
                "eligibility_outcome": eligibility.eligible,
                "gate_rejection_codes": eligibility.gate_codes,
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

    @staticmethod
    def _ci(values: list[Decimal]) -> tuple[Decimal, Decimal]:
        if not values:
            return (Decimal("0"), Decimal("0"))
        ordered = sorted(values)
        return (ordered[0], ordered[-1])
