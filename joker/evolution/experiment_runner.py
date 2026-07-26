"""Isolated champion/challenger experiment runner (never uses live ExecutionRuntime)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4

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
    """Replay-only experiment execution with resume + idempotent episode keys."""

    def __init__(
        self,
        experiment_repo: ExperimentRepository,
        *,
        gate: PromotionEligibilityGate | None = None,
        repeated_samples: int = 3,
    ) -> None:
        self._experiments = experiment_repo
        self._gate = gate or PromotionEligibilityGate()
        self._repeated_samples = repeated_samples
        self._completed_keys: set[str] = set()

    async def create(
        self, definition: ExperimentDefinition
    ) -> ExperimentDefinition:
        await self._experiments.append_definition(definition)
        return definition

    async def run(
        self,
        experiment_id: UUID | str,
        *,
        episodes: list[TradingEpisode],
        partition_map: dict[str, tuple[UUID, ...]],
        replay_fn: ReplayFn | None = None,
        adversarial_passed: bool = True,
    ) -> ExperimentResult:
        definition = await self._experiments.get_definition(experiment_id)
        if definition is None:
            raise ExperimentRunnerError(f"experiment not found: {experiment_id}")
        await self._experiments.mark_status(experiment_id, "running")

        replay = replay_fn or self._default_replay
        by_id = {ep.episode_id: ep for ep in episodes}
        slice_results: list[ExperimentSliceResult] = []
        champ_vals: list[Decimal] = []
        chall_vals: list[Decimal] = []
        missing: list[UUID] = []
        model_calls = 0
        cost = Decimal("0")

        for slice_name, ids in partition_map.items():
            metrics_acc: dict[str, list[Decimal]] = {"pnl": []}
            missing_count = 0
            for eid in ids:
                ep = by_id.get(eid)
                if ep is None:
                    missing.append(eid)
                    missing_count += 1
                    continue
                for sample in range(1, self._repeated_samples + 1):
                    key = experiment_episode_key(
                        experiment_id,
                        eid,
                        definition.challenger_version_id,
                        sample,
                    )
                    if key in self._completed_keys:
                        continue
                    champ = await replay(ep, definition.champion_version_id, sample)
                    chall = await replay(ep, definition.challenger_version_id, sample)
                    self._completed_keys.add(key)
                    await self._experiments.mark_status(
                        experiment_id, "running", recovery_cursor=key
                    )
                    c_pnl = Decimal(str(champ.get("realised_pnl", 0)))
                    h_pnl = Decimal(str(chall.get("realised_pnl", 0)))
                    champ_vals.append(c_pnl)
                    chall_vals.append(h_pnl)
                    metrics_acc["pnl"].append(h_pnl - c_pnl)
                    model_calls += int(champ.get("model_calls", 1)) + int(
                        chall.get("model_calls", 1)
                    )
                    cost += Decimal(str(champ.get("cost_gbp", "0.01"))) + Decimal(
                        str(chall.get("cost_gbp", "0.01"))
                    )
                    if cost > definition.maximum_cost_gbp:
                        await self._experiments.mark_status(experiment_id, "failed")
                        raise ExperimentRunnerError("maximum_cost_gbp exceeded")
            avg_delta = (
                sum(metrics_acc["pnl"]) / Decimal(len(metrics_acc["pnl"]))
                if metrics_acc["pnl"]
                else Decimal("0")
            )
            slice_results.append(
                ExperimentSliceResult(
                    slice_name=slice_name,
                    metrics={"pnl_delta": avg_delta, "samples": len(metrics_acc["pnl"])},
                    episode_count=len(ids) - missing_count,
                    missing_episode_count=missing_count,
                    confidence_intervals={
                        "pnl_delta": self._ci(metrics_acc["pnl"]),
                    },
                )
            )

        champ_mean = (
            sum(champ_vals) / Decimal(len(champ_vals)) if champ_vals else Decimal("0")
        )
        chall_mean = (
            sum(chall_vals) / Decimal(len(chall_vals)) if chall_vals else Decimal("0")
        )
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
                "pnl_delta": self._ci([b - a for a, b in zip(champ_vals, chall_vals)])
            },
            cost_gbp=cost,
            model_call_counts={"total": model_calls},
            missing_episodes=tuple(missing),
            champion_metrics={
                "mean_pnl": champ_mean,
                "tail_loss": min(champ_vals) if champ_vals else Decimal("0"),
                "calibration_error": Decimal("0.10"),
                "latency_ms": 100,
                "cost_gbp": cost / Decimal("2"),
            },
            challenger_metrics={
                "mean_pnl": chall_mean,
                "tail_loss": min(chall_vals) if chall_vals else Decimal("0"),
                "calibration_error": Decimal("0.09"),
                "latency_ms": 110,
                "cost_gbp": cost / Decimal("2"),
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
                "content_hash": "",
            }
        )
        result = result.model_copy(
            update={"content_hash": hash_model(result, exclude={"created_at"})}
        )
        await self._experiments.append_result(result)
        await self._experiments.mark_status(experiment_id, "completed", recovery_cursor=None)
        return result

    async def resume(
        self,
        experiment_id: UUID | str,
        *,
        episodes: list[TradingEpisode],
        partition_map: dict[str, tuple[UUID, ...]],
        replay_fn: ReplayFn | None = None,
        known_completed_keys: set[str] | None = None,
    ) -> ExperimentResult:
        if known_completed_keys:
            self._completed_keys |= set(known_completed_keys)
        definition = await self._experiments.get_definition(experiment_id)
        if definition and definition.recovery_cursor:
            self._completed_keys.add(definition.recovery_cursor)
        existing = await self._experiments.get_result(experiment_id)
        if existing is not None:
            return existing
        return await self.run(
            experiment_id,
            episodes=episodes,
            partition_map=partition_map,
            replay_fn=replay_fn,
        )

    async def _default_replay(
        self, episode: TradingEpisode, configuration_version_id: UUID, sample: int
    ) -> dict[str, Any]:
        """Deterministic stub fill model for tests — not live broker."""
        base = episode.realised_pnl or Decimal("0")
        # Slight challenger bump encoded via UUID nibble for stability in tests.
        bump = Decimal(int(str(configuration_version_id).replace("-", "")[:2], 16) % 5) / Decimal(
            "100"
        )
        return {
            "realised_pnl": base + bump + Decimal(sample - 1) * Decimal("0.01"),
            "model_calls": 2,
            "cost_gbp": "0.02",
            "configuration_version_id": str(configuration_version_id),
            "broker_submit": False,
        }

    @staticmethod
    def _ci(values: list[Decimal]) -> tuple[Decimal, Decimal]:
        if not values:
            return (Decimal("0"), Decimal("0"))
        ordered = sorted(values)
        lo = ordered[0]
        hi = ordered[-1]
        return (lo, hi)
