"""Immutable evaluation dataset construction with leakage controls."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Iterable
from uuid import UUID, uuid4

from joker.evolution.hashing import content_hash, hash_model, stable_json_dumps
from joker.evolution.repositories import DatasetRepository
from joker.evolution.schemas import EvaluationDataset, TradingEpisode


class DatasetBuilderError(ValueError):
    """Raised when dataset construction would leak or corrupt partitions."""


class DatasetBuilder:
    def __init__(self, dataset_repo: DatasetRepository) -> None:
        self._datasets = dataset_repo

    def build(
        self,
        episodes: Iterable[TradingEpisode],
        *,
        random_seed: int = 42,
        cutoff: datetime | None = None,
        proposal_time: datetime | None = None,
        allow_incomplete: bool = False,
        source_db_hashes: dict[str, str] | None = None,
        adversarial_ids: tuple[UUID, ...] = (),
        minimum_holdout: int = 1,
    ) -> EvaluationDataset:
        rng = random.Random(random_seed)
        exclusion: list[str] = []
        leakage: list[str] = []
        eligible: list[TradingEpisode] = []

        for ep in episodes:
            if not ep.completed and not allow_incomplete:
                exclusion.append(f"{ep.episode_id}:incomplete")
                continue
            if cutoff is not None and ep.created_at > cutoff:
                exclusion.append(f"{ep.episode_id}:after_cutoff")
                continue
            if proposal_time is not None and ep.created_at > proposal_time:
                leakage.append(
                    f"reject_post_proposal_episode:{ep.episode_id}"
                )
                continue
            eligible.append(ep)

        if leakage:
            raise DatasetBuilderError(
                "temporal leakage detected: " + "; ".join(leakage)
            )

        ids = [ep.episode_id for ep in eligible]
        rng.shuffle(ids)
        n = len(ids)
        if n == 0:
            raise DatasetBuilderError("no eligible episodes for dataset")

        # Partition: development 50%, validation 20%, holdout 20%, shadow 10%
        n_dev = max(1, int(n * 0.5))
        n_val = max(0, int(n * 0.2))
        n_hold = max(minimum_holdout, int(n * 0.2))
        if n_dev + n_val + n_hold > n:
            n_hold = max(1, n - n_dev - n_val)
        n_shadow = max(0, n - n_dev - n_val - n_hold)

        cursor = 0
        development = tuple(ids[cursor : cursor + n_dev])
        cursor += n_dev
        validation = tuple(ids[cursor : cursor + n_val])
        cursor += n_val
        holdout = tuple(ids[cursor : cursor + n_hold])
        cursor += n_hold
        shadow = tuple(ids[cursor : cursor + n_shadow])

        adv = tuple(adversarial_ids)
        # Ensure adversarial IDs do not overlap comparison partitions.
        comparison = set(development) | set(validation) | set(holdout) | set(shadow)
        overlap = comparison & set(adv)
        if overlap:
            raise DatasetBuilderError(f"adversarial overlap with partitions: {overlap}")

        partition_map = {
            "development": development,
            "validation": validation,
            "holdout": holdout,
            "adversarial": adv,
            "shadow": shadow,
        }

        regime_dist: dict[str, int] = {}
        outcome_dist: dict[str, int] = {}
        config_dist: dict[str, int] = {}
        for ep in eligible:
            for tag in ep.market_regime_tags or ("untagged",):
                regime_dist[tag] = regime_dist.get(tag, 0) + 1
            outcome_dist[ep.action_class] = outcome_dist.get(ep.action_class, 0) + 1
            ck = str(ep.configuration_version_id)
            config_dist[ck] = config_dist.get(ck, 0) + 1

        dataset = EvaluationDataset(
            dataset_id=uuid4(),
            construction_timestamp=datetime.now(timezone.utc),
            episode_ids=tuple(ids),
            partition_map=partition_map,
            regime_distribution=regime_dist,
            outcome_distribution=outcome_dist,
            configuration_distribution=config_dist,
            source_db_hashes=source_db_hashes or {},
            random_seed=random_seed,
            exclusion_reasons=tuple(exclusion),
            leakage_audit=tuple(leakage),
            content_hash="",
        )
        dataset = dataset.model_copy(
            update={
                "content_hash": content_hash(
                    stable_json_dumps(dataset.model_dump(mode="json", exclude={"created_at", "dataset_id"}))
                )
            }
        )
        return dataset

    async def build_and_persist(self, *args, **kwargs) -> EvaluationDataset:
        dataset = self.build(*args, **kwargs)
        await self._datasets.append(dataset)
        return dataset
