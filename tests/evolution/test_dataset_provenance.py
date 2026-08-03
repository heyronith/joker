"""Configuration dataset provenance for leakage-safe historical EV."""

from __future__ import annotations

from uuid import uuid4

import pytest

from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.improvement import ImprovementProposalService
from joker.evolution.repositories import build_evolution_repositories
from joker.evolution.schemas import (
    PromptPatch,
    resolve_dataset_provenance_status,
)
from joker.objectives.historical_schemas import HistoricalOutcomeQuery
from tests.objectives.historical_fixtures import (
    make_repo_backed_hist_service,
    persist_dataset_with_episodes,
    persist_positive_history,
)
from datetime import datetime, timedelta, timezone
from decimal import Decimal


def _patch() -> PromptPatch:
    return PromptPatch(
        role="falsifier",
        parent_prompt_version_id=uuid4(),
        replacement_template="Require evidence IDs.",
        change_rationale="grounding",
    )


@pytest.mark.asyncio
async def test_challenger_persists_training_dataset_ids(tmp_path) -> None:
    db = tmp_path / "prov.db"
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    repos = build_evolution_repositories(db)
    await repos["configurations"].initialize()
    svc = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    train_id = uuid4()
    _, challenger = await svc.propose(
        parent_champion=champ,
        weakness="w",
        hypothesis="h",
        patch=_patch(),
        training_dataset_ids=(train_id,),
    )
    assert challenger.training_dataset_ids == (train_id,)
    assert challenger.dataset_provenance_status == "resolved"
    stored = await repos["configurations"].get_by_id(challenger.configuration_version_id)
    assert stored is not None
    assert stored.training_dataset_ids == (train_id,)
    await registry.close()


@pytest.mark.asyncio
async def test_challenger_persists_evaluation_dataset_ids(tmp_path) -> None:
    db = tmp_path / "prov-eval.db"
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    repos = build_evolution_repositories(db)
    await repos["configurations"].initialize()
    svc = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    train_id, eval_id = uuid4(), uuid4()
    _, challenger = await svc.propose(
        parent_champion=champ,
        weakness="w",
        hypothesis="h",
        patch=_patch(),
        training_dataset_ids=(train_id,),
        evaluation_dataset_ids=(eval_id,),
    )
    assert challenger.evaluation_dataset_ids == (eval_id,)
    assert challenger.dataset_provenance_status == "resolved"
    await registry.close()


@pytest.mark.asyncio
async def test_promoted_champion_preserves_dataset_provenance(tmp_path) -> None:
    db = tmp_path / "prov-promo.db"
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    repos = build_evolution_repositories(db)
    await repos["configurations"].initialize()
    svc = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    train_id, chall_id, eval_id = uuid4(), uuid4(), uuid4()
    _, challenger = await svc.propose(
        parent_champion=champ,
        weakness="w",
        hypothesis="h",
        patch=_patch(),
        training_dataset_ids=(train_id,),
        challenger_dataset_ids=(chall_id,),
        evaluation_dataset_ids=(eval_id,),
    )
    await registry.promote(
        challenger=challenger,
        expected_champion_id=champ.configuration_version_id,
        reason="test",
    )
    current = await registry.get_current_champion()
    assert current is not None
    assert current.training_dataset_ids == (train_id,)
    assert current.challenger_dataset_ids == (chall_id,)
    assert current.evaluation_dataset_ids == (eval_id,)
    assert current.dataset_provenance_status == "resolved"
    await registry.close()


@pytest.mark.asyncio
async def test_bootstrap_configuration_can_be_not_applicable(tmp_path) -> None:
    registry = ChampionRegistry(tmp_path / "boot.db")
    champ = await registry.bootstrap_champion()
    assert champ.created_by == "bootstrap"
    assert champ.dataset_provenance_status == "not_applicable"
    assert champ.training_dataset_ids == ()
    assert (
        resolve_dataset_provenance_status(created_by="bootstrap") == "not_applicable"
    )
    await registry.close()


@pytest.mark.asyncio
async def test_non_bootstrap_empty_dataset_provenance_is_unknown(tmp_path) -> None:
    db = tmp_path / "unknown.db"
    registry = ChampionRegistry(db)
    champ = await registry.bootstrap_champion()
    repos = build_evolution_repositories(db)
    await repos["configurations"].initialize()
    svc = ImprovementProposalService(
        repos["proposals"], repos["configurations"], registry.policy_store
    )
    _, challenger = await svc.propose(
        parent_champion=champ,
        weakness="w",
        hypothesis="h",
        patch=_patch(),
        training_dataset_ids=(),
        challenger_dataset_ids=(),
        evaluation_dataset_ids=(),
    )
    assert challenger.dataset_provenance_status == "unknown"
    assert resolve_dataset_provenance_status(created_by="agent") == "unknown"
    await registry.close()


@pytest.mark.asyncio
async def test_unknown_active_configuration_dataset_provenance_blocks_ev(
    tmp_path,
) -> None:
    svc, _, ep_repo, ev_repo, _ = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime.now(timezone.utc)
    await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=8
    )
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            as_of_timestamp=as_of,
            configuration_version_id=uuid4(),
            configuration_dataset_provenance_resolved=False,
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
        )
    )
    assert report.safe is False
    assert summary.valid_for_ev is False
    assert any("unknown_configuration_dataset_provenance" in n for n in report.notes)


@pytest.mark.asyncio
async def test_active_training_episode_is_excluded(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, ds_repo = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime.now(timezone.utc)
    rows = await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=8
    )
    overlap_ids = tuple(r[0].episode_id for r in rows[:3])
    ds = await persist_dataset_with_episodes(
        ds_repo,
        episode_ids=overlap_ids,
        partition="train",
        time_end=as_of - timedelta(days=1),
    )
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
            configuration_version_id=uuid4(),
            blocked_training_dataset_ids=(ds.dataset_id,),
            configuration_dataset_provenance_resolved=True,
        )
    )
    assert len(report.excluded_dataset_overlap) >= 3
    assert summary.sample_count == 5


@pytest.mark.asyncio
async def test_unrelated_training_dataset_remains_eligible(tmp_path) -> None:
    svc, _, ep_repo, ev_repo, ds_repo = await make_repo_backed_hist_service(
        tmp_path, minimum_samples_for_ev=5
    )
    as_of = datetime.now(timezone.utc)
    rows = await persist_positive_history(
        episode_repo=ep_repo, evaluation_repo=ev_repo, as_of=as_of, n=8
    )
    overlap_ids = tuple(r[0].episode_id for r in rows[:3])
    await persist_dataset_with_episodes(
        ds_repo,
        episode_ids=overlap_ids,
        partition="train",
        time_end=as_of - timedelta(days=1),
    )
    summary, report, _ = await svc.query_comparable_outcomes(
        HistoricalOutcomeQuery(
            objective_id=uuid4(),
            strategy_id=uuid4(),
            snapshot_id=uuid4(),
            strategy_family="breakout_continuation",
            direction="bullish",
            maximum_samples=50,
            minimum_similarity=Decimal("0.10"),
            as_of_timestamp=as_of,
            configuration_version_id=uuid4(),
            blocked_training_dataset_ids=(),
            configuration_dataset_provenance_resolved=True,
        )
    )
    assert summary.sample_count == 8
    assert report.excluded_dataset_overlap == ()
