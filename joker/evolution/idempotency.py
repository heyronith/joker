"""Stable idempotency keys for Task 3 operations."""

from __future__ import annotations

from uuid import UUID

from joker.evolution.hashing import content_hash


def episode_idempotency_key(
    session_id: str,
    position_lifecycle_key: str,
    terminal_event_id: str,
) -> str:
    return content_hash("episode", session_id, position_lifecycle_key, terminal_event_id)


def evaluation_idempotency_key(
    episode_id: UUID | str,
    evaluator_version: str,
    configuration_version_id: UUID | str,
) -> str:
    return content_hash(
        "evaluation",
        str(episode_id),
        evaluator_version,
        str(configuration_version_id),
    )


def proposal_idempotency_key(
    evaluation_window_hash: str,
    parent_champion_id: UUID | str,
    proposal_content_hash: str,
) -> str:
    return content_hash(
        "proposal",
        evaluation_window_hash,
        str(parent_champion_id),
        proposal_content_hash,
    )


def experiment_episode_key(
    experiment_id: UUID | str,
    episode_id: UUID | str,
    configuration_version_id: UUID | str,
    sample_number: int,
) -> str:
    return content_hash(
        "experiment_episode",
        str(experiment_id),
        str(episode_id),
        str(configuration_version_id),
        str(sample_number),
    )


def promotion_idempotency_key(
    experiment_id: UUID | str,
    challenger_version_id: UUID | str,
    champion_version_id: UUID | str,
) -> str:
    return content_hash(
        "promotion",
        str(experiment_id),
        str(challenger_version_id),
        str(champion_version_id),
    )


def rollback_idempotency_key(
    trigger_id: str,
    rolled_back_version_id: UUID | str,
    restored_version_id: UUID | str,
) -> str:
    return content_hash(
        "rollback",
        trigger_id,
        str(rolled_back_version_id),
        str(restored_version_id),
    )
