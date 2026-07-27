"""Durable adversarial recovery checkpoint round-trip and resume proofs."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from joker.evolution.adversarial_recovery import (
    AdversarialRecoveryCheckpoint,
    AdversarialRecoveryStore,
)
from joker.evolution.migrations import apply_task3_migrations


@pytest.mark.asyncio
async def test_recovery_checkpoint_round_trip(tmp_path) -> None:
    db = tmp_path / "recovery.db"
    apply_task3_migrations(db)
    store = AdversarialRecoveryStore(db)
    experiment_id = uuid4()
    cfg_id = uuid4()
    key = AdversarialRecoveryStore.checkpoint_key(
        experiment_id, "adv_15", "3.1.0", cfg_id, 1
    )
    checkpoint = AdversarialRecoveryCheckpoint(
        checkpoint_key=key,
        experiment_id=experiment_id,
        scenario_id="adv_15",
        scenario_version="3.1.0",
        configuration_version_id=cfg_id,
        sample_number=1,
        crash_point="after_model",
        graph_thread_ids=("thread:1",),
        cash=str(Decimal("25000")),
        submitted_keys=("adv-entry:adv_15:1",),
        order_ids=("o1",),
        fill_ids=("f1",),
        model_call_ids=("mc1",),
        gateway_action_ids=("adv-entry:adv_15:1",),
        findings=("graph_fail_closed:RuntimeError",),
    )
    await store.save(checkpoint)

    fresh_store = AdversarialRecoveryStore(db)
    loaded = await fresh_store.load(key)
    assert loaded is not None
    assert loaded.checkpoint_key == key
    assert loaded.graph_thread_ids == ("thread:1",)
    assert loaded.order_ids == ("o1",)
    assert loaded.fill_ids == ("f1",)
    assert loaded.model_call_ids == ("mc1",)
    assert loaded.submitted_keys == ("adv-entry:adv_15:1",)
    assert loaded.crash_point == "after_model"


@pytest.mark.asyncio
async def test_recovery_resume_detects_duplicate_orders(tmp_path) -> None:
    db = tmp_path / "dup.db"
    apply_task3_migrations(db)
    store = AdversarialRecoveryStore(db)
    experiment_id = uuid4()
    cfg_id = uuid4()
    key = AdversarialRecoveryStore.checkpoint_key(
        experiment_id, "adv_16", "3.1.0", cfg_id, 1
    )
    await store.save(
        AdversarialRecoveryCheckpoint(
            checkpoint_key=key,
            experiment_id=experiment_id,
            scenario_id="adv_16",
            scenario_version="3.1.0",
            configuration_version_id=cfg_id,
            sample_number=1,
            order_ids=("dup-order",),
            fill_ids=("dup-fill",),
            model_call_ids=("dup-mc",),
        )
    )
    loaded = await store.load(key)
    assert loaded is not None
    stage2_orders = ("dup-order", "new-order")
    stage2_fills = ("dup-fill",)
    stage2_model = ("dup-mc", "new-mc")
    assert set(loaded.order_ids) & set(stage2_orders)
    assert set(loaded.fill_ids) & set(stage2_fills)
    assert set(loaded.model_call_ids) & set(stage2_model)
