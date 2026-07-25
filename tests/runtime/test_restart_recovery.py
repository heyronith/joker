
"""Checkpoint save/restore smoke."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from joker.graph.checkpoints import SqliteCheckpointStore
from joker.graph.state import JokerGraphState


def test_checkpoint_save_restore(tmp_path) -> None:
    async def _run() -> None:
        store = SqliteCheckpointStore(tmp_path / "ckpt.db")
        await store.initialize()
        state: JokerGraphState = {
            "session_id": "s1",
            "run_id": "r1",
            "exchange_time": datetime.now(timezone.utc),
            "pending_event_ids": [],
            "errors": [],
        }
        cid = await store.save(state, session_id="s1")
        latest = await store.load_latest("s1")
        assert latest is not None
        assert latest.checkpoint_id == cid
        assert latest.state["session_id"] == "s1"
        loaded = await store.load(cid)
        assert loaded is not None

    asyncio.run(_run())
