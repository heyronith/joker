"""Durable replay workflow execution state (orders, fills, checkpoints)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite

from joker.evolution.replay_execution import ReplayFill, ReplayOrder, ReplayPosition


_REPLAY_DDL = """
CREATE TABLE IF NOT EXISTS replay_workflows (
    replay_key TEXT PRIMARY KEY NOT NULL,
    experiment_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    configuration_version_id TEXT NOT NULL,
    sample_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    frame_index INTEGER NOT NULL DEFAULT 0,
    entry_cycle_id TEXT,
    entry_order_id TEXT,
    entry_decision_completed INTEGER NOT NULL DEFAULT 0,
    cash TEXT NOT NULL,
    realised_pnl TEXT NOT NULL DEFAULT '0',
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS replay_orders (
    replay_key TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (replay_key, client_order_id)
);

CREATE TABLE IF NOT EXISTS replay_fills (
    fill_id TEXT PRIMARY KEY NOT NULL,
    replay_key TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS replay_positions (
    replay_key TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (replay_key, contract_id)
);

CREATE TABLE IF NOT EXISTS replay_execution_checkpoints (
    replay_key TEXT PRIMARY KEY NOT NULL,
    frame_index INTEGER NOT NULL,
    submitted_keys_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def replay_key(
    experiment_id: UUID | str,
    episode_id: UUID | str,
    configuration_version_id: UUID | str,
    sample_number: int,
) -> str:
    return f"{experiment_id}:{episode_id}:{configuration_version_id}:{sample_number}"


class ReplayExecutionStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_REPLAY_DDL)
            await db.commit()

    async def save_checkpoint(
        self,
        *,
        key: str,
        experiment_id: str,
        episode_id: str,
        configuration_version_id: str,
        sample_number: int,
        status: str,
        frame_index: int,
        cash: Decimal,
        realised_pnl: Decimal,
        orders: dict[str, ReplayOrder],
        fills: list[ReplayFill],
        positions: dict[str, ReplayPosition],
        submitted_keys: set[str],
        entry_cycle_id: str | None = None,
        entry_order_id: str | None = None,
        entry_decision_completed: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        await self.initialize()
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "entry_cycle_id": entry_cycle_id,
            "entry_order_id": entry_order_id,
            "entry_decision_completed": entry_decision_completed,
            **(extra or {}),
        }
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO replay_workflows (
                    replay_key, experiment_id, episode_id, configuration_version_id,
                    sample_number, status, frame_index, entry_cycle_id, entry_order_id,
                    entry_decision_completed, cash, realised_pnl, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    experiment_id,
                    episode_id,
                    configuration_version_id,
                    sample_number,
                    status,
                    frame_index,
                    entry_cycle_id,
                    entry_order_id,
                    1 if entry_decision_completed else 0,
                    str(cash),
                    str(realised_pnl),
                    json.dumps(payload, default=str),
                    now,
                ),
            )
            await db.execute("DELETE FROM replay_orders WHERE replay_key = ?", (key,))
            for oid, order in orders.items():
                await db.execute(
                    """
                    INSERT INTO replay_orders (replay_key, client_order_id, payload_json)
                    VALUES (?, ?, ?)
                    """,
                    (key, oid, json.dumps(order.__dict__, default=str)),
                )
            await db.execute("DELETE FROM replay_fills WHERE replay_key = ?", (key,))
            for fill in fills:
                await db.execute(
                    """
                    INSERT INTO replay_fills (fill_id, replay_key, client_order_id, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        fill.fill_id,
                        key,
                        fill.client_order_id,
                        json.dumps(fill.__dict__, default=str),
                    ),
                )
            await db.execute("DELETE FROM replay_positions WHERE replay_key = ?", (key,))
            for cid, pos in positions.items():
                await db.execute(
                    """
                    INSERT INTO replay_positions (replay_key, contract_id, payload_json)
                    VALUES (?, ?, ?)
                    """,
                    (key, cid, json.dumps(pos.__dict__, default=str)),
                )
            await db.execute(
                """
                INSERT OR REPLACE INTO replay_execution_checkpoints (
                    replay_key, frame_index, submitted_keys_json, payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    key,
                    frame_index,
                    json.dumps(sorted(submitted_keys)),
                    json.dumps(payload, default=str),
                    now,
                ),
            )
            await db.commit()

    async def load_checkpoint(self, key: str) -> dict[str, Any] | None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM replay_workflows WHERE replay_key = ?", (key,)
            )
            row = await cur.fetchone()
            if row is None:
                return None
            orders_cur = await db.execute(
                "SELECT payload_json FROM replay_orders WHERE replay_key = ?", (key,)
            )
            fills_cur = await db.execute(
                "SELECT payload_json FROM replay_fills WHERE replay_key = ?", (key,)
            )
            pos_cur = await db.execute(
                "SELECT payload_json FROM replay_positions WHERE replay_key = ?", (key,)
            )
            ck_cur = await db.execute(
                "SELECT * FROM replay_execution_checkpoints WHERE replay_key = ?",
                (key,),
            )
            ck = await ck_cur.fetchone()
            order_rows = await orders_cur.fetchall()
            fill_rows = await fills_cur.fetchall()
            pos_rows = await pos_cur.fetchall()

        orders: dict[str, ReplayOrder] = {}
        for r in order_rows:
            data = json.loads(r[0])
            orders[data["client_order_id"]] = ReplayOrder(**data)
        fills = [ReplayFill(**json.loads(r[0])) for r in fill_rows]
        positions: dict[str, ReplayPosition] = {}
        for r in pos_rows:
            data = json.loads(r[0])
            cfg = data.get("configuration_version_id")
            positions[data["contract_id"]] = ReplayPosition(
                contract_id=str(data["contract_id"]),
                quantity=Decimal(str(data["quantity"])),
                avg_price=Decimal(str(data["avg_price"])),
                realised_pnl=Decimal(str(data.get("realised_pnl", "0"))),
                configuration_version_id=UUID(str(cfg)) if cfg else None,
                position_lifecycle_id=data.get("position_lifecycle_id"),
            )
        submitted = set(json.loads(ck["submitted_keys_json"])) if ck else set()
        payload = json.loads(row["payload_json"])
        return {
            "status": row["status"],
            "frame_index": int(row["frame_index"]),
            "cash": Decimal(row["cash"]),
            "realised_pnl": Decimal(row["realised_pnl"]),
            "entry_cycle_id": row["entry_cycle_id"],
            "entry_order_id": row["entry_order_id"],
            "entry_decision_completed": bool(row["entry_decision_completed"]),
            "orders": orders,
            "fills": fills,
            "positions": positions,
            "submitted_keys": submitted,
            "payload": payload,
        }
