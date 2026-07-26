"""Owned durable LangGraph checkpointers for Task 3 graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from joker.graph.langgraph_checkpointer import CognitiveCheckpointer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvolutionCheckpointers:
    evaluation: AsyncSqliteSaver
    improvement: AsyncSqliteSaver
    replay: AsyncSqliteSaver
    decision: AsyncSqliteSaver
    orchestrator: AsyncSqliteSaver


@dataclass
class EvolutionCheckpointerOwner:
    """Open once at EvolutionRuntime.prepare(); close before loop teardown."""

    db_path: Path
    _helpers: dict[str, CognitiveCheckpointer] = field(default_factory=dict)
    _savers: EvolutionCheckpointers | None = None
    _closed: bool = False

    def paths(self) -> dict[str, Path]:
        stem = self.db_path.stem
        parent = self.db_path.parent
        return {
            "evaluation": parent / f"{stem}_evolution_eval_ckpt.db",
            "improvement": parent / f"{stem}_evolution_improvement_ckpt.db",
            "replay": parent / f"{stem}_evolution_replay_ckpt.db",
            "decision": parent / f"{stem}_evolution_decision_ckpt.db",
            "orchestrator": parent / f"{stem}_evolution_orchestrator_ckpt.db",
        }

    async def open_all(self) -> EvolutionCheckpointers:
        if self._savers is not None and not self._closed:
            return self._savers
        self._closed = False
        opened: dict[str, AsyncSqliteSaver] = {}
        try:
            for name, path in self.paths().items():
                helper = CognitiveCheckpointer(path)
                saver = await helper.open()
                self._helpers[name] = helper
                opened[name] = saver
            self._savers = EvolutionCheckpointers(
                evaluation=opened["evaluation"],
                improvement=opened["improvement"],
                replay=opened["replay"],
                decision=opened["decision"],
                orchestrator=opened["orchestrator"],
            )
            return self._savers
        except Exception:
            await self.close_all()
            raise

    @property
    def savers(self) -> EvolutionCheckpointers:
        if self._savers is None:
            raise RuntimeError("EvolutionCheckpointerOwner is not open")
        return self._savers

    async def close_all(self) -> None:
        if self._closed and not self._helpers:
            return
        errors: list[str] = []
        for name, helper in list(self._helpers.items()):
            try:
                await helper.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}:{exc}")
                logger.exception("evolution_checkpointer_close_failed", extra={"name": name})
        self._helpers.clear()
        self._savers = None
        self._closed = True
        if errors:
            raise RuntimeError("failed closing evolution checkpointers: " + "; ".join(errors))
