"""LangGraph AsyncSqliteSaver lifecycle for cognitive decision/position graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from joker.persistence.aiosqlite_lifecycle import close_aiosqlite_connection

logger = logging.getLogger(__name__)


@dataclass
class CognitiveCheckpointer:
    """Owns a persistent aiosqlite connection for LangGraph checkpoints.

    Must be opened before graph compile/ainvoke and closed during runtime shutdown
    *before* the event loop is destroyed to avoid aiosqlite worker races.
    """

    db_path: Path
    _conn: aiosqlite.Connection | None = None
    _saver: AsyncSqliteSaver | None = None

    @property
    def saver(self) -> AsyncSqliteSaver:
        if self._saver is None:
            raise RuntimeError("CognitiveCheckpointer is not open")
        return self._saver

    async def open(self) -> AsyncSqliteSaver:
        if self._saver is not None:
            return self._saver
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._saver = AsyncSqliteSaver(self._conn)
        await self._saver.setup()
        logger.info(
            "cognitive_checkpointer_opened",
            extra={"db_path": str(self.db_path)},
        )
        return self._saver

    async def close(self) -> None:
        saver = self._saver
        conn = self._conn
        self._saver = None
        self._conn = None
        if conn is not None:
            await close_aiosqlite_connection(conn)
        logger.info(
            "cognitive_checkpointer_closed",
            extra={"db_path": str(self.db_path), "had_saver": saver is not None},
        )


def cognitive_thread_id(*, session_id: str, graph_kind: str, cycle_id: str) -> str:
    """Stable LangGraph thread id derived from session, graph kind, and cycle."""
    return f"{session_id}:{graph_kind}:{cycle_id}"


def ainvoke_config(*, session_id: str, graph_kind: str, cycle_id: str) -> dict:
    """Build the checkpoint configuration passed to every ainvoke."""
    return {
        "configurable": {
            "thread_id": cognitive_thread_id(
                session_id=session_id,
                graph_kind=graph_kind,
                cycle_id=cycle_id,
            )
        }
    }
