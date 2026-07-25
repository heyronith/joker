"""Thin persistence facades over market/ledger repositories."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from joker.ledger.schemas import LedgerEvent
from joker.ledger.store import SqliteLedgerStore
from joker.market.option_surface import OptionSurfaceRepository, OptionSurfaceSnapshot
from joker.market.snapshots import MarketSnapshot, SnapshotRepository


class SnapshotRepositoryFacade:
    """Facade wrapping ``joker.market.snapshots.SnapshotRepository``."""

    def __init__(self, db_path: str | Path | SnapshotRepository) -> None:
        if isinstance(db_path, SnapshotRepository):
            self._inner = db_path
        else:
            self._inner = SnapshotRepository(db_path)

    @property
    def inner(self) -> SnapshotRepository:
        return self._inner

    async def initialize(self) -> None:
        await self._inner.initialize()

    async def save(self, snapshot: MarketSnapshot) -> None:
        await self._inner.save(snapshot)

    async def get_by_id(self, snapshot_id: UUID | str) -> MarketSnapshot | None:
        return await self._inner.get_by_id(snapshot_id)

    async def list_by_trading_date(self, trading_date: date) -> list[MarketSnapshot]:
        return await self._inner.list_by_trading_date(trading_date)


class OptionSurfaceRepositoryFacade:
    """Facade wrapping ``joker.market.option_surface.OptionSurfaceRepository``."""

    def __init__(self, db_path: str | Path | OptionSurfaceRepository) -> None:
        if isinstance(db_path, OptionSurfaceRepository):
            self._inner = db_path
        else:
            self._inner = OptionSurfaceRepository(db_path)

    @property
    def inner(self) -> OptionSurfaceRepository:
        return self._inner

    async def initialize(self) -> None:
        await self._inner.initialize()

    async def save(self, surface: OptionSurfaceSnapshot) -> None:
        await self._inner.save(surface)

    async def get_by_id(self, surface_id: UUID | str) -> OptionSurfaceSnapshot | None:
        return await self._inner.get_by_id(surface_id)


class LedgerRepositoryFacade:
    """Facade wrapping ``joker.ledger.store.SqliteLedgerStore``."""

    def __init__(self, db_path: str | Path | SqliteLedgerStore) -> None:
        if isinstance(db_path, SqliteLedgerStore):
            self._inner = db_path
        else:
            self._inner = SqliteLedgerStore(db_path)

    @property
    def inner(self) -> SqliteLedgerStore:
        return self._inner

    async def initialize(self) -> None:
        await self._inner.initialize()

    async def close(self) -> None:
        await self._inner.close()

    async def append(self, event: LedgerEvent) -> bool:
        return await self._inner.append(event)

    async def get_by_session(self, session_id: str) -> list[LedgerEvent]:
        return await self._inner.get_by_session(session_id)

    async def get_by_order(self, client_order_id: str) -> list[LedgerEvent]:
        return await self._inner.get_by_order(client_order_id)

    async def get_by_contract(self, contract_id: str) -> list[LedgerEvent]:
        return await self._inner.get_by_contract(contract_id)

    async def get_by_position(self, position_id: str) -> list[LedgerEvent]:
        return await self._inner.get_by_position(position_id)


def open_task1_repositories(db_path: str | Path) -> dict[str, Any]:
    """Construct Task 1 repository facades sharing one database path."""
    path = Path(db_path)
    return {
        "snapshots": SnapshotRepositoryFacade(path),
        "surfaces": OptionSurfaceRepositoryFacade(path),
        "ledger": LedgerRepositoryFacade(path),
    }
