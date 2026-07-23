"""SQLite database initialization and repository access."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, Type, TypeVar

from sqlmodel import Session, SQLModel, create_engine, select

from joker.compliance.opra_sanitizer import sanitize_for_persistence

from joker.storage.models import (
    AgentDecisionRecord,
    FillRecord,
    MarketSnapshotRecord,
    OrderRecord,
    PositionRecord,
    RiskDecisionRecord,
    RunRecord,
    RunStatus,
    SystemEventRecord,
    TradeCandidateRecord,
    TradingDayStateRecord,
    UserMessageRecord,
    new_run_id,
    utc_now,
)

T = TypeVar("T", bound=SQLModel)


class StorageError(Exception):
    """Raised when storage operations fail safely."""


class Database:
    """SQLite-backed local storage."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{self.db_path}"
            self._engine = create_engine(url, connect_args={"check_same_thread": False})
        return self._engine

    def initialize(self) -> None:
        try:
            SQLModel.metadata.create_all(self.engine)
        except Exception as exc:
            raise StorageError(f"Failed to initialize database at {self.db_path}") from exc

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        with Session(self.engine) as session:
            yield session

    def create_run(
        self,
        mode: str,
        trading_day,
        config_snapshot: dict | None = None,
    ) -> RunRecord:
        run_id = new_run_id()
        record = RunRecord(
            run_id=run_id,
            mode=mode,
            trading_day=trading_day,
            status=RunStatus.RUNNING.value,
            config_snapshot=config_snapshot or {},
        )
        with self.session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.session() as session:
            return session.exec(select(RunRecord).where(RunRecord.run_id == run_id)).first()

    def save(self, record: SQLModel) -> SQLModel:
        record = self._sanitize_record(record)
        with self.session() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list_by_run(self, model: Type[T], run_id: str) -> list[T]:
        with self.session() as session:
            statement = select(model).where(model.run_id == run_id)  # type: ignore[attr-defined]
            return list(session.exec(statement).all())

    def _sanitize_record(self, record: SQLModel) -> SQLModel:
        for field_name in ("payload", "config_snapshot"):
            if not hasattr(record, field_name):
                continue
            value = getattr(record, field_name)
            if isinstance(value, dict):
                setattr(record, field_name, sanitize_for_persistence(value))
        return record

    def get_trading_day_state(self, trading_day) -> TradingDayStateRecord | None:
        with self.session() as session:
            return session.exec(
                select(TradingDayStateRecord).where(
                    TradingDayStateRecord.trading_day == trading_day
                )
            ).first()

    def upsert_trading_day_state(self, state: TradingDayStateRecord) -> TradingDayStateRecord:
        state.updated_at = utc_now()
        with self.session() as session:
            existing = session.exec(
                select(TradingDayStateRecord).where(
                    TradingDayStateRecord.trading_day == state.trading_day
                )
            ).first()
            if existing:
                for field in (
                    "run_id",
                    "mode",
                    "daily_pnl_usd",
                    "trades_count",
                    "open_positions",
                    "kill_switch",
                    "playbook_approved",
                    "payload",
                    "updated_at",
                ):
                    setattr(existing, field, getattr(state, field))
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing
            session.add(state)
            session.commit()
            session.refresh(state)
            return state

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


def ensure_database(db_path: Path) -> Database:
    db = Database(db_path)
    db.initialize()
    return db
