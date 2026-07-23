"""Local durable storage using SQLite via SQLModel."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    return str(uuid.uuid4())


class RunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class RunRecord(SQLModel, table=True):
    __tablename__ = "runs"

    id: str = Field(primary_key=True, default_factory=new_run_id)
    run_id: str = Field(index=True)
    mode: str
    trading_day: date = Field(index=True)
    status: str = RunStatus.CREATED.value
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: Optional[datetime] = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    schema_version: str = "1"


class AgentDecisionRecord(SQLModel, table=True):
    __tablename__ = "agent_decisions"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    agent_name: str
    decision_type: str
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1"


class MarketSnapshotRecord(SQLModel, table=True):
    __tablename__ = "market_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    symbol: str
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    captured_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1"


class TradeCandidateRecord(SQLModel, table=True):
    __tablename__ = "trade_candidates"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    candidate_id: str = Field(index=True)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1"


class RiskDecisionRecord(SQLModel, table=True):
    __tablename__ = "risk_decisions"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    candidate_id: str = Field(index=True)
    approved: bool
    reason_codes: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1"


class OrderRecord(SQLModel, table=True):
    __tablename__ = "orders"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    order_id: str = Field(index=True, unique=True)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1"


class FillRecord(SQLModel, table=True):
    __tablename__ = "fills"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    order_id: str = Field(index=True)
    fill_id: str = Field(index=True, unique=True)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    filled_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1"


class PositionRecord(SQLModel, table=True):
    __tablename__ = "positions"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    position_id: str = Field(index=True, unique=True)
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    is_open: bool = True
    opened_at: datetime = Field(default_factory=utc_now)
    closed_at: Optional[datetime] = None
    schema_version: str = "1"


class UserMessageRecord(SQLModel, table=True):
    __tablename__ = "user_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    role: str
    content: str = Field(sa_column=Column(Text))
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1"


class SystemEventRecord(SQLModel, table=True):
    __tablename__ = "system_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    event_type: str
    source: str
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1"


class TradingDayStateRecord(SQLModel, table=True):
    __tablename__ = "trading_day_state"

    id: Optional[int] = Field(default=None, primary_key=True)
    trading_day: date = Field(index=True, unique=True)
    run_id: str = Field(index=True)
    mode: str
    daily_pnl_usd: float = 0.0
    trades_count: int = 0
    open_positions: int = 0
    kill_switch: bool = False
    playbook_approved: bool = False
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "1"
