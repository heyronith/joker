"""Full option-surface snapshots and append-only persistence."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import UUID, uuid4

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field, field_validator

from joker.market.exceptions import MarketDataError, SnapshotError

SCHEMA_VERSION = "1"


def _reject_naive(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("Naive datetimes are not allowed on option surfaces")
    return ts


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _valid_bid_ask(bid: Decimal | None, ask: Decimal | None) -> bool:
    return (
        bid is not None
        and ask is not None
        and bid >= 0
        and ask >= 0
        and ask >= bid
    )


def compute_mid(bid: Decimal | None, ask: Decimal | None) -> Decimal | None:
    """Mid only when bid/ask are both valid and non-crossed."""
    if not _valid_bid_ask(bid, ask):
        return None
    assert bid is not None and ask is not None
    return (bid + ask) / Decimal("2")


def compute_relative_spread(bid: Decimal | None, ask: Decimal | None) -> Decimal | None:
    """(ask - bid) / mid when mid > 0; None otherwise."""
    mid = compute_mid(bid, ask)
    if mid is None or mid <= 0:
        return None
    assert bid is not None and ask is not None
    return (ask - bid) / mid


def compute_liquidity_score(
    *,
    bid_size: int | None,
    ask_size: int | None,
    volume: int | None,
    relative_spread: Decimal | None,
) -> float:
    """
    Transparent liquidity score in [0, 1].

    Components (equal weight when present):
    - size_score from min(bid_size, ask_size)
    - volume_score from session volume
    - spread_score inverse of relative spread
    Missing inputs reduce the average rather than inventing data.
    """
    scores: list[float] = []

    sizes = [s for s in (bid_size, ask_size) if s is not None and s >= 0]
    if sizes:
        top = min(sizes)
        # 100 contracts at the touch ≈ full size score.
        scores.append(min(1.0, top / 100.0))

    if volume is not None and volume >= 0:
        scores.append(min(1.0, volume / 1000.0))

    if relative_spread is not None and relative_spread >= 0:
        # 0% spread → 1.0; 10%+ spread → ~0
        spread_f = float(relative_spread)
        scores.append(max(0.0, 1.0 - min(spread_f / 0.10, 1.0)))

    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 6)


class OptionContractSnapshot(BaseModel):
    """Immutable single-contract surface row."""

    model_config = ConfigDict(frozen=True)

    contract_id: str
    symbol: str
    expiry: date
    strike: Decimal
    option_type: Literal["call", "put"]

    bid: Decimal | None = None
    ask: Decimal | None = None
    mid: Decimal | None = None
    last: Decimal | None = None

    bid_size: int | None = None
    ask_size: int | None = None
    volume: int | None = None
    open_interest: int | None = None

    implied_volatility: Decimal | None = None
    delta: Decimal | None = None
    gamma: Decimal | None = None
    theta: Decimal | None = None
    vega: Decimal | None = None

    quote_timestamp: datetime
    quote_age_ms: int
    relative_spread: Decimal | None = None
    liquidity_score: float = 0.0
    quality_flags: tuple[str, ...] = ()

    @field_validator("quote_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _reject_naive(value)


class OptionSurfaceSnapshot(BaseModel):
    """Full option surface at a point in exchange time."""

    model_config = ConfigDict(frozen=True)

    surface_id: UUID = Field(default_factory=uuid4)
    schema_version: str = SCHEMA_VERSION
    exchange_time: datetime
    trading_date: date
    underlying_symbol: str
    underlying_price: Decimal | None = None
    contracts: tuple[OptionContractSnapshot, ...] = ()
    source: str = "unknown"

    @field_validator("exchange_time")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _reject_naive(value)


class OptionSurfaceBuilder:
    """Build OptionSurfaceSnapshot from provider row dictionaries."""

    @classmethod
    def from_provider_rows(
        cls,
        *,
        rows: list[Mapping[str, Any]],
        exchange_time: datetime,
        trading_date: date,
        underlying_symbol: str,
        underlying_price: Decimal | None = None,
        source: str = "provider",
        surface_id: UUID | None = None,
    ) -> OptionSurfaceSnapshot:
        """
        Convert provider rows into a sorted, quality-flagged surface.

        Never fabricates Greeks. Mid/relative_spread only when bid/ask valid.
        """
        exchange_time = _reject_naive(exchange_time)
        contracts: list[OptionContractSnapshot] = []

        for row in rows:
            built = cls._row_to_contract(row, exchange_time=exchange_time)
            if built is not None:
                contracts.append(built)

        contracts.sort(
            key=lambda c: (
                c.expiry,
                0 if c.option_type == "call" else 1,
                c.strike,
                c.contract_id,
            )
        )

        return OptionSurfaceSnapshot(
            surface_id=surface_id or uuid4(),
            exchange_time=exchange_time,
            trading_date=trading_date,
            underlying_symbol=underlying_symbol,
            underlying_price=underlying_price,
            contracts=tuple(contracts),
            source=source,
        )

    @classmethod
    def _row_to_contract(
        cls,
        row: Mapping[str, Any],
        *,
        exchange_time: datetime,
    ) -> OptionContractSnapshot | None:
        contract_id = str(
            row.get("contract_id")
            or row.get("symbol")
            or row.get("ticker")
            or ""
        ).strip()
        symbol = str(row.get("symbol") or row.get("ticker") or contract_id).strip()
        if not contract_id or not symbol:
            return None

        expiry_raw = row.get("expiry") or row.get("expiration") or row.get("expire_date")
        if expiry_raw is None:
            return None
        if isinstance(expiry_raw, date) and not isinstance(expiry_raw, datetime):
            expiry = expiry_raw
        elif isinstance(expiry_raw, datetime):
            expiry = expiry_raw.date()
        else:
            expiry = date.fromisoformat(str(expiry_raw)[:10])

        strike = _to_decimal(row.get("strike"))
        if strike is None:
            return None

        opt_raw = str(row.get("option_type") or row.get("type") or row.get("direction") or "").lower()
        if opt_raw in {"c", "call"}:
            option_type: Literal["call", "put"] = "call"
        elif opt_raw in {"p", "put"}:
            option_type = "put"
        else:
            return None

        bid = _to_decimal(row.get("bid"))
        ask = _to_decimal(row.get("ask"))
        last = _to_decimal(row.get("last") or row.get("latestPrice"))
        bid_size = _to_int(row.get("bid_size") or row.get("bidSize"))
        ask_size = _to_int(row.get("ask_size") or row.get("askSize"))
        volume = _to_int(row.get("volume"))
        open_interest = _to_int(row.get("open_interest") or row.get("openInterest"))

        # Greeks — pass through only when present; never fabricate.
        iv = _to_decimal(row.get("implied_volatility") or row.get("impliedVolatility") or row.get("iv"))
        delta = _to_decimal(row.get("delta"))
        gamma = _to_decimal(row.get("gamma"))
        theta = _to_decimal(row.get("theta"))
        vega = _to_decimal(row.get("vega"))

        quote_ts_raw = (
            row.get("quote_timestamp")
            or row.get("quote_time")
            or row.get("timestamp")
            or row.get("quoteTime")
        )
        if isinstance(quote_ts_raw, datetime):
            quote_ts = _reject_naive(quote_ts_raw)
        elif quote_ts_raw is None:
            quote_ts = exchange_time
        else:
            text = str(quote_ts_raw).replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError as exc:
                raise MarketDataError(f"Invalid quote timestamp for {contract_id}") from exc
            if parsed.tzinfo is None:
                raise MarketDataError(
                    f"Naive quote timestamp rejected for {contract_id}"
                )
            quote_ts = parsed

        quote_age_ms = max(
            0,
            int((exchange_time - quote_ts.astimezone(exchange_time.tzinfo)).total_seconds() * 1000),
        )

        flags: list[str] = []
        mid = compute_mid(bid, ask)
        relative_spread = compute_relative_spread(bid, ask)

        if bid is None:
            flags.append("missing_bid")
        if ask is None:
            flags.append("missing_ask")
        if bid is not None and ask is not None and ask < bid:
            flags.append("crossed_market")
        if bid is not None and ask is not None and ask == bid:
            flags.append("locked_market")
        if last is None:
            flags.append("missing_last")
        if volume is None:
            flags.append("missing_volume")
        if open_interest is None:
            flags.append("missing_open_interest")
        if iv is None:
            flags.append("missing_iv")
        if delta is None:
            flags.append("missing_delta")
        if gamma is None:
            flags.append("missing_gamma")
        if theta is None:
            flags.append("missing_theta")
        if vega is None:
            flags.append("missing_vega")
        if mid is None:
            flags.append("mid_unavailable")

        liquidity_score = compute_liquidity_score(
            bid_size=bid_size,
            ask_size=ask_size,
            volume=volume,
            relative_spread=relative_spread,
        )

        return OptionContractSnapshot(
            contract_id=contract_id,
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            bid=bid,
            ask=ask,
            mid=mid,
            last=last,
            bid_size=bid_size,
            ask_size=ask_size,
            volume=volume,
            open_interest=open_interest,
            implied_volatility=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            quote_timestamp=quote_ts,
            quote_age_ms=quote_age_ms,
            relative_spread=relative_spread,
            liquidity_score=liquidity_score,
            quality_flags=tuple(flags),
        )


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS option_surfaces (
    surface_id TEXT PRIMARY KEY,
    trading_date TEXT NOT NULL,
    exchange_time TEXT NOT NULL,
    underlying_symbol TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_option_surfaces_trading_date
    ON option_surfaces(trading_date);
"""


class OptionSurfaceRepository:
    """Append-only aiosqlite repository for OptionSurfaceSnapshot rows."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_SQL)
            await db.commit()
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def save(self, surface: OptionSurfaceSnapshot) -> UUID:
        await self._ensure_initialized()
        payload = surface.model_dump_json()
        created_at = datetime.now(tz=surface.exchange_time.tzinfo).isoformat()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    INSERT INTO option_surfaces
                        (surface_id, trading_date, exchange_time, underlying_symbol,
                         payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(surface.surface_id),
                        surface.trading_date.isoformat(),
                        surface.exchange_time.isoformat(),
                        surface.underlying_symbol,
                        payload,
                        created_at,
                    ),
                )
                await db.commit()
        except aiosqlite.IntegrityError as exc:
            raise SnapshotError(
                f"Option surface {surface.surface_id} already exists (append-only)"
            ) from exc
        except aiosqlite.Error as exc:
            raise SnapshotError(f"Failed to save option surface {surface.surface_id}") from exc
        return surface.surface_id

    async def get_by_id(self, surface_id: UUID) -> OptionSurfaceSnapshot | None:
        await self._ensure_initialized()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT payload_json FROM option_surfaces WHERE surface_id = ?",
                    (str(surface_id),),
                ) as cursor:
                    row = await cursor.fetchone()
        except aiosqlite.Error as exc:
            raise SnapshotError(f"Failed to load option surface {surface_id}") from exc
        if row is None:
            return None
        return OptionSurfaceSnapshot.model_validate_json(row[0])

    async def list_by_trading_date(self, trading_date: date) -> list[OptionSurfaceSnapshot]:
        await self._ensure_initialized()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    """
                    SELECT payload_json FROM option_surfaces
                    WHERE trading_date = ?
                    ORDER BY exchange_time ASC
                    """,
                    (trading_date.isoformat(),),
                ) as cursor:
                    rows = await cursor.fetchall()
        except aiosqlite.Error as exc:
            raise SnapshotError(
                f"Failed to list option surfaces for {trading_date.isoformat()}"
            ) from exc
        return [OptionSurfaceSnapshot.model_validate_json(r[0]) for r in rows]
