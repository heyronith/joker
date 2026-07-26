"""Compile TradingEpisode artefacts from Task 1/2 authoritative sources."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID, uuid4

from joker.evolution.hashing import content_hash, hash_model
from joker.evolution.idempotency import episode_idempotency_key
from joker.evolution.repositories import (
    DecisionTraceRepository,
    TradingEpisodeRepository,
)
from joker.evolution.schemas import DecisionTraceSummary, TradingEpisode


ActionClass = Literal[
    "no_trade",
    "entry_rejected",
    "entry_cancelled",
    "closed_trade",
    "open_at_session_end",
]


class EpisodeCompiler:
    """Build immutable episodes from ledger/market/cognitive truth — never logs."""

    def __init__(
        self,
        episode_repo: TradingEpisodeRepository,
        trace_repo: DecisionTraceRepository | None = None,
    ) -> None:
        self._episodes = episode_repo
        self._traces = trace_repo

    async def compile_closed_trade(
        self,
        *,
        session_id: str,
        run_id: str,
        trading_date: date,
        configuration_version_id: UUID,
        initial_snapshot_id: UUID,
        terminal_snapshot_id: UUID | None,
        contract_id: str,
        direction: Literal["bullish", "bearish", "neutral", "none"],
        entry_order_ids: tuple[str, ...],
        exit_order_ids: tuple[str, ...],
        position_action_ids: tuple[str, ...] = (),
        entry_price: Decimal,
        exit_price: Decimal,
        entry_quantity: Decimal,
        exit_quantity: Decimal,
        remaining_quantity: Decimal,
        realised_pnl: Decimal,
        max_favourable_excursion: Decimal | None = None,
        max_adverse_excursion: Decimal | None = None,
        holding_seconds: int | None = None,
        entry_slippage: Decimal | None = None,
        exit_slippage: Decimal | None = None,
        total_fees: Decimal | None = None,
        source_event_ids: tuple[UUID, ...] = (),
        cognitive_artifact_ids: tuple[UUID, ...] = (),
        model_call_ids: tuple[UUID, ...] = (),
        data_quality_ids: tuple[UUID, ...] = (),
        option_surface_ids: tuple[UUID, ...] = (),
        entry_cycle_id: str | None = None,
        proposal_id: UUID | None = None,
        decision_id: UUID | None = None,
        parent_strategy_id: UUID | None = None,
        market_regime_tags: tuple[str, ...] = (),
        terminal_event_id: str,
        prompt_version_ids: tuple[UUID, ...] = (),
    ) -> TradingEpisode:
        findings: list[str] = []
        completed = True
        if remaining_quantity != 0:
            findings.append("remaining_quantity_nonzero")
            completed = False
        if entry_quantity - exit_quantity != remaining_quantity:
            findings.append("quantity_identity_mismatch")
            completed = False
        if not entry_order_ids or not exit_order_ids:
            findings.append("missing_order_ids")
            completed = False

        lifecycle = f"{contract_id}:{entry_order_ids[0] if entry_order_ids else 'none'}"
        key = episode_idempotency_key(session_id, lifecycle, terminal_event_id)
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            entry_cycle_id=entry_cycle_id,
            parent_strategy_id=parent_strategy_id,
            proposal_id=proposal_id,
            decision_id=decision_id,
            initial_snapshot_id=initial_snapshot_id,
            terminal_snapshot_id=terminal_snapshot_id,
            contract_id=contract_id,
            direction=direction,
            action_class="closed_trade",
            entry_order_ids=entry_order_ids,
            position_action_ids=position_action_ids,
            exit_order_ids=exit_order_ids,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=entry_quantity,
            realised_pnl=realised_pnl,
            max_favourable_excursion=max_favourable_excursion,
            max_adverse_excursion=max_adverse_excursion,
            holding_seconds=holding_seconds,
            entry_slippage=entry_slippage,
            exit_slippage=exit_slippage,
            total_fees=total_fees,
            market_regime_tags=market_regime_tags,
            data_quality_ids=data_quality_ids,
            option_surface_ids=option_surface_ids,
            source_event_ids=source_event_ids,
            cognitive_artifact_ids=cognitive_artifact_ids,
            model_call_ids=model_call_ids,
            prompt_version_ids=prompt_version_ids,
            configuration_version_id=configuration_version_id,
            completed=completed,
            completeness_findings=tuple(findings),
            idempotency_key=key,
        )
        await self._episodes.append(episode)
        return episode

    async def compile_no_trade(
        self,
        *,
        session_id: str,
        run_id: str,
        trading_date: date,
        configuration_version_id: UUID,
        initial_snapshot_id: UUID,
        terminal_event_id: str,
        entry_cycle_id: str | None = None,
        decision_id: UUID | None = None,
        rejection_codes: tuple[str, ...] = (),
        cognitive_artifact_ids: tuple[UUID, ...] = (),
        model_call_ids: tuple[UUID, ...] = (),
        data_quality_ids: tuple[UUID, ...] = (),
        option_surface_ids: tuple[UUID, ...] = (),
        source_event_ids: tuple[UUID, ...] = (),
        direction: Literal["bullish", "bearish", "neutral", "none"] = "none",
        market_regime_tags: tuple[str, ...] = (),
        findings: tuple[str, ...] = (),
        completed: bool = True,
        decision_rationale: str = "",
        confidence_values: dict[str, Decimal] | None = None,
    ) -> TradingEpisode:
        lifecycle = f"no_trade:{entry_cycle_id or initial_snapshot_id}"
        key = episode_idempotency_key(session_id, lifecycle, terminal_event_id)
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            entry_cycle_id=entry_cycle_id,
            decision_id=decision_id,
            initial_snapshot_id=initial_snapshot_id,
            direction=direction,
            action_class="no_trade",
            quantity=Decimal("0"),
            market_regime_tags=market_regime_tags,
            data_quality_ids=data_quality_ids,
            option_surface_ids=option_surface_ids,
            source_event_ids=source_event_ids,
            cognitive_artifact_ids=cognitive_artifact_ids,
            model_call_ids=model_call_ids,
            configuration_version_id=configuration_version_id,
            completed=completed,
            completeness_findings=findings,
            idempotency_key=key,
        )
        await self._episodes.append(episode)
        if self._traces is not None:
            summary = DecisionTraceSummary(
                episode_id=episode.episode_id,
                typed_conclusions=("no_trade",),
                rejection_codes=rejection_codes,
                decision_rationale=decision_rationale,
                confidence_values=confidence_values or {},
                content_hash="",
            )
            summary = summary.model_copy(
                update={"content_hash": hash_model(summary, exclude={"created_at"})}
            )
            await self._traces.append(summary)
        return episode

    async def compile_rejected_entry(
        self,
        *,
        session_id: str,
        run_id: str,
        trading_date: date,
        configuration_version_id: UUID,
        initial_snapshot_id: UUID,
        terminal_event_id: str,
        entry_order_ids: tuple[str, ...] = (),
        rejection_codes: tuple[str, ...] = (),
        findings: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> TradingEpisode:
        lifecycle = f"rejected:{entry_order_ids[0] if entry_order_ids else terminal_event_id}"
        key = episode_idempotency_key(session_id, lifecycle, terminal_event_id)
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            initial_snapshot_id=initial_snapshot_id,
            action_class="entry_rejected",
            entry_order_ids=entry_order_ids,
            quantity=Decimal("0"),
            configuration_version_id=configuration_version_id,
            completed=True,
            completeness_findings=tuple([*findings, *rejection_codes]),
            idempotency_key=key,
            **{k: v for k, v in kwargs.items() if k in TradingEpisode.model_fields},
        )
        await self._episodes.append(episode)
        return episode

    async def compile_open_at_session_end(
        self,
        *,
        session_id: str,
        run_id: str,
        trading_date: date,
        configuration_version_id: UUID,
        initial_snapshot_id: UUID,
        contract_id: str,
        quantity: Decimal,
        entry_order_ids: tuple[str, ...],
        terminal_event_id: str,
        **kwargs: Any,
    ) -> TradingEpisode:
        lifecycle = f"open_eod:{contract_id}"
        key = episode_idempotency_key(session_id, lifecycle, terminal_event_id)
        episode = TradingEpisode(
            episode_id=uuid4(),
            session_id=session_id,
            run_id=run_id,
            trading_date=trading_date,
            initial_snapshot_id=initial_snapshot_id,
            contract_id=contract_id,
            action_class="open_at_session_end",
            entry_order_ids=entry_order_ids,
            quantity=quantity,
            configuration_version_id=configuration_version_id,
            completed=False,
            completeness_findings=("open_at_session_end",),
            idempotency_key=key,
            **{k: v for k, v in kwargs.items() if k in TradingEpisode.model_fields},
        )
        await self._episodes.append(episode)
        return episode
