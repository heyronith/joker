"""Entry tactician and authoritative execution proposal validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.exceptions import CognitiveValidationError
from joker.cognition.schemas import (
    AgentRole,
    ExecutionProposal,
    MetaDecision,
    StrategyHypothesis,
)
from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.market.quality import DataQualityReport
from joker.market.snapshots import MarketSnapshot
from joker.models.router import ModelRouter
from joker.runtime.execution_runtime import ExecutionCommand
from joker.schemas.domain import OptionContract, OrderIntent


@dataclass(frozen=True)
class ProvenancedExecutionCommand:
    """ExecutionCommand with cognitive provenance metadata."""

    command: ExecutionCommand
    decision_id: str
    strategy_id: str
    proposal_id: str
    snapshot_id: str
    cycle_id: str
    evidence_ids: tuple[str, ...]
    max_quote_age_seconds: int | None = 3600
    estimate_id: str | None = None
    # Horizon-start anchor persisted into entry execution provenance.
    causation_event_id: str | None = None


@dataclass(frozen=True)
class ExecutionValidationConfig:
    """Hard constraints for execution proposals — no direction ranking."""

    allowed_symbol: str = "SPY"
    paper_mode_required: bool = True
    max_legs: int = 1
    allowed_sides: frozenset[str] = frozenset({"buy", "sell"})
    max_quantity: int = 20
    require_positive_quantity: bool = True


@dataclass(frozen=True)
class AuthoritativeMarketTruth:
    """Task 1 truth re-read immediately before submission."""

    snapshot: MarketSnapshot
    data_quality: DataQualityReport | None = None
    option_surface: OptionSurfaceSnapshot | None = None
    working_orders: tuple[Any, ...] = ()
    open_order_client_ids: frozenset[str] = field(default_factory=frozenset)
    open_position_contract_ids: frozenset[str] = field(default_factory=frozenset)
    already_submitted_proposal_ids: frozenset[str] = field(default_factory=frozenset)
    trading_mode: str = "PAPER"
    now: datetime | None = None


class EntryTacticianAgent(CognitiveAgent[ExecutionProposal]):
    role = AgentRole.ENTRY_TACTICIAN
    output_type = ExecutionProposal

    async def propose(
        self,
        context: ContextPackage,
        router: ModelRouter,
        *,
        decision: MetaDecision,
        strategy: StrategyHypothesis,
        evidence_ids: Sequence[UUID] = (),
    ) -> ExecutionProposal:
        """Produce an execution proposal from an approved meta-decision and strategy."""
        proposal = await self.run(
            context,
            router,
            extra_payload={
                "meta_decision": decision.model_dump(mode="json"),
                "strategy": strategy.model_dump(mode="json"),
                "evidence_ids": [str(eid) for eid in evidence_ids],
            },
        )
        return proposal.model_copy(
            update={
                "decision_id": decision.decision_id,
                "strategy_id": strategy.strategy_id,
                "session_id": context.session_id,
                "cycle_id": context.cycle_id,
                "snapshot_id": context.snapshot_id,
            }
        )


class ExecutionProposalValidator:
    """Validate hard execution constraints without ranking trade direction.

    When ``truth`` is supplied, re-reads Task 1 snapshot/surface/quality and
    applies authoritative checks. Does not alter direction, contract, quantity,
    or price for strategic reasons.
    """

    def __init__(self, config: ExecutionValidationConfig | None = None) -> None:
        self._config = config or ExecutionValidationConfig()

    def validate(
        self,
        proposal: ExecutionProposal,
        *,
        trading_mode: str = "PAPER",
        latest_snapshot_id: str | None = None,
        trading_date: date | None = None,
        truth: AuthoritativeMarketTruth | None = None,
    ) -> None:
        if truth is not None:
            self._validate_authoritative(proposal, truth=truth)
            return

        if latest_snapshot_id and str(proposal.snapshot_id) != latest_snapshot_id:
            raise CognitiveValidationError("execution proposal snapshot is stale")

        if self._config.paper_mode_required and trading_mode.upper() != "PAPER":
            raise CognitiveValidationError(
                f"execution proposals require PAPER mode; got {trading_mode!r}"
            )

        if not proposal.legs:
            raise CognitiveValidationError("execution proposal must include at least one leg")

        if len(proposal.legs) > self._config.max_legs:
            raise CognitiveValidationError(
                f"execution proposal exceeds max_legs={self._config.max_legs}"
            )

        if proposal.order_type not in {"limit", "market"}:
            raise CognitiveValidationError(f"unsupported order_type={proposal.order_type!r}")

        if trading_date is None:
            raise CognitiveValidationError(
                "trading_date is required to validate 0DTE contract expiry"
            )

        for leg in proposal.legs:
            self._validate_leg_basics(leg.quantity, leg.side)
            contract = parse_contract_id(leg.contract_id, trading_date=trading_date)
            if contract.symbol != self._config.allowed_symbol:
                raise CognitiveValidationError(
                    f"contract symbol must be {self._config.allowed_symbol!r}; "
                    f"got {contract.symbol!r}"
                )
            if not contract.is_0dte:
                raise CognitiveValidationError("only 0DTE contracts are supported")

    def _validate_authoritative(
        self,
        proposal: ExecutionProposal,
        *,
        truth: AuthoritativeMarketTruth,
    ) -> None:
        snapshot = truth.snapshot
        if snapshot.snapshot_id is None:
            raise CognitiveValidationError("proposal snapshot does not exist")
        if str(proposal.snapshot_id) != str(snapshot.snapshot_id):
            raise CognitiveValidationError(
                "execution proposal snapshot is stale or mismatched vs Task 1 truth"
            )

        if self._config.paper_mode_required and truth.trading_mode.upper() != "PAPER":
            raise CognitiveValidationError(
                f"execution proposals require PAPER mode; got {truth.trading_mode!r}"
            )

        dq = truth.data_quality
        if dq is None:
            raise CognitiveValidationError(
                "data quality report unavailable; execution blocked"
            )
        if not dq.usable_for_execution:
            raise CognitiveValidationError(
                "data quality does not permit execution "
                f"(severity={dq.severity.value})"
            )

        if truth.option_surface is None:
            raise CognitiveValidationError("referenced option surface does not exist")
        if (
            snapshot.option_surface_id is not None
            and truth.option_surface.surface_id != snapshot.option_surface_id
        ):
            raise CognitiveValidationError(
                "option surface id does not match snapshot.option_surface_id"
            )

        if str(proposal.proposal_id) in truth.already_submitted_proposal_ids:
            raise CognitiveValidationError(
                "proposal/decision has already been submitted"
            )

        if not proposal.legs:
            raise CognitiveValidationError("execution proposal must include at least one leg")
        if len(proposal.legs) > self._config.max_legs:
            raise CognitiveValidationError(
                f"execution proposal exceeds max_legs={self._config.max_legs}"
            )
        if proposal.order_type not in {"limit", "market"}:
            raise CognitiveValidationError(
                f"unsupported execution shape order_type={proposal.order_type!r}"
            )

        surface_by_id = {
            c.contract_id: c for c in truth.option_surface.contracts
        }
        now = truth.now or datetime.now(timezone.utc)

        for leg in proposal.legs:
            self._validate_leg_basics(leg.quantity, leg.side)
            contract = parse_contract_id(
                leg.contract_id,
                trading_date=snapshot.trading_date,
            )
            if contract.symbol != self._config.allowed_symbol:
                raise CognitiveValidationError(
                    f"contract symbol must be {self._config.allowed_symbol!r}; "
                    f"got {contract.symbol!r}"
                )
            if contract.expiration != snapshot.trading_date:
                raise CognitiveValidationError(
                    "contract expiry must equal the snapshot exchange trading date "
                    f"(expiry={contract.expiration.isoformat()}, "
                    f"trading_date={snapshot.trading_date.isoformat()})"
                )
            if not contract.is_0dte:
                raise CognitiveValidationError("only 0DTE contracts are supported")

            surface_row = surface_by_id.get(leg.contract_id)
            if surface_row is None:
                raise CognitiveValidationError(
                    f"contract_id {leg.contract_id!r} is absent from the option surface"
                )
            self._validate_quote_age(leg.max_quote_age_seconds, surface_row, now=now)

            if leg.side == "buy" and truth.open_position_contract_ids:
                if leg.contract_id in truth.open_position_contract_ids:
                    raise CognitiveValidationError(
                        "conflicting active position exists for contract"
                    )
            if truth.working_orders:
                from joker.runtime.order_action_gateway import has_working_entry_order

                if leg.side == "buy" and has_working_entry_order(truth.working_orders):
                    raise CognitiveValidationError(
                        "conflicting working entry order exists; no additional entry"
                    )
            elif truth.open_order_client_ids and leg.side == "buy":
                pass

        if truth.working_orders and proposal.action in {"execute", "probe"}:
            proposal_order_token = str(proposal.proposal_id)
            if any(
                getattr(o, "proposal_id", None) == proposal_order_token
                for o in truth.working_orders
            ):
                raise CognitiveValidationError(
                    "conflicting active order exists for proposal"
                )
        elif truth.open_order_client_ids and proposal.action in {"execute", "probe"}:
            # Block when an open working order already covers this proposal id.
            proposal_order_token = str(proposal.proposal_id)
            if any(proposal_order_token in oid for oid in truth.open_order_client_ids):
                raise CognitiveValidationError(
                    "conflicting active order exists for proposal"
                )

    def _validate_leg_basics(self, quantity: int, side: str) -> None:
        if self._config.require_positive_quantity and quantity <= 0:
            raise CognitiveValidationError("leg quantity must be positive")
        if quantity > self._config.max_quantity:
            raise CognitiveValidationError(
                f"leg quantity {quantity} exceeds max_quantity={self._config.max_quantity}"
            )
        if side not in self._config.allowed_sides:
            raise CognitiveValidationError(f"unsupported leg side={side!r}")

    @staticmethod
    def _validate_quote_age(
        max_quote_age_seconds: int | None,
        row: OptionContractSnapshot,
        *,
        now: datetime,
    ) -> None:
        if max_quote_age_seconds is None:
            return
        if row.quote_age_ms is not None:
            if row.quote_age_ms > max_quote_age_seconds * 1000:
                raise CognitiveValidationError(
                    f"quote age {row.quote_age_ms}ms exceeds proposal limit "
                    f"{max_quote_age_seconds}s"
                )
            return
        if row.quote_timestamp is None:
            raise CognitiveValidationError("quote timestamp missing")
        if row.quote_timestamp.tzinfo is None:
            raise CognitiveValidationError("quote timestamp must be timezone-aware")
        age = (now - row.quote_timestamp).total_seconds()
        if age < -1:
            raise CognitiveValidationError(
                "quote timestamp is materially in the future"
            )
        if age > max_quote_age_seconds:
            raise CognitiveValidationError(
                f"quote age {age:.1f}s exceeds proposal limit {max_quote_age_seconds}s"
            )


class ExecutionCommandCompiler:
    """Compile a validated proposal into a Task 1 ExecutionCommand."""

    def __init__(self, *, broker_account_id: str = "default") -> None:
        self._broker_account_id = broker_account_id

    def compile(
        self,
        proposal: ExecutionProposal,
        *,
        evidence_ids: Sequence[UUID] = (),
        client_order_id: str | None = None,
        trading_date: date | None = None,
    ) -> ProvenancedExecutionCommand:
        if len(proposal.legs) != 1:
            raise CognitiveValidationError(
                "compiler currently supports single-leg proposals only"
            )

        leg = proposal.legs[0]
        contract = parse_contract_id(leg.contract_id, trading_date=trading_date)
        order_id = client_order_id or f"cog-{proposal.proposal_id}-{leg.leg_id}"
        intent = OrderIntent(
            intent_id=order_id,
            candidate_id=str(proposal.proposal_id),
            contract=contract,
            side=leg.side,
            order_type=proposal.order_type,
            quantity=leg.quantity,
            limit_price=float(leg.limit_price) if leg.limit_price is not None else None,
        )
        command = ExecutionCommand(
            client_order_id=order_id,
            intent=intent,
            broker_account_id=self._broker_account_id,
        )
        return ProvenancedExecutionCommand(
            command=command,
            decision_id=str(proposal.decision_id),
            strategy_id=str(proposal.strategy_id),
            proposal_id=str(proposal.proposal_id),
            snapshot_id=str(proposal.snapshot_id),
            cycle_id=proposal.cycle_id,
            evidence_ids=tuple(str(eid) for eid in evidence_ids),
            max_quote_age_seconds=leg.max_quote_age_seconds,
        )


def parse_contract_id(
    contract_id: str,
    *,
    trading_date: date | None = None,
) -> OptionContract:
    """Parse ``SYMBOL:YYYY-MM-DD:strike:type`` and derive 0DTE from trading date."""
    parts = contract_id.split(":")
    if len(parts) != 4:
        raise CognitiveValidationError(
            f"unsupported contract_id format: {contract_id!r}; expected SYMBOL:date:strike:type"
        )
    symbol, expiry_raw, strike_raw, option_type = parts
    if option_type not in {"call", "put"}:
        raise CognitiveValidationError(f"invalid option_type in contract_id: {contract_id!r}")
    try:
        expiry = date.fromisoformat(expiry_raw)
        strike = float(strike_raw)
    except ValueError as exc:
        raise CognitiveValidationError(f"invalid contract_id: {contract_id!r}") from exc
    if trading_date is None:
        # Without an authoritative trading date, refuse to invent 0DTE status.
        raise CognitiveValidationError(
            "trading_date is required to derive 0DTE status for contract_id"
        )
    if expiry != trading_date:
        raise CognitiveValidationError(
            "contract expiry must equal the snapshot exchange trading date "
            f"(expiry={expiry.isoformat()}, trading_date={trading_date.isoformat()}); "
            "only 0DTE contracts are supported"
        )
    return OptionContract(
        symbol=symbol,
        expiration=expiry,
        strike=strike,
        option_type=option_type,  # type: ignore[arg-type]
        is_0dte=True,
    )


async def run_entry_tactician(
    *,
    state,
    router: ModelRouter,
    context: ContextPackage,
    meta_decision: MetaDecision,
    strategy: StrategyHypothesis,
) -> ExecutionProposal:
    """Graph-facing entry tactician wrapper."""
    evidence_ids = tuple(e.evidence_id for e in (state.get("evidence") or []))
    agent = EntryTacticianAgent()
    return await agent.propose(
        context,
        router,
        decision=meta_decision,
        strategy=strategy,
        evidence_ids=evidence_ids,
    )


def validate_and_compile_proposal(
    proposal: ExecutionProposal,
    *,
    latest_snapshot_id: str | None = None,
    broker_account_id: str = "default",
    evidence_ids: Sequence[UUID] = (),
    truth: AuthoritativeMarketTruth | None = None,
    config: ExecutionValidationConfig | None = None,
    trading_date: date | None = None,
) -> ProvenancedExecutionCommand:
    """Validate proposal and compile to Task 1 ExecutionCommand."""
    validator = ExecutionProposalValidator(config)
    if truth is not None:
        validator.validate(proposal, truth=truth)
        resolved_date = truth.snapshot.trading_date
    else:
        validator.validate(
            proposal,
            latest_snapshot_id=latest_snapshot_id,
            trading_date=trading_date,
        )
        resolved_date = trading_date
    return ExecutionCommandCompiler(broker_account_id=broker_account_id).compile(
        proposal,
        evidence_ids=evidence_ids,
        trading_date=resolved_date,
    )


def build_truth_from_deps(
    *,
    snapshot: MarketSnapshot,
    data_quality: DataQualityReport | None,
    option_surface: OptionSurfaceSnapshot | None,
    projection: Any | None = None,
    already_submitted_proposal_ids: Sequence[str] = (),
    trading_mode: str = "PAPER",
    now: datetime | None = None,
) -> AuthoritativeMarketTruth:
    """Assemble authoritative truth from Task 1 repositories / projection."""
    from joker.runtime.order_action_gateway import (
        open_positions_from_projection,
        working_orders_from_projection,
    )

    working = working_orders_from_projection(projection)
    open_positions = open_positions_from_projection(projection)
    return AuthoritativeMarketTruth(
        snapshot=snapshot,
        data_quality=data_quality,
        option_surface=option_surface,
        working_orders=working,
        open_order_client_ids=frozenset(o.client_order_id for o in working),
        open_position_contract_ids=frozenset(open_positions.keys()),
        already_submitted_proposal_ids=frozenset(str(x) for x in already_submitted_proposal_ids),
        trading_mode=trading_mode,
        now=now,
    )
