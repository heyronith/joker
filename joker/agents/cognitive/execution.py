"""Entry tactician and execution proposal compilation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence
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


@dataclass(frozen=True)
class ExecutionValidationConfig:
    """Hard constraints for execution proposals — no direction ranking."""

    allowed_symbol: str = "SPY"
    paper_mode_required: bool = True
    max_legs: int = 1
    allowed_sides: frozenset[str] = frozenset({"buy"})


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
    """Validate hard execution constraints without ranking trade direction."""

    def __init__(self, config: ExecutionValidationConfig | None = None) -> None:
        self._config = config or ExecutionValidationConfig()

    def validate(
        self,
        proposal: ExecutionProposal,
        *,
        trading_mode: str = "PAPER",
        latest_snapshot_id: str | None = None,
    ) -> None:
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

        for leg in proposal.legs:
            if leg.quantity <= 0:
                raise CognitiveValidationError("leg quantity must be positive")
            if leg.side not in self._config.allowed_sides:
                raise CognitiveValidationError(f"unsupported leg side={leg.side!r}")
            contract = parse_contract_id(leg.contract_id)
            if contract.symbol != self._config.allowed_symbol:
                raise CognitiveValidationError(
                    f"contract symbol must be {self._config.allowed_symbol!r}; "
                    f"got {contract.symbol!r}"
                )
            if not contract.is_0dte:
                raise CognitiveValidationError("only 0DTE contracts are supported")


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
    ) -> ProvenancedExecutionCommand:
        if len(proposal.legs) != 1:
            raise CognitiveValidationError(
                "compiler currently supports single-leg proposals only"
            )

        leg = proposal.legs[0]
        contract = parse_contract_id(leg.contract_id)
        intent = OrderIntent(
            candidate_id=str(proposal.proposal_id),
            contract=contract,
            side=leg.side,
            order_type=proposal.order_type,
            quantity=leg.quantity,
            limit_price=float(leg.limit_price) if leg.limit_price is not None else None,
        )
        order_id = client_order_id or f"cog-{proposal.proposal_id}-{leg.leg_id}"
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
        )


def parse_contract_id(contract_id: str) -> OptionContract:
    """Parse ``SYMBOL:YYYY-MM-DD:strike:type`` contract IDs."""
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
) -> ProvenancedExecutionCommand:
    """Validate proposal and compile to Task 1 ExecutionCommand."""
    ExecutionProposalValidator().validate(
        proposal,
        latest_snapshot_id=latest_snapshot_id,
    )
    return ExecutionCommandCompiler(broker_account_id=broker_account_id).compile(
        proposal,
        evidence_ids=evidence_ids,
    )
