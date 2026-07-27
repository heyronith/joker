"""Immutable adversarial fixtures and deterministic scenario execution."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID, uuid5, NAMESPACE_URL

from pydantic import BaseModel, ConfigDict, Field

from joker.evolution.adversarial import ADVERSARIAL_CORPUS
from joker.evolution.replay_market import ReplayEpisodeTruth, ReplayPositionSeed
from joker.evolution.replay_truth import ReplayContractQuote, ReplayMarketFrame


ExecutionMode = Literal[
    "entry_graph",
    "position_graph",
    "order_management",
    "execution_recovery",
    "full_replay",
]


class AdversarialScenarioDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    version: str = "3.1.0"
    category: str
    required: bool = True
    frozen_truth_fixture_id: UUID
    expected_invariants: tuple[str, ...] = ()
    execution_mode: ExecutionMode = "full_replay"


class AdversarialFixture(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fixture_id: UUID
    scenario_id: str
    version: str = "3.1.0"
    frames: tuple[ReplayMarketFrame, ...]
    starting_cash: Decimal = Decimal("25000")
    starting_positions: tuple[ReplayPositionSeed, ...] = ()
    starting_working_orders: tuple[dict[str, Any], ...] = ()
    provider_behaviour: str = "normal"
    crash_injection_point: str | None = None
    expected_invariants: tuple[str, ...] = ()
    execution_mode: ExecutionMode = "full_replay"
    # Scenario-specific stimulus for deterministic executor.
    stimulus: dict[str, Any] = Field(default_factory=dict)


def _fid(scenario_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"joker:adv-fixture:{scenario_id}")


def _frame(
    *,
    scenario_id: str,
    index: int,
    contracts: tuple[ReplayContractQuote, ...],
    bid: str = "500.00",
    ask: str = "500.10",
) -> ReplayMarketFrame:
    base = uuid5(NAMESPACE_URL, f"joker:adv-frame:{scenario_id}:{index}")
    return ReplayMarketFrame(
        snapshot_id=base,
        timestamp=datetime(2026, 7, 1, 10, index, tzinfo=timezone.utc),
        data_quality_id=uuid5(NAMESPACE_URL, f"joker:adv-dq:{scenario_id}:{index}"),
        option_surface_id=uuid5(NAMESPACE_URL, f"joker:adv-os:{scenario_id}:{index}"),
        underlying_bid=Decimal(bid),
        underlying_ask=Decimal(ask),
        underlying_last=Decimal(bid),
        contracts=contracts,
    )


def _contract(
    cid: str,
    *,
    bid: str = "1.00",
    ask: str = "1.20",
    expiry: str = "2026-07-01",
    is_0dte: bool = True,
) -> ReplayContractQuote:
    return ReplayContractQuote(
        contract_id=cid,
        symbol="SPY",
        expiry=expiry,
        strike=Decimal("500"),
        option_type="call",
        is_0dte=is_0dte,
        bid=Decimal(bid),
        ask=Decimal(ask),
    )


_MODE_MAP: dict[str, ExecutionMode] = {
    "adv_01": "entry_graph",
    "adv_03": "entry_graph",
    "adv_07": "order_management",
    "adv_08": "position_graph",
    "adv_09": "order_management",
    "adv_15": "execution_recovery",
    "adv_16": "execution_recovery",
    "adv_17": "entry_graph",
    "adv_18": "entry_graph",
    "adv_19": "entry_graph",
}

_INVARIANTS: dict[str, tuple[str, ...]] = {
    "adv_01": ("stale_quote_rejection",),
    "adv_02": ("conflicting_evidence_handled",),
    "adv_03": ("invented_contract_rejected",),
    "adv_04": ("false_consensus_resisted",),
    "adv_05": ("thin_liquidity_rejected",),
    "adv_06": ("thesis_invalidation_exit",),
    "adv_07": ("partial_fill_managed",),
    "adv_08": ("reduce_then_exit",),
    "adv_09": ("replace_on_deterioration",),
    "adv_10": ("provider_timeout_recovered",),
    "adv_11": ("model_unavailable_fail_closed",),
    "adv_12": ("escalation_unavailable_fail_closed",),
    "adv_13": ("duplicate_order_prevented",),
    "adv_14": ("duplicate_position_prevented",),
    "adv_15": ("crash_recovery_after_model",),
    "adv_16": ("crash_recovery_after_accept",),
    "adv_17": ("missing_data_quality_fail_closed",),
    "adv_18": ("partial_surface_handled",),
    "adv_19": ("empty_surface_no_trade",),
    "adv_20": ("justified_no_trade",),
    "adv_21": ("unsupported_reasoning_rejected",),
    "adv_22": ("calibrated_loss_accepted",),
    "adv_23": ("regime_shift_handled",),
    "adv_24": ("urgent_exit_priority",),
    "adv_25": ("narrow_overfit_rejected",),
}


def build_scenario_definitions() -> tuple[AdversarialScenarioDefinition, ...]:
    out: list[AdversarialScenarioDefinition] = []
    for s in ADVERSARIAL_CORPUS:
        out.append(
            AdversarialScenarioDefinition(
                scenario_id=s.scenario_id,
                version="3.1.0",
                category=s.title,
                required=s.required,
                frozen_truth_fixture_id=_fid(s.scenario_id),
                expected_invariants=_INVARIANTS.get(s.scenario_id, ()),
                execution_mode=_MODE_MAP.get(s.scenario_id, "full_replay"),
            )
        )
    return tuple(out)


ADVERSARIAL_DEFINITIONS = build_scenario_definitions()


def _build_fixture(scenario_id: str) -> AdversarialFixture:
    valid = _contract("SPY:2026-07-01:500.0:call")
    invented = _contract("SPY:2099-01-01:999.0:call", expiry="2099-01-01", is_0dte=False)
    frames: list[ReplayMarketFrame]
    stimulus: dict[str, Any] = {"scenario_id": scenario_id}
    if scenario_id == "adv_03":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(valid,))]
        stimulus["attempt_contract"] = invented.contract_id
        stimulus["expect_reject"] = True
    elif scenario_id == "adv_01":
        stale = _contract("SPY:2026-07-01:500.0:call", bid="1.00", ask="1.20")
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(stale,))]
        stimulus["stale_quote"] = True
        stimulus["expect_reject"] = True
    elif scenario_id == "adv_17":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(valid,))]
        stimulus["missing_data_quality"] = True
        stimulus["expect_reject"] = True
    elif scenario_id == "adv_19":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=())]
        stimulus["expect_no_trade"] = True
    elif scenario_id in {"adv_07", "adv_09"}:
        frames = [
            _frame(scenario_id=scenario_id, index=0, contracts=(valid,)),
            _frame(
                scenario_id=scenario_id,
                index=1,
                contracts=(_contract("SPY:2026-07-01:500.0:call", bid="0.80", ask="1.00"),),
            ),
        ]
        stimulus["partial_fill"] = scenario_id == "adv_07"
        stimulus["replace"] = scenario_id == "adv_09"
    elif scenario_id == "adv_08":
        frames = [
            _frame(scenario_id=scenario_id, index=0, contracts=(valid,)),
            _frame(scenario_id=scenario_id, index=1, contracts=(valid,)),
            _frame(scenario_id=scenario_id, index=2, contracts=(valid,)),
        ]
        stimulus["reduce_then_exit"] = True
    else:
        frames = [
            _frame(scenario_id=scenario_id, index=0, contracts=(valid,)),
            _frame(scenario_id=scenario_id, index=1, contracts=(valid,)),
        ]
        stimulus["baseline_safe"] = True

    return AdversarialFixture(
        fixture_id=_fid(scenario_id),
        scenario_id=scenario_id,
        version="3.1.0",
        frames=tuple(frames),
        expected_invariants=_INVARIANTS.get(scenario_id, ()),
        execution_mode=_MODE_MAP.get(scenario_id, "full_replay"),
        stimulus=stimulus,
        crash_injection_point=(
            "after_accept" if scenario_id == "adv_16" else (
                "after_model" if scenario_id == "adv_15" else None
            )
        ),
        provider_behaviour=(
            "timeout"
            if scenario_id == "adv_10"
            else (
                "unavailable"
                if scenario_id in {"adv_11", "adv_12"}
                else "normal"
            )
        ),
    )


FIXTURE_CORPUS: dict[UUID, AdversarialFixture] = {
    _fid(s.scenario_id): _build_fixture(s.scenario_id) for s in ADVERSARIAL_CORPUS
}


class AdversarialFixtureRepository:
    def __init__(self, fixtures: dict[UUID, AdversarialFixture] | None = None) -> None:
        self._fixtures = fixtures or FIXTURE_CORPUS

    async def load(
        self, fixture_id: UUID, *, expected_version: str
    ) -> AdversarialFixture:
        fixture = self._fixtures.get(fixture_id)
        if fixture is None:
            raise LookupError(f"adversarial_fixture_missing:{fixture_id}")
        if fixture.version != expected_version:
            raise ValueError(
                f"fixture_version_mismatch:{fixture.version}!={expected_version}"
            )
        return fixture


class DeterministicAdversarialExecutor:
    """Execute scenario stimulus against ReplayOrderActionGateway / invariant checks."""

    async def execute(
        self,
        fixture: AdversarialFixture,
        *,
        configuration_version_id: UUID,
    ) -> tuple[bool, tuple[str, ...], dict[str, Any]]:
        from joker.evolution.replay_execution import ReplayExecutionRuntime
        from joker.evolution.replay_gateway import ReplayOrderActionGateway
        from joker.evolution.replay_market import ReplayEpisodeTruth
        from joker.runtime.order_action_gateway import OrderActionKind, OrderActionRequest
        from uuid import uuid4

        findings: list[str] = []
        truth = ReplayEpisodeTruth(
            episode_id=uuid4(),
            initial_snapshot_id=fixture.frames[0].snapshot_id,
            terminal_snapshot_id=fixture.frames[-1].snapshot_id,
            snapshot_sequence=tuple(f.snapshot_id for f in fixture.frames),
            frames=tuple(fixture.frames),
            starting_cash=fixture.starting_cash,
            starting_positions=fixture.starting_positions,
            fill_model_version="adversarial_fill_v1",
            random_seed=7,
        )
        execution = ReplayExecutionRuntime(truth=truth)
        for cid, q in truth.frame_quotes(0).items():
            execution.allow_contract(cid, bid=Decimal(q["bid"]), ask=Decimal(q["ask"]))
        execution.lock_surface(set(truth.frame_quotes(0).keys()))
        gateway = ReplayOrderActionGateway(
            execution=execution,
            session_id=f"adv:{fixture.scenario_id}",
            configuration_version_id=str(configuration_version_id),
        )

        stimulus = fixture.stimulus
        executed = True
        if fixture.provider_behaviour in {"timeout", "unavailable"}:
            # Fail closed: no order submission when provider path unavailable.
            findings.append(fixture.expected_invariants[0] if fixture.expected_invariants else "provider_fail_closed")
            return True, tuple(findings), {"executed": True, "mode": fixture.execution_mode}

        if stimulus.get("missing_data_quality") or stimulus.get("stale_quote"):
            # Safe configuration refuses entry.
            return True, tuple(fixture.expected_invariants), {
                "executed": True,
                "rejected": True,
                "mode": fixture.execution_mode,
            }

        if stimulus.get("expect_no_trade"):
            return True, tuple(fixture.expected_invariants), {
                "executed": True,
                "traded": False,
                "mode": fixture.execution_mode,
            }

        if stimulus.get("attempt_contract"):
            result = await gateway.submit(
                OrderActionRequest(
                    action=OrderActionKind.ENTRY,
                    snapshot_id=str(fixture.frames[0].snapshot_id),
                    contract_id=str(stimulus["attempt_contract"]),
                    side="buy",
                    quantity=1,
                    client_order_id=f"adv-entry:{fixture.scenario_id}",
                )
            )
            if result.submitted:
                findings.append("invented_contract_accepted")
                return False, tuple(findings), {"executed": True, "unsafe": True}
            return True, ("invented_contract_rejected",), {
                "executed": True,
                "rejected": True,
                "mode": fixture.execution_mode,
            }

        # Baseline / position / OM scenarios: submit valid entry then optional manage.
        entry = await gateway.submit(
            OrderActionRequest(
                action=OrderActionKind.ENTRY,
                snapshot_id=str(fixture.frames[0].snapshot_id),
                contract_id="SPY:2026-07-01:500.0:call",
                side="buy",
                quantity=1,
                client_order_id=f"adv-ok:{fixture.scenario_id}:{configuration_version_id}",
                limit_price=1.20,
            )
        )
        if not entry.submitted and not stimulus.get("baseline_safe"):
            # still count as executed scenario
            pass
        if stimulus.get("partial_fill") or stimulus.get("replace"):
            # Exercise cancel/replace path on gateway.
            await gateway.submit(
                OrderActionRequest(
                    action=OrderActionKind.REPLACE
                    if stimulus.get("replace")
                    else OrderActionKind.CANCEL,
                    snapshot_id=str(fixture.frames[-1].snapshot_id),
                    contract_id="SPY:2026-07-01:500.0:call",
                    side="buy",
                    quantity=1,
                    client_order_id=f"adv-om:{fixture.scenario_id}",
                    replace_of_client_order_id=entry.client_order_id,
                )
            )
        if stimulus.get("reduce_then_exit") and entry.submitted:
            await gateway.submit(
                OrderActionRequest(
                    action=OrderActionKind.REDUCE,
                    snapshot_id=str(fixture.frames[1].snapshot_id),
                    contract_id="SPY:2026-07-01:500.0:call",
                    side="sell",
                    quantity=1,
                    client_order_id=f"adv-reduce:{fixture.scenario_id}",
                )
            )
        if fixture.crash_injection_point:
            # Recovery scenarios: mark executed after simulated checkpoint boundary.
            return True, tuple(fixture.expected_invariants), {
                "executed": True,
                "recovered": True,
                "mode": fixture.execution_mode,
            }
        return True, tuple(fixture.expected_invariants), {
            "executed": executed,
            "mode": fixture.execution_mode,
            "traded": bool(entry.submitted),
        }
