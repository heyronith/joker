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


# Every required scenario must map to a runner that can observe its invariant.
# Do not default unmapped scenarios to full_replay — that previously allowed
# label-only "passes" from a clean generic replay.
_MODE_MAP: dict[str, ExecutionMode] = {
    "adv_01": "entry_graph",
    "adv_02": "entry_graph",
    "adv_03": "entry_graph",
    "adv_04": "entry_graph",
    "adv_05": "entry_graph",
    "adv_06": "position_graph",
    "adv_07": "order_management",
    "adv_08": "position_graph",
    "adv_09": "order_management",
    "adv_10": "entry_graph",
    "adv_11": "entry_graph",
    "adv_12": "entry_graph",
    "adv_13": "entry_graph",
    "adv_14": "entry_graph",
    "adv_15": "execution_recovery",
    "adv_16": "execution_recovery",
    "adv_17": "entry_graph",
    "adv_18": "entry_graph",
    "adv_19": "entry_graph",
    "adv_20": "entry_graph",
    "adv_21": "entry_graph",
    "adv_22": "full_replay",
    "adv_23": "full_replay",
    "adv_24": "position_graph",
    "adv_25": "entry_graph",
    **{f"adv_obj_{i:02d}": "entry_graph" for i in range(1, 21)},
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
    **{
        f"adv_obj_{i:02d}": ("objective_invariant",)
        for i in range(1, 21)
    },
}


def build_scenario_definitions() -> tuple[AdversarialScenarioDefinition, ...]:
    out: list[AdversarialScenarioDefinition] = []
    missing = [s.scenario_id for s in ADVERSARIAL_CORPUS if s.scenario_id not in _MODE_MAP]
    if missing:
        raise RuntimeError(f"adversarial_scenarios_missing_execution_mode:{missing}")
    for s in ADVERSARIAL_CORPUS:
        out.append(
            AdversarialScenarioDefinition(
                scenario_id=s.scenario_id,
                version="3.1.0",
                category=s.title,
                required=s.required,
                frozen_truth_fixture_id=_fid(s.scenario_id),
                expected_invariants=_INVARIANTS.get(s.scenario_id, ()),
                execution_mode=_MODE_MAP[s.scenario_id],
            )
        )
    return tuple(out)


ADVERSARIAL_DEFINITIONS = build_scenario_definitions()


def _build_fixture(scenario_id: str) -> AdversarialFixture:
    if scenario_id not in _MODE_MAP:
        raise RuntimeError(f"adversarial_fixture_missing_execution_mode:{scenario_id}")
    valid = _contract("SPY:2026-07-01:500.0:call")
    invented = _contract("SPY:2099-01-01:999.0:call", expiry="2099-01-01", is_0dte=False)
    thin = _contract("SPY:2026-07-01:500.0:call", bid="0.05", ask="2.50")
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
    elif scenario_id == "adv_02":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(valid,))]
        stimulus["conflicting_evidence"] = True
        stimulus["expect_no_trade"] = True
    elif scenario_id == "adv_04":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(valid,))]
        stimulus["false_consensus"] = True
        stimulus["expect_no_trade"] = True
    elif scenario_id == "adv_05":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(thin,))]
        stimulus["thin_liquidity"] = True
        stimulus["expect_no_trade"] = True
    elif scenario_id == "adv_06":
        frames = [
            _frame(scenario_id=scenario_id, index=0, contracts=(valid,)),
            _frame(scenario_id=scenario_id, index=1, contracts=(valid,)),
        ]
        stimulus["thesis_invalidation_exit"] = True
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
    elif scenario_id == "adv_18":
        frames = [
            _frame(
                scenario_id=scenario_id,
                index=0,
                contracts=(
                    valid,
                    _contract("SPY:2026-07-01:505.0:call"),
                ),
            ),
        ]
    elif scenario_id == "adv_08":
        frames = [
            _frame(scenario_id=scenario_id, index=0, contracts=(valid,)),
            _frame(scenario_id=scenario_id, index=1, contracts=(valid,)),
            _frame(scenario_id=scenario_id, index=2, contracts=(valid,)),
        ]
        stimulus["reduce_then_exit"] = True
    elif scenario_id == "adv_13":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(valid,))]
        stimulus["duplicate_order"] = True
    elif scenario_id == "adv_14":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(valid,))]
        stimulus["duplicate_position"] = True
    elif scenario_id == "adv_20":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(valid,))]
        stimulus["expect_no_trade"] = True
        stimulus["justified_no_trade"] = True
    elif scenario_id == "adv_21":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(valid,))]
        stimulus["unsupported_reasoning"] = True
        stimulus["expect_no_trade"] = True
    elif scenario_id == "adv_24":
        frames = [
            _frame(scenario_id=scenario_id, index=0, contracts=(valid,)),
            _frame(scenario_id=scenario_id, index=1, contracts=(valid,)),
        ]
        stimulus["urgent_exit"] = True
    elif scenario_id == "adv_25":
        frames = [_frame(scenario_id=scenario_id, index=0, contracts=(valid,))]
        stimulus["narrow_overfit"] = True
        stimulus["expect_no_trade"] = True
    elif scenario_id in {"adv_22", "adv_23"}:
        # Entry on frame 0 at ~1.20 ask; adverse frame 1 exits near 0.40 bid → loss.
        adverse = _contract("SPY:2026-07-01:500.0:call", bid="0.40", ask="0.60")
        frames = [
            _frame(scenario_id=scenario_id, index=0, contracts=(valid,), bid="500.00", ask="500.10"),
            _frame(
                scenario_id=scenario_id,
                index=1,
                contracts=(adverse,),
                bid="498.00",
                ask="498.20",
            ),
        ]
        key = "full_replay_regime" if scenario_id == "adv_23" else "full_replay_calibration"
        stimulus[key] = True
        stimulus["full_replay_exit"] = True
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
        execution_mode=_MODE_MAP[scenario_id],
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
    """Deprecated label-only executor — use AdversarialRunnerDispatcher."""

    async def execute(
        self,
        fixture: AdversarialFixture,
        *,
        configuration_version_id: UUID,
    ) -> tuple[bool, tuple[str, ...], dict[str, Any]]:
        raise NotImplementedError(
            "DeterministicAdversarialExecutor is retired; "
            "use AdversarialRunnerDispatcher mode runners"
        )
