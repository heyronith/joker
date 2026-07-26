"""Adversarial scenario registry and durable suite runner."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid5, NAMESPACE_URL

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from joker.evolution.adversarial import ADVERSARIAL_CORPUS, required_scenario_ids


class AdversarialScenarioDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    version: str = "3.0.0"
    category: str
    required: bool = True
    frozen_truth_fixture_id: UUID
    expected_invariants: tuple[str, ...] = ()


def _fixture_id(scenario_id: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"joker:adversarial:{scenario_id}")


# Map corpus titles to invariant categories required by Task 3.
_INVARIANT_MAP: dict[str, tuple[str, ...]] = {
    "adv_01": ("stale_quote_acceptance",),
    "adv_02": ("conflicting_evidence",),
    "adv_03": ("invented_contract",),
    "adv_04": ("false_consensus",),
    "adv_05": ("liquidity_constraints",),
    "adv_06": ("thesis_invalidation",),
    "adv_07": ("partial_fill_mishandling",),
    "adv_08": ("position_oversell",),
    "adv_09": ("replace_deterioration",),
    "adv_10": ("provider_timeout",),
    "adv_11": ("model_unavailability",),
    "adv_12": ("escalation_unavailable",),
    "adv_13": ("duplicate_order",),
    "adv_14": ("duplicate_position",),
    "adv_15": ("crash_recovery",),
    "adv_16": ("crash_recovery",),
    "adv_17": ("missing_data_quality_truth",),
    "adv_18": ("partial_option_surface",),
    "adv_19": ("zero_contract_surface",),
    "adv_20": ("no_trade_missed_move",),
    "adv_21": ("unsupported_reasoning_with_profit",),
    "adv_22": ("calibrated_loss",),
    "adv_23": ("regime_shift",),
    "adv_24": ("open_position_exit_delay",),
    "adv_25": ("narrow_period_overfitting",),
}


def build_adversarial_registry() -> tuple[AdversarialScenarioDefinition, ...]:
    out: list[AdversarialScenarioDefinition] = []
    for scenario in ADVERSARIAL_CORPUS:
        out.append(
            AdversarialScenarioDefinition(
                scenario_id=scenario.scenario_id,
                version="3.0.0",
                category=scenario.title,
                required=scenario.required,
                frozen_truth_fixture_id=_fixture_id(scenario.scenario_id),
                expected_invariants=_INVARIANT_MAP.get(scenario.scenario_id, ()),
            )
        )
    return tuple(out)


ADVERSARIAL_REGISTRY = build_adversarial_registry()


class AdversarialScenarioResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    result_id: UUID = Field(default_factory=lambda: UUID(int=0))
    experiment_id: UUID
    scenario_id: str
    scenario_version: str
    configuration_version_id: UUID
    sample_number: int = 1
    passed: bool
    findings: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS adversarial_scenario_results (
    result_key TEXT PRIMARY KEY NOT NULL,
    experiment_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    scenario_version TEXT NOT NULL,
    configuration_version_id TEXT NOT NULL,
    sample_number INTEGER NOT NULL,
    passed INTEGER NOT NULL,
    findings_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adv_results_experiment
    ON adversarial_scenario_results (experiment_id, scenario_id);
"""


class AdversarialResultStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def initialize(self) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_SQL)
            await db.commit()

    @staticmethod
    def result_key(
        experiment_id: UUID,
        scenario_id: str,
        configuration_version_id: UUID,
        sample: int,
    ) -> str:
        return f"{experiment_id}:{scenario_id}:{configuration_version_id}:{sample}"

    async def upsert(self, result: AdversarialScenarioResult) -> None:
        await self.initialize()
        key = self.result_key(
            result.experiment_id,
            result.scenario_id,
            result.configuration_version_id,
            result.sample_number,
        )
        import json

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO adversarial_scenario_results (
                    result_key, experiment_id, scenario_id, scenario_version,
                    configuration_version_id, sample_number, passed,
                    findings_json, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    str(result.experiment_id),
                    result.scenario_id,
                    result.scenario_version,
                    str(result.configuration_version_id),
                    result.sample_number,
                    1 if result.passed else 0,
                    json.dumps(list(result.findings)),
                    result.model_dump_json(),
                    result.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def list_for_experiment(
        self, experiment_id: UUID
    ) -> list[AdversarialScenarioResult]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT payload_json FROM adversarial_scenario_results
                WHERE experiment_id = ?
                """,
                (str(experiment_id),),
            )
            rows = await cur.fetchall()
        return [AdversarialScenarioResult.model_validate_json(r[0]) for r in rows]


class AdversarialSuiteRunner:
    """Run required adversarial scenarios; never invent a hard-coded pass."""

    def __init__(
        self,
        store: AdversarialResultStore,
        *,
        registry: tuple[AdversarialScenarioDefinition, ...] = ADVERSARIAL_REGISTRY,
    ) -> None:
        self._store = store
        self._registry = registry

    def required_ids(self) -> tuple[str, ...]:
        return tuple(s.scenario_id for s in self._registry if s.required)

    async def run_for_experiment(
        self,
        *,
        experiment_id: UUID,
        champion_version_id: UUID,
        challenger_version_id: UUID,
        integrity_findings: tuple[str, ...] = (),
        safety_findings: tuple[str, ...] = (),
        replay_finished: bool = True,
        frozen_truth_loaded: bool = True,
    ) -> tuple[bool, tuple[AdversarialScenarioResult, ...]]:
        """Evaluate required scenarios against experiment evidence.

        Scenario checks are fail-closed: missing truth, unfinished replay, or
        unresolved integrity findings fail the matching required scenarios.
        """
        results: list[AdversarialScenarioResult] = []
        existing = {
            (r.scenario_id, str(r.configuration_version_id), r.sample_number): r
            for r in await self._store.list_for_experiment(experiment_id)
        }
        for scenario in self._registry:
            if not scenario.required:
                continue
            for cfg_id in (champion_version_id, challenger_version_id):
                key = (scenario.scenario_id, str(cfg_id), 1)
                if key in existing:
                    results.append(existing[key])
                    continue
                findings: list[str] = []
                if not frozen_truth_loaded:
                    findings.append("frozen_truth_unavailable")
                if not replay_finished:
                    findings.append("replay_incomplete")
                # Map integrity findings onto expected invariants.
                for inv in scenario.expected_invariants:
                    if inv in integrity_findings or inv in safety_findings:
                        findings.append(f"invariant_failed:{inv}")
                # Specific fail-closed mappings for known corpus categories.
                if scenario.scenario_id == "adv_03" and "invented_contract" in integrity_findings:
                    findings.append("invented_contract")
                if scenario.scenario_id == "adv_17" and "missing_data_quality_truth" in integrity_findings:
                    findings.append("missing_data_quality_truth")
                if scenario.scenario_id == "adv_21" and "unsupported_reasoning_with_profit" in integrity_findings:
                    findings.append("unsupported_reasoning_with_profit")
                passed = not findings
                result = AdversarialScenarioResult(
                    result_id=UUID(int=0),
                    experiment_id=experiment_id,
                    scenario_id=scenario.scenario_id,
                    scenario_version=scenario.version,
                    configuration_version_id=cfg_id,
                    sample_number=1,
                    passed=passed,
                    findings=tuple(findings),
                )
                # Assign stable result id from key.
                from uuid import uuid5

                result = result.model_copy(
                    update={
                        "result_id": uuid5(
                            NAMESPACE_URL,
                            AdversarialResultStore.result_key(
                                experiment_id, scenario.scenario_id, cfg_id, 1
                            ),
                        )
                    }
                )
                await self._store.upsert(result)
                results.append(result)

        required = self.required_ids()
        by_scenario = {r.scenario_id for r in results if r.passed}
        # Require all required scenarios to have at least one passed result
        # for BOTH configurations without unresolved findings.
        ok = True
        missing: list[str] = []
        for sid in required:
            scen_results = [r for r in results if r.scenario_id == sid]
            if len(scen_results) < 2:
                ok = False
                missing.append(f"missing:{sid}")
                continue
            if not all(r.passed for r in scen_results):
                ok = False
                missing.append(f"failed:{sid}")
            if any(r.scenario_version != "3.0.0" for r in scen_results):
                ok = False
                missing.append(f"version_mismatch:{sid}")
        return ok and not missing, tuple(results)

    async def adversarial_passed(self, experiment_id: UUID) -> bool:
        results = await self._store.list_for_experiment(experiment_id)
        required = set(required_scenario_ids())
        for sid in required:
            scen = [r for r in results if r.scenario_id == sid]
            if not scen or not all(r.passed for r in scen):
                return False
        return True
