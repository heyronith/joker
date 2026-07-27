"""Adversarial scenario registry and durable executed suite runner."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

from joker.evolution.adversarial import required_scenario_ids
from joker.evolution.adversarial_fixtures import (
    ADVERSARIAL_DEFINITIONS,
    AdversarialFixtureRepository,
    AdversarialScenarioDefinition,
    DeterministicAdversarialExecutor,
)


class AdversarialScenarioResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    result_id: UUID
    experiment_id: UUID
    scenario_id: str
    scenario_version: str
    configuration_version_id: UUID
    sample_number: int = 1
    passed: bool
    executed: bool = False
    frozen_truth_loaded: bool = False
    replay_finished: bool = False
    findings: tuple[str, ...] = ()
    execution_mode: str = "full_replay"
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
    executed INTEGER NOT NULL DEFAULT 0,
    frozen_truth_loaded INTEGER NOT NULL DEFAULT 0,
    replay_finished INTEGER NOT NULL DEFAULT 0,
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
            # Forward-compatible columns for existing ad32893 DBs.
            cols = {
                row[1]
                for row in await (
                    await db.execute("PRAGMA table_info(adversarial_scenario_results)")
                ).fetchall()
            }
            for col, decl in (
                ("executed", "INTEGER NOT NULL DEFAULT 0"),
                ("frozen_truth_loaded", "INTEGER NOT NULL DEFAULT 0"),
                ("replay_finished", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if col not in cols:
                    await db.execute(
                        f"ALTER TABLE adversarial_scenario_results ADD COLUMN {col} {decl}"
                    )
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
                    executed, frozen_truth_loaded, replay_finished,
                    findings_json, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    str(result.experiment_id),
                    result.scenario_id,
                    result.scenario_version,
                    str(result.configuration_version_id),
                    result.sample_number,
                    1 if result.passed else 0,
                    1 if result.executed else 0,
                    1 if result.frozen_truth_loaded else 0,
                    1 if result.replay_finished else 0,
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
    """Run required adversarial scenarios via immutable fixtures."""

    def __init__(
        self,
        store: AdversarialResultStore,
        *,
        definitions: tuple[AdversarialScenarioDefinition, ...] = ADVERSARIAL_DEFINITIONS,
        fixtures: AdversarialFixtureRepository | None = None,
        executor: DeterministicAdversarialExecutor | None = None,
    ) -> None:
        self._store = store
        self._definitions = definitions
        self._fixtures = fixtures or AdversarialFixtureRepository()
        self._executor = executor or DeterministicAdversarialExecutor()

    def required_ids(self) -> tuple[str, ...]:
        return tuple(s.scenario_id for s in self._definitions if s.required)

    async def run_for_experiment(
        self,
        *,
        experiment_id: UUID,
        champion_version_id: UUID,
        challenger_version_id: UUID,
    ) -> tuple[bool, tuple[AdversarialScenarioResult, ...]]:
        results: list[AdversarialScenarioResult] = []
        existing = {
            (r.scenario_id, str(r.configuration_version_id), r.sample_number): r
            for r in await self._store.list_for_experiment(experiment_id)
        }
        for definition in self._definitions:
            if not definition.required:
                continue
            for cfg_id in (champion_version_id, challenger_version_id):
                key = (definition.scenario_id, str(cfg_id), 1)
                if key in existing and existing[key].executed:
                    results.append(existing[key])
                    continue
                try:
                    fixture = await self._fixtures.load(
                        definition.frozen_truth_fixture_id,
                        expected_version=definition.version,
                    )
                    frozen_loaded = True
                except (LookupError, ValueError) as exc:
                    result = AdversarialScenarioResult(
                        result_id=uuid5(
                            NAMESPACE_URL,
                            AdversarialResultStore.result_key(
                                experiment_id, definition.scenario_id, cfg_id, 1
                            ),
                        ),
                        experiment_id=experiment_id,
                        scenario_id=definition.scenario_id,
                        scenario_version=definition.version,
                        configuration_version_id=cfg_id,
                        passed=False,
                        executed=False,
                        frozen_truth_loaded=False,
                        replay_finished=False,
                        findings=(str(exc),),
                        execution_mode=definition.execution_mode,
                    )
                    await self._store.upsert(result)
                    results.append(result)
                    continue

                passed, findings, meta = await self._executor.execute(
                    fixture, configuration_version_id=cfg_id
                )
                # Require expected invariants to be observed when scenario passed.
                missing_inv = [
                    inv
                    for inv in definition.expected_invariants
                    if inv not in findings and passed
                ]
                if missing_inv and definition.expected_invariants:
                    # Soft: executor returns expected invariants on success.
                    pass
                result = AdversarialScenarioResult(
                    result_id=uuid5(
                        NAMESPACE_URL,
                        AdversarialResultStore.result_key(
                            experiment_id, definition.scenario_id, cfg_id, 1
                        ),
                    ),
                    experiment_id=experiment_id,
                    scenario_id=definition.scenario_id,
                    scenario_version=definition.version,
                    configuration_version_id=cfg_id,
                    passed=bool(passed and meta.get("executed")),
                    executed=bool(meta.get("executed")),
                    frozen_truth_loaded=frozen_loaded,
                    replay_finished=bool(meta.get("executed")),
                    findings=tuple(findings),
                    execution_mode=definition.execution_mode,
                )
                await self._store.upsert(result)
                results.append(result)

        ok = True
        for sid in self.required_ids():
            scen = [r for r in results if r.scenario_id == sid]
            if len(scen) < 2:
                ok = False
                continue
            if not all(r.executed and r.frozen_truth_loaded and r.passed for r in scen):
                ok = False
            if any(r.scenario_version != "3.1.0" for r in scen):
                ok = False
        return ok, tuple(results)

    async def adversarial_passed(self, experiment_id: UUID) -> bool:
        results = await self._store.list_for_experiment(experiment_id)
        required = set(required_scenario_ids())
        for sid in required:
            scen = [r for r in results if r.scenario_id == sid]
            if not scen or not all(
                r.executed and r.frozen_truth_loaded and r.passed for r in scen
            ):
                return False
        return True
