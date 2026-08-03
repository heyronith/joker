# Joker Project Status

| Item | Status |
| --- | --- |
| Task 1 | approved |
| Task 2 | approved |
| Task 3 | approved project foundation |
| Goal-driven objective layer | implemented |
| Current phase | historical EV entry-anchor / training-provenance correction (**not approved**) |
| Current branch | `task3-agent-evolution` |
| Comparison base | `33678b1c6ac1b2a8da4232e3d4645061264ce09a` |
| Verified code commit | `c58aa933f6e36a2f483010711cbf321c2f2f555e` |
| Acceptance path | **local** SHA-tied evidence under `artifacts/local-verify/<sha>/` |
| GitHub Actions | `workflow_dispatch` only — **not** an acceptance gate |
| Latest local evidence (3.12) | `c58aa93` — **PASS** |
| Paper-only status | enforced (live trading disabled) |
| Textual UI status | not started / not modified in this phase |
| Known unresolved items | independent review of production EV path at `c58aa93`; phase remains unapproved |

## Local verification evidence (`c58aa93`)

| Field | Value |
| --- | --- |
| Verified code SHA | `c58aa933f6e36a2f483010711cbf321c2f2f555e` |
| Result | **PASS** |
| Manifest | `artifacts/local-verify/c58aa933f6e36a2f483010711cbf321c2f2f555e/manifest.json` |
| Python | 3.12 (`.venv`) |
| Focused | 48 passed ×3 (dataset provenance, episode compiler, horizon verify, strategy family, live runner) |
| Full suite | 861 passed ×1 |
| Soak | 9 passed ×1 (`live_runner` + `production_promotion_rollback`) |
| Ruff | ok |
| CLI | `joker --help`, `joker evolve --help` ok |
| SQLite integrity | ok |
| Git during verify | clean implementation tree at `c58aa93` |

## Correction summary

1. `verify_event_horizon` fails closed when `entry_event_id` or `terminal_event_id` is `None` (`authoritative_horizon_entry_missing` / `authoritative_horizon_terminal_missing`); no skip-on-None.
2. EpisodeCompiler no longer invents entry from the first window event; missing entry causation → incomplete, EV/promotion ineligible, truth degraded.
3. Entry `causation_event_id` flows `OrderActionRequest` → `ExecutionProvenanceRecord` (decision-completed → proposal → cycle trigger; never a fill).
4. Production fixtures (`persist_compiler_produced_history`) use exact factual entry/terminal anchors; horizon loaders do not mint missing IDs.
5. `resolve_dataset_provenance_status`: non-bootstrap `resolved` requires `training_dataset_ids`; challenger/evaluation IDs alone → `unknown`. Explicit `construction_method` (`bootstrap` / `human_static`) for genuine no-training cases; agent challengers never auto `not_applicable`.

## Historical-EV phase status

**Not complete / not approved.** Independent review should verify:

```text
explicit entry causation event
→ exact entry-to-terminal horizon
→ anchored and contiguous factual episode
→ explicit strategy family
→ active configuration with known training datasets
→ leakage-safe historical sample
→ supported positive EV
→ price-covered paper execution
```

GitHub Actions is not required for acceptance. Do not claim real-money readiness.
Do not merge into `main`.
