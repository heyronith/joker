# Joker Project Status

| Item | Status |
| --- | --- |
| Task 1 | approved |
| Task 2 | approved |
| Task 3 | approved project foundation |
| Goal-driven objective layer | implemented |
| Current phase | historical EV leakage/provenance correction (**not approved**) |
| Current branch | `task3-agent-evolution` |
| Comparison base | `fde7a2aec88b2a9e9ca963f837f162479e202cca` |
| Verified code commit | `33678b1c6ac1b2a8da4232e3d4645061264ce09a` |
| Acceptance path | **local** `python scripts/local_verify.py` → `artifacts/local-verify/<sha>/` |
| GitHub Actions | `workflow_dispatch` only — **not** an acceptance gate |
| Latest local evidence (3.12) | `33678b1` — **PASS** |
| Paper-only status | enforced (live trading disabled) |
| Textual UI status | not started / not modified in this phase |
| Known unresolved items | independent review of production EV path at `33678b1`; phase remains unapproved |

## Local verification evidence (`33678b1`)

| Field | Value |
| --- | --- |
| Verified code SHA | `33678b1c6ac1b2a8da4232e3d4645061264ce09a` |
| Result | **PASS** |
| Manifest | `artifacts/local-verify/33678b1c6ac1b2a8da4232e3d4645061264ce09a/manifest.json` |
| Python | 3.12 (`.venv`) |
| Focused | 67 passed ×3 |
| Full suite | 849 passed ×1 |
| Soak | 8 passed ×1 (`live_runner` + `production_promotion_rollback`) |
| Ruff | ok |
| CLI | `joker --help`, `joker evolve --help` ok |
| SQLite integrity | ok |
| Git during verify | clean implementation tree at `33678b1` |

## Correction summary

1. Challengers persist `training_dataset_ids` / `challenger_dataset_ids` / `evaluation_dataset_ids` with `dataset_provenance_status` (`not_applicable` \| `resolved` \| `unknown`). Empty tuples never auto-resolve for non-bootstrap configs. Unknown active provenance fail-closes historical EV. Promotion preserves IDs unchanged.
2. Strategy family is never invented from agent role; missing family → `historical_strategy_family_missing` / EV ineligible.
3. Anchored `verify_event_horizon` with sequence policy; session index assigns exchange-time-ordered sequences; loader prefers entry→terminal sequence range.

## Historical-EV phase status

**Not complete / not approved.** Independent review should verify:

```text
Task 3 dataset
→ configuration records exact dataset provenance
→ promotion preserves provenance
→ active configuration exposes blocked training datasets
→ historical loader excludes only configuration-related leakage
→ compiler uses explicit strategy family
→ anchored complete event horizon
→ supported historical EV
→ price-safe paper execution
```

GitHub Actions is not required for acceptance. Do not claim real-money readiness.
