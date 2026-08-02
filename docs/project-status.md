# Joker Project Status

| Item | Status |
| --- | --- |
| Task 1 | approved |
| Task 2 | approved |
| Task 3 | approved project foundation |
| Goal-driven objective layer | implemented |
| Current phase | historical EV production truth correction (not approved) |
| Current branch | `task3-agent-evolution` |
| Comparison base (pre this correction) | `8cb47c69b8bbdd2888f3e12b06c892f90cbe4c63` |
| Prior verified code tip | `569a2235a2e713bd5de9b84c3b416f913416cd99` |
| Acceptance path | **local** `python scripts/local_verify.py` (SHA-tied evidence under `artifacts/local-verify/<sha>/`) |
| GitHub Actions | optional `workflow_dispatch` only — **not** an acceptance gate |
| Tip commit (this correction) | pending clean commit after verify tooling |
| Latest local evidence | pending `scripts/local_verify.py` on clean tip |
| Paper-only status | enforced (live trading disabled) |
| Textual UI status | not started / not modified in this phase |
| Known unresolved items | independent review; phase remains unapproved |

## Local verification (acceptance)

Do not treat GitHub Actions push/PR results as acceptance. Reproduce on a clean
working tree at the exact commit SHA:

```bash
# Python 3.12 (example)
.venv/bin/python scripts/local_verify.py

# Python 3.11 (example)
.venv311/bin/python scripts/local_verify.py --python .venv311/bin/python
```

Defaults (same warning gates as former CI): focused ×5, full suite ×3, soak ×3,
Ruff, CLI help smoke, fresh-DB SQLite `PRAGMA integrity_check`.

Evidence: `artifacts/local-verify/<commit_sha>/manifest.json` + `summary.txt`.

## Historical-EV phase status

**Not complete / not approved.** This correction requires production `EpisodeCompiler` factual metadata, fail-closed authoritative event horizons, quote+limit worst-case fill safety, configuration-scoped dataset leakage, and a public session-factory PaperBroker path.

Do not mark the historical-EV phase complete until independent review verifies:

```text
real EpisodeCompiler output
→ complete factual metadata
→ authoritative event horizon
→ leakage-safe historical summary
→ positive EV at maximum execution price
→ fully covered reservation
→ one public-path PaperBroker submission
```

plus green **local** verification evidence for that exact tip SHA on Python 3.11 and 3.12 (GitHub Actions is not required).
