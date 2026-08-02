# Joker Project Status

| Item | Status |
| --- | --- |
| Task 1 | approved |
| Task 2 | approved |
| Task 3 | approved project foundation |
| Goal-driven objective layer | implemented |
| Current phase | historical EV production truth correction (**not approved**) |
| Current branch | `task3-agent-evolution` |
| Comparison base (pre this correction) | `8cb47c69b8bbdd2888f3e12b06c892f90cbe4c63` |
| Verified code commit | `fde7a2aec88b2a9e9ca963f837f162479e202cca` |
| Acceptance path | **local** `python scripts/local_verify.py` → `artifacts/local-verify/<sha>/` |
| GitHub Actions | `workflow_dispatch` only — **not** an acceptance gate |
| Latest local evidence (3.12) | verified code `fde7a2a` — **PASS** (`overall_ok: true`) |
| Docs commit after evidence | documentation-only; does **not** require its own full suite |
| Python 3.11 on verified code | not re-run (optional; prior tip had 830×3 on 3.11) |
| Paper-only status | enforced (live trading disabled) |
| Textual UI status | not started / not modified in this phase |
| Known unresolved items | independent review of production EV path at `fde7a2a`; phase remains unapproved |

## What changed (this acceptance move)

1. **CI is no longer a gate** — `.github/workflows/ci.yml` is `workflow_dispatch` only (no push/PR triggers).
2. **Local acceptance command** — `scripts/local_verify.py` records a SHA-tied `manifest.json` + `summary.txt` (Ruff, CLI smoke, SQLite integrity, focused×5, full×3, soak×3). Same warning gates as former CI (`ResourceWarning` / unhandled thread exceptions as errors). Gates were not weakened.
3. **Harness/runtime soak fix** — `pause_workers` cancels only episode/eval workers (not orchestrator), tracks in-flight compile/eval, and settle waits for persisted evaluations before the next entry so cancel cannot drop dequeued jobs.

## Local verification evidence (`fde7a2a`)

Command:

```bash
.venv/bin/python -u scripts/local_verify.py --python .venv/bin/python
```

| Field | Value |
| --- | --- |
| Verified code SHA (`manifest.commit_sha`) | `fde7a2aec88b2a9e9ca963f837f162479e202cca` |
| Result | **PASS** |
| Manifest (local, gitignored) | `artifacts/local-verify/fde7a2aec88b2a9e9ca963f837f162479e202cca/manifest.json` |
| Python | 3.12.13 (CPython), `.venv/bin/python` |
| Platform | macOS-26.5.2-arm64-arm-64bit |
| SQLite | 3.53.2 — `PRAGMA integrity_check` = **ok** (58 tables after Task1+objectives+Task3 migrations; 0 FK violations) |
| Ruff | ok |
| CLI `python -m joker --help` | ok |
| Focused | 65 passed ×5 |
| Full suite | 830 passed ×3 |
| Soak | 15 passed ×3 (recovery matrix + live runner + kill-switch paths) |
| Git during verify | clean (`git_dirty=false`, porcelain empty) |
| Note | A later documentation-only commit may dirty/advance the tip; independent review targets `fde7a2a`, not the docs commit |

Reproduce later:

```bash
git checkout fde7a2aec88b2a9e9ca963f837f162479e202cca
.venv/bin/python scripts/local_verify.py
```

## Historical-EV phase status

**Not complete / not approved.** Production path still needs independent review:

```text
real EpisodeCompiler output
→ complete factual metadata
→ authoritative event horizon
→ leakage-safe historical summary
→ positive EV at maximum execution price
→ fully covered reservation
→ one public-path PaperBroker submission
```

Green local evidence on the tip SHA is recorded above for Python 3.12. GitHub Actions is not required for acceptance.
