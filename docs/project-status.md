# Joker Project Status

| Item | Status |
| --- | --- |
| Task 1 | approved |
| Task 2 | approved |
| Task 3 | approved project foundation |
| Goal-driven objective layer | implemented |
| Historical-EV architecture | implemented at `c58aa93` (pending independent review) |
| Current phase | live trading implementation Steps 1+2 (**not operationally approved**) |
| Current branch | `live-trading-implementation` |
| Approved historical-EV base | `c58aa933f6e36a2f483010711cbf321c2f2f555e` |
| Step 1 SHA | `778d7d9d873d016538b8d21245bcf9137e6659a3` |
| Step 2 SHA / verified tip | `e36f09f6980dfa637dd167d880f6fbc592a91cca` |
| Acceptance path | **local** SHA-tied evidence under `artifacts/local-verify/<sha>/` |
| GitHub Actions | `workflow_dispatch` only — **not** an acceptance gate |
| Latest local evidence (3.12) | `e36f09f` — **PASS** |
| Paper-only default | enforced unless LIVE_GATED + live flags + LiveActivation |
| Textual UI / paper-live selector | not built |
| Real-money operational approval | **not claimed** — market-open acceptance pending |

## Local verification evidence (`e36f09f`)

| Field | Value |
| --- | --- |
| Verified code SHA | `e36f09f6980dfa637dd167d880f6fbc592a91cca` |
| Result | **PASS** |
| Manifest | `artifacts/local-verify/e36f09f6980dfa637dd167d880f6fbc592a91cca/manifest.json` |
| Python | 3.12 (`.venv`) |
| Focused | 46 passed ×3 |
| Full suite | 907 passed ×1 |
| Recovery soak | 9 passed ×1 |
| Ruff | ok |
| CLI | `joker --help`, `broker --help`, `live --help`, `live preflight --help` ok |
| SQLite integrity | ok |
| Git during verify | clean at `e36f09f` |

## Implementation summary

### Step 1 — Production Webull execution subsystem
- Separate `WebullLiveClient` (paper `WebullClient` unchanged / still refuses live)
- Explicit `WEBULL_LIVE_*` credentials via `live_credentials_env()` (no paper fallback)
- Order preview, options `position_intent`, durable `broker_submission_journal`
- Ambiguous timeout → `submission_unknown` + order-detail reconcile
- `BrokerReconciliationService` with typed mismatch classifications

### Step 2 — Agentic live runtime
- Shared `prepare_agentic_trading_session` used by paper and live wrappers
- `LiveTradingRunner` + process-local `LiveActivation`
- Live preview inside `OrderActionGateway` (no agent bypass)
- Read-only `joker live preflight`
- Paper/live command equivalence tests (capture mode)

## Capability matrix

| Capability | Offline verified | Sandbox verified | Production read-only verified | Production order verified |
| --- | ---: | ---: | ---: | ---: |
| Authentication | yes | pending | pending | no |
| Account identity | yes | pending | pending | no |
| Preview | yes | pending | pending | no |
| Placement | yes (mock) | pending | n/a | no |
| Cancellation | yes (mock) | pending | n/a | no |
| Partial fill | yes (mock/status) | pending | n/a | no |
| Position reconciliation | yes | pending | pending | no |
| Restart recovery | yes | pending | n/a | no |
| Agent-managed exit | yes (intent/payload) | pending | n/a | no |
| Episode compilation | yes (existing path) | pending | n/a | no |

## Not claimed

- Real-money operational readiness
- Market-open production order verification
- Final CLI paper/live selector UI
- Merge to `main`
