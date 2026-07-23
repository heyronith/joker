# Phase 0–14 Scaffold Audit

Audit date: 2026-07-01  
Scope: joker trading research system before Phase 15 OpenAI integration.

## What is real (production-intended)

| Component | Status | Notes |
|-----------|--------|-------|
| Config loading (YAML + env) | Real | Merged profiles, validation, secret redaction |
| Safety modes (PAPER/SHADOW/LIVE_GATED) | Real | Fail-closed; live requires explicit opt-in |
| SQLite storage | Real | Runs, decisions, orders, fills, positions, day state |
| JSONL event log | Real | Append-only, redacts secrets |
| Pydantic domain schemas | Real | Typed contracts with `schema_version` |
| Paper broker | Real | Local simulation with slippage, fills, P&L |
| Risk governor | Real | Deterministic; agent override ignored |
| Feature engine | Real | VWAP, levels, momentum, stale detection |
| Reactive state machine | Real | Validated transitions, risk handoff |
| Playbook patch validation | Real | Cannot loosen hard risk / kill switch |
| Shadow runtime | Real | Records would-trades, no broker submit |
| Postmarket reports | Real | Generated from DB fixtures |
| Textual TUI shell | Real | Panels, navigation, communicator input |

## What is stubbed

| Component | Status | Notes |
|-----------|--------|-------|
| Webull adapter | Stub | `LIVE_CALLS_ENABLED = False`; endpoints unverified |
| Live broker execution | Stub | No live order path implemented |
| Market data streaming | Stub | `mock_spy_snapshot()` only |
| Intraday council scheduling | Stub | Patch logic exists; no live scheduler |
| Replay fixture provider | Partial | E2E uses mock snapshot, not full replay files |

## What is mocked (pre–Phase 15)

| Component | Status | Notes |
|-----------|--------|-------|
| Agent council (Phase 7) | Mock | Deterministic rules, not OpenAI |
| CommunicatorAgent | Mock | Template strings from local state |
| Premarket market context | Mock | Feature engine on synthetic snapshot |

Phase 15 replaces mock council with OpenAI-backed agents when `mock_agents: false`.

## Test quality assessment (pre–Phase 15)

### Strong tests (behavioral)

- `test_risk.py` — exercises each rejection reason code
- `test_paper_broker.py` — fill, cancel, P&L, no Webull import
- `test_storage.py` — DB roundtrip, secret redaction in JSONL
- `test_reactive.py` — state transitions, risk blocks orders
- `test_e2e_paper_day.py` — full premarket → trade → report path
- `test_config.py` — mode fail-closed, live gating, model validation

### Shallow tests ( strengthened in Phase 15)

- `test_tui.py` — was mostly import/state; now checks panel content, kill switch, navigation wrap
- `test_agents.py` — was source-string grep; now validates playbook rules, factory mock default
- `test_webull.py` — env gating only (appropriate for stub adapter)

### Acceptable smoke tests

- `test_schemas.py` — parsing/roundtrip (appropriate for schema layer)
- `test_features.py` — deterministic VWAP (appropriate)

## Unsafe for production

1. **Webull live calls** — disabled by design; endpoints not verified.
2. **Mock market data** — prices/regimes not from live feed.
3. **Mock agents (default)** — deterministic, not market-adaptive AI.
4. **No authentication** — local single-user assumed.
5. **SQLite** — fine for local research; not multi-user production DB.
6. **Communicator** — must not invent prices; Phase 15 enforces local-state-only answers.

## Before shadow mode (operational)

- [ ] Replace mock market snapshots with real or recorded replay data
- [ ] Wire TUI to live runtime state (DB + event log)
- [ ] Run full session in SHADOW with real-time data feed
- [ ] Verify shadow runtime never calls `broker.submit_order`
- [ ] Validate quote freshness thresholds against real SPY 0DTE spreads
- [ ] Enable OpenAI agents (`mock_agents: false`) and validate playbook quality manually

## Before live-gated mode

- [ ] Complete Webull endpoint verification with paper account tests
- [ ] Implement order status / partial fill handling
- [ ] Add explicit user confirmation step before each live order
- [ ] Pen-test prompt injection on all agent inputs
- [ ] Run shadow mode for minimum agreed observation period
- [ ] Document and test kill switch / max loss halt procedures
- [ ] Set `live_trading_enabled: true` only in verified `live.yaml`
- [ ] Never allow LLM output to skip RiskGovernor

## Phase 15 additions

- `LLMClient` interface with `MockLLMClient` and `OpenAILLMClient`
- `mock_agents` config flag (default `true`)
- Structured Pydantic outputs for all OpenAI agent calls
- Prompt-injection detection on user input
- Agent output validation (no BrokerOrder, no risk loosening)
- Communicator uses local state only; refuses trade advice
