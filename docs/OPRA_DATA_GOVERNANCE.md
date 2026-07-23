# OPRA Data Governance (Phase 21A / 21A.1)

This document describes how joker complies with Webull OPRA Non-Display guidance for personal algorithmic use, and how Webull **stock** data is kept separate from **OPRA options** data.

## Webull support interpretation

Webull confirmed the following for this project:

| Activity | Allowed |
|----------|---------|
| In-memory processing of OPRA for private personal algo calculations | Yes |
| Local private terminal display of raw/derived OPRA for the user | Yes |
| Persistent local storage of raw OPRA market data | **No** |
| Storage of non-price decision metadata | Yes |
| Sending raw OPRA to third-party AI APIs (OpenAI) | **No** |
| Sending non-price summaries, decision metadata, and app state to AI | Yes |
| In-memory-only raw OPRA architecture | Yes |

## Source taxonomy

| Source label | Meaning | Classification |
|--------------|---------|----------------|
| `webull_stock` | Real Webull SPY stock quotes/candles | `STOCK_MARKET_DATA` |
| `webull_opra` | Real Webull OPRA option snapshots/quotes | `RAW_OPRA` |
| `webull_opra_safe` | Non-price metadata derived from OPRA | `NON_PRICE_DECISION_METADATA` |
| `synthetic_stock` | Synthetic SPY replay stock events | `SYNTHETIC_DATA` |
| `synthetic_option` | Synthetic option replay events | `SYNTHETIC_DATA` |
| `mock_stock` / `mock_option` | Offline mock fixtures | `SYNTHETIC_DATA` |

**Important:** `source="webull"` alone is ambiguous and must not be used for new events. Stock and OPRA paths use explicit labels.

## Joker implementation rules

1. **Classification** — `classify_market_event()` uses explicit `data_classification`, event type, asset class, and source label. Generic `webull` is not treated as OPRA.
2. **Persistence boundary** — JSONL, SQLite, captures, and reports call `sanitize_for_persistence()`. Stock bid/ask may persist; OPRA bid/ask may not.
3. **OpenAI boundary** — Agent context passes through `audit_and_sanitize_openai_context()`. Stock regime summaries are allowed; OPRA prices are not.
4. **TUI** — Raw OPRA values and exact contract IDs exist in `display_state_ephemeral()` only. `persisted_state_safe()` has no prices or OSI/contract IDs.
5. **Shadow mode** — OPRA used in memory; only safe metadata persisted.

## Allowed persisted stock metadata (examples)

- SPY price, bid, ask (when `source=webull_stock`, `STOCK_MARKET_DATA`)
- Candle OHLCV
- Stock-derived technical features: trend label, VWAP distance, momentum
- Stock regime summaries for agents

## Allowed persisted OPRA metadata (examples)

- `spread_check`: PASS / FAIL
- `freshness_check`: PASS / FAIL
- `bid_ask_available`, `greeks_available`, `iv_available`, `volume_available`, `open_interest_available`
- `contract_quality`: PASS / FAIL
- `contract_role`: ATM_CALL / ATM_PUT / etc.
- `selected_direction`: long_call / long_put
- `expiration_type`: 0DTE
- `moneyness_bucket`: ATM / ITM / OTM
- `contract_selected`: true / false
- `risk_reason_code`, `setup_id`, `shadow_result_label`, `exit_reason`

## Not allowed persisted OPRA values

- `bid`, `ask`, `mid`, `last`, `spread_pct`
- `volume`, `open_interest`, `implied_volatility`
- `delta`, `gamma`, `theta`, `vega`
- Option `quote_timestamp`
- Exact `contract_id`, OSI symbol, or `instrument_id` (TUI may show ephemerally)
- Exact simulated option entry/exit prices or P&L from real OPRA sessions

Synthetic replay labeled `synthetic_option` may still contain synthetic option quotes for offline testing.

## Compliance commands

```bash
joker compliance scan-local-opra
joker compliance quarantine-opra-artifacts
joker shadow preflight --provider webull --symbol SPY
joker shadow run --provider webull --symbol SPY --duration-minutes 30
```

The scanner auto-discovers SQLite DB paths from config and `data/*.db`. Output categories:

- `possible_raw_opra` — needs review/quarantine
- `stock_data_not_opra` — Webull stock bid/ask (not a violation)
- `synthetic_ignored` — labeled synthetic replay
- `safe_metadata` — pass/fail and availability booleans only

## Key modules

- `joker/compliance/data_classification.py` — source taxonomy and `classify_market_event()`
- `joker/compliance/opra_sanitizer.py` — persistence/OpenAI/report sanitizers
- `joker/compliance/opra_scanner.py` — local artifact scanner with categories
- `joker/compliance/openai_audit.py` — OpenAI input audit events

## Before real shadow mode

1. `joker compliance scan-local-opra` — zero `possible_raw_opra` violations
2. `joker shadow preflight --provider webull --symbol SPY`
3. `pytest` — all governance tests pass
