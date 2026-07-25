# Webull OpenAPI Market-Data Endpoint Audit

Last updated: 2026-07-01 (Phase 20)

This document records joker's Webull market-data endpoint contract audit. **No trading, account, or order endpoints** are included.

Base URLs (from Webull docs):

| Environment | Base URL |
|-------------|----------|
| UAT | `https://us-openapi-alb.uat.webullbroker.com` |
| Production | `https://api.webull.com` |

Authentication for market-data v2 uses **HMAC-SHA1 signed headers** (`x-app-key`, `x-app-secret`, `x-timestamp`, `x-signature-nonce`, `x-access-token`, `x-version`, `x-signature`). Legacy OAuth token fetch uses unsigned `POST /openapi/oauth/token`.

Central registry: `joker/data/webull_endpoints.py`

---

## Stock snapshot

| Field | Value |
|-------|-------|
| Method | `GET` |
| Path | `/openapi/market-data/stock/snapshot` |
| Required params | `symbols` (comma-separated), `category=US_STOCK` |
| Required headers | Signed market-data v2 headers |
| Response shape | JSON **array** of objects: `symbol`, `price`, `bid`, `ask`, `quote_time`, `volume`, … |
| Rate limit | 60 req/min (per docs) |
| Verified | **true** |
| Docs | https://developer.webull.com/apis/docs/reference/stock-snapshot |

### joker fields

**Required:** `price` (or `close`/`last`), `bid`, `ask`, `quote_time`/`timestamp`

**Optional:** `volume`, delayed flag

---

## Stock candles

| Field | Value |
|-------|-------|
| Method | `GET` |
| Path | `/openapi/market-data/stock/bars` |
| Required params | `symbol` (singular), `category=US_STOCK`, `timespan` (`M1`, …) |
| Required headers | Signed market-data v2 headers |
| Response shape | Array of OHLCV bar objects (often newest-first) |
| Rate limit | 60 req/min |
| Verified | **true** (live-checked 2026-07-24: `symbol`+`M1` works; `symbols` returns 400) |
| Docs | https://developer.webull.com/apis/docs/reference/stock-historical-bars |

joker maps `1m` → `M1` via `normalize_stock_timespan`.

---

## Stock streaming

| Field | Value |
|-------|-------|
| Method | `MQTT` |
| Path | `data-api.webull.com` |
| Verified | **true** (documented; joker uses HTTP polling via stock snapshot) |
| Docs | https://developer.webull.com/apis/market-data-api/getting-started |

---

## Option contract discovery

| Field | Value |
|-------|-------|
| Method | `GET` |
| Path | *(no official endpoint in Webull OpenAPI docs as of 2026-03)* |
| Verified | **false** |
| Docs | unavailable |

**joker fallback:** OSI symbol construction (`joker/data/webull_option_symbols.py`) for verification/capture when chain API is unverified. Format: `SPY260701C00550000` (underlying + YYMMDD + C/P + strike×1000 padded to 8 digits).

---

## Option snapshot

| Field | Value |
|-------|-------|
| Method | `GET` |
| Path | `/openapi/market-data/option/snapshot` |
| Required params | `symbols` (OSI codes, comma-separated, max 20), `category=US_OPTION` |
| Required headers | Signed market-data v2 headers |
| Response shape | JSON **array**: `symbol`, `instrument_id`, `bid`, `ask`, `price`, `quote_time`, `volume`, `open_interest`, `imp_vol`, `delta`, `gamma`, `theta`, `vega`, `strike_price` |
| Rate limit | 60 req/min |
| Verified | **true** |
| Docs | https://developer.webull.com/apis/docs/reference/option-snapshot |

Request identifier: **OSI symbol** in `symbols` param (not legacy `/market/option/quote/{id}`).

---

## Option batch snapshot

Same endpoint as option snapshot — pass up to 20 comma-separated OSI symbols.

| Verified | **true** |

---

## Option bars

| Field | Value |
|-------|-------|
| Method | `GET` |
| Path | `/openapi/market-data/option/bars` |
| Required params | `symbols`, `category=US_OPTION`, `timespan` (`M1`, `M5`, …) |
| Rate limit | 1 call/sec per App Key (per docs) |
| Verified | **true** |
| Docs | https://developer.webull.com/apis/docs/reference/option-historical-bars |

---

## Option ticks

| Field | Value |
|-------|-------|
| Method | `GET` |
| Path | `/openapi/market-data/option/tick` |
| Required params | `symbols`, `category=US_OPTION` |
| Verified | **true** |
| Docs | https://developer.webull.com/apis/docs/reference/option-tick |

---

## Known limitations

1. **No official option chain API** — contract discovery via chain endpoint is disabled; use OSI construction for 0DTE verification.
2. **Stock bars** verified with `symbol` + `timespan=M1` (do not use `symbols`).
3. **MQTT streaming** not implemented in joker; polling uses stock snapshot.
4. **Shadow mode** refuses options unless `data/capabilities/webull_options_capability.json` says `usable_for_shadow=true` or in-session verification succeeds.
5. **Rate limits** enforced client-side via `RateLimiter` (60/min default).

---

## Subscription / permission requirements

- Webull OpenAPI **market-data subscription** (Advanced Quotes) required for live bid/ask on options.
- HTTP 403 / permission errors classified as `subscription/permission` (not endpoint mismatch).
- HTTP 404 classified as `endpoint mismatch`.

---

## Fields joker requires (options shadow)

| Field | Required |
|-------|----------|
| Contract identity (OSI symbol) | yes |
| Bid | yes |
| Ask | yes |
| Quote timestamp | yes |
| Volume | optional |
| Open interest | optional |
| IV (`imp_vol`) | optional |
| Greeks (delta, gamma, theta, vega) | optional |
| Delayed flag | informational (warning, not blocker for diagnostics) |

---

## Verification commands

```bash
joker data diagnose-options --provider webull --symbol SPY
joker data capture-webull-contract --symbol SPY --include-options
joker data verify-webull-options --symbol SPY
```

Outputs:

- `data/captures/webull_contract/` — redacted response-shape captures
- `reports/webull/options_verification_<date>.md` — verification report
- `data/capabilities/webull_options_capability.json` — persisted capability cache
