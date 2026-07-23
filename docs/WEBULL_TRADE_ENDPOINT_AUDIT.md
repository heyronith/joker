# Webull Trade Endpoint Audit (paper account)

This document tracks Trade API endpoints used for **Webull paper-account** order
placement. Real-money live trading remains disabled (`WEBULL_LIVE_TRADING_ENABLED=false`).

## Safety gates

| Flag | Meaning |
|------|---------|
| `WEBULL_PAPER_TRADING_ENABLED=true` | Allow paper-account orders |
| `WEBULL_PAPER_ACCOUNT_ID` | Explicit account id (must be paper/sandbox) |
| `WEBULL_LIVE_TRADING_ENABLED` | **Must stay false** — rejected by env validator |

Orders are refused unless the request `account_id` matches `WEBULL_PAPER_ACCOUNT_ID`.

## Documented endpoints (US OpenAPI reference)

| Registry name | Method | Path | Notes |
|---------------|--------|------|-------|
| `broker_account_list` | GET | `/openapi/account/list` | List accounts; pick paper id |
| `broker_account_balance` | GET | `/openapi/assets/balance` | `account_id` query |
| `broker_account_positions` | GET | `/openapi/assets/positions` | `account_id` query |
| `broker_order_place` | POST | `/openapi/trade/order/place` | Body `{account_id, new_orders}` |
| `broker_order_cancel` | POST | `/openapi/trade/order/cancel` | Body `{account_id, client_order_id}` |
| `broker_order_detail` | GET | `/openapi/trade/order/detail` | `account_id` + `client_order_id` |
| `broker_open_orders` | GET | `/openapi/trade/order/open` | `account_id` (+ paging) |

Sources: [Account List](https://developer.webull.com/apis/docs/reference/account-list),
[Place](https://developer.webull.com/apis/docs/reference/common-order-place),
[Cancel](https://developer.webull.com/apis/docs/reference/common-order-cancel),
[Options](https://developer.webull.com/apis/docs/trade-api/options.md).

## Options constraints (fail closed)

- Options: **LIMIT only** (no MARKET)
- Single-leg: `instrument_type=OPTION`, `option_strategy=SINGLE`, `legs[]`
- `client_order_id` ≤ 32 characters (uuid hex)

## Manual smoke (orders only)

```bash
joker broker accounts
joker broker preflight
joker broker smoke-place --strike 600 --option-type call --limit-price 0.01
joker broker smoke-place --strike 600 --option-type call --limit-price 0.01 --confirm-place
```

## Full agentic loop (recommended test)

Once paper account env is set, the live paper session monitors, decides, logs, and
auto-places on the paper account:

```bash
joker paper preflight --require-webull-paper
joker paper run --duration-minutes 30 --use-openai --require-webull-paper
```

`WEBULL_PAPER_TRADING_ENABLED=true` + `WEBULL_PAPER_ACCOUNT_ID` causes
`joker paper run` to use the Webull paper broker automatically.
