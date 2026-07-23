# Webull Options Verification — 2026-07-01

Symbol: **SPY**
Checked at: 2026-07-01T23:55:56.841462+00:00

## Credentials & Auth
- Credentials present: yes
- Auth passed: no

## Endpoint Status
- option_bars: **verified**
- option_chain: **unverified**
- option_snapshot: **verified**
- option_tick: **verified**
- stock_bars: **unverified**
- stock_snapshot: **verified**
- stock_streaming: **verified**

## Market Data Results
- SPY stock snapshot: fail
- Same-day expiration found: no
- ATM call snapshot: fail
- ATM put snapshot: fail

## Required Fields
- Bid/ask: no
- Quote timestamp: no
- Contract identity: no

## Optional Fields
- Volume: no
- Open interest: no
- IV: no
- Greeks: no

## Usability
- Usable for joker shadow mode: **no**
- Usable for paper simulation: **no**
- External options provider needed: **yes**

## Issues
- Likely issue: auth failure
- Missing required: bid/ask, ATM call/put snapshot, same-day expiration contracts

## Next Action
Fix authentication — verify token and signed request headers

> No secrets are included in this report.
