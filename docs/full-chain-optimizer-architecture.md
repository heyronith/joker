# Goal-conditioned full-chain optimizer

## Scope and safety

This architecture is enabled only by an explicit `full_chain_optimizer.enabled`
configuration. The repository default remains disabled; `config/paper.yaml`
enables it for paper/replay research. It does not enable live trading,
`WEBULL_LIVE_TRADING_ENABLED`, or any production order path.

The optimizer estimates, but does not guarantee, the probability of reaching a
session objective. Its probabilities are heuristic unless a result explicitly
identifies contract-level empirical calibration.

## Authority boundaries

1. Strategy inventor agents own a market thesis: family, direction, expected
   underlying and adverse paths, horizon, conditions, confidence, and evidence.
   `candidate_legs` remain backward-compatible hints only.
2. `full_chain_universe.py` owns executable contract discovery. Every contract
   ID comes from the exact option surface linked to the graph snapshot.
3. `contract_outcomes.py` owns contract-specific scenario estimates.
4. `portfolio_search.py` owns quantity expansion, bounded portfolio ranking,
   and ENTER versus WAIT selection.
5. Debate and meta-decision agents review evidence. They may oppose or force a
   recalculation, but cannot replace an authorized strategy, contract,
   quantity, position count, or capital allocation.
6. The entry tactician owns limit/timing tactics only.
7. Deterministic execution validation and the order gateway retain final
   authority over quote freshness, capital, objective version, position limits,
   broker identity, data quality, idempotency, and kill switches.

## Full-chain truth boundary

The universe builder validates all linked SPY 0DTE surface rows before ranking:

- non-empty contract identity;
- SPY underlying and current trading-date expiry;
- positive, non-crossed bid/ask;
- quote age and execution quality;
- configured relative spread;
- single-contract affordability.

Input order never determines output order. If the eligible set exceeds the
configured evaluation bound, deterministic round-robin stratification preserves
representation across option type, ITM/ATM/OTM, premium, delta, distance from
spot, and liquidity buckets. Cheap valid contracts are not excluded because of
price. Full raw surfaces are not sent to model providers.

## Contract outcome estimation

Each strategy × contract pair is evaluated separately. When provider Greeks and
implied volatility are present, a bounded shared-underlying scenario grid uses
delta, gamma, theta, intrinsic value, quote spread, strike, premium, and horizon.
When Greeks are missing, a conservative intrinsic-value/quote fallback is used
and labeled `quote_intrinsic_fallback`. If required truth or a defensible
scenario grid is unavailable, the estimate is `unknown` and excluded from
ranking.

No strategy-level win rate is copied to every contract. Contract-level empirical
rates may be blended only when an actual sample count and evidence are supplied.
The implementation never invents Greeks, sample counts, volatility, evidence,
or historical observations.

## Quantity and portfolio search

Every affordable quantity from one through the configured contract cap is
retained in the auditable grid. Each row records capital, maximum loss,
scenario-based goal probability, WAIT probability, uncertainty, feasibility,
and selection state.

Portfolio search supports long, single-leg SPY 0DTE components. It does not
create naked shorts or option spreads. A deterministic bounded search combines
component P&L on common underlying scenarios; it does not multiply independent
contract probabilities. Capital, available position slots, duplicate exposure,
concentration, liquidity, spread, resolution, and tail-risk tie-breaks are
enforced. Search bounds are configuration and evidence fields.

## WAIT comparison

WAIT is generated on every eligible cycle with zero deployed capital. Its
opportunity estimate decays from the durable objective duration and current
exchange-clock time remaining. ENTER requires at least the configured
probability improvement over WAIT. Decisions persist both probabilities,
their delta, reason codes, objective version, snapshot ID, and time remaining.

## Authoritative decision and execution

`AuthorizedPositionTuple`, `PortfolioAttainmentEvaluation`, and
`TargetPortfolioDecision` preserve exact strategy, contract, quantity,
evaluation premium, allocation, decision, objective-version, and snapshot
provenance. The first tuple is also projected into legacy target-attainment
channels when only one position is authorized.

For multiple components, proposals are compiled into deterministic single-leg
child intents and submitted sequentially. Before each child, objective,
remaining capital, account/order projection, snapshot-linked surface, data
quality, and quote truth are refreshed. A changed tuple, material repricing,
working-order conflict, stale snapshot, deadline, achieved target, or capital
shortfall stops the remaining sequence and requests full reoptimization.
Components are never silently resized, substituted, dropped, or added.

## Observable graph CLI

`joker paper run --graph-view compact|verbose|json` renders the same structured
event payloads used by the JSONL evidence stream. Verbose output includes goal,
market, thesis, contract, portfolio, debate, decision, and execution evidence.
The stream contains summaries and cited evidence identifiers, not hidden
chain-of-thought, raw prompts, credentials, account IDs, or raw OPRA payloads.

Stable events include:

- `graph.cycle.started`
- `strategy.thesis.generated`
- `chain.universe.built`
- `contract.outcome.estimated`
- `contract.grid.scored`
- `portfolio.grid.scored`
- `debate.review.completed`
- `target.portfolio.selected`
- `target.wait.selected`
- `execution.revalidation`
- `execution.reoptimization_required`
- `graph.cycle.completed`

## Approximation and calibration limitations

- Scenario probabilities are coarse heuristic weights, not statistically
  calibrated forecasts.
- Greek-based local approximations can be inaccurate for large 0DTE moves,
  volatility jumps, discontinuous liquidity, and near-expiry gamma.
- The fallback model intentionally understates residual time value and may rank
  valid opportunities as unknown or unattractive.
- Fill probability, queue position, slippage, halts, latency, partial fills, and
  stop execution remain execution risks; estimated option-price scenarios are
  not fill promises.
- Bounded search can omit the global optimum when configured beam or candidate
  limits are too small. Limits and resulting candidate counts are auditable.
- Portfolio scenario dependence is shared-underlying correlation, not a full
  joint volatility-surface model.

These limitations must be addressed with replay calibration and paper evidence
before any separate live-money review. Passing tests does not constitute paper
or production approval.
