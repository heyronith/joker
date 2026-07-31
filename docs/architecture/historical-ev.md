# Historical EV

## Data sources

Persisted Task 3 `TradingEpisode` + `EpisodeEvaluation` rows.

Production live estimates use factual completed paper `closed_trade` episodes only.

## Eligibility

Episodes must be:

* completed and terminal-linked
* factual fills and realised P&L
* approved evaluator version
* as-of safe (no future terminals/evaluations)
* not the current episode
* not duplicate independence keys
* not truth-degraded / incomplete horizons
* not synthetic replay samples (unless research mode)

Independence key:

```text
authoritative_episode_id + entry lifecycle + terminal lifecycle
```

## Similarity

Deterministic weighted policy (`similarity_policy_version`).

Configuration-driven weights; no embedding/LLM-only similarity.

## Statistics

* mean / median / std of realised P&L
* optional similarity-weighted EV
* effective sample size `(Σw)² / Σw²`
* lower confidence bound at configured confidence level

`valid_for_ev` requires sample count, effective N, leakage safety, and (when configured) positive LCB.

Missing samples → `expected_value_usd = None` (no defaults).

## Estimate provenance

`StrategyObjectiveEstimate` stores query/summary IDs, episode IDs, evaluation IDs, sample counts, LCB, similarity policy version, and TTL.

## Repricing

Method: `long_option_entry_cost_adjust_v1`

```text
gross = original_ev + original_entry_cost
repriced_ev = gross - current_entry_cost
```

Gateway rejects missing/non-positive repriced EV for ENTRY, PROBE, and ADD.

## Cold start

Status `insufficient_historical_evidence`: observe and score, but do not submit.

## Kill switch

`CognitiveGraphDeps.kill_switch` is wired from `risk.kill_switch` into `OrderActionGateway`.
ENTRY / PROBE / ADD / REPLACE are rejected with `KILL_SWITCH` before reservation or broker submit.
The kill switch remains stronger than objective confirmation and positive historical EV.

## Exploration mode

Disabled by default. Operator-approved, paper-only, separately labelled; never treated as production positive-EV validation automatically.
