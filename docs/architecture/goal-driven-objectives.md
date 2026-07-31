# Goal-Driven Objectives

Task 1 owns objective capital exposure, fills, and P&L projection.

Task 2 owns strategy invention, debate, and selection — but cannot bypass deterministic EV, capital, deadline, or truth gates.

Task 3 owns evaluated historical episodes used as factual evidence for EV.

## Production path

```text
Task 1 snapshot
→ Task 2 strategy hypothesis
→ historical analogue query (Task 3)
→ typed StrategyObjectiveEstimate
→ ObjectiveStrategyScore (+ no-trade)
→ meta-decision (validated)
→ deterministic sizing
→ entry tactician
→ gateway quote repricing
→ paper ExecutionRuntime
```

## Hard gates

* Confirmed objective required
* Positive supported EV required when `require_positive_expected_value: true`
* Available capital = authorised − working reservation − filled exposure
* Truth degradation blocks new entries
* Kill switch remains stronger than objective approval

See also: [historical-ev.md](historical-ev.md).
