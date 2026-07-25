# Trade lifecycle (agent_led paper)

Engineer reference for how joker decides, sizes, enters, and exits SPY 0DTE paper trades.

## Design split

- **Deterministic code** owns speed: features, confirm gates, capital ceilings, EV size clamps, exits.
- **LLM (DecisionAgent)** owns judgment: edge hypothesis, direction, aggression hints, stop/TP.
- Soft risk caps are advisory under `agent_led`. Hard floors always apply (kill switch, capital, 0DTE, feed silent, stop/TP present).

## Flow

```
SessionConfirm(capital, goal)
  → Webull warmup (quotes/candles)
  → Premarket council → Playbook
  → Poll loop (~1s)
       → FeatureEngine
       → ExitManager (if open)
       → EdgePrefilter → DecisionAgent (propose/confirm) when candidate
       → confirm_gate → EV gates → CapitalBudget.allocate
       → ATM OptionSelector → RiskGovernor → sized order
       → SessionMicroMemory + capital PnL feedback
```

## Stages

| Stage | Role |
|-------|------|
| Capital confirm | User authorizes daily premium budget + target profit % |
| Features | VWAP, momentum, opening range, bands, day-part (ET) |
| DecisionAgent | `hold` / `propose` / `confirm` / `abandon` with p(win), EV |
| Sizing | EV-capped fraction of available capital; never above authorized |
| Exits | Stop, TP, trailing, time stop, EOD (America/New_York 15:55) |

## Latency notes

1. LLM is the main bottleneck — skipped when EdgePrefilter finds no candidate.
2. Fast confirm path re-evaluates pending proposals without waiting the full decision interval.
3. Stock poll ~1s; ATM option fetches on decision ticks and while a position is open.

## Goal / reward

Session objective = `authorized_usd × target_profit_pct`. Aggression adapts to goal gap and minutes-to-close. Entries pause when `goal_met` if configured. This is a reward pressure objective, not a guaranteed return.

## Market bars note

Live paper prefers Webull 1m `stock_bars` (`symbol` + `timespan=M1`). If bars fail, quote-derived 1m candles are used with equal-weight VWAP proxy so EdgePrefilter can still run.
