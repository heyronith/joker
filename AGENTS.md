# Joker Repository Instructions

## Repository identity

- Repository: `heyronith/joker`
- Current feature branch: `goal-conditioned-full-chain-optimizer`
- Required base SHA for the current correction lineage: `69e39fcb937bacf2e01bfee291b6663c0be034c1`
- Do not assume `docs/project-status.md` reflects the currently checked-out branch. Verify branch and SHA directly with Git before every implementation task.

## System purpose

This project implements a goal-conditioned, multi-agent graph system for SPY 0DTE option research and paper trading.

The target-attainment workflow is:

```text
capital + profit target + deadline
→ current market and account truth
→ strategy-thesis generation
→ full SPY 0DTE option-chain analysis
→ contract-specific outcome estimation
→ contract × quantity grid
→ bounded portfolio search
→ ENTER versus WAIT comparison
→ adversarial review
→ deterministic validation
→ paper-broker execution
→ reconciliation and objective reassessment
```

## Authority hierarchy

- Market, account, order, position, broker and reconciliation truth are authoritative.
- Agents may generate, criticize and review market theses.
- Agents may not fabricate executable contract IDs.
- Agents may not silently modify contracts, quantities, component counts or capital allocations.
- The deterministic optimizer owns the authoritative portfolio decision.
- Execution must preserve the exact authorized portfolio tuples.
- Material truth changes require reoptimization.
- `WAIT` is an explicit competing decision.
- The system must never claim that a profit target is guaranteed.

## Safety requirements

- The full-chain optimizer is approved only for paper and replay modes.
- It must remain blocked in `LIVE_GATED` mode and with a `webull_live` broker.
- `WEBULL_LIVE_TRADING_ENABLED` must remain false.
- Do not run `joker paper run`, any live session, or any external broker-order command during code implementation or tests.
- Do not place real or paper-account orders.
- Do not alter production activation defaults.
- Do not merge feature branches unless the user explicitly authorizes it.
- Do not force-push or rewrite existing branch history.
- Do not commit secrets, credentials, account IDs, tokens, cookies, databases, caches, broker responses or restricted raw OPRA artifacts.

## Development process

Before editing:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git rev-parse origin/goal-conditioned-full-chain-optimizer
```

Stop without editing if:

- The working tree is not clean.
- The branch is unexpected.
- Local and remote SHA differ.
- The requested starting SHA does not match.

During implementation:

- Inspect existing graph, execution, persistence and reconciliation abstractions before adding new ones.
- Prefer extending existing durable mechanisms over creating competing stores.
- Keep changes narrowly scoped.
- Preserve backward compatibility unless the task explicitly requires a migration.
- Add typed schemas, deterministic behavior, explicit error codes and fail-closed paths.
- Never weaken or delete tests merely to make a suite pass.
- Tests for execution paths must exercise the compiled graph and real deterministic gateway where requested, not only helper functions or unconditional approval stubs.

Before committing:

- Review the complete diff.
- Run focused tests.
- Run the full test suite.
- Run `ruff check .`.
- Run `python scripts/local_verify.py`.
- Run SQLite integrity and foreign-key checks.
- Confirm no trading session or external order was executed.
- Confirm no live activation was enabled.
- Confirm no prohibited file or secret is staged.

Before pushing:

- Show the working-tree status, commit list and diff statistics.
- Push only the requested feature branch.
- Do not merge.
- Return the exact remote SHA for independent review.
