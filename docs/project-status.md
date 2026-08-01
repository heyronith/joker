# Joker Project Status

| Item | Status |
| --- | --- |
| Task 1 | approved |
| Task 2 | approved |
| Task 3 | approved project foundation |
| Goal-driven objective layer | implemented |
| Current phase | historical EV production wiring + authoritative quote repricing (correction; not approved) |
| Current branch | `task3-agent-evolution` |
| Comparison base (pre-correction) | `e5cd91cba5ef0b21e16841674124fdda28280def` |
| Primary implementation commit (pre-correction) | `e5cd91cba5ef0b21e16841674124fdda28280def` |
| Tip commit (this correction) | `27bf157489415ad01bdcb4fcfd2af8224a69508e` |
| Latest verified CI run (comparison base) | `30629186951` (781 on Python 3.11 + 3.12; Ruff passed) |
| Latest verified CI run (tip) | *(set after exact-tip CI)* |
| Local Python 3.12 | 803 passed ×3 (`ResourceWarning` as error) |
| Local Python 3.11 | 803 passed ×3 (`ResourceWarning` as error) |
| Focused historical-EV / gateway / graph | 44 passed ×5 |
| Kill-switch soak | compiled-graph proof: positive EV estimate built; gateway blocked; no paper order/reservation |
| Paper-only status | enforced (live trading disabled) |
| Textual UI status | not started / not modified in this phase |
| Known unresolved items | exact-tip CI for this correction; independent review; phase remains unapproved until green tip CI |

## Historical-EV phase status

**Not complete / not approved.** Prior tip `e5cd91c` connected Task 3 outcomes into estimates and gateway arithmetic, but production repository wiring guessed undefined settings paths, tests seeded private historical-service state, and the gateway treated proposed limit price as quote truth. This correction passes Task 3 repositories by DI, reloads authoritative Task 1 quotes before reserve/submit, and proves the public graph path without private seeding.

Do not mark the historical-EV phase complete until this correction has green exact-tip CI on Python 3.11 and 3.12 and independent review.
