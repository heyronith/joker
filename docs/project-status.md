# Joker Project Status

| Item | Status |
| --- | --- |
| Task 1 | approved |
| Task 2 | approved |
| Task 3 | approved project foundation |
| Goal-driven objective layer | implemented |
| Current phase | historical EV production truth correction (not approved) |
| Current branch | `task3-agent-evolution` |
| Comparison base (pre this correction) | `8cb47c69b8bbdd2888f3e12b06c892f90cbe4c63` |
| Prior verified code tip | `569a2235a2e713bd5de9b84c3b416f913416cd99` |
| Latest verified CI run (prior tip) | `30707444746` (803 on Python 3.11 + 3.12; Ruff passed) |
| Tip commit (this correction) | pending commit / tip CI |
| Latest verified CI run (this tip) | pending |
| Local Python 3.12 | 830 passed ×3 (`ResourceWarning` as error) |
| Local Python 3.11 | 830 passed ×3 (`ResourceWarning` as error) |
| Focused historical-EV / compiler / gateway / public runner | 65 passed ×5 (`ResourceWarning` as error) |
| Paper-only status | enforced (live trading disabled) |
| Textual UI status | not started / not modified in this phase |
| Known unresolved items | exact-tip CI for this correction; independent review; phase remains unapproved |

## Historical-EV phase status

**Not complete / not approved.** This correction requires production `EpisodeCompiler` factual metadata, fail-closed authoritative event horizons, quote+limit worst-case fill safety, configuration-scoped dataset leakage, and a public session-factory PaperBroker path.

Do not mark the historical-EV phase complete until independent review verifies:

```text
real EpisodeCompiler output
→ complete factual metadata
→ authoritative event horizon
→ leakage-safe historical summary
→ positive EV at maximum execution price
→ fully covered reservation
→ one public-path PaperBroker submission
```

plus green exact-tip CI on Python 3.11 and 3.12.
