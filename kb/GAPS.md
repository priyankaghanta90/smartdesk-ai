# Deliberate Knowledge-Base Gaps

These topics are **intentionally not covered** by the KB, so the agent's escalation logic can be tested.
For any of them, SmartDesk must say it doesn't have the information and offer to raise a ticket. It must never invent an answer.

| Topic | Example question | Expected route |
|---|---|---|
| Monitor flickering / display hardware faults | "My monitor has been flickering for two days" | Escalate → IT ticket |
| Printer jams / printer setup | "The 3rd-floor printer keeps jamming" | Escalate → IT ticket |
| Office parking | "How do I get a parking slot at HQ?" | Escalate → HR/Facilities ticket |
| Cafeteria menu / food coupons | "What's on the cafeteria menu today?" | Escalate → HR ticket |
| Relocation / transfer allowance | "Do I get relocation support if I move to Denver?" | Escalate → HR ticket |
| Shuttle bus routes | "Is there a company shuttle from St. Paul?" | Escalate → HR ticket |
| Employee loans / salary advance | "Can I get a salary advance?" | Escalate → HR ticket |

Not covered by design: nothing in `kb/` or `kb/faq.json` mentions these topics (checked by `tests/test_kb.py`).

> This file is documentation only. `scripts/build_index.py` does not index it.
