# kb/

## Why this folder exists
The knowledge base is **data, not code**: the source of truth SmartDesk answers from. Keeping it as plain Markdown
and JSON means non-developers (HR, IT) can edit policies, changes are reviewable in git, and the indexer rebuilds
everything from these files (`python scripts/build_index.py`). The subfolder decides the domain: `it/` is answered
by the IT specialist, the others by HR.

Each policy file has YAML front matter (`title`, `domain`, `category`) and `##` sections; each section becomes a
retrieval chunk, so a section should be one self-contained rule.

## Files

| Path | What it is | Why it exists |
|---|---|---|
| [`it/`](it/) | 7 IT policies | IT specialist's knowledge |
| [`hr/`](hr/) | 6 HR policies | HR specialist's knowledge |
| [`onboarding/`](onboarding/) | 4 onboarding guides | New-hire questions, answered by HR |
| [`payroll/`](payroll/) | 4 payroll and compensation policies | Pay and tax questions, answered by HR |
| `faq.json` | 44 short Q&A pairs across all areas | Short, direct answers that match how employees phrase questions; each pair is its own chunk |
| `GAPS.md` | Topics deliberately left out (not indexed) | Lets us test that SmartDesk escalates instead of guessing; `tests/test_kb.py` checks they stay absent |
| `FACTS.md` | Company facts shared across documents (not indexed) | Keeps numbers, names and formats consistent across all files (e.g. the employee-ID format) |

`README.md` files are skipped by the indexer, so these guides never appear in answers.
