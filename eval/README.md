# eval/

## Why this folder exists
LLM behaviour can't be fully checked by unit tests: answers vary, and a prompt change can quietly break routing or
grounding. This folder measures quality as numbers, runs the measurements as **LangSmith experiments** (so runs
can be compared over time, with every question's trace and the judges' reasoning), and keeps a summary in the repo.

It is separate from `tests/` because evals are statistical and cost money (LLM calls), while tests are
pass/fail and should run on every change. Datasets are synced from the files here before every run, so the repo
stays the source of truth.

## Files

| File | What it does | Why it exists |
|---|---|---|
| `run.py` | Entry point: `python eval/run.py --suite rag \| router \| agent` | One command per suite, with consistent reporting. RAG quality, routing and whole conversations are measured separately so a router bug can't hide behind good retrieval numbers |
| `golden.json` | 48 golden questions: 36 answerable (reference answer + source chunk ids), 12 deliberate gaps | The ground truth for retrieval, escalation and answer quality. Paraphrased, so it tests understanding rather than copy-matching |
| `conversations.json` | 16 scripted multi-turn conversations with expected outcomes and accepted graph paths | Ground truth for agent behaviour across turns: tickets, status, cancellations, identity and safety cases |
| `agent_suite.py` | Router and conversation targets plus evaluators (`agentevals` path match, outcome, two safety checks) | Agent-specific scoring is kept apart from the RAG metrics; the safety checks encode rules that must hold 100% of the time |
| `metrics.py` | hit@k, MRR, escalation precision/recall/F1, threshold sweep | Plain-Python metrics shared by `run.py` and `sweep_k.py`, so a metric is defined exactly once |
| `langsmith_setup.py` | LangSmith client; dataset sync (add / update / delete examples) | Keeps LangSmith in step with the repo and passes keys from `.env` to the SDKs, which read the process environment |
| `sync_datasets.py` | Uploads datasets without running an eval | Useful when only the data changed |
| `sweep_k.py` | Retrieval-only sweep over candidate pool size and chunks sent to the LLM | Tunes `CANDIDATE_K` / `FINAL_K` cheaply (no chat-model calls) |
| `results.md` | Latest RAG suite report | Keeps a reviewable record in git next to the code that produced it |
| `results_router.md` | Latest router suite report | Same, for routing |
| `results_agent.md` | Latest conversation suite report (pass rates over repetitions) | Same, for whole conversations |
