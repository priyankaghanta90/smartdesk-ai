# smartdesk/rag/

## Why this folder exists
Answers must come **only** from NovaTech's knowledge base, with citations, and SmartDesk must know when it *can't*
answer. This folder owns that whole pipeline, from turning documents into chunks to producing a grounded answer
or an escalation signal. It deliberately knows nothing about agents or tickets, so it can be:

- evaluated on its own (hit@k, MRR, groundedness) without running conversations;
- swapped or tuned (retriever mode, threshold, re-ranker) without touching the agent flow.

It runs in two phases: **A, indexing** (once, `scripts/build_index.py`) and **B, answering** (every question).

## Files

| File | What it does | Why it exists |
|---|---|---|
| `ingest.py` | Splits `kb/*/*.md` by heading and `kb/faq.json` by question into `Chunk`s with metadata; skips `README.md` | Policy sections are self-contained rules, so splitting by heading keeps a rule with its exceptions. Isolating ingestion means the chunking strategy can change without touching retrieval |
| `retrieval.py` | `build_index()` (Chroma + `chunks.json`) and `HybridRetriever` (dense + BM25 → RRF → cross-encoder re-rank, filtered by IT/HR) | Dense search catches paraphrases and BM25 catches exact terms; fusion + re-ranking gets the best of both, and the re-rank score doubles as the confidence signal. Rebuilding the index also clears the answer cache so old policy text can't be served |
| `confidence.py` | The escalation gate: no chunks, or a top score below `RERANK_THRESHOLD` | Escalation is the core safety property ("don't guess"), so its rule lives in a tiny module that's easy to test and calibrate (the eval sweep picks the threshold) |
| `answer.py` | `answer_question()`: retrieve → gate → grounded LLM answer with `[S1]` citations and a sources footer; cross-domain retry; `INSUFFICIENT_CONTEXT` → escalate | The one function both the agent and the eval call, so what's evaluated is exactly what users get. The cross-domain retry stops a wrong IT/HR guess from becoming a false escalation |
| `prompts.py` | The grounded-answer prompt | Separated from the agent prompts because it's the RAG contract: answer only from context, cite sources, or say `INSUFFICIENT_CONTEXT` |
| `semantic_cache.py` | Shared answer cache: SQLite + float32 vectors, an in-memory matrix per IT/HR scope | Repeated or paraphrased questions skip retrieval and the LLM (much faster and cheaper). It's in `rag/` because it stores RAG answers; its privacy rules live in `guardrails/` |
