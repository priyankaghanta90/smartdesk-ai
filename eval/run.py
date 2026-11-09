"""LangSmith evaluation suites.

    python eval/run.py --suite rag                        # all retriever modes, with LLM judges
    python eval/run.py --suite rag --modes hybrid_rerank  # one mode
    python eval/run.py --suite rag --no-judge             # retrieval / escalation / latency only (cheap)
    python eval/run.py --suite router                     # does every message reach the right agent?
    python eval/run.py --suite agent --repetitions 3      # whole conversations: paths, outcomes, safety

Each mode is one LangSmith experiment on the `smartdesk-rag` dataset (synced from eval/golden.json
first), so modes and runs can be compared side by side in LangSmith. A summary is also written to
eval/results.md so the repo keeps a record.

RAG suite metrics
  code (no LLM):  hit@1, hit@3, MRR, escalation precision/recall/F1, answered gaps, latency,
                  best RERANK_THRESHOLD (hybrid_rerank)
  LLM judges:     groundedness, helpfulness, correctness (vs the golden reference), retrieval
                  relevance; openevals prompts + JUDGE_SYSTEM, EVAL_JUDGE_MODEL (gpt-4o) judge,
                  pass/fail per answered question

Router suite (eval/results_router.md): the 48 golden questions plus labelled status / ticket /
small-talk / out-of-scope messages; route accuracy overall and per destination node.

Agent suite (eval/results_agent.md): scripted multi-turn conversations (eval/conversations.json) run
through the real agent with an in-memory ticket store, each repeated N times so flaky behaviour shows
up as a pass rate. Scored on the path through the graph (agentevals strict graph-trajectory match),
the outcome (tickets created, final state, reply content) and two safety checks that must be 100%:
no ticket without confirmation, and no other employee's tickets in a reply.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_openai import ChatOpenAI  # noqa: E402
from openevals.llm import create_llm_as_judge  # noqa: E402
from openevals.prompts import (  # noqa: E402
    CORRECTNESS_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
    RAG_HELPFULNESS_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)

from eval import agent_suite  # noqa: E402
from eval.langsmith_setup import RAG_DATASET, client, rag_example, sync_dataset  # noqa: E402
from eval.metrics import (  # noqa: E402
    EVAL_DIR, escalation_scores, hit_at_k, load_golden, pct, reciprocal_rank, threshold_curve,
)
from smartdesk.agent import hr as agent_hr, it as agent_it  # noqa: E402
from smartdesk.config import get_settings  # noqa: E402
from smartdesk.llm import embeddings  # noqa: E402
from smartdesk.rag.answer import answer_question  # noqa: E402
from smartdesk.rag.retrieval import MODES, HybridRetriever  # noqa: E402
from smartdesk.rag.semantic_cache import SemanticCache  # noqa: E402

PERSONAS = {"IT": agent_it.PERSONA, "HR": agent_hr.PERSONA}  # same personas the live agents use
JUDGE_KEYS = ["groundedness", "helpfulness", "correctness", "retrieval_relevance"]
SKIP = {"results": []}  # an evaluator result that records nothing (metric doesn't apply to this row)


# --- target ----------------------------------------------------------------------------
def rag_target(mode: str, retriever: HybridRetriever):
    def target(inputs: dict) -> dict:
        t0 = time.perf_counter()
        ans = answer_question(inputs["question"], retriever, group=inputs["group"], mode=mode,
                              persona=PERSONAS[inputs["group"]])
        return {
            "mode": mode,
            "answer": ans.answer,
            "escalated": ans.escalate,
            "reason": ans.reason,
            "confidence": ans.confidence if math.isfinite(ans.confidence) else None,
            "retrieved_ids": [s["chunk_id"] for s in ans.sources],
            "contexts": ans.contexts,
            "latency_s": time.perf_counter() - t0,
        }

    return target


def _answered(outputs: dict, reference_outputs: dict) -> bool:
    return reference_outputs["answerable"] and not outputs["escalated"]


# --- row evaluators (code) ---------------------------------------------------------------
def escalation_correct(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "escalation_correct", "score": outputs["escalated"] == (not reference_outputs["answerable"])}


def answered_gap(outputs: dict, reference_outputs: dict) -> dict:
    """1 = a deliberate KB gap got an answer instead of an escalation (must never happen)."""
    if reference_outputs["answerable"]:
        return SKIP
    return {"key": "answered_gap", "score": not outputs["escalated"]}


def retrieval(outputs: dict, reference_outputs: dict) -> dict:
    if not reference_outputs["answerable"]:
        return SKIP
    got, want = outputs["retrieved_ids"], reference_outputs["reference_source_ids"]
    return {"results": [
        {"key": "hit@1", "score": hit_at_k(got, want, 1)},
        {"key": "hit@3", "score": hit_at_k(got, want, 3)},
        {"key": "reciprocal_rank", "score": reciprocal_rank(got, want)},
    ]}


# --- row evaluators (LLM judges) -----------------------------------------------------------
def _context(outputs: dict) -> str:
    return "\n\n".join(f"[S{i}] {c}" for i, c in enumerate(outputs["contexts"], start=1))


# Without this, the judge graded against "how most companies do it" and failed a correct "No, personal
# devices cannot use the VPN", and marked context that answers a question with "no" as irrelevant.
JUDGE_SYSTEM = (
    "You are grading NovaTech's internal IT & HR help-desk assistant. Grade ONLY against the context and "
    "reference you are given; NovaTech's policies are the truth here, even where they differ from common "
    "practice or your own knowledge. Do not penalise missing details that are not in the reference or context. "
    "A context or answer that settles the question with 'no' or 'not allowed' is relevant and can be complete. "
    "Ignore the assistant's closing offer of more help, its [S1]-style citations and its sources footer."
)


def make_judges(judge_model: str | None = None) -> list:
    s = get_settings()
    model = ChatOpenAI(model=judge_model or s.eval_judge_model, temperature=0, api_key=s.openai_api_key)
    specs = {  # key -> (openevals prompt, how to fill its variables)
        "groundedness": (RAG_GROUNDEDNESS_PROMPT, lambda i, o, r: {"context": _context(o), "outputs": o["answer"]}),
        "helpfulness": (RAG_HELPFULNESS_PROMPT, lambda i, o, r: {"inputs": i["question"], "outputs": o["answer"]}),
        "correctness": (CORRECTNESS_PROMPT, lambda i, o, r: {
            "inputs": i["question"], "outputs": o["answer"], "reference_outputs": r["reference"]}),
        "retrieval_relevance": (RAG_RETRIEVAL_RELEVANCE_PROMPT, lambda i, o, r: {
            "inputs": i["question"], "context": _context(o)}),
    }
    judges = []
    for key, (prompt, fill) in specs.items():
        judge = create_llm_as_judge(prompt=prompt, feedback_key=key, judge=model, system=JUDGE_SYSTEM)

        def evaluator(inputs: dict, outputs: dict, reference_outputs: dict, _judge=judge, _fill=fill) -> dict:
            if not _answered(outputs, reference_outputs):  # judged only on answered, answerable questions
                return SKIP
            return _judge(**_fill(inputs, outputs, reference_outputs))

        evaluator.__name__ = key
        judges.append(evaluator)
    return judges


# --- summary evaluators (whole experiment) -----------------------------------------------
def rag_summary(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    s = summarise(outputs, reference_outputs)
    keys = ["hit@1", "hit@3", "mrr", "answer_rate", "esc_precision", "esc_recall", "esc_f1",
            "latency_p50_s", "latency_p95_s"]
    results = [{"key": k, "score": s[k]} for k in keys]
    if "best_threshold" in s:
        results += [{"key": "best_threshold", "score": s["best_threshold"]},
                    {"key": "best_threshold_f1", "score": s["best_threshold_f1"]}]
    return {"results": results}


def summarise(outputs: list[dict], references: list[dict], judge_scores: dict | None = None) -> dict:
    rows = list(zip(outputs, references))
    answerable = [(o, r) for o, r in rows if r["answerable"]]
    should, did = [not r["answerable"] for _, r in rows], [o["escalated"] for o, _ in rows]
    esc = escalation_scores(should, did)
    s = {
        "hit@1": sum(hit_at_k(o["retrieved_ids"], r["reference_source_ids"], 1) for o, r in answerable) / len(answerable),
        "hit@3": sum(hit_at_k(o["retrieved_ids"], r["reference_source_ids"], 3) for o, r in answerable) / len(answerable),
        "mrr": sum(reciprocal_rank(o["retrieved_ids"], r["reference_source_ids"]) for o, r in answerable) / len(answerable),
        "answer_rate": sum(not o["escalated"] for o, _ in answerable) / len(answerable),
        **{f"esc_{k}": v for k, v in esc.items()},
        "latency_p50_s": pct([o["latency_s"] for o in outputs], 50),
        "latency_p95_s": pct([o["latency_s"] for o in outputs], 95),
    }
    if all(o["confidence"] is not None or o["escalated"] for o in outputs) and outputs[0].get("mode") == "hybrid_rerank":
        conf = [o["confidence"] if o["confidence"] is not None else -math.inf for o in outputs]
        s["best_threshold"], s["best_threshold_f1"], s["threshold_curve"] = threshold_curve(conf, should)
    for key, scores in (judge_scores or {}).items():
        s[key] = sum(scores) / len(scores) if scores else float("nan")
    return s


# --- local extras ------------------------------------------------------------------------
def cache_latency(retriever: HybridRetriever, golden: list[dict]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        cache = SemanticCache(Path(tmp) / "cache.db", embeddings(), get_settings().cache_similarity)
        sample = [r for r in golden if r["answerable"]][:10]
        for r in sample:
            ans = answer_question(r["user_input"], retriever, group=r["group"], persona=PERSONAS[r["group"]])
            cache.store(r["user_input"], r["group"], ans.answer or "-", ans.sources)
        timings, hits = [], 0
        for r in sample:
            t0 = time.perf_counter()
            hits += cache.lookup(r["user_input"], r["group"]) is not None
            timings.append(time.perf_counter() - t0)
    return {"cache_hit_rate": hits / len(sample), "cache_latency_p50_s": pct(timings, 50)}


def write_report(summaries: list[dict], cache_stats: dict, judged: bool, path: Path) -> None:
    lines = ["# SmartDesk evaluation results", "",
             "Golden set: `eval/golden.json` (LangSmith dataset `smartdesk-rag`). Generated by "
             "`python eval/run.py --suite rag`. Per-question results are in each LangSmith experiment.", "",
             "## Retrieval & escalation by retriever mode", "",
             "| mode | experiment | hit@1 | hit@3 | MRR | answer rate | escalation P | escalation R | escalation F1 | p50 latency (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['mode']} | `{s['experiment']}` | {s['hit@1']:.2f} | {s['hit@3']:.2f} | {s['mrr']:.2f} | "
                     f"{s['answer_rate']:.2f} | {s['esc_precision']:.2f} | {s['esc_recall']:.2f} | {s['esc_f1']:.2f} | "
                     f"{s['latency_p50_s']:.2f} |")
    if judged:
        lines += ["", f"## LLM judges (openevals, {summaries[0]['judge_model']} judge; pass rate on answered, answerable questions)", "",
                  "| mode | Groundedness | Helpfulness | Correctness | Retrieval relevance |", "|---|---|---|---|---|"]
        for s in summaries:
            lines.append(f"| {s['mode']} | " + " | ".join(f"{s.get(k, float('nan')):.2f}" for k in JUDGE_KEYS) + " |")
    for s in summaries:
        if "threshold_curve" in s:
            lines += ["", "## Escalation threshold calibration (hybrid_rerank, score gate only)", "",
                      f"Best `RERANK_THRESHOLD` = **{s['best_threshold']:.1f}** (escalation F1 {s['best_threshold_f1']:.2f}); "
                      f"configured {get_settings().rerank_threshold}.", "", "| threshold | F1 |", "|---|---|"]
            lines += [f"| {t:.1f} | {f:.2f} |" for t, f in s["threshold_curve"] if t == int(t)]
    lines += ["", "## Semantic cache", "",
              f"Hit rate on repeated questions: {cache_stats['cache_hit_rate']:.0%}; "
              f"p50 lookup latency {cache_stats['cache_latency_p50_s'] * 1000:.0f} ms (vs full RAG p50 above).", "",
              "## Unanswerable questions that were answered (should be empty)", ""]
    for s in summaries:
        lines.append(f"- **{s['mode']}**: {', '.join(s['answered_gaps']) or 'none'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- suites -------------------------------------------------------------------------------
def run_rag(modes: list[str], judged: bool, concurrency: int, judge_model: str | None = None) -> None:
    from langsmith import evaluate

    ls = client()
    golden = load_golden()
    print(sync_dataset(ls, RAG_DATASET, [rag_example(r) for r in golden],
                       "SmartDesk golden questions (eval/golden.json): 36 answerable, 12 deliberate KB gaps."))
    retriever = HybridRetriever()
    retriever.retrieve("warm up the reranker")
    judges = make_judges(judge_model) if judged else []

    summaries = []
    for mode in modes:
        print(f"Evaluating mode={mode}…")
        results = evaluate(
            rag_target(mode, retriever),
            data=RAG_DATASET,
            evaluators=[escalation_correct, answered_gap, retrieval, *judges],
            summary_evaluators=[rag_summary],
            experiment_prefix=f"rag-{mode}",
            metadata={"suite": "rag", "mode": mode, "rerank_threshold": get_settings().rerank_threshold,
                      "chat_model": get_settings().openai_chat_model, "judged": judged,
                      "judge_model": (judge_model or get_settings().eval_judge_model) if judged else None},
            max_concurrency=concurrency,
            client=ls,
        )
        outputs, refs, questions, judge_scores = [], [], [], {k: [] for k in JUDGE_KEYS}
        for row in results:
            outputs.append(row["run"].outputs or {})
            refs.append(row["example"].outputs)
            questions.append(row["example"].inputs["question"])
            for res in row["evaluation_results"]["results"]:
                if res.key in judge_scores and res.score is not None:
                    judge_scores[res.key].append(float(res.score))
        s = summarise(outputs, refs, judge_scores if judged else None)
        s.update(mode=mode, experiment=results.experiment_name, judge_model=judge_model or get_settings().eval_judge_model,
                 answered_gaps=[q for q, o, r in zip(questions, outputs, refs) if not r["answerable"] and not o["escalated"]])
        summaries.append(s)
        shown = {k: round(v, 3) for k, v in s.items() if isinstance(v, float) and k != "threshold_curve"}
        print(f"  {results.experiment_name}: {shown}")

    cache_stats = cache_latency(retriever, golden)
    write_report(summaries, cache_stats, judged, EVAL_DIR / "results.md")
    print(f"Wrote {EVAL_DIR / 'results.md'}")


def _rows_by_example(results) -> dict[str, list[dict]]:
    """Experiment rows grouped by example (several per example when repeated)."""
    grouped: dict[str, list[dict]] = {}
    for row in results:
        grouped.setdefault(row["example"].inputs["question"], []).append(row)
    return grouped


def _scores(row) -> dict[str, float]:
    return {r.key: float(r.score) for r in row["evaluation_results"]["results"] if r.score is not None}


def run_router(concurrency: int) -> None:
    from langsmith import evaluate

    ls = client()
    print(sync_dataset(ls, agent_suite.ROUTER_DATASET, agent_suite.router_examples(),
                       "Router labels: golden KB questions (it_agent / hr_agent) plus status, ticket and small-talk messages."))
    results = evaluate(agent_suite.router_target, data=agent_suite.ROUTER_DATASET,
                       evaluators=[agent_suite.route_correct], summary_evaluators=[agent_suite.router_summary],
                       experiment_prefix="router", metadata={"suite": "router"}, max_concurrency=concurrency, client=ls)
    rows = [(r["example"].inputs["question"], r["run"].outputs or {}, r["example"].outputs) for r in results]
    by_route: dict[str, list[bool]] = {}
    for _, out, ref in rows:
        by_route.setdefault(ref["route"], []).append(out.get("route") == ref["route"])
    misses = [(q, ref["route"], out.get("route")) for q, out, ref in rows if out.get("route") != ref["route"]]
    total = sum(sum(v) for v in by_route.values()) / len(rows)
    lines = ["# SmartDesk router evaluation", "",
             f"Experiment `{results.experiment_name}` on LangSmith dataset `{agent_suite.ROUTER_DATASET}` "
             f"({len(rows)} messages). Generated by `python eval/run.py --suite router`.", "",
             f"**Route accuracy: {total:.0%}**", "", "| expected route | messages | accuracy |", "|---|---|---|"]
    lines += [f"| `{k}` | {len(v)} | {sum(v) / len(v):.0%} |" for k, v in sorted(by_route.items())]
    lines += ["", "## Misrouted messages", ""] + ([f"- {q!r}: expected `{e}`, got `{g}`" for q, e, g in misses] or ["- none"])
    (EVAL_DIR / "results_router.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  route accuracy {total:.1%}; {len(misses)} misrouted. Wrote {EVAL_DIR / 'results_router.md'}")


AGENT_KEYS = ["trajectory_match", "outcome_correct", "safety_confirmation", "safety_no_foreign_tickets"]


def run_agent(concurrency: int, repetitions: int) -> None:
    from langsmith import evaluate

    ls = client()
    print(sync_dataset(ls, agent_suite.AGENT_DATASET, agent_suite.conversation_examples(),
                       "Scripted SmartDesk conversations (eval/conversations.json) with accepted graph paths and outcomes."))
    retriever = HybridRetriever()
    retriever.retrieve("warm up the reranker")
    results = evaluate(agent_suite.make_conversation_target(retriever), data=agent_suite.AGENT_DATASET,
                       evaluators=agent_suite.AGENT_EVALUATORS, experiment_prefix="agent",
                       metadata={"suite": "agent", "repetitions": repetitions,
                                 "chat_model": get_settings().openai_chat_model},
                       num_repetitions=repetitions, max_concurrency=concurrency, client=ls)
    grouped = _rows_by_example(results)
    lines = ["# SmartDesk agent (conversation) evaluation", "",
             f"Experiment `{results.experiment_name}` on LangSmith dataset `{agent_suite.AGENT_DATASET}`: "
             f"{len(grouped)} scripted conversations, each run {repetitions}× through the real agent "
             "(in-memory ticket store). Generated by `python eval/run.py --suite agent`. Cells are pass rates.", "",
             "| conversation | path (agentevals) | outcome | safety: confirmation | safety: no foreign tickets |",
             "|---|---|---|---|---|"]
    totals: dict[str, list[float]] = {k: [] for k in AGENT_KEYS}
    for name, rows in grouped.items():
        cells = []
        for key in AGENT_KEYS:
            vals = [_scores(r)[key] for r in rows if key in _scores(r)]
            totals[key] += vals
            cells.append(f"{sum(vals) / len(vals):.0%}" if vals else "–")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    overall = {k: (sum(v) / len(v) if v else float("nan")) for k, v in totals.items()}
    lines.insert(4, "**Overall:** " + ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in overall.items()) + "\n")
    (EVAL_DIR / "results_agent.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  " + ", ".join(f"{k} {v:.1%}" for k, v in overall.items()) + f". Wrote {EVAL_DIR / 'results_agent.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["rag", "router", "agent"], default="rag")
    parser.add_argument("--repetitions", type=int, default=3, help="agent suite: runs per conversation")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judges (retrieval/escalation only)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--judge-model", help="override EVAL_JUDGE_MODEL for this run")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.suite == "router":
        run_router(args.concurrency)
    elif args.suite == "agent":
        run_agent(args.concurrency, args.repetitions)
    else:
        run_rag(args.modes, judged=not args.no_judge, concurrency=args.concurrency, judge_model=args.judge_model)


if __name__ == "__main__":
    main()
