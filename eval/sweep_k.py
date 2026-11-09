"""Retrieval-only sweep over CANDIDATE_K (pool size per retriever) and FINAL_K (chunks sent to the LLM).

No chat-model calls: only query embeddings, BM25 and the cross-encoder. Mirrors HybridRetriever.retrieve,
but fetches each query's candidates once at the largest K and slices, so the sweep stays cheap.

    python eval/sweep_k.py
    python eval/sweep_k.py --candidate-ks 5 10 20 30 --final-ks 1 2 3 4 6 8 --show-misses
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import hit_at_k, load_golden, reciprocal_rank  # noqa: E402
from smartdesk.rag.retrieval import MODES, HybridRetriever, rrf_fuse  # noqa: E402


def rank_for(mode: str, dense_ids: list[str], sparse_ids: list[str], ck: int, logits: dict[str, float]) -> list[str]:
    dense_ids, sparse_ids = dense_ids[:ck], sparse_ids[:ck]
    if mode == "dense":
        return dense_ids
    if mode == "bm25":
        return sparse_ids
    fused = rrf_fuse([dense_ids, sparse_ids])[:ck]
    if mode == "hybrid":
        return fused
    return sorted(fused, key=lambda cid: logits[cid], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--candidate-ks", nargs="+", type=int, default=[5, 10, 20, 30])
    parser.add_argument("--final-ks", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--show-misses", action="store_true", help="list questions whose ground truth is missed")
    args = parser.parse_args()

    retriever = HybridRetriever()
    golden = load_golden()
    max_ck = max(args.candidate_ks)

    # Fetch every query's candidates once, at the largest K.
    cached = []
    for r in golden:
        q, group = r["user_input"], r["group"]
        dense_ids = [cid for cid, _ in retriever.dense(q, max_ck, group)]
        sparse_ids = retriever.sparse(q, max_ck, group)
        pool = list(dict.fromkeys(dense_ids + sparse_ids))
        logits = dict(zip(pool, map(float, retriever.reranker.predict([(q, retriever.by_id[c].text) for c in pool]))))
        cached.append((r, dense_ids, sparse_ids, logits))

    answerable = [c for c in cached if c[0]["answerable"]]
    print(f"{len(answerable)} answerable / {len(cached) - len(answerable)} unanswerable golden questions\n")
    print(f"{'mode':<14}{'cand_k':>7}{'final_k':>8}{'hit@final':>11}{'hit@1':>7}{'mrr':>7}")
    misses: dict[tuple, list[str]] = {}
    for mode in args.modes:
        for ck in args.candidate_ks:
            ranked = [(r, rank_for(mode, d, s, ck, lg)) for r, d, s, lg in answerable]
            for fk in args.final_ks:
                hits = [hit_at_k(ids, r["reference_source_ids"], fk) for r, ids in ranked]
                h1 = statistics.mean(hit_at_k(ids, r["reference_source_ids"], 1) for r, ids in ranked)
                mrr = statistics.mean(reciprocal_rank(ids[:fk], r["reference_source_ids"]) for r, ids in ranked)
                print(f"{mode:<14}{ck:>7}{fk:>8}{statistics.mean(hits):>11.3f}{h1:>7.3f}{mrr:>7.3f}")
                misses[(mode, ck, fk)] = [r["user_input"] for (r, _), h in zip(ranked, hits) if not h]
        print()

    # Escalation separation: the gate uses the top rerank score (see smartdesk/rag/confidence.py).
    if "hybrid_rerank" in args.modes:
        print("Top rerank score (gate confidence) by candidate_k:")
        for ck in args.candidate_ks:
            tops = {True: [], False: []}
            for r, d, s, lg in cached:
                ids = rank_for("hybrid_rerank", d, s, ck, lg)
                tops[r["answerable"]].append(lg[ids[0]] if ids else float("-inf"))
            print(f"  cand_k={ck:<3} answerable min={min(tops[True]):6.2f} median={statistics.median(tops[True]):6.2f}"
                  f" | unanswerable max={max(tops[False]):6.2f} median={statistics.median(tops[False]):6.2f}")

    if args.show_misses:
        print("\nMissed ground truth:")
        for key, qs in misses.items():
            if qs:
                print(f"  {key}: {qs}")


if __name__ == "__main__":
    main()
