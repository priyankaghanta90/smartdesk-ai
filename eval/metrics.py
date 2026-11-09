"""Metric helpers shared by the eval scripts (no LLM calls)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent


def load_golden() -> list[dict]:
    return json.loads((EVAL_DIR / "golden.json").read_text(encoding="utf-8"))


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


# --- retrieval metrics ---------------------------------------------------------------
def hit_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return float(any(r in relevant for r in retrieved[:k]))


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


# --- escalation metrics ----------------------------------------------------------------
def escalation_scores(should: list[bool], did: list[bool]) -> dict:
    tp = sum(s and d for s, d in zip(should, did))
    fp = sum((not s) and d for s, d in zip(should, did))
    fn = sum(s and (not d) for s, d in zip(should, did))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "false_escalations": fp, "missed_escalations": fn}


def threshold_curve(confidences: list[float], should: list[bool]) -> tuple[float, float, list[tuple[float, float]]]:
    """Escalation F1 of the score gate alone at each candidate RERANK_THRESHOLD; returns (best_t, best_f1, curve)."""
    curve = []
    for t in np.arange(-8.0, 6.01, 0.5):
        f1 = escalation_scores(should, [bool(c < t) for c in confidences])["f1"]
        curve.append((float(t), float(f1)))
    best_t, best_f1 = max(curve, key=lambda x: (x[1], -abs(x[0])))
    return best_t, best_f1, curve
