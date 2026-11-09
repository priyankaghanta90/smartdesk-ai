"""Escalation gate: decides whether retrieved context is good enough to answer from.

Layer 1  no chunks retrieved                     -> escalate
Layer 2  top score below threshold               -> escalate
Layer 3  LLM replies INSUFFICIENT_CONTEXT        -> escalate   (checked in answer.py)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from .retrieval import RetrievalResult

INSUFFICIENT = "INSUFFICIENT_CONTEXT"


@dataclass
class GateDecision:
    passed: bool
    reason: str


def threshold_for(mode: str, s: Settings) -> float:
    return s.rerank_threshold if mode == "hybrid_rerank" else s.dense_threshold


def retrieval_gate(result: RetrievalResult, settings: Settings | None = None) -> GateDecision:
    s = settings or get_settings()
    if not result.chunks:
        return GateDecision(False, "no matching knowledge-base articles")
    threshold = threshold_for(result.mode, s)
    if result.confidence < threshold:
        return GateDecision(
            False, f"best match score {result.confidence:.2f} is below threshold {threshold:.2f}"
        )
    return GateDecision(True, f"best match score {result.confidence:.2f}")


def is_insufficient(answer: str) -> bool:
    return INSUFFICIENT in answer.upper().replace(" ", "_")
