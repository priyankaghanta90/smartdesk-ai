"""Grounded answering over the KB, with escalation signals for the agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from .confidence import is_insufficient, retrieval_gate
from .prompts import ANSWER_SYSTEM
from .retrieval import HybridRetriever, RetrievalResult


@dataclass
class RagAnswer:
    answer: str
    escalate: bool
    reason: str
    confidence: float
    sources: list[dict] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)  # raw chunk texts (used by the eval judges)


def format_context(result: RetrievalResult) -> str:
    return "\n\n".join(f"[S{i}] {rc.chunk.text}" for i, rc in enumerate(result.chunks, start=1))


def _sources(result: RetrievalResult) -> list[dict]:
    return [
        {
            "ref": f"S{i}",
            "chunk_id": rc.chunk_id,
            "title": rc.chunk.title,
            "section": rc.chunk.section,
            "source": rc.chunk.source,
            "score": round(rc.rerank_score if rc.rerank_score is not None else (rc.dense_score or 0.0), 3),
        }
        for i, rc in enumerate(result.chunks, start=1)
    ]


def with_source_footer(reply: str, sources: list[dict]) -> str:
    """Append a deterministic citation footer: the sources the model cited, else the top source."""
    cited = set(re.findall(r"\[(S\d+)\]", reply))
    used = [s for s in sources if s["ref"] in cited] or sources[:1]
    lines = [f"[{s['ref']}] {s['title']} › {s['section']}" for s in used]
    return f"{reply}\n\n_Sources: {'; '.join(lines)}_"


def answer_question(
    question: str,
    retriever: HybridRetriever,
    group: str | None = None,
    history: str = "",
    mode: str | None = None,
    persona: str = "",
) -> RagAnswer:
    result = retriever.retrieve(question, group=group, mode=mode)
    gate = retrieval_gate(result)

    # The router can mis-classify IT vs HR; retry once across the whole KB before giving up.
    if not gate.passed and group is not None:
        wider = retriever.retrieve(question, group=None, mode=mode)
        wider_gate = retrieval_gate(wider)
        if wider_gate.passed:
            result, gate = wider, wider_gate

    sources, contexts = _sources(result), [rc.chunk.text for rc in result.chunks]
    if not gate.passed:
        return RagAnswer("", True, gate.reason, result.confidence, sources, contexts)

    messages = [
        SystemMessage(content=ANSWER_SYSTEM.format(persona=persona, context=format_context(result))),
        HumanMessage(content=(f"Recent conversation:\n{history}\n\n" if history else "") + f"Question: {question}"),
    ]
    reply = llm.invoke_safely(llm.chat(), messages).content.strip()

    if is_insufficient(reply):
        return RagAnswer("", True, "the knowledge base does not cover this question", result.confidence, sources, contexts)
    return RagAnswer(with_source_footer(reply, sources), False, gate.reason, result.confidence, sources, contexts)
