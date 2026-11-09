"""Shared by all agents: dependencies, constants, message helpers, reply classification and error handling."""

from __future__ import annotations

import functools
import re
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Literal, Optional

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel

from .. import llm
from ..config import get_settings
from ..guardrails.pii import PiiGuard, guarded_store
from ..llm import LLMUnavailable
from ..rag.answer import answer_question
from ..rag.semantic_cache import SemanticCache
from ..rag.retrieval import HybridRetriever
from ..ticketing import TicketClient, TicketingError, TicketingUnavailable
from . import prompts
from .state import SmartDeskState

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
MAX_INPUT_CHARS = 2000
TEAM = {"IT": "IT Support", "HR": "HR"}
YES = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "go ahead", "please do", "yes please", "do it", "confirm"}
NO = {"no", "n", "nope", "cancel", "no thanks", "not now", "don't", "dont", "stop"}
# A reply made only of these words is unambiguous ("no, cancel", "yes ok go ahead") and skips the LLM.
YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "go", "ahead", "please", "do", "it", "confirm"}
NO_WORDS = {"no", "n", "nope", "cancel", "stop", "don't", "dont", "not", "now", "thanks"}

LLM_DOWN_MSG = (
    "I'm having trouble reaching my language service right now. Please try again in a minute, "
    "or call the IT Service Desk on ext. 2020 if it's urgent."
)


@dataclass
class AgentDeps:
    """External services the agents use. Built once in flow.py and handed to the agents that need them."""

    retriever: HybridRetriever
    tickets: TicketClient
    cache: SemanticCache | None = None
    guard: PiiGuard | None = None  # screens answers before they enter the shared cache
    background: Executor | None = None  # runs guarded cache writes off the response path; None = inline


class ReplyDecision(BaseModel):
    action: Literal["yes", "no", "edit", "other"]
    details: Optional[str] = None
    priority: Optional[Literal["Low", "Medium", "High", "Critical"]] = None


# --- message helpers --------------------------------------------------------------
def extract_email(text: str) -> str | None:
    match = EMAIL_RE.search(text or "")
    return match.group(0).lower() if match else None


def typed_email_allowed(state: SmartDeskState) -> bool:
    """An email typed in chat is only trusted in local dev mode and never overrides a signed-in one."""
    return not state.get("email_verified") and not get_settings().require_verified_email


SIGN_IN_MSG = "Please sign in so I can {purpose}. For your privacy I can't use an email address typed in the chat."


def last_human(state: SmartDeskState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content)[:MAX_INPUT_CHARS]
    return ""


def format_history(messages: list[AnyMessage], limit: int = 8) -> str:
    lines = []
    for msg in messages[-limit:]:
        who = "Employee" if isinstance(msg, HumanMessage) else "SmartDesk"
        lines.append(f"{who}: {str(msg.content)[:600]}")
    return "\n".join(lines)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z' ]", "", text.lower()).strip()


def is_yes(text: str) -> bool:
    norm = _normalise(text)
    return norm in YES or (bool(norm) and set(norm.split()) <= YES_WORDS)


def is_no(text: str) -> bool:
    norm = _normalise(text)
    words = set(norm.split())
    # "thanks" alone is gratitude, not a refusal, so it only counts next to a real no-word.
    return norm in NO or (bool(words) and words <= NO_WORDS and words != {"thanks"})


def classify_reply(question: str, reply: str) -> ReplyDecision:
    if is_yes(reply):
        return ReplyDecision(action="yes")
    if is_no(reply):
        return ReplyDecision(action="no")
    return llm.invoke_safely(
        llm.structured(ReplyDecision),
        [SystemMessage(content=prompts.REPLY_SYSTEM.format(question=question)), HumanMessage(content=reply)],
    )


# --- shared by the IT and HR specialists -------------------------------------------
def answer_from_kb(state: SmartDeskState, deps: AgentDeps, group: str, persona: str) -> Command:
    """Cache -> RAG answer, or hand over to the ticket agent when the KB can't answer."""
    query = state.get("query") or last_human(state)
    if deps.cache and (hit := deps.cache.lookup(query, group)):
        return Command(goto=END, update={
            "messages": [AIMessage(content=hit.answer)],
            "last_sources": hit.sources, "cache_hit": True,
        })

    history = format_history(state["messages"][:-1], 4)
    ans = answer_question(query, deps.retriever, group=group, persona=persona, history=history)
    update = {"last_sources": ans.sources, "last_confidence": ans.confidence, "domain": group}
    if ans.escalate:
        update["escalation_reason"] = ans.reason
        return Command(goto="escalate", update=update)
    # The cache is shared and keyed by the question alone, so only cache answers that didn't depend on
    # earlier turns (e.g. "I'm on a Mac" would otherwise leak Mac-only steps to a Windows user).
    if deps.cache and not history:
        _cache_answer(deps, query, group, ans.answer, ans.sources)
    update["messages"] = [AIMessage(content=ans.answer)]
    return Command(goto=END, update=update)


def _cache_answer(deps: AgentDeps, query: str, group: str, answer: str, sources: list[dict]) -> None:
    if not deps.guard:
        deps.cache.store(query, group, answer, sources)
    elif deps.background:
        deps.background.submit(guarded_store, deps.cache, deps.guard, query, group, answer, sources)
    else:
        guarded_store(deps.cache, deps.guard, query, group, answer, sources)


# --- error handling ---------------------------------------------------------------
def graceful(node):
    """Turn LLM / ticketing failures into a polite reply instead of a crash.

    GraphInterrupt (raised by interrupt()) is not caught, so pausing still works.
    """

    @functools.wraps(node)
    def wrapper(state):
        try:
            return node(state)
        except LLMUnavailable:
            return Command(goto=END, update={"messages": [AIMessage(content=LLM_DOWN_MSG)]})
        except TicketingUnavailable:
            return Command(goto=END, update={"messages": [AIMessage(content=(
                "I'm sorry — our ticketing system isn't reachable right now. "
                "Please try again in a few minutes; just say \"try again\"."
            ))]})
        except TicketingError as exc:
            return Command(goto=END, update={"messages": [AIMessage(content=(
                "I ran into a problem with the ticketing system, so I couldn't complete that. "
                f"({exc}). Please contact the IT Service Desk on ext. 2020."
            ))]})

    return wrapper
