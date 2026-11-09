"""Router agent: reads the latest message and decides which agent handles it."""

from __future__ import annotations

import re
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field

from .. import llm
from ..llm import LLMUnavailable
from . import prompts
from .common import _normalise, extract_email, format_history, graceful, last_human, typed_email_allowed
from .state import SmartDeskState

# Whole words only, so "chrome" doesn't match "hr", "display" doesn't match "pay", "syntax" doesn't match "tax".
HR_WORDS = re.compile(
    r"(?<![\w-])(leave|pto|sick|salary|pay|payroll|payday|tax|taxes|w-2|w-4|401k|401\(k\)|insurance|benefits"
    r"|wfh|reimburse\w*|stock options?|introductory|hr)(?![\w-])"
)


class RouteDecision(BaseModel):
    intent: Literal["kb_question", "create_ticket", "ticket_status", "chitchat", "out_of_scope"]
    domain: Optional[Literal["IT", "HR"]] = None
    standalone_query: str = Field(description="latest message rewritten as a self-contained question")


def heuristic_route(text: str) -> RouteDecision:
    """Keyword router used only when the LLM is unavailable."""
    t = text.lower()
    if any(w in t for w in ("status", "update on", "my ticket", "any update")):
        intent = "ticket_status"
    elif any(w in t for w in ("raise a ticket", "create a ticket", "open a ticket", "log a ticket")):
        intent = "create_ticket"
    elif _normalise(t) in {"hi", "hello", "hey", "thanks", "thank you", "bye"}:
        intent = "chitchat"
    else:
        intent = "kb_question"
    domain = "HR" if HR_WORDS.search(t) else "IT"
    return RouteDecision(intent=intent, domain=domain, standalone_query=text)


SELF_ID_RE = re.compile(r"\b(?:my (?:e-?mail|address)(?: address)? is|i am|i'm|im|this is|it's|email)\W*$", re.I)


def self_identified_email(text: str) -> str | None:
    """The email only if the employee gives it as their own ("jane@x.com", "my email is jane@x.com"),
    not one that is merely mentioned ("my manager bob@x.com approved it")."""
    email = extract_email(text)
    if not email:
        return None
    before = text[: text.lower().find(email)].strip()
    return email if not before or SELF_ID_RE.search(before) else None


@graceful
def router(state: SmartDeskState) -> Command[Literal["it_agent", "hr_agent", "collect", "status", "chitchat", "__end__"]]:
    text = last_human(state)
    update: dict = {"escalation_reason": None, "cache_hit": False, "last_sources": [], "last_confidence": None}
    if typed_email_allowed(state) and (email := self_identified_email(text)):
        update["email"] = email
    if not text.strip():
        update["messages"] = [AIMessage(content="I didn't catch that — could you type your question?")]
        return Command(goto=END, update=update)

    try:
        decision = llm.invoke_safely(
            llm.structured(RouteDecision),
            [
                SystemMessage(content=prompts.ROUTER_SYSTEM),
                HumanMessage(content=f"Conversation so far:\n{format_history(state['messages'][:-1])}\n\nLatest message: {text}"),
            ],
        )
    except LLMUnavailable:
        decision = heuristic_route(text)

    update["domain"] = decision.domain or state.get("domain")
    update["query"] = decision.standalone_query or text
    goto = {
        "kb_question": "hr_agent" if update["domain"] == "HR" else "it_agent",
        "create_ticket": "collect",
        "ticket_status": "status",
        "chitchat": "chitchat",
        "out_of_scope": "chitchat",
    }[decision.intent]
    return Command(goto=goto, update=update)
