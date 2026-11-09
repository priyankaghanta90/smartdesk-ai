"""Status agent: finds the employee's tickets in Jira and shows status and support-team comments.

Nodes: status -> (get_email if no email yet) -> choose_ticket when there are several tickets.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from .. import llm
from ..ticketing import Ticket
from . import prompts
from .common import KEY_RE, SIGN_IN_MSG, AgentDeps, _normalise, graceful, is_no, last_human, typed_email_allowed
from .state import SmartDeskState

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": -1}


class TicketChoice(BaseModel):
    # The key, not the list position: with keys like IT-8 in a 5-item list the model returned 8 as the "number".
    key: Optional[str] = Field(None, description="Key of the ticket the employee means, e.g. IT-8, or null")


def format_ticket_details(t: Ticket) -> str:
    lines = [f"**{t.key}: {t.summary}** — Status: **{t.status}**"]
    meta = [x for x in (f"Priority: {t.priority}" if t.priority else "", f"Last updated: {t.updated[:10]}" if t.updated else "") if x]
    if meta:
        lines.append(" · ".join(meta))
    if t.comments:
        lines.append("\nLatest updates from the support team:")
        lines += [f"- \"{c.body.strip()}\" — {c.author}, {c.created[:10]}" for c in t.comments[:3]]
    else:
        lines.append("\nNo comments from the support team yet.")
    lines.append(f"\nTrack it here: {t.url}\n\nIs there anything else I can help you with?")
    return "\n".join(lines)


def resolve_choice(reply: str, candidates: list[dict]) -> int | None:
    """Return a 0-based index into candidates, or None."""
    keys = [c["key"] for c in candidates]
    named = KEY_RE.findall(reply.upper())
    for key in named:
        if key in keys:
            return keys.index(key)
    if named:  # e.g. "SD-3" when the list is SD-10..SD-12: not one of theirs, don't read "3" as a position
        return None
    norm = _normalise(reply)
    digits = re.findall(r"\b(\d{1,2})\b", reply)
    # Only a short reply ("2", "number 2 please") is a position; "I have 2 questions about leave" is not.
    if digits and len(reply.split()) <= 3 and 1 <= int(digits[0]) <= len(candidates):
        return int(digits[0]) - 1
    for word, n in ORDINALS.items():
        if word in norm.split():
            return len(candidates) - 1 if n == -1 else (n - 1 if n <= len(candidates) else None)
    listing = "\n".join(f"- {c['key']}: {c['summary']} ({c['status']})" for c in candidates)
    choice = llm.invoke_safely(
        llm.structured(TicketChoice),
        [SystemMessage(content=prompts.TICKET_CHOICE_SYSTEM.format(tickets=listing)), HumanMessage(content=reply)],
    )
    picked = (choice.key or "").strip().upper()
    return keys.index(picked) if picked in keys else None


def make_status(deps: AgentDeps):
    @graceful
    def status(state: SmartDeskState) -> Command[Literal["get_email", "choose_ticket", "__end__"]]:
        email = state.get("email")
        if not email:
            if not typed_email_allowed(state):
                msg = SIGN_IN_MSG.format(purpose="look up your tickets")
                return Command(goto=END, update={"messages": [AIMessage(content=msg)]})
            return Command(goto="get_email", update={"after_email": "status"})
        tickets = deps.tickets.list_tickets(email)

        if not tickets:
            msg = (
                f"I couldn't find any tickets linked to **{email}**. If you raised one with a different email, "
                "share it and I'll check again — or I can create a new ticket for you."
            )
            return Command(goto=END, update={"messages": [AIMessage(content=msg)]})

        mentioned = [k for k in KEY_RE.findall(last_human(state).upper()) if k in {t.key for t in tickets}]
        if len(tickets) == 1 or mentioned:
            key = mentioned[0] if mentioned else tickets[0].key
            prefix = "I found one ticket for you:\n\n" if len(tickets) == 1 else ""
            details = deps.tickets.get_ticket(key)
            return Command(goto=END, update={"messages": [AIMessage(content=prefix + format_ticket_details(details))]})

        candidates = [{"key": t.key, "summary": t.summary, "status": t.status} for t in tickets]
        return Command(goto="choose_ticket", update={"ticket_candidates": candidates})

    return status


def make_choose_ticket(deps: AgentDeps):
    @graceful
    def choose_ticket(state: SmartDeskState) -> Command[Literal["router", "__end__"]]:
        candidates = state["ticket_candidates"]
        listing = "\n".join(f"{i}. **{c['key']}**: {c['summary']} — {c['status']}" for i, c in enumerate(candidates, 1))
        question = f"I found {len(candidates)} tickets linked to {state['email']}:\n\n{listing}\n\nWhich one would you like details on?"
        reply = interrupt({"kind": "choose_ticket", "message": question, "tickets": candidates})
        msgs = [AIMessage(content=question), HumanMessage(content=reply)]

        index = resolve_choice(reply, candidates)
        if index is None:
            if is_no(reply):
                msgs.append(AIMessage(content="Alright. Is there anything else I can help you with?"))
                return Command(goto=END, update={"messages": msgs})
            return Command(goto="router", update={"messages": msgs})
        details = deps.tickets.get_ticket(candidates[index]["key"])
        msgs.append(AIMessage(content=format_ticket_details(details)))
        return Command(goto=END, update={"messages": msgs})

    return choose_ticket
