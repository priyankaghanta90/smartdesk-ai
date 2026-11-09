"""Ticket agent: offers a ticket when the KB can't help, collects the details, confirms, and creates it in Jira.

Nodes: escalate -> collect -> (get_email | get_details) -> confirm -> create.
get_email is also used by the status agent, which needs the email to look tickets up.

Every question to the employee is a LangGraph `interrupt()`: the graph pauses, the UI shows the
question, and the employee's reply resumes the same node. A ticket is created only from `confirm`.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from .. import llm
from ..ticketing import TicketDraft, TicketingError, TicketingUnavailable
from . import prompts
from .common import (
    SIGN_IN_MSG, TEAM, AgentDeps, _normalise, classify_reply, extract_email, format_history, graceful, is_no,
    last_human, typed_email_allowed,
)
from .state import SmartDeskState

# The whole message must be a retry request: "my monitor is flickering again" is a new ticket, not a retry.
RETRY_RE = re.compile(r"^\W*(?:(?:yes|ok|okay|please)\W+)*(?:try (?:it )?again|retry|again)(?:\W+please)?\W*$", re.I)
FRESH_DRAFT = {"draft_ticket": None, "details_asked": False}


class DraftSchema(BaseModel):
    summary: str
    description: str
    category: Literal["IT", "HR"]
    priority: Literal["Low", "Medium", "High", "Critical"]
    needs_details: bool = False


def format_draft(d: dict) -> str:
    return (
        "Here is the ticket I'll create for you:\n\n"
        f"- **Title:** {d['summary']}\n"
        f"- **Description:** {d['description']}\n"
        f"- **Category:** {TEAM.get(d['category'], d['category'])}\n"
        f"- **Priority:** {d['priority']}\n"
        f"- **Email:** {d['email']}\n\n"
        "Shall I go ahead and create it? You can also tell me anything you'd like to add or change."
    )


@graceful
def escalate(state: SmartDeskState) -> Command[Literal["collect", "router", "__end__"]]:
    team = TEAM.get(state.get("domain") or "IT", "IT Support")
    offer = (
        "I'm sorry, I don't have information about this in NovaTech's knowledge base, "
        "and I don't want to give you a wrong answer. "
        f"Would you like me to raise a support ticket with the **{team}** team so someone can help you?"
    )
    reply = interrupt({"kind": "offer_ticket", "message": offer})
    msgs = [AIMessage(content=offer), HumanMessage(content=reply)]
    decision = classify_reply(offer, reply)
    if decision.action in ("yes", "edit"):  # a new problem: never reuse an older draft
        return Command(goto="collect", update={"messages": msgs, **FRESH_DRAFT})
    if decision.action == "no":
        msgs.append(AIMessage(content="No problem. Is there anything else I can help you with?"))
        return Command(goto=END, update={"messages": msgs})
    return Command(goto="router", update={"messages": msgs})


@graceful
def collect(state: SmartDeskState) -> Command[Literal["collect", "get_email", "get_details", "confirm", "__end__"]]:
    if not state.get("email"):
        if not typed_email_allowed(state):
            msg = SIGN_IN_MSG.format(purpose="raise a ticket for you")
            return Command(goto=END, update={"messages": [AIMessage(content=msg)]})
        return Command(goto="get_email", update={"after_email": "collect"})
    if state.get("draft_ticket"):
        if RETRY_RE.search(last_human(state)):  # "try again" after Jira was unreachable
            return Command(goto="confirm")
        return Command(goto="collect", update=FRESH_DRAFT)  # a different request: draft from scratch

    context = format_history(state["messages"], 12)
    hint = f"Likely category: {state.get('domain') or 'unknown'}. Assistant note: {state.get('escalation_reason') or 'employee asked for a ticket'}."
    draft = llm.invoke_safely(
        llm.structured(DraftSchema),
        [SystemMessage(content=prompts.TICKET_DRAFT_SYSTEM), HumanMessage(content=f"{hint}\n\nConversation:\n{context}")],
    )
    if draft.needs_details and not state.get("details_asked"):
        return Command(goto="get_details")
    ticket = TicketDraft(state["email"], draft.summary, draft.description, draft.category, draft.priority)
    return Command(goto="confirm", update={"draft_ticket": ticket.as_dict()})


@graceful
def get_details(state: SmartDeskState) -> Command[Literal["collect", "__end__"]]:
    question = "Sure — could you briefly describe the issue or request, so the team has everything they need?"
    reply = interrupt({"kind": "ask_details", "message": question})
    msgs = [AIMessage(content=question), HumanMessage(content=reply)]
    if is_no(reply):  # "no, cancel" is a refusal, not the description
        msgs.append(AIMessage(content="Okay, I haven't created a ticket. Is there anything else I can help you with?"))
        return Command(goto=END, update={"messages": msgs, **FRESH_DRAFT})
    return Command(goto="collect", update={"details_asked": True, "messages": msgs})


@graceful
def get_email(state: SmartDeskState) -> Command[Literal["collect", "status", "router", "__end__"]]:
    purpose = "look up your tickets" if state.get("after_email") == "status" else "link the ticket to you"
    question = f"Could you please share your NovaTech email address so I can {purpose}?"
    msgs: list[AnyMessage] = []
    for _ in range(3):
        reply = interrupt({"kind": "ask_email", "message": question})
        msgs += [AIMessage(content=question), HumanMessage(content=reply)]
        if email := extract_email(reply):
            return Command(goto=state.get("after_email") or "router", update={"email": email, "messages": msgs})
        norm = _normalise(reply)
        if is_no(reply):
            msgs.append(AIMessage(content="No problem. Let me know if there's anything else I can help with."))
            return Command(goto=END, update={"messages": msgs})
        if "?" in reply or len(norm.split()) > 6:  # changed topic
            return Command(goto="router", update={"messages": msgs})
        question = "That doesn't look like a valid email address. Could you share it in the format name@novatech.com?"
    msgs.append(AIMessage(content="I still couldn't read a valid email address, so I'll stop here. Feel free to try again anytime."))
    return Command(goto=END, update={"messages": msgs})


@graceful
def confirm(state: SmartDeskState) -> Command[Literal["create", "confirm", "router", "__end__"]]:
    draft = state["draft_ticket"]
    preview = format_draft(draft)
    reply = interrupt({"kind": "confirm_ticket", "message": preview, "draft": draft})
    msgs = [AIMessage(content=preview), HumanMessage(content=reply)]
    decision = classify_reply(preview, reply)

    if decision.action == "yes":
        return Command(goto="create", update={"messages": msgs})
    if decision.action == "no":
        msgs.append(AIMessage(content="Okay, I haven't created the ticket. Is there anything else I can help you with?"))
        return Command(goto=END, update={"messages": msgs, "draft_ticket": None, "details_asked": False})
    if decision.action == "edit":
        updated = llm.invoke_safely(
            llm.structured(DraftSchema),
            [SystemMessage(content=prompts.TICKET_EDIT_SYSTEM),
             HumanMessage(content=f"Current ticket: {draft}\n\nEmployee's change request: {reply}")],
        )
        new_draft = TicketDraft(
            (typed_email_allowed(state) and extract_email(reply)) or draft["email"], updated.summary, updated.description,
            updated.category, decision.priority or updated.priority,
        ).as_dict()
        return Command(goto="confirm", update={"messages": msgs, "draft_ticket": new_draft})
    return Command(goto="router", update={"messages": msgs, "draft_ticket": None, "details_asked": False})


def make_create(deps: AgentDeps):
    def create(state: SmartDeskState) -> Command[Literal["__end__"]]:
        draft = TicketDraft(**state["draft_ticket"])
        if state.get("email_verified"):  # a signed-in employee can only raise tickets as themselves
            draft.email = state["email"]
        try:
            ticket = deps.tickets.create_ticket(draft)
        except TicketingUnavailable:
            # Keep the draft so "try again" goes straight back to confirmation.
            return Command(goto=END, update={"messages": [AIMessage(content=(
                "I'm sorry — our ticketing system (Jira) isn't reachable right now, so the ticket wasn't created. "
                "I've kept your details; just say \"try again\" in a few minutes."
            ))]})
        except TicketingError as exc:  # not transient, so a retry won't help: drop the draft
            return Command(goto=END, update={**FRESH_DRAFT, "messages": [AIMessage(content=(
                f"I couldn't create the ticket because of a ticketing configuration problem ({exc}). "
                "Please contact the IT Service Desk on ext. 2020."
            ))]})
        msg = (
            f"Done! Ticket **{ticket.key}** has been created for the {TEAM.get(draft.category, draft.category)} team.\n\n"
            f"You can track it here: {ticket.url}\n\nAsk me for an update anytime. Is there anything else I can help with?"
        )
        return Command(goto=END, update={"messages": [AIMessage(content=msg)], "draft_ticket": None, "details_asked": False})

    return create
