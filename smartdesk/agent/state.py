"""Conversation state persisted per session (thread_id) by the LangGraph checkpointer."""

from typing import Optional

from langgraph.graph import MessagesState


class SmartDeskState(MessagesState):
    email: Optional[str]  # remembered for the whole session once given
    email_verified: bool  # True when email came from sign-in, not from chat text
    domain: Optional[str]  # "IT" | "HR" from the router
    query: Optional[str]  # standalone rewrite of the latest question
    escalation_reason: Optional[str]
    draft_ticket: Optional[dict]  # TicketDraft fields awaiting confirmation
    details_asked: bool
    ticket_candidates: list[dict]
    after_email: Optional[str]  # node to return to once the email is collected
    # diagnostics for the UI
    last_sources: list[dict]
    last_confidence: Optional[float]
    cache_hit: bool
