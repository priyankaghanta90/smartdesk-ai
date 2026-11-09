"""The complete SmartDesk flow: which agents exist, how they connect, and the session API used by the
CLI, Streamlit app, tests and eval.

Every message enters at the router. Each node returns a `Command(goto=...)` that picks the next node,
so the arrows below are decided at runtime; the `Literal[...]` return types on each node list where it can go.

    START ─► router ─┬─► it_agent ─┐                        (router.py, it.py)
                     ├─► hr_agent ─┴─► answer ─► END        (hr.py; RAG lives in smartdesk/rag/)
                     │                 └─ KB can't answer ─► escalate
                     │
                     ├─► collect ◄───────────── escalate ("raise a ticket?" yes)   (ticket.py)
                     │     ├─► get_email ─► collect     (dev mode only; normally the email comes from sign-in)
                     │     ├─► get_details ─► collect
                     │     └─► confirm ─► create ─► END     (ticket created only after "yes")
                     │
                     ├─► status ─┬─► get_email ─► status    (status.py)
                     │           ├─► choose_ticket ─► END   (several tickets)
                     │           └─► END                    (none or one ticket)
                     │
                     └─► chitchat ─► END                    (chitchat.py; greetings / out of scope)

Pauses: escalate, get_email, get_details, confirm and choose_ticket ask the employee a question with
`interrupt()`. The graph stops, `send()` returns the question, and the next `send()` resumes that node.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from ..config import get_settings
from ..guardrails.pii import PiiGuard
from ..llm import embeddings
from ..rag.semantic_cache import SemanticCache
from ..rag.retrieval import HybridRetriever
from ..ticketing import TicketClient, make_ticket_client
from . import chitchat, hr, it, router, status, ticket
from .common import AgentDeps
from .state import SmartDeskState


def build_graph(deps: AgentDeps, checkpointer=None):
    graph = StateGraph(SmartDeskState)

    # 1. Router: every message starts here.
    graph.add_node("router", router.router)

    # 2. Knowledge-base specialists (RAG).
    graph.add_node("it_agent", it.make_it_agent(deps))
    graph.add_node("hr_agent", hr.make_hr_agent(deps))

    # 3. Ticket creation (escalation, human-in-the-loop).
    graph.add_node("escalate", ticket.escalate)
    graph.add_node("collect", ticket.collect)
    graph.add_node("get_email", ticket.get_email)
    graph.add_node("get_details", ticket.get_details)
    graph.add_node("confirm", ticket.confirm)
    graph.add_node("create", ticket.make_create(deps))

    # 4. Ticket status.
    graph.add_node("status", status.make_status(deps))
    graph.add_node("choose_ticket", status.make_choose_ticket(deps))

    # 5. Small talk and out-of-scope requests.
    graph.add_node("chitchat", chitchat.chitchat)

    graph.add_edge(START, "router")
    return graph.compile(checkpointer=checkpointer or MemorySaver())


@dataclass
class TurnResult:
    reply: str
    pending: str | None = None  # interrupt kind awaiting the employee: offer_ticket | ask_email | confirm_ticket | ...
    sources: list[dict] = field(default_factory=list)
    confidence: float | None = None
    cache_hit: bool = False
    email: str | None = None
    escalation_reason: str | None = None


class SmartDeskAgent:
    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        tickets: TicketClient | None = None,
        cache: SemanticCache | None | bool = True,
    ):
        s = get_settings()
        if cache is True:
            cache = SemanticCache(s.sqlite_path, embeddings(), s.cache_similarity)
        self.tickets = tickets or make_ticket_client()
        retriever = retriever or HybridRetriever()
        guard = background = None
        if cache:
            # Cache writes are PII-screened on a background thread, so the reply never waits for the check.
            guard = PiiGuard(c.text for c in retriever.chunks)
            background = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guardrail")
            threading.Thread(target=guard.warm_up, daemon=True).start()  # ~30 s model load, off the request path
        self.deps = AgentDeps(retriever, self.tickets, cache or None, guard, background)
        self.graph = build_graph(self.deps)

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _pending_interrupt(self, thread_id: str):
        snapshot = self.graph.get_state(self._config(thread_id))
        for task in snapshot.tasks:
            if task.interrupts:
                return task.interrupts[0].value
        return None

    def send(self, thread_id: str, text: str, user_email: str | None = None) -> TurnResult:
        """`user_email` is the signed-in identity; it is re-asserted every turn and can't be changed from chat."""
        config = self._config(thread_id)
        identity = {"email": user_email.lower(), "email_verified": True} if user_email else {}
        if self._pending_interrupt(thread_id) is not None:
            self.graph.invoke(Command(resume=text, update=identity or None), config)
        else:
            self.graph.invoke({"messages": [HumanMessage(content=text)], **identity}, config)

        state = self.graph.get_state(config).values
        pending = self._pending_interrupt(thread_id)
        if pending is not None:
            reply, kind = pending["message"], pending["kind"]
        else:
            last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
            reply, kind = (last_ai.content if last_ai else ""), None
        return TurnResult(
            reply=reply,
            pending=kind,
            sources=state.get("last_sources") or [],
            confidence=state.get("last_confidence"),
            cache_hit=bool(state.get("cache_hit")),
            email=state.get("email"),
            escalation_reason=state.get("escalation_reason"),
        )

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
