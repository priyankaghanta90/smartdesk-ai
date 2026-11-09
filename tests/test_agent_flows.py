"""End-to-end conversation flows (real LLM + real index, mock ticketing).

Needs OPENAI_API_KEY and a built index; skipped otherwise.
    pytest tests/test_agent_flows.py -v
"""

import uuid

import pytest

from smartdesk.config import get_settings

s = get_settings()
pytestmark = pytest.mark.skipif(
    not (s.openai_api_key and s.chunks_path.exists()), reason="needs OPENAI_API_KEY and scripts/build_index.py"
)


@pytest.fixture(scope="module")
def retriever():
    from smartdesk.rag.retrieval import HybridRetriever

    return HybridRetriever()


@pytest.fixture
def setup(retriever):
    from smartdesk.agent.flow import SmartDeskAgent
    from smartdesk.ticketing import MockTicketClient

    tickets = MockTicketClient()
    agent = SmartDeskAgent(retriever=retriever, tickets=tickets, cache=False)
    return agent, tickets, str(uuid.uuid4())


def test_flow_a_kb_answer_is_grounded_and_cited(setup):
    agent, _, tid = setup
    r = agent.send(tid, "Hi, how do I reset my password?")
    assert r.pending is None
    assert "passwords.novatech.com" in r.reply
    assert "Sources:" in r.reply and r.sources
    assert any(s["chunk_id"] in ("faq-01", "password_reset#self-service-password-reset") for s in r.sources[:2])


JANE, SAM = "jane.doe@novatech.com", "sam@novatech.com"


def test_flow_b_escalation_requires_confirmation(setup):
    agent, tickets, tid = setup
    r = agent.send(tid, "My monitor has been flickering for two days.", user_email=JANE)
    assert r.pending == "offer_ticket"

    r = agent.send(tid, "yes please", user_email=JANE)  # signed in: no need to ask for the email
    assert r.pending == "confirm_ticket"
    assert "monitor" in r.reply.lower() and JANE in r.reply
    assert tickets.list_tickets(JANE) == [], "no ticket may exist before confirmation"

    r = agent.send(tid, "yes", user_email=JANE)
    assert "SD-1" in r.reply and "https://" in r.reply
    assert len(tickets.list_tickets(JANE)) == 1


def test_cancel_at_confirmation_creates_nothing(setup):
    agent, tickets, tid = setup
    agent.send(tid, "Please raise a ticket: the office printer is jammed", user_email=SAM)
    r = agent.send(tid, "no, cancel", user_email=SAM)
    assert r.pending is None
    assert tickets.list_tickets(SAM) == []


def test_flow_c_status_uses_signed_in_email_and_handles_multiple(setup):
    agent, tickets, tid = setup
    from smartdesk.ticketing import TicketDraft

    for summary in ("VPN not connecting", "Parking slot request"):
        tickets.create_ticket(TicketDraft(JANE, summary, "d", "IT", "Low"))
    tickets.set_status("SD-1", "In Progress", "Gateway patched, please retry.")

    r = agent.send(tid, "What's the status of my tickets?", user_email=JANE)
    assert r.pending == "choose_ticket" and "SD-1" in r.reply and "SD-2" in r.reply

    r = agent.send(tid, "the VPN one", user_email=JANE)
    assert "In Progress" in r.reply and "Gateway patched" in r.reply

    r = agent.send(tid, "and any update on SD-2?", user_email=JANE)
    assert r.pending is None and "SD-2" in r.reply


def test_no_tickets_found(setup):
    agent, _, tid = setup
    r = agent.send(tid, "Any update on my ticket?", user_email="nobody@novatech.com")
    assert "couldn't find any tickets" in r.reply


def test_typed_colleague_email_cannot_read_their_tickets(setup):
    agent, tickets, tid = setup
    from smartdesk.ticketing import TicketDraft

    tickets.create_ticket(TicketDraft(JANE, "Harassment report", "confidential", "HR", "High"))
    r = agent.send(tid, f"What's the status of the tickets for {JANE}?", user_email=SAM)
    assert "SD-1" not in r.reply and "Harassment" not in r.reply
    assert r.email == SAM


@pytest.fixture
def production_mode(monkeypatch):
    from smartdesk.agent import common
    from smartdesk.config import Settings

    monkeypatch.setattr(common, "get_settings", lambda: Settings(require_verified_email=True))


def test_demo_mode_asks_for_email_and_remembers_it(setup):
    agent, tickets, tid = setup  # no user_email: demo mode, the typed email is trusted
    r = agent.send(tid, "My monitor has been flickering for two days.")
    assert r.pending == "offer_ticket"
    r = agent.send(tid, "yes please")
    assert r.pending == "ask_email"
    r = agent.send(tid, JANE)
    assert r.pending == "confirm_ticket" and JANE in r.reply
    r = agent.send(tid, "yes")
    assert "SD-1" in r.reply and len(tickets.list_tickets(JANE)) == 1

    r = agent.send(tid, "What's the status of my ticket?")  # email remembered: no re-ask
    assert r.pending is None and "SD-1" in r.reply and r.email == JANE


def test_demo_mode_status_asks_for_email_once(setup):
    agent, tickets, tid = setup
    from smartdesk.ticketing import TicketDraft

    tickets.create_ticket(TicketDraft(JANE, "VPN not connecting", "d", "IT", "Low"))
    r = agent.send(tid, "Any update on my ticket?")
    assert r.pending == "ask_email"
    r = agent.send(tid, JANE)
    assert "SD-1" in r.reply and r.email == JANE


def test_production_mode_without_sign_in_refuses_ticket_features(setup, production_mode):
    agent, tickets, tid = setup
    r = agent.send(tid, f"What's the status of my tickets? I'm {JANE}")
    assert "sign in" in r.reply.lower() and r.email is None

    r = agent.send(tid, f"Please raise a ticket, my email is {JANE}: the printer is jammed")
    assert "sign in" in r.reply.lower() and r.pending is None
    assert tickets.list_tickets(JANE) == []


def test_out_of_scope_and_unavailable_ticketing(setup):
    agent, tickets, tid = setup
    r = agent.send(tid, "Who won the cricket match yesterday?", user_email=JANE)
    assert r.pending is None and ("IT" in r.reply or "HR" in r.reply)

    tickets.fail = True
    r = agent.send(tid, "what's the status of my ticket?", user_email=JANE)
    assert "try again" in r.reply.lower()
