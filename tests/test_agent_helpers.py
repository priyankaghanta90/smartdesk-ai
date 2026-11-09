from smartdesk.agent.common import extract_email
from smartdesk.agent.router import heuristic_route
from smartdesk.agent.status import resolve_choice
from smartdesk.agent.ticket import format_draft

CANDIDATES = [
    {"key": "SD-3", "summary": "Monitor flickering", "status": "In Progress"},
    {"key": "SD-2", "summary": "Parking slot", "status": "To Do"},
    {"key": "SD-1", "summary": "Printer jam", "status": "Done"},
]


def test_extract_email():
    assert extract_email("sure, Jane.Doe@NovaTech.com thanks") == "jane.doe@novatech.com"
    assert extract_email("my email is jane at novatech") is None


def test_resolve_choice_by_key_number_and_ordinal():
    assert resolve_choice("sd-2 please", CANDIDATES) == 1
    assert resolve_choice("3", CANDIDATES) == 2
    assert resolve_choice("the first one", CANDIDATES) == 0
    assert resolve_choice("last", CANDIDATES) == 2


def test_draft_preview_lists_all_mandatory_fields():
    text = format_draft({"summary": "Monitor flickering", "description": "Two days.", "category": "IT",
                         "priority": "Medium", "email": "jane.doe@novatech.com"})
    for part in ("Title", "Description", "Category", "Priority", "jane.doe@novatech.com", "Shall I go ahead"):
        assert part in text


def test_heuristic_router_used_when_llm_down():
    assert heuristic_route("any update on my ticket?").intent == "ticket_status"
    assert heuristic_route("how many sick leaves do I get").domain == "HR"


def test_ticket_key_not_in_list_is_not_read_as_a_position():
    candidates = [{"key": f"SD-{n}", "summary": "x", "status": "To Do"} for n in (10, 11, 12)]
    assert resolve_choice("SD-3", candidates) is None  # not "candidate #3"
    assert resolve_choice("SD-11", candidates) == 1


def test_heuristic_router_matches_whole_words_only():
    for it_question in ("Chrome keeps crashing", "my display is blank", "syntax error in my Outlook rule"):
        assert heuristic_route(it_question).domain == "IT", it_question
    for hr_question in ("when is payday?", "question about my 401(k)", "who do I ask in HR?", "W-2 copy"):
        assert heuristic_route(hr_question).domain == "HR", hr_question


def test_yes_no_replies_are_decided_without_the_llm():
    from smartdesk.agent.common import is_no, is_yes

    for reply in ("no, cancel", "No thanks!", "nope", "stop", "don't"):
        assert is_no(reply) and not is_yes(reply), reply
    for reply in ("yes", "Yes please, go ahead", "ok do it"):
        assert is_yes(reply) and not is_no(reply), reply
    for ambiguous in ("thanks", "no, change the priority to high", "yes but add my laptop model"):
        assert not is_yes(ambiguous) and not is_no(ambiguous), ambiguous


def test_new_ticket_request_does_not_reuse_a_stale_draft():
    from langchain_core.messages import HumanMessage

    from smartdesk.agent.ticket import collect

    stale = {"summary": "VPN broken", "description": "d", "category": "IT", "priority": "Medium",
             "email": "jane.doe@novatech.com", "request_id": "abc"}
    state = {"email": "jane.doe@novatech.com", "email_verified": True, "draft_ticket": stale,
             "messages": [HumanMessage(content="please raise a ticket, the printer keeps jamming")]}
    cmd = collect(state)
    assert cmd.goto == "collect" and cmd.update["draft_ticket"] is None  # re-drafted from scratch

    state["messages"] = [HumanMessage(content="try again")]
    assert collect(state).goto == "confirm"  # explicit retry keeps the draft


def test_answers_that_depend_on_history_are_not_cached(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    from smartdesk.agent import common
    from smartdesk.rag.answer import RagAnswer

    stored = []

    class FakeCache:
        def lookup(self, *a):
            return None

        def store(self, *a):
            stored.append(a)

    monkeypatch.setattr(common, "answer_question", lambda *a, **k: RagAnswer("Use GlobalProtect [S1].", False, "ok", 5.0))
    deps = common.AgentDeps(retriever=None, tickets=None, cache=FakeCache())

    first = {"query": "How do I install the VPN client?", "messages": [HumanMessage(content="How do I install the VPN?")]}
    common.answer_from_kb(first, deps, "IT", "persona")
    assert len(stored) == 1

    follow_up = {"query": "How do I install the VPN client on a Mac?", "messages": [
        HumanMessage(content="I'm on a Mac"), AIMessage(content="Noted."), HumanMessage(content="how do I install the VPN?")]}
    common.answer_from_kb(follow_up, deps, "IT", "persona")
    assert len(stored) == 1  # not cached: the answer used earlier turns


def test_saying_no_when_asked_for_details_cancels(monkeypatch):
    from smartdesk.agent import ticket

    monkeypatch.setattr(ticket, "interrupt", lambda payload: "no, cancel")
    cmd = ticket.get_details({"messages": []})
    assert cmd.goto == "__end__" and cmd.update["draft_ticket"] is None
    assert "haven't created a ticket" in cmd.update["messages"][-1].content


def test_retry_only_when_the_whole_message_is_a_retry():
    from smartdesk.agent.ticket import RETRY_RE

    for text in ("try again", "Try again please", "yes, try again", "retry", "again!", "ok try it again"):
        assert RETRY_RE.search(text), text
    for text in ("my monitor is flickering again, please raise a ticket", "can you try again to explain PTO?"):
        assert not RETRY_RE.search(text), text


def test_only_a_self_identified_email_is_adopted():
    from smartdesk.agent.router import self_identified_email

    assert self_identified_email("jane.doe@novatech.com") == "jane.doe@novatech.com"
    assert self_identified_email("My email is Jane.Doe@NovaTech.com") == "jane.doe@novatech.com"
    assert self_identified_email("I'm jane.doe@novatech.com, any update on my ticket?") == "jane.doe@novatech.com"
    assert self_identified_email("my manager bob@novatech.com approved it, how do I claim?") is None
    assert self_identified_email("what's the status of the tickets for jane.doe@novatech.com?") is None


def test_a_number_in_a_long_reply_is_not_a_ticket_choice(monkeypatch):
    from smartdesk.agent import status

    monkeypatch.setattr(status.llm, "invoke_safely", lambda *a, **k: status.TicketChoice(key=None))
    assert resolve_choice("number 2 please", CANDIDATES) == 1
    assert resolve_choice("actually I have 2 questions about my leave balance", CANDIDATES) is None


def test_follow_up_without_a_domain_stays_with_the_previous_specialist(monkeypatch):
    from langchain_core.messages import HumanMessage

    from smartdesk.agent import router

    decision = router.RouteDecision(intent="kb_question", domain=None, standalone_query="How do I carry PTO forward?")
    monkeypatch.setattr(router.llm, "invoke_safely", lambda *a, **k: decision)
    cmd = router.router({"domain": "HR", "email_verified": True,
                         "messages": [HumanMessage(content="and how do I carry it forward?")]})
    assert cmd.goto == "hr_agent" and cmd.update["domain"] == "HR"
