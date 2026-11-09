import json

import pytest
import requests
import responses

from smartdesk.ticketing import (
    EmailIndex,
    JiraTicketClient,
    TicketDraft,
    TicketingConfigError,
    TicketingUnavailable,
)
from smartdesk.ticketing.base import request_label, requester_label

BASE = "https://example.atlassian.net"
SEARCH = f"{BASE}/rest/api/3/search/jql"
JANE = "jane.doe@novatech.com"


@pytest.fixture
def client(tmp_path):
    return JiraTicketClient(BASE, "bot@example.com", "token", "SD", email_index=EmailIndex(tmp_path / "t.db"))


def draft():
    return TicketDraft(JANE, "Monitor flickering", "Flickers for two days.", "IT", "Medium", request_id="req1")


def no_existing_request():
    """The idempotency lookup that runs before every create finds nothing."""
    responses.post(SEARCH, json={"issues": []})


def creates(calls) -> list:
    return [c for c in calls if c.request.method == "POST" and c.request.url.endswith("/rest/api/3/issue")]


def test_requester_labels_are_unique_per_email():
    labels = {requester_label(e) for e in (JANE, "jane_doe@novatech.com", "jane-doe@novatech.com")}
    assert len(labels) == 3
    assert requester_label(" Jane.Doe@NovaTech.com ") == requester_label(JANE)  # case/space-insensitive
    assert all(" " not in label for label in labels)


@responses.activate
def test_create_ticket_sends_labels_and_returns_key(client):
    no_existing_request()
    responses.post(f"{BASE}/rest/api/3/issue", json={"key": "SD-7"}, status=201)
    ticket = client.create_ticket(draft())

    assert ticket.key == "SD-7" and ticket.url == f"{BASE}/browse/SD-7"
    fields = json.loads(creates(responses.calls)[0].request.body)["fields"]
    assert requester_label(JANE) in fields["labels"] and "cat-it" in fields["labels"]
    assert request_label("req1") in fields["labels"]
    assert fields["description"]["type"] == "doc"
    assert client.email_index.keys_for(JANE) == ["SD-7"]


@responses.activate
def test_create_falls_back_when_priority_field_unsupported(client):
    no_existing_request()
    responses.post(f"{BASE}/rest/api/3/issue", json={"errors": {"priority": "cannot be set"}}, status=400)
    responses.post(f"{BASE}/rest/api/3/issue", json={"key": "SD-8"}, status=201)
    assert client.create_ticket(draft()).key == "SD-8"
    retry_fields = json.loads(creates(responses.calls)[1].request.body)["fields"]
    assert "priority" not in retry_fields and "prio-medium" in retry_fields["labels"]


@responses.activate
def test_create_is_not_blindly_retried_after_a_server_error(client):
    responses.post(SEARCH, json={"issues": []})  # before the create
    responses.post(f"{BASE}/rest/api/3/issue", status=503)
    responses.post(SEARCH, json={"issues": []})  # after the ambiguous failure: still not there
    with pytest.raises(TicketingUnavailable):
        client.create_ticket(draft())
    assert len(creates(responses.calls)) == 1, "a 5xx may mean Jira created it; re-POSTing would duplicate"


@responses.activate
def test_ambiguous_create_returns_the_ticket_jira_actually_made(client):
    made = {"key": "SD-12", "fields": {"summary": "Monitor flickering", "status": {"name": "To Do"}, "labels": ["cat-it"]}}
    responses.post(SEARCH, json={"issues": []})
    responses.post(f"{BASE}/rest/api/3/issue", body=requests.ReadTimeout("read timed out"))
    responses.post(SEARCH, json={"issues": [made]})
    assert client.create_ticket(draft()).key == "SD-12"
    assert len(creates(responses.calls)) == 1
    assert client.email_index.keys_for(JANE) == ["SD-12"]


@responses.activate
def test_try_again_after_ambiguous_failure_does_not_duplicate(client):
    made = {"key": "SD-12", "fields": {"summary": "Monitor flickering", "status": {"name": "To Do"}}}
    responses.post(SEARCH, json={"issues": [made]})  # the earlier attempt did reach Jira
    assert client.create_ticket(draft()).key == "SD-12"
    assert creates(responses.calls) == []


@responses.activate
def test_connect_errors_on_create_are_retried(client):
    no_existing_request()
    responses.post(f"{BASE}/rest/api/3/issue", body=requests.ConnectTimeout("no route"))
    responses.post(f"{BASE}/rest/api/3/issue", json={"key": "SD-9"}, status=201)
    assert client.create_ticket(draft()).key == "SD-9"  # the request never left, so retrying is safe


@responses.activate
def test_bad_credentials_is_config_error(client):
    responses.post(SEARCH, status=401)
    with pytest.raises(TicketingConfigError):
        client.list_tickets(JANE)


@responses.activate
def test_list_and_get_ticket_with_comments(client):
    issue = {"key": "SD-1", "fields": {"summary": "VPN down", "status": {"name": "In Progress"},
                                       "priority": {"name": "High"}, "labels": ["cat-it"], "updated": "2026-10-20T10:00"}}
    responses.post(SEARCH, json={"issues": [issue]})
    responses.get(f"{BASE}/rest/api/3/issue/SD-1", json=issue)
    responses.get(f"{BASE}/rest/api/3/issue/SD-1/comment", json={"comments": [{
        "author": {"displayName": "Ravi (IT)"}, "created": "2026-10-21",
        "body": {"type": "doc", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Gateway patched."}]}]},
    }]})

    tickets = client.list_tickets(JANE)
    assert [t.key for t in tickets] == ["SD-1"] and tickets[0].category == "IT"
    jql = json.loads(responses.calls[0].request.body)["jql"]
    assert f'labels = "{requester_label(JANE)}"' in jql

    detail = client.get_ticket("SD-1")
    assert detail.status == "In Progress"
    assert detail.comments[0].body == "Gateway patched." and detail.comments[0].author == "Ravi (IT)"


@responses.activate
def test_list_merges_locally_known_keys_missing_from_search(client):
    client.email_index.add(JANE, "SD-9")
    responses.post(SEARCH, json={"issues": []})
    responses.get(f"{BASE}/rest/api/3/issue/SD-9", json={"key": "SD-9", "fields": {"summary": "New", "status": {"name": "To Do"}}})
    assert [t.key for t in client.list_tickets(JANE)] == ["SD-9"]


@responses.activate
def test_deleted_ticket_in_local_index_does_not_break_the_lookup(client):
    client.email_index.add(JANE, "SD-5")  # deleted in Jira since
    client.email_index.add(JANE, "SD-6")
    responses.post(SEARCH, json={"issues": []})
    responses.get(f"{BASE}/rest/api/3/issue/SD-5", status=404)
    responses.get(f"{BASE}/rest/api/3/issue/SD-6", json={"key": "SD-6", "fields": {"summary": "Wi-Fi", "status": {"name": "To Do"}}})

    assert [t.key for t in client.list_tickets(JANE)] == ["SD-6"]
    assert client.email_index.keys_for(JANE) == ["SD-6"], "the deleted key is pruned"


def test_missing_credentials_rejected():
    with pytest.raises(TicketingConfigError):
        JiraTicketClient("", "", "", "SD")


@responses.activate
def test_get_ticket_other_4xx_is_a_ticketing_error(client):
    from smartdesk.ticketing import TicketingError

    responses.get(f"{BASE}/rest/api/3/issue/SD-5", json={"errorMessages": ["Bad request"]}, status=400)
    with pytest.raises(TicketingError):
        client.get_ticket("SD-5")
