"""Jira Cloud REST v3 client.

Ownership: every SmartDesk ticket carries the label `requester-<email>` (plus `smartdesk`,
`cat-it|cat-hr`), so "my tickets" is a JQL label query. The email is also written in the
description and mirrored in the local EmailIndex as a fallback.
"""

from __future__ import annotations

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .base import (
    Comment,
    Ticket,
    TicketDraft,
    TicketingConfigError,
    TicketingError,
    TicketingUnavailable,
    request_label,
    requester_label,
)
from .email_index import EmailIndex

FIELDS = ["summary", "status", "priority", "labels", "created", "updated"]
MAX_INDEX_LOOKUPS = 10  # bounds the per-key fallback in list_tickets


class _Retryable(Exception):
    pass


class _Ambiguous(Exception):
    """A create may or may not have reached Jira; never blindly retried."""


def _adf(text: str) -> dict:
    """Plain text -> Atlassian Document Format (one paragraph per line)."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": line}]}
            for line in text.splitlines()
            if line.strip()
        ],
    }


def _adf_to_text(node) -> str:
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("text", "")
    parts = [_adf_to_text(child) for child in node.get("content", [])]
    sep = "\n" if node.get("type") in ("doc", "bulletList", "orderedList") else ""
    return sep.join(p for p in parts if p)


class JiraTicketClient:
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        project_key: str,
        issue_type: str = "Task",
        timeout_s: float = 10.0,
        email_index: EmailIndex | None = None,
    ):
        if not (base_url and email and api_token):
            raise TicketingConfigError("JIRA_BASE_URL, JIRA_EMAIL and JIRA_API_TOKEN must be set")
        self.base_url = base_url.rstrip("/")
        self.project_key = project_key
        self.issue_type = issue_type
        self.timeout_s = timeout_s
        self.email_index = email_index
        self.session = requests.Session()
        self.session.auth = (email, api_token)
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    # --- HTTP with retries -----------------------------------------------------
    @retry(
        retry=retry_if_exception_type(_Retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _send(self, method: str, path: str, idempotent: bool = True, **kwargs) -> requests.Response:
        """Non-idempotent calls (creating an issue) are only retried when Jira certainly didn't process them:
        the connection never opened, or a 429. A read timeout or 5xx may mean the issue was created."""
        try:
            resp = self.session.request(method, f"{self.base_url}{path}", timeout=self.timeout_s, **kwargs)
        except requests.ConnectTimeout as exc:
            raise _Retryable(str(exc)) from exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            if idempotent:
                raise _Retryable(str(exc)) from exc
            raise _Ambiguous(str(exc)) from exc
        if resp.status_code == 429 or (resp.status_code >= 500 and idempotent):
            raise _Retryable(f"HTTP {resp.status_code}")
        if resp.status_code >= 500:
            raise _Ambiguous(f"HTTP {resp.status_code}")
        return resp

    def _request(self, method: str, path: str, idempotent: bool = True, **kwargs) -> requests.Response:
        try:
            resp = self._send(method, path, idempotent=idempotent, **kwargs)
        except (_Retryable, _Ambiguous) as exc:
            raise TicketingUnavailable(f"Jira unreachable: {exc}") from exc
        if resp.status_code in (401, 403):
            raise TicketingConfigError("Jira rejected the credentials (check JIRA_EMAIL / JIRA_API_TOKEN)")
        return resp

    def _url(self, key: str) -> str:
        return f"{self.base_url}/browse/{key}"

    def _to_ticket(self, issue: dict) -> Ticket:
        f = issue.get("fields", {})
        labels = f.get("labels") or []
        category = "IT" if "cat-it" in labels else "HR" if "cat-hr" in labels else ""
        return Ticket(
            key=issue["key"],
            summary=f.get("summary", ""),
            status=(f.get("status") or {}).get("name", "Unknown"),
            url=self._url(issue["key"]),
            category=category,
            priority=(f.get("priority") or {}).get("name", ""),
            created=f.get("created", ""),
            updated=f.get("updated", ""),
        )

    # --- public API ------------------------------------------------------------------
    def create_ticket(self, draft: TicketDraft) -> Ticket:
        # A previous attempt of this same draft may have reached Jira (timeout / 5xx, then "try again").
        if existing := self._find_request(draft):
            return existing
        try:
            return self._create(draft)
        except TicketingUnavailable:
            if existing := self._find_request(draft):
                return existing
            raise

    def _find_request(self, draft: TicketDraft) -> Ticket | None:
        """Best effort: Jira's search index can lag a few seconds behind a create."""
        try:
            found = self._search(f'project = "{self.project_key}" AND labels = "{request_label(draft.request_id)}"')
        except TicketingError:  # the lookup is only a safety net; never let it block creating the ticket
            return None
        if found and self.email_index:
            self.email_index.add(draft.email, found[0].key)
        return found[0] if found else None

    def _create(self, draft: TicketDraft) -> Ticket:
        description = (
            f"{draft.description}\n"
            f"Reported by: {draft.email}\n"
            f"Category: {draft.category}\n"
            f"Priority (requested): {draft.priority}\n"
            f"Created by SmartDesk AI."
        )
        labels = [
            "smartdesk", f"cat-{draft.category.lower()}", requester_label(draft.email), request_label(draft.request_id),
        ]
        fields = {
            "project": {"key": self.project_key},
            "issuetype": {"name": self.issue_type},
            "summary": draft.summary[:250],
            "description": _adf(description),
            "labels": labels,
            "priority": {"name": draft.priority},
        }
        resp = self._request("POST", "/rest/api/3/issue", idempotent=False, json={"fields": fields})
        if resp.status_code == 400 and "priority" in resp.text.lower():
            # Team-managed projects often don't expose the priority field: keep it as a label instead.
            fields.pop("priority")
            fields["labels"] = labels + [f"prio-{draft.priority.lower()}"]
            resp = self._request("POST", "/rest/api/3/issue", idempotent=False, json={"fields": fields})
        if resp.status_code == 404:
            raise TicketingConfigError(f"Jira project {self.project_key!r} not found")
        if resp.status_code >= 400:
            raise TicketingConfigError(f"Jira refused the ticket: HTTP {resp.status_code} {resp.text[:300]}")

        key = resp.json()["key"]
        if self.email_index:
            self.email_index.add(draft.email, key)
        return Ticket(
            key=key, summary=draft.summary, status="To Do", url=self._url(key),
            category=draft.category, priority=draft.priority,
        )

    def _search(self, jql: str) -> list[Ticket]:
        resp = self._request(
            "POST", "/rest/api/3/search/jql", json={"jql": jql, "fields": FIELDS, "maxResults": 20}
        )
        if resp.status_code >= 400:
            raise TicketingConfigError(f"Jira search failed: HTTP {resp.status_code} {resp.text[:300]}")
        return [self._to_ticket(issue) for issue in resp.json().get("issues", [])]

    def list_tickets(self, email: str) -> list[Ticket]:
        tickets = self._search(
            f'project = "{self.project_key}" AND labels = "{requester_label(email)}" ORDER BY updated DESC'
        )
        # Search indexing can lag a few seconds after creation; merge in locally known keys.
        # Fetched one by one: a single deleted key in a `key in (...)` JQL makes Jira reject the whole query.
        known = {t.key for t in tickets}
        missing = [k for k in (self.email_index.keys_for(email) if self.email_index else []) if k not in known]
        for key in missing[:MAX_INDEX_LOOKUPS]:
            resp = self._request("GET", f"/rest/api/3/issue/{key}", params={"fields": ",".join(FIELDS)})
            if resp.status_code == 404:  # deleted, or moved out of reach: stop looking it up
                self.email_index.remove(email, key)
            elif resp.status_code < 400 and (ticket := self._to_ticket(resp.json())).key not in known:
                known.add(ticket.key)
                tickets.append(ticket)
        return tickets

    def get_ticket(self, key: str) -> Ticket:
        resp = self._request("GET", f"/rest/api/3/issue/{key}", params={"fields": ",".join(FIELDS)})
        if resp.status_code == 404:
            raise TicketingConfigError(f"Ticket {key} not found")
        if resp.status_code >= 400:
            raise TicketingError(f"Jira returned {resp.status_code} for ticket {key}")
        ticket = self._to_ticket(resp.json())
        comments = self._request("GET", f"/rest/api/3/issue/{key}/comment", params={"orderBy": "-created"})
        if comments.status_code < 400:
            for c in comments.json().get("comments", [])[:5]:
                ticket.comments.append(
                    Comment(
                        author=(c.get("author") or {}).get("displayName", "Support team"),
                        body=_adf_to_text(c.get("body")),
                        created=c.get("created", ""),
                    )
                )
        return ticket

    # --- used by scripts/seed_jira.py -----------------------------------------------
    def add_comment(self, key: str, text: str) -> None:
        self._request("POST", f"/rest/api/3/issue/{key}/comment", idempotent=False, json={"body": _adf(text)})

    def transition(self, key: str, status_name: str) -> bool:
        resp = self._request("GET", f"/rest/api/3/issue/{key}/transitions")
        for t in resp.json().get("transitions", []):
            if t["to"]["name"].lower() == status_name.lower() or t["name"].lower() == status_name.lower():
                self._request("POST", f"/rest/api/3/issue/{key}/transitions", json={"transition": {"id": t["id"]}})
                return True
        return False
