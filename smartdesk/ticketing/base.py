"""Ticketing abstractions shared by the Jira and mock clients."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Protocol


class TicketingError(RuntimeError):
    """Base error for ticketing problems."""


class TicketingUnavailable(TicketingError):
    """The platform could not be reached (network, timeouts, 5xx) after retries."""


class TicketingConfigError(TicketingError):
    """Credentials / project configuration is wrong (401, 403, 404)."""


@dataclass
class Comment:
    author: str
    body: str
    created: str


@dataclass
class Ticket:
    key: str
    summary: str
    status: str
    url: str
    category: str = ""
    priority: str = ""
    created: str = ""
    updated: str = ""
    comments: list[Comment] = field(default_factory=list)


@dataclass
class TicketDraft:
    email: str
    summary: str
    description: str
    category: str  # IT | HR
    priority: str = "Medium"  # Low | Medium | High | Critical
    # Stays the same across "try again" retries of this draft, so a retry can't create a duplicate.
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class TicketClient(Protocol):
    def create_ticket(self, draft: TicketDraft) -> Ticket: ...

    def list_tickets(self, email: str) -> list[Ticket]: ...

    def get_ticket(self, key: str) -> Ticket: ...


def requester_label(email: str) -> str:
    """Jira ownership label: readable prefix + hash of the exact email.

    The readable part maps every symbol to "_", so jane.doe@, jane_doe@ and jane-doe@ would share it;
    the hash keeps each employee's label unique.
    """
    email = email.strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in email)[:60]
    digest = hashlib.sha256(email.encode("utf-8")).hexdigest()[:12]
    return f"requester-{safe}-{digest}"


def request_label(request_id: str) -> str:
    """Idempotency label: lets a retried create find the ticket an earlier attempt already made."""
    return f"sd-req-{request_id}"
