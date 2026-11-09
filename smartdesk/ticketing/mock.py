"""In-memory ticket client for tests and fully offline demos (TICKETING_BACKEND=mock)."""

from __future__ import annotations

from datetime import datetime, timezone

from .base import Comment, Ticket, TicketDraft, TicketingConfigError, TicketingUnavailable


class MockTicketClient:
    def __init__(self, project_key: str = "SD", fail: bool = False):
        self.project_key = project_key
        self.fail = fail  # simulate an outage
        self._tickets: dict[str, Ticket] = {}
        self._owners: dict[str, str] = {}
        self._next = 1

    def _check(self) -> None:
        if self.fail:
            raise TicketingUnavailable("mock outage")

    def create_ticket(self, draft: TicketDraft) -> Ticket:
        self._check()
        key = f"{self.project_key}-{self._next}"
        self._next += 1
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        ticket = Ticket(
            key=key, summary=draft.summary, status="To Do", url=f"https://mock.local/browse/{key}",
            category=draft.category, priority=draft.priority, created=now, updated=now,
        )
        self._tickets[key] = ticket
        self._owners[key] = draft.email.lower()
        return ticket

    def list_tickets(self, email: str) -> list[Ticket]:
        self._check()
        return [t for k, t in reversed(self._tickets.items()) if self._owners[k] == email.lower()]

    def get_ticket(self, key: str) -> Ticket:
        self._check()
        if key.upper() not in self._tickets:
            raise TicketingConfigError(f"ticket {key} not found")
        return self._tickets[key.upper()]

    # test helpers
    def set_status(self, key: str, status: str, comment: str | None = None) -> None:
        ticket = self._tickets[key]
        ticket.status = status
        if comment:
            ticket.comments.append(Comment("IT Support", comment, datetime.now(timezone.utc).isoformat()))
