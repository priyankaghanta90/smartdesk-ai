from ..config import get_settings
from .base import (
    Comment,
    Ticket,
    TicketClient,
    TicketDraft,
    TicketingConfigError,
    TicketingError,
    TicketingUnavailable,
)
from .email_index import EmailIndex
from .jira import JiraTicketClient
from .mock import MockTicketClient


def make_ticket_client() -> TicketClient:
    s = get_settings()
    if s.ticketing_backend == "mock":
        return MockTicketClient(project_key=s.jira_project_key)
    return JiraTicketClient(
        base_url=s.jira_base_url,
        email=s.jira_email,
        api_token=s.jira_api_token,
        project_key=s.jira_project_key,
        issue_type=s.jira_issue_type,
        timeout_s=s.jira_timeout_s,
        email_index=EmailIndex(s.sqlite_path),
    )


__all__ = [
    "Comment", "Ticket", "TicketClient", "TicketDraft", "TicketingError", "TicketingUnavailable",
    "TicketingConfigError", "EmailIndex", "JiraTicketClient", "MockTicketClient", "make_ticket_client",
]
