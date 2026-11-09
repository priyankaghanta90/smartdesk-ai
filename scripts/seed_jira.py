"""Pre-populate Jira with demo tickets for the status-check flow.

    python scripts/seed_jira.py [--email jane.doe@novatech.com] [--limit N]

Creates 3 tickets for the email, moves them to different statuses and adds support-team
comments, so "what's the status of my ticket?" shows multiple tickets with updates.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartdesk.ticketing import JiraTicketClient, TicketDraft, make_ticket_client  # noqa: E402

DEMO = [
    (
        TicketDraft("", "Monitor flickering on office desk", "Employee reports the external monitor at their desk "
                    "has been flickering for two days. No KB article covers display hardware faults.", "IT", "Medium"),
        "In Progress",
        "Replacement monitor ordered, delivery by Thursday.",
    ),
    (
        TicketDraft("", "Parking slot request at Minneapolis HQ", "Employee would like to know how to get a parking "
                    "slot at the Minneapolis HQ. The KB has no parking policy.", "HR", "Low"),
        "To Do",
        None,
    ),
    (
        TicketDraft("", "3rd-floor printer keeps jamming", "Employee reports the 3rd-floor printer jams on every "
                    "double-sided print job.", "IT", "Low"),
        "Done",
        "Fuser unit replaced by the vendor; printer tested OK.",
    ),
]


# Software spaces say "In Progress"; Jira Service Management workflows say "Work in progress".
STATUS_NAMES = {"In Progress": ["In Progress", "Work in progress"], "Done": ["Done", "Resolved"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", default="jane.doe@novatech.com")
    parser.add_argument("--limit", type=int, default=len(DEMO), help="seed only the first N demo tickets")
    args = parser.parse_args()

    client = make_ticket_client()
    if not isinstance(client, JiraTicketClient):
        sys.exit("TICKETING_BACKEND must be 'jira' to seed tickets.")

    for draft, status, comment in DEMO[: args.limit]:
        draft.email = args.email
        ticket = client.create_ticket(draft)
        moved = status == "To Do" or any(client.transition(ticket.key, name) for name in STATUS_NAMES[status])
        if comment:
            client.add_comment(ticket.key, comment)
        print(f"{ticket.key}  {draft.summary:<45} -> {status}{'' if moved else ' (transition not found)'}  {ticket.url}")


if __name__ == "__main__":
    main()
