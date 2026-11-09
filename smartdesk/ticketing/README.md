# smartdesk/ticketing/

## Why this folder exists
The agent needs three operations: create a ticket, list an employee's tickets, get one ticket's details. This folder
puts them behind one interface (`TicketClient`) so the agent never deals with HTTP, Jira's data formats or
retries. That gives:

- **swappable backends:** real Jira in production, an in-memory mock for tests, evals and offline demos
  (`TICKETING_BACKEND`);
- **one place for the hard parts:** ownership labels, retries without duplicate tickets, and search-index lag.

Import everything from `smartdesk.ticketing`, not from the individual files.

## Files

| File | What it does | Why it exists |
|---|---|---|
| `__init__.py` | `make_ticket_client()` (Jira or mock, from settings) and re-exports | Callers get the right client from configuration and import from one stable path, so moving internals never breaks them |
| `base.py` | `Ticket`, `TicketDraft`, `Comment`, the `TicketClient` protocol, error types, ownership and idempotency label helpers | The shared contract both clients implement. Typed errors (`TicketingUnavailable` vs `TicketingConfigError`) let the agent reply appropriately ("try again later" vs "contact IT"). Labels are hashed so look-alike emails can't share an owner |
| `jira.py` | Jira Cloud REST v3 client | Real ticketing: creates issues with owner and request labels, searches by owner, fetches comments, and retries safely. Creating is only retried when Jira certainly didn't receive it, and a request label lets a retry find a ticket that was already created |
| `mock.py` | In-memory client | Tests, evals and demos run without a Jira account and without creating real tickets; `fail=True` simulates an outage |
| `email_index.py` | Local SQLite map: email → ticket keys | Jira's search lags a few seconds behind a new ticket; this fallback makes "status of my ticket" work right after creation. Keys Jira no longer has are pruned |
