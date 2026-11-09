# smartdesk/agent/

## Why this folder exists
The help desk is a multi-step conversation, not a single question → answer call: it routes, answers, escalates,
collects ticket details over several turns, waits for confirmation, and looks tickets up. This folder models that as
a **LangGraph state machine** so that:

- each step is a small node with one job, testable on its own;
- the conversation can **pause** (`interrupt()`) for the employee's reply and resume exactly where it stopped;
- rules like "a ticket is created only after yes" are enforced by the graph's edges, not by prompt wording.

The agent layer depends on `rag/`, `ticketing/` and `guardrails/`, never the other way round.

## Files

| File | What it does | Why it exists |
|---|---|---|
| `flow.py` | Wires every node into one graph; `SmartDeskAgent.send(thread_id, text, user_email)` runs a turn and returns a `TurnResult` | A single entry point for the UIs, tests and evals. Keeping the wiring in one file makes the whole flow readable at a glance (its docstring draws it) |
| `state.py` | `SmartDeskState`: messages, email, draft ticket, ticket list, diagnostics | The graph's memory is a typed schema shared by all nodes; one definition avoids nodes disagreeing on field names |
| `router.py` | Picks the intent (KB question / new ticket / status / small talk) and IT vs HR; rewrites follow-ups into standalone questions; keyword fallback | Routing is the first decision on every turn and is evaluated separately (router suite). The keyword fallback keeps SmartDesk usable when the LLM is down |
| `it.py`, `hr.py` | The IT and HR specialists: a persona plus a domain | Separate specialists give each domain its own persona, a domain-filtered retriever and its own cache scope. They're thin because the shared logic lives in `common.py` |
| `ticket.py` | `escalate` → `collect` → `get_email` / `get_details` → `confirm` → `create` | Ticket creation is the riskiest flow (it writes to Jira), so each step is a node and `create` is only reachable from `confirm` after "yes". Stale drafts and "no, cancel" are handled explicitly |
| `status.py` | `status` → `choose_ticket` | Status lookups have their own branching (no tickets / one / several → pick one) and must only ever show the employee's own tickets |
| `chitchat.py` | Greetings, thanks, off-topic replies | Keeps small talk out of the RAG path, so "hi" doesn't trigger retrieval or a false escalation |
| `common.py` | `AgentDeps`, cache → RAG answering, yes/no detection, identity rules, `graceful` error wrapper | Code shared by several nodes lives once. Identity rules (`typed_email_allowed`) and deterministic yes/no detection are security- and reliability-critical, so they're in one audited place |
| `prompts.py` | Router, reply-classification, ticket-draft and chitchat prompts | Prompts are tuned more often than code; keeping them together makes prompt changes easy to find, review and evaluate |
