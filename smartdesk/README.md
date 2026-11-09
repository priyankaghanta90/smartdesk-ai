# smartdesk/

## Why this folder exists
All application logic lives in one installable Python package, separate from the entry points (`app/`, `scripts/`,
`eval/`) that use it. That keeps the logic testable without a UI, lets every entry point import the same code, and
gives imports a unique namespace (`smartdesk.config`, not a bare `config` that could clash with a PyPI package).

The package is split into layers, and **imports only go downwards**:

```
agent/  ──►  rag/  ticketing/  guardrails/  ──►  config.py  llm.py
```

The RAG layer never imports the agent layer, so retrieval can be evaluated on its own (`eval/run.py --suite rag`)
and the agent can be changed without touching retrieval.

## Files

| Path | What it does | Why it exists |
|---|---|---|
| `__init__.py` | Marks the package; its docstring maps every module | One place to see the whole layout without opening each folder |
| `config.py` | `Settings` read from environment variables / `.env`, plus `get_settings()` | Secrets and tunables (models, thresholds, Jira, sign-in rules) must never be hard-coded. One typed settings object means every layer reads the same values, and tests can build a `Settings(...)` with overrides |
| `llm.py` | Chat models with retries + fallback model; `structured()` for typed outputs; shared embeddings with a small query memo | Every LLM call needs the same reliability rules (timeouts, retries, fallback to gpt-4o, `LLMUnavailable` on failure). Centralising them stops each agent re-implementing error handling, and the memo stops the same question being embedded twice per turn |
| [`agent/`](agent/) | LangGraph conversation flow | Orchestration is a separate concern from knowledge retrieval and ticketing |
| [`rag/`](rag/) | Knowledge-base indexing, retrieval and grounded answering | Can be built, tested and evaluated without any agent code |
| [`ticketing/`](ticketing/) | Jira client, mock client, email index | Hides the ticketing system behind one interface so the agent never deals with HTTP |
| [`guardrails/`](guardrails/) | PII screening | Privacy rules are a cross-cutting policy, kept out of business logic so they can evolve independently |
