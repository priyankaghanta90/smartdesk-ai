# tests/

## Why this folder exists
Fast, deterministic checks that run on every change (`pytest -q`). They guard behaviour that must never regress:
tickets only after confirmation, nobody sees a colleague's tickets, deleted Jira tickets don't break lookups, KB gaps
stay gaps. External services are faked wherever possible (mocked HTTP, mock ticket store, fake embedder) so most
tests are free and offline. The end-to-end flows use the real LLM and skip themselves without an API key.

## Files

| File | What it covers | Why it exists | Needs API? |
|---|---|---|---|
| `test_agent_flows.py` | Whole conversations through the real agent (mock tickets): answers, escalation, confirm/cancel, status, demo vs signed-in identity, colleague privacy | Proves the rubric flows work end to end with the real LLM | yes |
| `test_agent_helpers.py` | Router keyword fallback, ticket choice, yes/no detection, stale-draft handling, cache rules, details refusal | Pins down the deterministic logic around the LLM, including bugs found earlier, so they can't come back | no |
| `test_agent_evaluators.py` | The agent-eval scorers fail on bad behaviour | An eval that always passes is worthless; this proves a skipped confirmation or a leaked ticket would be caught | no |
| `test_jira_client.py` | Jira client against mocked HTTP | Covers labels, safe retries without duplicates, deleted tickets and bad credentials without a Jira account | no |
| `test_kb.py` | KB size and domains, metadata, gaps really absent, golden ids exist, folder READMEs not indexed | Keeps the knowledge base and the golden set consistent with each other | no |
| `test_retrieval_logic.py` | Tokenizer, RRF fusion, escalation gate | Checks the retrieval maths without loading models | no |
| `test_semantic_cache.py` | Storage, scopes, invalidation, legacy rows | The cache is shared state; these prove it never serves stale or wrong-scope answers | no |
| `test_pii_guardrail.py` | PII detection, public KB values allowed, guarded writes | Proves personal data is kept out of the shared cache while normal answers still cache | no (uses the spaCy model from requirements.txt) |
| `conftest.py` | Puts the repo root on the import path | So `import smartdesk` works when pytest runs from any directory | – |
