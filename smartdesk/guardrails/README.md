# smartdesk/guardrails/

## Why this folder exists
Privacy rules are a cross-cutting policy, not part of retrieval or conversation logic. Keeping them here means the
policy can be reviewed, tested and strengthened in one place, and new guardrails (output filters, rate limits,
prompt-injection checks) have an obvious home.

Today's concern: the semantic cache is **shared by all employees**, so an answer containing someone's personal data
must never be stored and served to a colleague.

## Files

| File | What it does | Why it exists |
|---|---|---|
| `pii.py` | `PiiGuard` finds names, emails, phone numbers, SSNs, card/bank numbers, IPs and NovaTech employee IDs (Presidio + spaCy), ignoring values already published in the KB. `guarded_store()` caches an answer only if the question and answer are PII-free | Stops personal data leaking between employees through the cache. It runs on a background thread after the reply is sent, so it adds no latency; KB values like `payroll@novatech.com` are allowed so normal answers still cache |
