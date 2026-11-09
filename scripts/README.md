# scripts/

## Why this folder exists
One-off operational commands that change data, as opposed to the app (serves users) and evals (measure quality).
Keeping them as small scripts makes these steps explicit and repeatable instead of hidden side effects of starting
the app.

## Files

| File | What it does | Why it exists |
|---|---|---|
| `build_index.py` | Chunks the KB, embeds it into Chroma, writes `data/chunks.json`, clears the answer cache; `--query "..."` tests retrieval | Indexing is slow and costs embedding calls, so it runs once after KB changes rather than on every app start. Clearing the cache stops answers based on old policy text |
| `seed_jira.py` | Creates demo tickets for one email in different statuses, with support comments (`--email`, `--limit`) | The status flow needs realistic data (several tickets, different statuses, comments). Works with Jira Software and Jira Service Management status names |
