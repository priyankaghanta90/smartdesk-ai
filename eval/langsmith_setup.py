"""LangSmith client + dataset sync for the eval scripts.

The app reads secrets from .env through pydantic settings, which doesn't export them; the LangSmith
and OpenAI SDKs read the process environment, so `client()` copies the keys over.
"""

from __future__ import annotations

import os

from langsmith import Client

from smartdesk.config import get_settings

RAG_DATASET = "smartdesk-rag"


def client() -> Client:
    s = get_settings()
    if not s.langsmith_api_key:
        raise SystemExit("LANGSMITH_API_KEY is not set in .env")
    os.environ.setdefault("LANGSMITH_API_KEY", s.langsmith_api_key)
    os.environ.setdefault("OPENAI_API_KEY", s.openai_api_key)  # openevals judges build their own client
    os.environ.setdefault("LANGSMITH_PROJECT", s.langsmith_project)
    if s.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", s.langsmith_endpoint)
    return Client()


def rag_example(r: dict) -> dict:
    """golden.json row -> LangSmith example. `user_input` is the stable identity of an example."""
    return {
        "inputs": {"question": r["user_input"], "group": r["group"]},
        "outputs": {
            "answerable": r["answerable"],
            "reference": r.get("reference", ""),
            "reference_source_ids": r.get("reference_source_ids", []),
        },
        "metadata": {"group": r["group"], "answerable": r["answerable"]},
    }


def sync_dataset(ls: Client, name: str, rows: list[dict], description: str) -> str:
    """Make the LangSmith dataset match `rows` (the repo stays the source of truth).

    Adds new questions, updates changed ones and deletes removed ones; LangSmith keeps the old
    versions, so earlier experiments still point at the data they ran on. Returns a change summary.
    """
    wanted = {r["inputs"]["question"]: r for r in rows}
    if ls.has_dataset(dataset_name=name):
        dataset = ls.read_dataset(dataset_name=name)
    else:
        dataset = ls.create_dataset(name, description=description)
    existing = {e.inputs["question"]: e for e in ls.list_examples(dataset_id=dataset.id)}

    new = [wanted[q] for q in wanted if q not in existing]
    changed = [
        (existing[q], wanted[q]) for q in wanted
        if q in existing and (existing[q].inputs != wanted[q]["inputs"] or existing[q].outputs != wanted[q]["outputs"])
    ]
    removed = [e for q, e in existing.items() if q not in wanted]

    if new:
        ls.create_examples(dataset_id=dataset.id, examples=new)
    for example, row in changed:
        ls.update_example(example.id, inputs=row["inputs"], outputs=row["outputs"], metadata=row["metadata"])
    if removed:
        ls.delete_examples([e.id for e in removed])
    return f"{name}: {len(new)} added, {len(changed)} updated, {len(removed)} removed, {len(wanted)} total"
