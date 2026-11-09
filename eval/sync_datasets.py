"""Upload the repo's eval datasets to LangSmith (eval/run.py also does this before every run).

    python eval/sync_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.langsmith_setup import RAG_DATASET, client, rag_example, sync_dataset  # noqa: E402
from eval.metrics import load_golden  # noqa: E402


def main() -> None:
    ls = client()
    print(sync_dataset(ls, RAG_DATASET, [rag_example(r) for r in load_golden()],
                       "SmartDesk golden questions (eval/golden.json): 36 answerable, 12 deliberate KB gaps."))


if __name__ == "__main__":
    main()
