"""Build the knowledge-base indexes (Chroma + BM25 chunk file).

Usage:
    python scripts/build_index.py                  # build
    python scripts/build_index.py --query "How many sick leaves?" [--mode dense]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartdesk.rag.retrieval import MODES, HybridRetriever, build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="run a test query instead of building")
    parser.add_argument("--mode", choices=MODES, default=None)
    parser.add_argument("--group", choices=["IT", "HR"], default=None)
    args = parser.parse_args()

    if args.query:
        result = HybridRetriever().retrieve(args.query, group=args.group, mode=args.mode)
        print(f"mode={result.mode} confidence={result.confidence:.3f}")
        for rc in result.chunks:
            print(f"  {rc.chunk_id:<45} dense={rc.dense_score or 0:.3f} rerank={rc.rerank_score}")
        return

    chunks = build_index()
    print(f"Indexed {len(chunks)} chunks")
    print("  by group:", dict(Counter(c.group for c in chunks)))
    print("  by domain:", dict(Counter(c.domain for c in chunks)))


if __name__ == "__main__":
    main()
