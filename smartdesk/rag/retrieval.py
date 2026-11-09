"""Hybrid retrieval: dense (Chroma) + sparse (BM25) -> Reciprocal Rank Fusion -> cross-encoder rerank.

`mode` lets the evaluation compare each stage:
    dense          Chroma cosine search only
    bm25           BM25 only
    hybrid         dense + BM25 fused with RRF
    hybrid_rerank  hybrid candidates re-scored by a cross-encoder (default)
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from functools import cached_property

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from ..config import Settings, get_settings
from .ingest import Chunk, load_kb, read_chunks, save_chunks
from .semantic_cache import SemanticCache
from ..llm import embeddings

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
COLLECTION = "smartdesk_kb"
RRF_K = 60

_STOPWORDS = set(
    "a an and are as at be by can do does for from how i in is it my of on or the to what when "
    "where which who will with you your me we our get".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_d)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    dense_score: float | None = None  # Chroma relevance in [0, 1]
    rerank_score: float | None = None  # cross-encoder logit

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    confidence: float  # rerank logit (hybrid_rerank) or best dense relevance (other modes)
    mode: str


def build_index(settings: Settings | None = None) -> list[Chunk]:
    """(Re)build the Chroma collection and the chunk file BM25 is built from."""
    s = settings or get_settings()
    chunks = load_kb(s.kb_dir)
    if s.chroma_dir.exists():
        shutil.rmtree(s.chroma_dir)
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings(),
        persist_directory=str(s.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_texts(
        texts=[c.text for c in chunks],
        metadatas=[c.metadata() for c in chunks],
        ids=[c.chunk_id for c in chunks],
    )
    save_chunks(chunks, s.chunks_path)
    # Cached answers quote the old policy text; drop them so the next question is answered from the new KB.
    SemanticCache(s.sqlite_path, embedder=None).clear()
    return chunks


class HybridRetriever:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        if not self.s.chunks_path.exists():
            raise FileNotFoundError("Index not built. Run: python scripts/build_index.py")
        self.chunks = read_chunks(self.s.chunks_path)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self.store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings(),
            persist_directory=str(self.s.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    @cached_property
    def reranker(self):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # noisy on Windows without Developer Mode
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        from sentence_transformers import CrossEncoder  # heavy import, load lazily

        return CrossEncoder(self.s.rerank_model)

    # --- individual retrievers -------------------------------------------------
    def dense(self, query: str, k: int, group: str | None = None) -> list[tuple[str, float]]:
        where = {"group": group} if group else None
        hits = self.store.similarity_search_with_relevance_scores(query, k=k, filter=where)
        return [(doc.metadata["chunk_id"], float(score)) for doc, score in hits]

    def sparse(self, query: str, k: int, group: str | None = None) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        ids = [
            self.chunks[i].chunk_id
            for i in order
            if scores[i] > 0 and (group is None or self.chunks[i].group == group)
        ]
        return ids[:k]

    # --- main entry point ---------------------------------------------------------
    def retrieve(self, query: str, group: str | None = None, mode: str | None = None) -> RetrievalResult:
        mode = mode or self.s.retrieval_mode
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; choose from {MODES}")
        k, final_k = self.s.candidate_k, self.s.final_k

        dense_hits = self.dense(query, k, group)
        dense_scores = dict(dense_hits)
        best_dense = max(dense_scores.values(), default=0.0)

        if mode == "dense":
            ranked = [cid for cid, _ in dense_hits]
        elif mode == "bm25":
            ranked = self.sparse(query, k, group)
        else:
            ranked = rrf_fuse([[cid for cid, _ in dense_hits], self.sparse(query, k, group)])[:k]

        if mode != "hybrid_rerank":
            results = [RetrievedChunk(self.by_id[cid], dense_scores.get(cid)) for cid in ranked[:final_k]]
            return RetrievalResult(results, best_dense, mode)

        if not ranked:
            return RetrievalResult([], float("-inf"), mode)
        logits = self.reranker.predict([(query, self.by_id[cid].text) for cid in ranked])
        rescored = sorted(zip(ranked, logits), key=lambda x: x[1], reverse=True)[:final_k]
        results = [
            RetrievedChunk(self.by_id[cid], dense_scores.get(cid), float(logit)) for cid, logit in rescored
        ]
        return RetrievalResult(results, results[0].rerank_score, mode)
