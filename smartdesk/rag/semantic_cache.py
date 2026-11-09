"""Semantic answer cache (SQLite + embeddings).

A repeated or paraphrased question ("how do I reset my pwd?" vs "how can I reset my
password") whose embedding has cosine ≥ CACHE_SIMILARITY with a cached question in the
same IT/HR scope returns the stored grounded answer without retrieval or an LLM call.
Only answered (non-escalated) responses are cached.

Embeddings are stored as float32 BLOBs. Each scope's matrix is kept in memory and reloaded
only when that scope's rows change (checked with one cheap COUNT/MAX query per lookup), so a
lookup doesn't re-read and re-parse every cached vector.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CacheHit:
    question: str
    answer: str
    sources: list[dict]
    similarity: float


@dataclass
class _ScopeRows:
    version: tuple  # (row count, max id) when loaded
    ids: list[int]
    matrix: np.ndarray  # one normalised embedding per row


def _to_vector(stored) -> np.ndarray:
    if isinstance(stored, str):  # rows written before embeddings were stored as BLOBs
        return np.asarray(json.loads(stored), dtype=np.float32)
    return np.frombuffer(stored, dtype=np.float32)


class SemanticCache:
    def __init__(self, db_path: Path, embedder, threshold: float = 0.95):
        self.db_path = db_path
        self.embedder = embedder
        self.threshold = threshold
        self._scopes: dict[tuple[str, int], _ScopeRows] = {}  # (scope, vector size) -> rows
        self._lock = threading.Lock()  # lookups run on the request thread, stores on the guardrail thread
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS answer_cache ("
                " id INTEGER PRIMARY KEY, scope TEXT, question TEXT, embedding BLOB,"
                " answer TEXT, sources TEXT)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # `with sqlite3.connect()` commits but never closes; an open handle locks the file on Windows.
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _embed(self, text: str) -> np.ndarray:
        vec = np.asarray(self.embedder.embed_query(text), dtype=np.float32)
        return vec / (np.linalg.norm(vec) or 1.0)

    def _version(self, conn: sqlite3.Connection, scope: str) -> tuple:
        return conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM answer_cache WHERE scope = ?", (scope,)
        ).fetchone()

    def _rows(self, conn: sqlite3.Connection, scope: str, version: tuple, dim: int) -> _ScopeRows | None:
        """Rows whose vectors match `dim`. Rows from a previous embedding model (another size) are skipped,
        so changing OPENAI_EMBEDDING_MODEL gives cache misses instead of a shape error."""
        with self._lock:
            cached = self._scopes.get((scope, dim))
            if cached and cached.version == version:
                return cached
        rows = conn.execute("SELECT id, embedding FROM answer_cache WHERE scope = ? ORDER BY id", (scope,)).fetchall()
        kept = [(rid, vec) for rid, vec in ((r[0], _to_vector(r[1])) for r in rows) if vec.shape[0] == dim]
        if not kept:
            return None
        loaded = _ScopeRows(version, [rid for rid, _ in kept], np.vstack([vec for _, vec in kept]))
        with self._lock:
            self._scopes[(scope, dim)] = loaded
        return loaded

    def lookup(self, question: str, scope: str) -> CacheHit | None:
        with self._connect() as conn:
            version = self._version(conn, scope)
            if version[0] == 0:  # empty scope: skip the embedding call
                return None
            vec = self._embed(question)
            rows = self._rows(conn, scope, version, vec.shape[0])
            if rows is None:
                return None
            sims = rows.matrix @ vec
            best = int(np.argmax(sims))
            if sims[best] < self.threshold:
                return None
            row = conn.execute(
                "SELECT question, answer, sources FROM answer_cache WHERE id = ?", (rows.ids[best],)
            ).fetchone()
        if row is None:  # cleared since the matrix was loaded
            return None
        q, answer, sources = row
        return CacheHit(q, answer, json.loads(sources), float(sims[best]))

    def store(self, question: str, scope: str, answer: str, sources: list[dict]) -> None:
        vec = self._embed(question)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO answer_cache (scope, question, embedding, answer, sources) VALUES (?, ?, ?, ?, ?)",
                (scope, question, vec.astype(np.float32).tobytes(), answer, json.dumps(sources)),
            )

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM answer_cache")
        with self._lock:
            self._scopes.clear()
