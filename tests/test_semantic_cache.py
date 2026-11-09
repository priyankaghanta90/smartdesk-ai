"""Semantic cache storage and invalidation (fake embedder, no API calls)."""

import json
import sqlite3
from contextlib import closing

from smartdesk.rag.semantic_cache import SemanticCache

VECTORS = {"How much PTO do I get?": [1.0, 0.0], "How do I reset my password?": [0.0, 1.0]}


class FakeEmbedder:
    def embed_query(self, text):
        return VECTORS.get(text, [0.7, 0.7])


def test_store_and_lookup_by_scope(tmp_path):
    cache = SemanticCache(tmp_path / "c.db", FakeEmbedder())
    cache.store("How much PTO do I get?", "HR", "15 days [S1].", [{"ref": "S1"}])
    hit = cache.lookup("How much PTO do I get?", "HR")
    assert hit.answer == "15 days [S1]." and hit.sources == [{"ref": "S1"}]
    assert cache.lookup("How much PTO do I get?", "IT") is None  # other scope
    assert cache.lookup("How do I reset my password?", "HR") is None  # not similar


def test_new_rows_are_seen_after_the_matrix_was_loaded(tmp_path):
    cache = SemanticCache(tmp_path / "c.db", FakeEmbedder())
    cache.store("How much PTO do I get?", "HR", "15 days", [])
    assert cache.lookup("How do I reset my password?", "HR") is None  # loads the in-memory matrix
    cache.store("How do I reset my password?", "HR", "Use the portal", [])
    assert cache.lookup("How do I reset my password?", "HR").answer == "Use the portal"


def test_writes_from_another_process_invalidate_the_matrix(tmp_path):
    db = tmp_path / "c.db"
    reader, writer = SemanticCache(db, FakeEmbedder()), SemanticCache(db, FakeEmbedder())
    writer.store("How much PTO do I get?", "HR", "15 days", [])
    assert reader.lookup("How much PTO do I get?", "HR").answer == "15 days"
    writer.clear()
    writer.store("How much PTO do I get?", "HR", "18 days", [])  # same count, new id
    assert reader.lookup("How much PTO do I get?", "HR").answer == "18 days"


def test_clear_empties_the_cache(tmp_path):
    cache = SemanticCache(tmp_path / "c.db", FakeEmbedder())
    cache.store("How much PTO do I get?", "HR", "15 days", [])
    cache.clear()
    assert cache.lookup("How much PTO do I get?", "HR") is None


def test_rows_written_as_json_before_the_blob_format_still_load(tmp_path):
    db = tmp_path / "c.db"
    cache = SemanticCache(db, FakeEmbedder())
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("INSERT INTO answer_cache (scope, question, embedding, answer, sources) VALUES (?, ?, ?, ?, ?)",
                     ("HR", "How much PTO do I get?", json.dumps([1.0, 0.0]), "15 days", "[]"))
    assert cache.lookup("How much PTO do I get?", "HR").answer == "15 days"


def test_rows_from_another_embedding_size_are_misses_not_errors(tmp_path):
    class Wide:
        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    SemanticCache(tmp_path / "c.db", FakeEmbedder()).store("How much PTO do I get?", "HR", "15 days", [])
    wide = SemanticCache(tmp_path / "c.db", Wide())
    assert wide.lookup("How much PTO do I get?", "HR") is None  # old 2-d row skipped, no shape error
    wide.store("How much PTO do I get?", "HR", "15 days", [])
    assert wide.lookup("How much PTO do I get?", "HR").answer == "15 days"
