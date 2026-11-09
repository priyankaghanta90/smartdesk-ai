from smartdesk.config import Settings
from smartdesk.rag.confidence import is_insufficient, retrieval_gate
from smartdesk.rag.ingest import Chunk
from smartdesk.rag.retrieval import RetrievalResult, RetrievedChunk, rrf_fuse, tokenize


def _chunk(cid="a"):
    return Chunk(cid, "text", "src", "T", "S", "it", "IT", "c")


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("How do I reset my VPN password?") == ["reset", "vpn", "password"]


def test_rrf_rewards_documents_ranked_by_both_retrievers():
    fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]])
    assert fused[0] == "a"  # rank 1 + rank 2 beats everything else
    assert set(fused) == {"a", "b", "c", "d"}


def test_gate_escalates_on_no_chunks_and_low_scores():
    s = Settings(rerank_threshold=0.0, dense_threshold=0.4)
    assert not retrieval_gate(RetrievalResult([], float("-inf"), "hybrid_rerank"), s).passed
    low = RetrievalResult([RetrievedChunk(_chunk(), 0.9, -3.0)], -3.0, "hybrid_rerank")
    assert not retrieval_gate(low, s).passed
    ok = RetrievalResult([RetrievedChunk(_chunk(), 0.9, 4.2)], 4.2, "hybrid_rerank")
    assert retrieval_gate(ok, s).passed
    dense_low = RetrievalResult([RetrievedChunk(_chunk(), 0.2)], 0.2, "dense")
    assert not retrieval_gate(dense_low, s).passed


def test_insufficient_marker_detection():
    assert is_insufficient("INSUFFICIENT_CONTEXT")
    assert is_insufficient("insufficient context")
    assert not is_insufficient("Sick leave is 8 days [S1].")
