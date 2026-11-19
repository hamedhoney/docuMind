from unittest.mock import Mock

import pytest

from documind.retrieval.hybrid import HybridRetriever


def make_chunk(chunk_id: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "test-document",
        "page_number": 1,
        "text": f"Sample passage {chunk_id}",
        "score": score,
    }


@pytest.fixture
def retriever():
    semantic = Mock()
    bm25 = Mock()
    reranker = Mock()

    semantic.search.return_value = [
        make_chunk("A", 0.9),
        make_chunk("B", 0.8),
    ]

    bm25.search.return_value = [
        make_chunk("B", 5.0),
        make_chunk("C", 3.0),
    ]

    def fake_rerank(query, passages, top_k):
        return [
            {
                **passage,
                "rerank_score": float(len(passages) - i),
            }
            for i, passage in enumerate(passages[:top_k])
        ]

    reranker.rerank.side_effect = fake_rerank

    return HybridRetriever(semantic, bm25, reranker)


def test_dense_trace(retriever):
    passages, trace = retriever.search_with_trace(
        "test question",
        mode="dense",
    )

    assert len(passages) == 2
    assert len(trace.dense) == 2
    assert trace.bm25 == []
    assert trace.fusion == []
    assert trace.timings.total_ms >= 0


def test_hybrid_trace(retriever):
    passages, trace = retriever.search_with_trace(
        "test question",
        mode="hybrid",
    )

    assert len(passages) == 3
    assert len(trace.dense) == 2
    assert len(trace.bm25) == 2
    assert len(trace.fusion) == 3
    assert trace.fusion[0].chunk_id == "B"


def test_reranking_trace(retriever):
    passages, trace = retriever.search_with_trace(
        "test question",
        mode="hybrid_rerank",
    )

    assert len(passages) == 3
    assert len(trace.reranked) == 3
    assert len(trace.selected) == 3
    assert trace.timings.rerank_ms >= 0


def test_invalid_top_k(retriever):
    with pytest.raises(ValueError):
        retriever.search_with_trace(
            "test",
            top_k=0,
        )
