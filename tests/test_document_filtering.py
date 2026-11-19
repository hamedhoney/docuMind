from unittest.mock import Mock

from documind.retrieval.hybrid import HybridRetriever


def test_hybrid_passes_document_filter():
    semantic = Mock()
    bm25 = Mock()
    reranker = Mock()

    semantic.search.return_value = []
    bm25.search.return_value = []

    retriever = HybridRetriever(
        semantic=semantic,
        bm25=bm25,
        reranker=reranker,
    )

    passages, trace = retriever.search_with_trace(
        query="test question",
        top_k=5,
        mode="hybrid",
        document_id="document-123",
    )

    assert passages == []
    assert trace.selected == []

    assert semantic.search.call_args.kwargs["document_id"] == "document-123"

    assert bm25.search.call_args.kwargs["document_id"] == "document-123"
