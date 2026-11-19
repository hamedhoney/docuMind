from documind.retrieval.bm25 import BM25Retriever
from documind.retrieval.fusion import reciprocal_rank_fusion


def sample_chunks():
    return [
        {
            "chunk_id": "1",
            "document_id": "doc1",
            "page_number": 1,
            "text": "Peripheral artery disease affects blood flow.",
        },
        {
            "chunk_id": "2",
            "document_id": "doc1",
            "page_number": 2,
            "text": "Ankle exoskeletons assist human walking.",
        },
        {
            "chunk_id": "3",
            "document_id": "doc2",
            "page_number": 1,
            "text": "Neural networks learn from training data.",
        },
    ]


def test_bm25_returns_relevant_document():
    retriever = BM25Retriever()
    retriever.build(sample_chunks())

    results = retriever.search("ankle exoskeletons")

    assert results
    assert results[0]["chunk_id"] == "2"


def test_bm25_empty_index():
    retriever = BM25Retriever()

    assert retriever.search("anything") == []


def test_rrf_combines_rankings():
    chunks = sample_chunks()

    dense = [chunks[0], chunks[1]]
    lexical = [chunks[1], chunks[0]]

    results = reciprocal_rank_fusion([dense, lexical])

    assert len(results) == 2
    assert results[0]["rrf_score"] > 0


def test_rrf_deduplicates_documents():
    chunks = sample_chunks()

    results = reciprocal_rank_fusion([[chunks[0]], [chunks[0]]])

    assert len(results) == 1
    assert results[0]["chunk_id"] == "1"
