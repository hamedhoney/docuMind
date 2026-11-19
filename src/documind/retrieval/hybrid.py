from typing import Literal

from documind.retrieval.bm25 import BM25Retriever
from documind.retrieval.fusion import reciprocal_rank_fusion
from documind.retrieval.reranker import Reranker
from documind.retrieval.semantic import SemanticRetriever

RetrievalMode = Literal[
    "dense",
    "bm25",
    "hybrid",
    "hybrid_rerank",
]


class HybridRetriever:
    def __init__(
        self,
        semantic: SemanticRetriever,
        bm25: BM25Retriever,
        reranker: Reranker,
    ):
        self.semantic = semantic
        self.bm25 = bm25
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid_rerank",
    ) -> list[dict]:
        if mode == "dense":
            return self.semantic.search(query, top_k)

        if mode == "bm25":
            return self.bm25.search(query, top_k)

        candidate_count = max(top_k * 4, 20)

        dense_results = self.semantic.search(query, candidate_count)
        bm25_results = self.bm25.search(query, candidate_count)

        fused = reciprocal_rank_fusion([dense_results, bm25_results])

        if mode == "hybrid":
            return fused[:top_k]

        if mode == "hybrid_rerank":
            return self.reranker.rerank(
                query,
                fused[:candidate_count],
                top_k,
            )

        raise ValueError(f"Unknown retrieval mode: {mode}")
