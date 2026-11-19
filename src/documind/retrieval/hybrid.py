from time import perf_counter
from typing import Literal

from documind.retrieval.bm25 import BM25Retriever
from documind.retrieval.fusion import reciprocal_rank_fusion
from documind.retrieval.reranker import Reranker
from documind.retrieval.semantic import SemanticRetriever
from documind.retrieval.trace import (
    RankedChunk,
    RetrievalTimings,
    RetrievalTrace,
)

RetrievalMode = Literal[
    "dense",
    "bm25",
    "hybrid",
    "hybrid_rerank",
]


def ranked_chunks(
    passages: list[dict],
    score_key: str,
) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=p["chunk_id"],
            document_id=p["document_id"],
            page_number=p["page_number"],
            text=p["text"],
            rank=rank,
            score=float(p[score_key]),
        )
        for rank, p in enumerate(passages, start=1)
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

    def search_with_trace(
        self,
        query: str,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid_rerank",
        document_id: str | None = None,
    ) -> tuple[list[dict], RetrievalTrace]:

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")

        if mode not in ("dense", "bm25", "hybrid", "hybrid_rerank"):
            raise ValueError(f"Unknown mode: {mode}")

        started = perf_counter()
        timings = RetrievalTimings()

        dense_results = []
        bm25_results = []
        fused_results = []
        reranked_results = []

        candidate_count = max(top_k * 4, 20)

        if mode != "bm25":
            t0 = perf_counter()
            dense_results = self.semantic.search(
                query,
                candidate_count if mode.startswith("hybrid") else top_k,
                document_id=document_id,
            )
            timings.dense_ms = (perf_counter() - t0) * 1000

        if mode != "dense":
            t0 = perf_counter()
            bm25_results = self.bm25.search(
                query,
                candidate_count if mode.startswith("hybrid") else top_k,
                document_id=document_id,
            )
            timings.bm25_ms = (perf_counter() - t0) * 1000

        if mode.startswith("hybrid"):
            t0 = perf_counter()
            fused_results = reciprocal_rank_fusion([dense_results, bm25_results])
            timings.fusion_ms = (perf_counter() - t0) * 1000

        if mode == "dense":
            selected = dense_results[:top_k]
            score_key = "score"

        elif mode == "bm25":
            selected = bm25_results[:top_k]
            score_key = "score"

        elif mode == "hybrid":
            selected = fused_results[:top_k]
            score_key = "rrf_score"

        else:
            t0 = perf_counter()
            reranked_results = self.reranker.rerank(
                query,
                fused_results[:candidate_count],
                top_k=top_k,
            )
            timings.rerank_ms = (perf_counter() - t0) * 1000

            selected = reranked_results
            score_key = "rerank_score"

        timings.total_ms = (perf_counter() - started) * 1000

        trace = RetrievalTrace(
            query=query,
            mode=mode,
            top_k=top_k,
            timings=timings,
            dense=ranked_chunks(dense_results, "score"),
            bm25=ranked_chunks(bm25_results, "score"),
            fusion=ranked_chunks(fused_results, "rrf_score"),
            reranked=ranked_chunks(reranked_results, "rerank_score"),
            selected=ranked_chunks(selected, score_key),
        )

        return selected, trace

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: RetrievalMode = "hybrid_rerank",
        document_id: str | None = None,
    ) -> list[dict]:
        passages, _ = self.search_with_trace(
            query=query,
            top_k=top_k,
            mode=mode,
            document_id=document_id,
        )
        return passages
