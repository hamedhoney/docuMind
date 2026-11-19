from pydantic import BaseModel, Field


class RankedChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_number: int
    text: str
    rank: int = Field(ge=1)
    score: float


class RetrievalTimings(BaseModel):
    dense_ms: float = 0.0
    bm25_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


class RetrievalTrace(BaseModel):
    query: str
    mode: str
    top_k: int
    timings: RetrievalTimings
    dense: list[RankedChunk] = []
    bm25: list[RankedChunk] = []
    fusion: list[RankedChunk] = []
    reranked: list[RankedChunk] = []
    selected: list[RankedChunk] = []
