import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from documind.embeddings.encoder import EmbeddingEncoder
from documind.generation.answer import AnswerGenerator
from documind.ingestion.chunker import chunk_pages
from documind.ingestion.pdf import extract_pdf
from documind.retrieval.bm25 import BM25Retriever
from documind.retrieval.hybrid import HybridRetriever
from documind.retrieval.reranker import Reranker
from documind.retrieval.semantic import SemanticRetriever
from documind.retrieval.trace import RetrievalTrace
from documind.storage.vector_store import VectorStore
from documind.verification.models import VerificationReport
from documind.verification.verifier import EvidenceVerifier

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class HealthResponse(BaseModel):
    status: str
    service: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None
    retrieval_mode: Literal[
        "dense",
        "bm25",
        "hybrid",
        "hybrid_rerank",
    ] = "hybrid_rerank"
    verify: bool = False


class RetrievalResponse(BaseModel):
    trace: RetrievalTrace


class DocumentResponse(BaseModel):
    document_id: str
    pages: int
    chunks: int


class QuestionResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_trace: RetrievalTrace | None = None
    verification: VerificationReport | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    encoder = EmbeddingEncoder()
    store = VectorStore()
    store.initialize()

    semantic = SemanticRetriever(encoder, store)

    bm25 = BM25Retriever()
    bm25.build(store.get_all_chunks())

    reranker = Reranker()

    app.state.encoder = encoder
    app.state.store = store
    app.state.bm25 = bm25
    app.state.retriever = HybridRetriever(
        semantic=semantic,
        bm25=bm25,
        reranker=reranker,
    )
    app.state.generator = AnswerGenerator()
    app.state.verifier = EvidenceVerifier(app.state.generator)

    try:
        yield
    finally:
        store.close()


app = FastAPI(
    title="DocuMind API",
    description="Explainable, self-evaluating RAG platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.get("/")
def root():
    return {
        "name": "DocuMind",
        "version": "0.1.0",
        "description": "Explainable, self-evaluating RAG platform",
        "documentation": "/docs",
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="documind",
    )


@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve_documents(
    request: QuestionRequest,
) -> RetrievalResponse:
    try:
        _, trace = app.state.retriever.search_with_trace(
            query=request.question,
            top_k=request.top_k,
            mode=request.retrieval_mode,
            document_id=request.document_id,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Retrieval service unavailable",
        ) from exc

    return RetrievalResponse(trace=trace)


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=201,
)
def upload_document(
    file: Annotated[UploadFile, File()],
) -> DocumentResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    with TemporaryDirectory() as directory:
        path = Path(directory) / "document.pdf"
        total_bytes = 0

        with path.open("wb") as output:
            while block := file.file.read(1024 * 1024):
                total_bytes += len(block)

                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 20 MB limit",
                    )

                output.write(block)

        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        document_id = file_hash

        try:
            pages = extract_pdf(path, document_id)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unreadable PDF document",
            ) from exc

    chunks = chunk_pages(pages)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found",
        )

    try:
        vectors = app.state.encoder.encode([chunk.text for chunk in chunks])

        app.state.store.add(
            chunks,
            vectors.tolist(),
        )
        app.state.bm25.build(app.state.store.get_all_chunks())
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Document indexing unavailable",
        ) from exc

    return DocumentResponse(
        document_id=document_id,
        pages=len(pages),
        chunks=len(chunks),
    )


@app.post("/ask", response_model=QuestionResponse)
def ask_question(
    request: QuestionRequest,
) -> QuestionResponse:
    try:
        passages, trace = app.state.retriever.search_with_trace(
            query=request.question,
            top_k=request.top_k,
            mode=request.retrieval_mode,
            document_id=request.document_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Document retrieval unavailable",
        ) from exc

    try:
        answer = app.state.generator.generate(
            request.question,
            passages,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Answer generation unavailable",
        ) from exc

    verification = None

    if request.verify:
        try:
            verification = app.state.verifier.verify(
                answer=answer,
                passages=passages,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Evidence verification unavailable",
            ) from exc

    return QuestionResponse(
        answer=answer,
        sources=passages,
        retrieval_trace=trace,
        verification=verification,
    )
