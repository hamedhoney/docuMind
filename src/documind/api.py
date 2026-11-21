import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from documind.config import get_settings
from documind.embeddings.encoder import create_encoder
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

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

DEMO_DOCUMENT_ID = "24b26eb7b9bba731af9746974c452a23f870814d31b7d55cd681b771dfe02fdc"


# --------------------------------------------------
# Response and request models
# --------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    service: str


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    document_id: str | None = None

    retrieval_mode: Literal[
        "dense",
        "bm25",
        "hybrid",
        "hybrid_rerank",
    ] = "hybrid"

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


# --------------------------------------------------
# Public demo safeguards
# --------------------------------------------------


def validate_document_access(
    document_id: str | None,
) -> str | None:
    settings = get_settings()

    if settings.app_env != "cloud":
        return document_id

    if document_id is None:
        return DEMO_DOCUMENT_ID

    if document_id != DEMO_DOCUMENT_ID:
        raise HTTPException(
            status_code=403,
            detail="Document unavailable in public demo",
        )

    return document_id


def validate_retrieval_mode(
    mode: str,
) -> None:
    settings = get_settings()

    if mode == "hybrid_rerank" and settings.reranker_provider == "none":
        raise HTTPException(
            status_code=422,
            detail=(
                "Cross-encoder reranking is unavailable. Use hybrid retrieval instead."
            ),
        )


# --------------------------------------------------
# Application lifecycle
# --------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    encoder = create_encoder()

    store = VectorStore()
    store.initialize(dimension=encoder.dimension)

    semantic = SemanticRetriever(encoder, store)

    bm25 = BM25Retriever()
    bm25.build(store.get_all_chunks())

    reranker = Reranker() if settings.reranker_provider == "local" else None

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


# --------------------------------------------------
# FastAPI initialization
# --------------------------------------------------


app = FastAPI(
    title="DocuMind API",
    description="Explainable, self-evaluating RAG platform",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        get_settings().frontend_origin,
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------
# Rate limiting
# --------------------------------------------------


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
)

app.state.limiter = limiter


def rate_limit_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": ("Rate limit exceeded. Please try again later.")},
    )


app.add_exception_handler(
    RateLimitExceeded,
    rate_limit_handler,
)

app.add_middleware(SlowAPIMiddleware)


# --------------------------------------------------
# Root and health
# --------------------------------------------------


@app.get("/")
def root() -> dict:
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


# --------------------------------------------------
# Retrieval
# --------------------------------------------------


@app.post(
    "/retrieve",
    response_model=RetrievalResponse,
)
@limiter.limit("15/minute")
def retrieve_documents(
    request: Request,
    body: QuestionRequest,
) -> RetrievalResponse:
    document_id = validate_document_access(body.document_id)

    validate_retrieval_mode(body.retrieval_mode)

    try:
        _, trace = app.state.retriever.search_with_trace(
            query=body.question,
            top_k=body.top_k,
            mode=body.retrieval_mode,
            document_id=document_id,
        )

    except Exception as exc:
        logger.exception("Retrieval failed")

        raise HTTPException(
            status_code=503,
            detail="Retrieval service unavailable",
        ) from exc

    return RetrievalResponse(trace=trace)


# --------------------------------------------------
# Document ingestion
# --------------------------------------------------


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=201,
)
def upload_document(
    file: Annotated[UploadFile, File()],
) -> DocumentResponse:
    settings = get_settings()

    if settings.app_env == "cloud" and not settings.public_uploads_enabled:
        raise HTTPException(
            status_code=403,
            detail="Public document uploads are disabled",
        )

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

        document_id = hashlib.sha256(path.read_bytes()).hexdigest()

        try:
            pages = extract_pdf(
                path,
                document_id,
            )

        except Exception as exc:
            logger.warning(
                "PDF extraction failed: %s",
                exc,
            )

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
        vectors = app.state.encoder.encode(
            [chunk.text for chunk in chunks],
            task_type="RETRIEVAL_DOCUMENT",
        )

        app.state.store.add(
            chunks,
            vectors.tolist(),
        )

        app.state.bm25.build(app.state.store.get_all_chunks())

    except Exception as exc:
        logger.exception("Document indexing failed")

        raise HTTPException(
            status_code=503,
            detail="Document indexing unavailable",
        ) from exc

    return DocumentResponse(
        document_id=document_id,
        pages=len(pages),
        chunks=len(chunks),
    )


# --------------------------------------------------
# Question answering
# --------------------------------------------------


@app.post(
    "/ask",
    response_model=QuestionResponse,
)
@limiter.limit("5/minute")
def ask_question(
    request: Request,
    body: QuestionRequest,
) -> QuestionResponse:
    settings = get_settings()

    document_id = validate_document_access(body.document_id)

    validate_retrieval_mode(body.retrieval_mode)

    # Verification is disabled for the public demo.
    if settings.app_env == "cloud" and body.verify:
        raise HTTPException(
            status_code=403,
            detail=("Evidence verification is currently disabled in the public demo."),
        )

    try:
        passages, trace = app.state.retriever.search_with_trace(
            query=body.question,
            top_k=body.top_k,
            mode=body.retrieval_mode,
            document_id=document_id,
        )

    except Exception as exc:
        logger.exception("Question retrieval failed")

        raise HTTPException(
            status_code=503,
            detail="Document retrieval unavailable",
        ) from exc

    try:
        answer = app.state.generator.generate(
            body.question,
            passages,
        )

    except Exception as exc:
        logger.exception("Answer generation failed")

        raise HTTPException(
            status_code=502,
            detail="Answer generation unavailable",
        ) from exc

    verification = None

    if body.verify:
        try:
            verification = app.state.verifier.verify(
                answer=answer,
                passages=passages,
            )

        except Exception as exc:
            logger.exception("Evidence verification failed")

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
