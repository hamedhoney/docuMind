
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="DocuMind API",
    description="Explainable, self-evaluating RAG platform",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str
    service: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="documind",
    )

from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi import File, HTTPException, UploadFile
from pydantic import BaseModel

from documind.embeddings.encoder import EmbeddingEncoder
from documind.generation.answer import AnswerGenerator
from documind.ingestion.chunker import chunk_pages
from documind.ingestion.pdf import extract_pdf
from documind.retrieval.semantic import SemanticRetriever
from documind.storage.vector_store import VectorStore


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


encoder = EmbeddingEncoder()
store = VectorStore()
generator = AnswerGenerator()
retriever = SemanticRetriever(encoder, store)


@app.on_event("startup")
def initialize_vector_store():
    store.initialize()


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    document_id = str(uuid4())

    with TemporaryDirectory() as directory:
        path = Path(directory) / "document.pdf"

        with path.open("wb") as output:
            while block := file.file.read(1024 * 1024):
                output.write(block)

        try:
            pages = extract_pdf(path, document_id)
        except Exception as exc:
            raise HTTPException(400, "Invalid PDF document") from exc

    chunks = chunk_pages(pages)

    if not chunks:
        raise HTTPException(422, "No extractable text found")

    vectors = encoder.encode([chunk.text for chunk in chunks])
    store.add(chunks, vectors.tolist())

    return {
        "document_id": document_id,
        "pages": len(pages),
        "chunks": len(chunks),
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    passages = retriever.search(
        request.question,
        top_k=request.top_k,
    )

    answer = generator.generate(request.question, passages)

    return {
        "answer": answer,
        "sources": passages,
    }
