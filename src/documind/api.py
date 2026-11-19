
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
