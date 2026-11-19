
# DocuMind

### An Explainable, Self-Evaluating RAG Platform

DocuMind is an open-source document intelligence
platform focused on transparent retrieval,
evidence-grounded generation, and reproducible
evaluation.

## Planned Features

- [ ] Document ingestion and processing
- [ ] Dense and BM25 retrieval
- [ ] Hybrid search and reranking
- [ ] Retrieval X-Ray visualization
- [ ] Claim-level evidence verification
- [ ] RAG regression testing
- [ ] Interactive document chat
- [ ] Evaluation dashboard

## Tech Stack

Python · FastAPI · Qdrant · PostgreSQL
Next.js · Docker · Ragas · uv

## Quick Start

Install dependencies:

    uv sync

Start the backend:

    uv run uvicorn documind.api:app --reload

Run tests:

    uv run pytest

## Project Status

Early development. The initial API and
automated testing infrastructure are available.

## License

MIT
