# DocuMind

**Explainable, evaluation-driven Retrieval-Augmented Generation (RAG)**

[Live Demo](https://docu-mind-iota-self.vercel.app/) · [Evaluation Dashboard](https://docu-mind-iota-self.vercel.app/evaluation) · [API Docs](https://documind-api-mcnz.onrender.com/docs)

DocuMind is a full-stack document question-answering platform that makes retrieval observable. It combines dense semantic search, BM25 keyword search, Reciprocal Rank Fusion (RRF), source-grounded LLM answers, and a Retrieval X-Ray interface that shows which passages were found, how they ranked, and how long each retrieval stage took. A separate evaluation dashboard publishes measured retrieval quality and per-question failures.

> **Public demo:** A curated, pre-indexed research paper, *The Common Pile*, is available for questions. Uploads, cross-encoder reranking, and claim-level verification are restricted in the public cloud configuration; the full local configuration retains these capabilities.

## Features

| Feature | Public cloud demo | Full local setup |
| --- | --- | --- |
| PDF text extraction, chunking, SHA-256 document IDs | Pre-indexed documents; public uploads disabled | PDF upload and indexing |
| Dense semantic retrieval | Gemini embeddings | Sentence Transformers |
| BM25 lexical retrieval | Yes | Yes |
| Hybrid retrieval with RRF | Yes | Yes |
| Cross-encoder reranking | Disabled | Available |
| Gemini-grounded question answering | Yes | Yes |
| Source citations and clickable Citation Explorer | Yes | Yes |
| Retrieval X-Ray (stage rankings, passages, timing) | Yes | Yes |
| Claim-level evidence verification | Disabled in public demo | Available |
| Retrieval benchmarks and regression utilities | Precomputed results displayed | CLI evaluation tools |
| Interactive evaluation dashboard | Yes | Yes |
| API rate limiting and curated document access | Initial safeguards | Configurable |

### Retrieval X-Ray

Inspect dense, BM25, RRF fusion, and selected results, including source page, chunk ID, score, text, and stage latency. The Citation Explorer links inline answer citations (for example, `[2]`) to the corresponding retrieved passage.

### Evaluation dashboard

Compare Precision@5, Recall@5, MRR, nDCG@5, and latency across retrieval modes; filter per-question results and inspect failures. Dashboard data is exported to static JSON, so browsing the dashboard does **not** incur Gemini inference requests.

## Architecture

```text
Browser
  |
  v
Next.js frontend (Vercel)
  |
  v
FastAPI backend (Render)
  |-- PDF ingestion and chunking (local/full mode)
  |-- Dense query embeddings --> Gemini Embedding API (cloud)
  |-- Dense vector search ----> Qdrant Cloud
  |-- BM25 lexical search
  |-- Reciprocal Rank Fusion
  |-- Optional local cross-encoder reranking
  |-- Answer generation -----> Gemini API
  `-- Optional claim-level evidence verification

Offline benchmark runner --> benchmark JSON --> Next.js /evaluation
```

The same Python codebase supports two configurations:

- **Cloud:** Gemini embeddings, hosted Qdrant, no local PyTorch model, and a restricted public demo.
- **Local:** Sentence Transformers embeddings, embedded Qdrant, and optional cross-encoder reranking.

Embedding models and dimensions must remain consistent within each Qdrant collection. The cloud collection uses 768-dimensional Gemini embeddings; do not reuse it for local MiniLM vectors.

## Technology

- **Backend:** Python 3.12+, FastAPI, Pydantic, uv, PyMuPDF, Qdrant, rank-bm25, SlowAPI.
- **ML:** Gemini API, Sentence Transformers (optional local dependency), cross-encoder reranking, RRF.
- **Frontend:** Next.js, TypeScript, Tailwind CSS, Recharts.
- **Deployment:** Vercel (frontend), Render (API), Qdrant Cloud (vectors).
- **Quality:** pytest, Ruff, TypeScript, ESLint, reproducible retrieval benchmarks.

## Quick start — full local setup

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), Node.js, npm, and a Gemini API key.

```bash
git clone https://github.com/hamedhoney/docuMind.git
cd docuMind
uv sync --extra local-ml
cp .env.example .env
```

Edit `.env` with your actual Gemini key and local settings. Example (adjust model names and paths to your configuration):

```dotenv
APP_ENV=local
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
RERANKER_PROVIDER=local
QDRANT_COLLECTION=documind
FRONTEND_ORIGIN=http://localhost:3000
```

Start the backend:

```bash
uv run uvicorn documind.api:app --reload
```

In another terminal:

```bash
cd apps/web
npm install
printf 'NEXT_PUBLIC_API_URL=http://localhost:8000\n' > .env.local
npm run dev
```

Open **http://localhost:3000** for the frontend and **http://localhost:8000/docs** for the interactive API reference.

> The local frontend may be configured for the curated public demo document. To use other locally indexed documents, ensure the frontend supplies their IDs and the backend is in local mode.

## Cloud configuration

The cloud backend uses a separate Qdrant Cloud collection and Gemini embeddings. Copy `.env.cloud.example` as a reference; set real credentials only in your deployment provider's environment settings.

```dotenv
APP_ENV=cloud
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSION=768
RERANKER_PROVIDER=none
QDRANT_URL=https://your-qdrant-endpoint
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=documind_cloud
PUBLIC_UPLOADS_ENABLED=false
FRONTEND_ORIGIN=https://docu-mind-iota-self.vercel.app
```

The Vercel frontend uses:

```dotenv
NEXT_PUBLIC_API_URL=https://documind-api-mcnz.onrender.com
```

`NEXT_PUBLIC_API_URL` is intentionally browser-visible; **never** put Gemini or Qdrant API keys in `NEXT_PUBLIC_*` variables. Changing this value requires a new frontend build/deployment.

The public deployment uses a pre-indexed document and restricts expensive operations. Basic per-IP limits are a starting safeguard, **not** a guarantee against quota abuse: free-hosting proxies, process restarts, and multiple instances can affect in-memory rate limiting.

## API overview

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Service health |
| GET | `/docs` | Interactive OpenAPI documentation |
| POST | `/documents` | PDF ingestion (restricted in cloud) |
| POST | `/retrieve` | Ranked evidence and Retrieval X-Ray trace |
| POST | `/ask` | Grounded answer, sources, trace, optional verification |

Example question payload:

```json
{
  "question": "What is the primary purpose of the Common Pile?",
  "top_k": 5,
  "retrieval_mode": "hybrid",
  "verify": false
}
```

Retrieval modes: `dense`, `bm25`, `hybrid`, and `hybrid_rerank` (local reranker required). See `/docs` for the current request and response schemas.

## Retrieval benchmark

**Dataset:** 10 manually curated Common Pile questions with relevant chunk IDs; **top-k:** 5. These are preliminary, small-sample results and depend on the quality and completeness of relevance annotations.

| Metric | Dense | BM25 | Hybrid (RRF) |
| --- | ---: | ---: | ---: |
| Precision@5 | 0.080 | 0.080 | **0.140** |
| Recall@5 | 0.283 | 0.150 | **0.450** |
| MRR | 0.333 | 0.150 | **0.400** |
| nDCG@5 | 0.285 | 0.135 | **0.388** |
| Mean retrieval latency | 418.8 ms | 2.8 ms | 370.8 ms |

**Observations:** Hybrid improves average retrieval quality in this benchmark, but five questions (`q002`, `q003`, `q006`, `q009`, `q010`) have zero Recall@5 across all three strategies. These cases are candidates for relevance-label auditing, chunking improvements, and retrieval failure analysis. The benchmark is too small for broad performance claims.

Measured data: [`benchmarks/results/cloud_retrieval.json`](benchmarks/results/cloud_retrieval.json). Interactive charts: [Evaluation Dashboard](https://docu-mind-iota-self.vercel.app/evaluation).

### Reproduce the cloud benchmark

Configure `.env` for the existing cloud Qdrant collection and Gemini embedding model, and ensure the benchmark's relevant chunk IDs match the indexed document:

```bash
uv run python -m documind.evaluation.run \
  --dataset benchmarks/datasets/common_pile_questions.json \
  --output benchmarks/results/cloud_retrieval.json
```

Export results for the Next.js dashboard:

```bash
uv run python scripts/export_dashboard_results.py
```

The export produces `apps/web/src/data/retrieval-results.json`. Commit this file for the static dashboard; keep local Qdrant storage ignored. Cloud benchmarking consumes Gemini embedding quota and may encounter rate limits.

Other evaluation utilities include `scripts/validate_benchmarks.py`, `scripts/check_regression.py`, and `scripts/plot_benchmark.py`.

## Development checks

```bash
uv run ruff check .
uv run pytest -v
cd apps/web
npm run lint
npx tsc --noEmit
npm run build
```

## Repository layout

```text
apps/web/                    Next.js UI, Citation Explorer, Retrieval X-Ray, evaluation
src/documind/api.py          FastAPI application
src/documind/config.py       Local/cloud configuration
src/documind/ingestion/      PDF extraction and chunking
src/documind/embeddings/     Local and Gemini embedding providers
src/documind/retrieval/      Dense, BM25, RRF, reranking, traces
src/documind/storage/        Qdrant vector storage
src/documind/generation/     Grounded answer generation
src/documind/verification/   Claim-level evidence verification
src/documind/evaluation/     Metrics, benchmark runner, regression checks
benchmarks/datasets/         Benchmark questions and labels
benchmarks/results/          Measured benchmark reports
scripts/                     Indexing, validation, plotting, export utilities
```

## Limitations and next steps

- Public uploads and verification are disabled to control free-tier costs; the full local pipeline supports them.
- Free hosting may sleep or suspend inactive services, causing cold starts or temporary unavailability.
- Basic rate limiting is not equivalent to authentication or a shared, production-grade quota gateway.
- Citations expose the retrieved passages, but do not independently prove claim support; verification is a separate feature.
- Benchmark labels should be audited and expanded before drawing strong retrieval-quality conclusions.
- Planned improvements include retrieval failure analysis, chunk-boundary improvements, benchmark expansion, and stronger API protections.

## License

MIT. See [LICENSE](LICENSE).
