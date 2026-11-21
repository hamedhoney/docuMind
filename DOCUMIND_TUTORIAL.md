# How DocuMind Works: A Practical RAG Tutorial

**Companion to [README.md](README.md)**  
**Live demo:** https://docu-mind-iota-self.vercel.app/  
**Evaluation dashboard:** https://docu-mind-iota-self.vercel.app/evaluation

This tutorial explains the ideas behind DocuMind, the end-to-end data flow, the retrieval and evaluation methods, and how to reason about failures. It follows the architecture and behavior documented in the project README and the implementation discussed during development. Where exact implementation parameters or function signatures have not been verified from the repository, examples are explicitly illustrative; check the referenced source files before treating them as authoritative.

## 1. What problem does RAG solve?

A language model generates text from its learned parameters. It may not know a specific paper, may be outdated, and can make unsupported claims. **Retrieval-Augmented Generation (RAG)** supplies relevant passages from a document at question time and instructs the model to answer using that evidence.

The fundamental pipeline is:

```text
PDF -> Extract pages -> Split into chunks -> Embed chunks -> Store vectors
                                                            |
User question -> Embed query -> Dense search -----------------+
              -> BM25 search -------------------------------+  |
                                                           v  v
                                              Rank fusion / rerank
                                                       |
                                                Top-k passages
                                                       |
                                         Gemini generates cited answer
                                                       |
                                   Citation Explorer + Retrieval X-Ray
```

DocuMind is *explainable* because it exposes retrieved passages, rankings, scores, and timing instead of showing only the generated answer. It is *evaluation-driven* because it compares retrieval strategies against labeled questions.

## 2. The two operating configurations

| Capability | Public cloud demo | Full local configuration |
|---|---|---|
| Frontend | Next.js on Vercel | Next.js on your machine |
| API | FastAPI on Render | FastAPI locally |
| Embeddings | Gemini embedding API | Sentence Transformers / MiniLM |
| Vector database | Qdrant Cloud | Embedded Qdrant |
| Vector size used in this project | 768 | 384 |
| Lexical search | BM25 | BM25 |
| Rank fusion | RRF | RRF |
| Cross-encoder reranking | Disabled | Available |
| Generation | Gemini | Gemini |
| Claim-level verification | Disabled for public visitors | Available |
| Uploads | Restricted | Available |

**Do not mix embedding spaces.** A 384-dimensional MiniLM vector cannot be searched against a 768-dimensional Gemini collection; even embeddings with matching dimensions from different models generally are not semantically compatible. Use separate collections and re-embed documents when changing models.

## 3. Document ingestion: turning a PDF into searchable evidence

### 3.1 PDF extraction

**Relevant module:** `src/documind/ingestion/pdf.py`

A PDF is parsed into page-associated text. Page metadata matters because an answer can later point to the original page. Image-only/scanned PDFs may have no extractable text unless an OCR stage is added; do not assume that every PDF can be indexed.

### 3.2 Stable document identity

**Relevant endpoint:** `POST /documents` in `src/documind/api.py`

DocuMind computes a SHA-256 hash of the uploaded PDF bytes and uses it as a document identifier:

```python
import hashlib

document_id = hashlib.sha256(pdf_bytes).hexdigest()
```

The same exact file bytes produce the same identifier. A modified PDF produces a different hash. A hash identifies content; it is not a substitute for access control.

### 3.3 Chunking

**Relevant module:** `src/documind/ingestion/chunker.py`

Long documents are divided into shorter passages called **chunks**. Each chunk retains text, a chunk ID, a document ID, and a page number. Chunk size and overlap control the tradeoff:

- **Small chunks:** more precise passages, but may lose context.
- **Large chunks:** more context, but may contain distracting information and consume more tokens.
- **Overlap:** can preserve information crossing boundaries, but creates duplicate-like candidates and increases indexing cost.

Some passages in the current demo begin or end mid-word; improving boundary-aware chunking is a useful future experiment. Do not change chunking without rebuilding the vector index and revisiting relevance labels, because chunk IDs and evaluation outcomes may change.

### 3.4 Embedding and indexing

**Relevant modules:** `src/documind/embeddings/encoder.py`, `src/documind/storage/vector_store.py`

An embedding model maps a passage into a numerical vector:

\[
\mathbf{v}_i = E_{\mathrm{document}}(c_i) \in \mathbb{R}^{d}
\]

The cloud setup uses Gemini embeddings with a configured output dimension of 768. The local setup uses Sentence Transformers with a 384-dimensional MiniLM model. Vectors and chunk metadata are stored in Qdrant. The `document_id` field is indexed as a keyword payload to support document-scoped filtering in Qdrant Cloud.

**Example concept (not a drop-in API call):**

```python
chunks = chunk_pages(pages)
vectors = encoder.encode(
    [chunk.text for chunk in chunks],
    task_type="RETRIEVAL_DOCUMENT",
)
store.add(chunks, vectors.tolist())
```

Gemini indexing may encounter `429 RESOURCE_EXHAUSTED` due to rate limits. A resumable indexing workflow should skip existing chunk IDs, save batches, and respect provider retry delays.

## 4. Question answering: retrieval before generation

### 4.1 Query embedding and dense retrieval

**Relevant module:** `src/documind/retrieval/semantic.py`

A question is embedded with the **query** task type:

\[
\mathbf{q} = E_{\mathrm{query}}(\text{question})
\]

Qdrant searches for document vectors close to the query vector. When vectors are normalized and cosine similarity is used, the similarity is:

\[
\operatorname{cos}(\mathbf{q},\mathbf{v}) =
\frac{\mathbf{q}\cdot\mathbf{v}}{\|\mathbf{q}\|\,\|\mathbf{v}\|}
\]

Dense retrieval can match semantically similar phrases even when their words differ. Its weaknesses include embedding-model mismatch, semantic ambiguity, and external embedding API latency or quotas.

### 4.2 BM25 lexical retrieval

**Relevant module:** `src/documind/retrieval/bm25.py`

BM25 ranks chunks based on query-term occurrences, inverse document frequency, and document-length normalization. A common form is:

\[
\operatorname{BM25}(q,d) = \sum_{t\in q}
\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\mathrm{avgdl}}\right)}
\]

Here, `f(t,d)` is term frequency, `|d|` is passage length, `avgdl` is average passage length, and `k1`/`b` control term-frequency saturation and length normalization. This is the general BM25 method; inspect `bm25.py` for the library's actual tokenization and parameters.

BM25 is especially useful for exact terminology, abbreviations, and rare names. It can miss paraphrases and semantically equivalent wording.

### 4.3 Reciprocal Rank Fusion (RRF)

**Relevant module:** `src/documind/retrieval/fusion.py`

Dense and BM25 scores are not directly comparable: they have different scales. **RRF combines ranks, not raw scores**:

\[
\operatorname{RRF}(d) = \sum_{r\in\{\mathrm{dense},\mathrm{BM25}\}}
\frac{1}{k+\operatorname{rank}_r(d)}
\]

The constant `k` dampens the advantage of very high ranks. A chunk appearing near the top of both lists often rises in the fused list. Check `fusion.py` for DocuMind's exact `k` and handling of duplicate chunk IDs.

**Example (illustrative):** A chunk ranked #2 by Dense and #3 by BM25 receives a contribution from both systems. A chunk present in only one list receives one contribution. This makes fusion robust when one retriever misses useful evidence.

### 4.4 Optional cross-encoder reranking

**Relevant module:** `src/documind/retrieval/reranker.py`

A bi-encoder embeds query and document separately, making it efficient for large-scale search. A **cross-encoder** processes the query and candidate passage together and produces a relevance score. It is usually more expensive per candidate, but can improve ordering within a small candidate set.

```text
Dense + BM25 -> RRF candidate pool -> Cross-encoder -> Top-k
```

DocuMind's `hybrid_rerank` mode is available in the full local configuration. It is disabled in the lightweight cloud demo to avoid loading PyTorch and Sentence Transformers on Render. Plain `hybrid` must not require a reranker.

### 4.5 Retrieval modes

**Relevant module:** `src/documind/retrieval/hybrid.py`

| Mode | Processing | Best use |
|---|---|---|
| `dense` | Query embedding + Qdrant search | Semantic matches and paraphrases |
| `bm25` | Keyword scoring | Exact terms and rare phrases |
| `hybrid` | Dense + BM25 + RRF | General-purpose retrieval |
| `hybrid_rerank` | Hybrid + cross-encoder | More precise candidate ordering when local compute is available |

The retriever produces both selected passages and a **retrieval trace** containing stage-specific ranked results and timing. A candidate pool larger than the final `top_k` is used for hybrid retrieval; the exact selection logic is in `hybrid.py`.

## 5. Grounded generation and citation numbering

**Relevant module:** `src/documind/generation/answer.py`

The selected top-k passages are sent to Gemini along with the question. The generator asks for an answer grounded in the supplied evidence. Citations such as `[2]` refer to the second selected passage, not necessarily to page 2 of the PDF.

```text
Question: What is the purpose of the Common Pile?
Retrieved passages: [1], [2], [3], [4], [5]
Generated answer: ... openly licensed training data [2, 4].
```

A citation indicates which retrieved passage the generator referred to; **it does not by itself prove that the claim is supported**. The Citation Explorer displays the cited passage and page, while the optional verification stage assesses support separately.

## 6. Evidence verification

**Relevant modules:** `src/documind/verification/verifier.py`, `src/documind/verification/models.py`

The full local configuration can inspect claims in an answer against retrieved evidence and report statuses such as:

- **Supported:** retrieved evidence backs the claim.
- **Contradicted:** retrieved evidence conflicts with the claim.
- **Insufficient evidence:** the available passages do not establish the claim.

The verification report can include individual claims, cited source IDs, explanations, and aggregate counts. This is a model-assisted assessment, not a formal proof of factual correctness. The public demo currently disables verification to control API usage.

## 7. Retrieval X-Ray and Citation Explorer

**Frontend modules:** `apps/web/src/components/retrieval-xray.tsx` and `apps/web/src/components/citation-explorer.tsx`

**Retrieval X-Ray** displays Dense, BM25, Fusion, and Selected rankings, plus stage timings, scores, chunk IDs, pages, and expandable text. It helps answer *why did this passage appear?* and *which stage took time?*

**Citation Explorer** makes `[n]` citations clickable. Citation `[n]` maps to `sources[n - 1]`, provided the answer generator and API maintain this ordering. The interface shows source text and page number; it is not a PDF viewer or independent verifier.

**Interpret scores carefully:** BM25, cosine similarity, RRF, and cross-encoder scores have different meanings. Do not compare their numerical values across stages as if they were calibrated probabilities.

## 8. Measuring retrieval quality

**Relevant modules:** `src/documind/evaluation/metrics.py`, `benchmark.py`, `run.py`

A benchmark contains questions and **manually checked relevant chunk IDs**. Each retrieval strategy returns a ranked list; metrics compare that list against the labels. DocuMind's evaluation dashboard reads a precomputed JSON export, avoiding live Gemini requests while visitors browse charts.

### 8.1 Precision@K

\[
P@K = \frac{|\mathrm{Retrieved}_{K}\cap\mathrm{Relevant}|}{K}
\]

Of the top K returned chunks, what fraction are labeled relevant?

### 8.2 Recall@K

\[
R@K = \frac{|\mathrm{Retrieved}_{K}\cap\mathrm{Relevant}|}{|\mathrm{Relevant}|}
\]

Of all labeled relevant chunks, what fraction appear in the top K? This depends strongly on having sufficiently complete ground-truth labels.

### 8.3 Mean Reciprocal Rank (MRR)

\[
\mathrm{MRR} = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i}
\]

Here `r_i` is the rank of the **first** relevant result for question `i`; if none is retrieved, its contribution is zero. MRR rewards finding at least one relevant passage early.

### 8.4 nDCG@K

For binary relevance, discounted cumulative gain is:

\[
\mathrm{DCG}@K = \sum_{i=1}^{K}\frac{\mathrm{rel}_i}{\log_2(i+1)},
\qquad
\mathrm{nDCG}@K = \frac{\mathrm{DCG}@K}{\mathrm{IDCG}@K}
\]

nDCG measures how well relevant passages are ordered. `IDCG` is the best possible DCG for the same labels. Inspect `metrics.py` for the exact implementation and edge-case behavior.

### 8.5 Latency

The retriever records elapsed time for Dense, BM25, fusion, optional reranking, and total retrieval. These numbers do **not** include the full browser-to-answer time or necessarily Gemini generation latency. Mean latency is the arithmetic average; P95 is a high-percentile statistic useful for understanding slower queries.

## 9. Current measured benchmark: Common Pile

**Dataset:** 10 labeled questions; **K:** 5; **Cloud:** Gemini embeddings + Qdrant Cloud. Values below are taken from the recorded `benchmarks/results/cloud_retrieval.json` shared during development.

| Metric | Dense | BM25 | Hybrid |
|---|---:|---:|---:|
| Precision@5 | 0.080 | 0.080 | **0.140** |
| Recall@5 | 0.283 | 0.150 | **0.450** |
| MRR | 0.333 | 0.150 | **0.400** |
| nDCG@5 | 0.285 | 0.135 | **0.388** |
| Mean retrieval latency | 418.8 ms | **2.8 ms** | 370.8 ms |

Hybrid improves Recall@5 over Dense by about **58.8% relative** (0.45 versus 0.2833), but it is far slower than BM25 alone because the dense stage requires external embedding and vector search.

Five questions — `q002`, `q003`, `q006`, `q009`, and `q010` — had zero Recall@5 under all three strategies. This may reflect retrieval misses, incomplete labels, or chunk-boundary effects. These are **small-sample baseline results**, not broad evidence of system performance.

### Reproducing the evaluation

From the repository root, configure the cloud `.env` to point at the indexed collection, then run:

```bash
uv run python -m documind.evaluation.run \
  --dataset benchmarks/datasets/common_pile_questions.json \
  --output benchmarks/results/cloud_retrieval.json
```

The runner uses Dense, BM25, and Hybrid in cloud mode, and can include `hybrid_rerank` when the local reranker is enabled. The benchmark calls the embedding provider for Dense and Hybrid, so it can consume API quota.

Export the static dashboard data using the project's export script:

```bash
uv run python scripts/export_dashboard_results.py
```

The frontend loads `apps/web/src/data/retrieval-results.json` at `/evaluation`. Re-export after rerunning the benchmark and commit the new data file if you want the public dashboard updated.

**Label hygiene:** Nonempty `relevant_chunk_ids` are not sufficient evidence that the labels are correct. Verify that IDs exist in the target collection and that each passage truly supports the question. Never automatically label a retriever's own top results as ground truth.

## 10. API walkthrough

**Relevant module:** `src/documind/api.py`

| Endpoint | Role | Public demo notes |
|---|---|---|
| `GET /health` | Basic service health | Does not itself prove Gemini/Qdrant availability |
| `POST /documents` | Upload, extract, chunk, embed, index | Public uploads disabled |
| `POST /retrieve` | Return retrieval trace | Rate-limited in the demo configuration |
| `POST /ask` | Retrieve, generate answer, optionally verify | Rate-limited; verification disabled in public mode |
| `GET /docs` | Interactive OpenAPI documentation | FastAPI-generated |

**Illustrative request** (check `/docs` for the currently deployed schema):

```json
{
  "question": "What is the primary purpose of the Common Pile?",
  "top_k": 5,
  "retrieval_mode": "hybrid",
  "document_id": "24b26eb7b9bba731af9746974c452a23f870814d31b7d55cd681b771dfe02fdc",
  "verify": false
}
```

The response contains `answer`, `sources`, `retrieval_trace`, and optionally `verification`. The frontend uses `apps/web/src/lib/api.ts` to call these endpoints.

## 11. How to run the full local version

Prerequisites: Python 3.12+, `uv`, Node.js/npm, and Gemini credentials. See [README.md](README.md) for the latest environment templates and deployment guidance.

```bash
git clone https://github.com/hamedhoney/docuMind.git
cd docuMind
uv sync --extra local-ml
cp .env.example .env
# Edit .env locally; never commit secrets.
uv run uvicorn documind.api:app --reload
```

In another terminal:

```bash
cd apps/web
npm install
npm run dev
```

Open `http://localhost:3000` and `http://localhost:8000/docs`. Local uploads and reranking depend on the selected environment settings. The public Next.js UI is optimized for the curated cloud demo; full local backend features can also be exercised through FastAPI `/docs`.

## 12. How the hosted demo works

```text
Vercel (Next.js browser UI)
    | NEXT_PUBLIC_API_URL
    v
Render (FastAPI)
    |-- Gemini Embedding API: question vectors
    |-- Qdrant Cloud: dense retrieval + metadata
    |-- BM25 in API process: lexical retrieval
    |-- RRF: combined ranking
    `-- Gemini: cited answer
```

The frontend exposes only the **public API URL**. Gemini and Qdrant API keys remain on the Render backend. CORS allows approved browser origins but **does not authenticate API callers**. Per-IP in-memory rate limiting is a preliminary safeguard, not a hard spending cap; reverse proxies and process restarts can affect enforcement. Public upload restrictions and a curated document allowlist reduce exposure, but public endpoints still require quota monitoring.

## 13. Troubleshooting guide

| Symptom | Likely cause | First check |
|---|---|---|
| `Module not found: @/lib/api` on Vercel | Git ignores `src/lib/` | Check root `.gitignore` for unanchored `lib/` |
| `Module not found: @/data/retrieval-results.json` | Git ignores frontend data | Use `/data/` to ignore only root Qdrant storage |
| Browser `Failed to fetch` | Wrong API URL, sleeping backend, CORS | Inspect Network request and preflight |
| `OPTIONS /ask` returns 400 | CORS origin mismatch | Match exact Vercel origin in Render settings |
| Qdrant 400 on `document_id` | Missing keyword payload index | Inspect Qdrant collection payload schema |
| 503 with `hybrid` | Reranker guard applied too broadly or upstream failure | Inspect server traceback; only `hybrid_rerank` needs reranker |
| `ModuleNotFoundError: sentence_transformers` in cloud | Eager local-model import | Move CrossEncoder import inside local reranker constructor |
| Gemini 429 | Provider quota/rate limit | Respect retry delay; use resumable indexing |
| Benchmark validation error | Empty or invalid relevance labels | Audit benchmark dataset and indexed chunk IDs |
| Benchmark shows unexpectedly low recall | Missing labels, chunking, or ranking issues | Review per-question retrieved passages |

## 14. Suggested experiments and extensions

1. **Audit relevance labels** for the five zero-recall questions before tuning retrieval.
2. **Chunking ablation:** compare fixed-size chunks against sentence-aware boundaries and multiple overlaps. Reindex and relabel where needed.
3. **Fusion tuning:** vary RRF's damping constant and candidate pool size while holding the test set fixed.
4. **Reranking study:** compare Hybrid and Hybrid+CrossEncoder in the local setup, measuring both quality and latency.
5. **Error analysis:** add per-question evidence previews and identify whether misses originate in Dense, BM25, fusion, or labeling.
6. **Operational improvements:** durable rate limiting, request quotas, deployment health checks, and benchmark CI.

For a credible improvement claim, record the configuration, model, collection, labels, random seeds where applicable, metrics, and latency conditions. Keep a held-out set so repeated tuning does not overfit the ten demo questions.

---

**Key idea:** DocuMind is not just *ask a PDF*. It is an inspectable retrieval system: ingest documents, compare retrieval methods, fuse rankings, generate grounded answers, expose evidence, and measure what works — including failures.
