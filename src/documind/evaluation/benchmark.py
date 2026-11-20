
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from pydantic import BaseModel, Field

from documind.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class EvaluationQuestion(BaseModel):
    id: str
    question: str
    relevant_chunk_ids: list[str] = Field(min_length=1)
    category: str = "general"
    document_id: str | None = None


class BenchmarkResult(BaseModel):
    question_id: str
    mode: str
    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    latency_ms: float


MODES = (
    "dense",
    "bm25",
    "hybrid",
    "hybrid_rerank",
)


def run_benchmark(
    retriever,
    dataset_path: Path,
    output_path: Path,
) -> dict:
    questions = [
        EvaluationQuestion.model_validate(item)
        for item in json.loads(dataset_path.read_text())
    ]

    if not questions:
        raise ValueError("Evaluation dataset is empty")

    results = []

    for question in questions:
        relevant = set(question.relevant_chunk_ids)

        for mode in MODES:
            start = perf_counter()

            passages = retriever.search(
                query=question.question,
                top_k=5,
                mode=mode,
                document_id=question.document_id,
            )

            latency_ms = (perf_counter() - start) * 1000

            retrieved_ids = [
                passage["chunk_id"]
                for passage in passages
            ]

            results.append(
                BenchmarkResult(
                    question_id=question.id,
                    mode=mode,
                    precision_at_5=precision_at_k(
                        retrieved_ids, relevant, 5
                    ),
                    recall_at_5=recall_at_k(
                        retrieved_ids, relevant, 5
                    ),
                    mrr=reciprocal_rank(
                        retrieved_ids, relevant
                    ),
                    ndcg_at_5=ndcg_at_k(
                        retrieved_ids, relevant, 5
                    ),
                    latency_ms=latency_ms,
                )
            )

    summary = {}

    for mode in MODES:
        mode_results = [
            result for result in results
            if result.mode == mode
        ]

        summary[mode] = {
            "precision_at_5": mean(
                r.precision_at_5 for r in mode_results
            ),
            "recall_at_5": mean(
                r.recall_at_5 for r in mode_results
            ),
            "mrr": mean(r.mrr for r in mode_results),
            "ndcg_at_5": mean(
                r.ndcg_at_5 for r in mode_results
            ),
            "mean_latency_ms": mean(
                r.latency_ms for r in mode_results
            ),
        }

    report = {
        "num_questions": len(questions),
        "summary": summary,
        "results": [
            result.model_dump()
            for result in results
        ],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(report, indent=2)
    )

    return report
