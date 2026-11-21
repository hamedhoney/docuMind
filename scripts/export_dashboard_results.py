
import json
from pathlib import Path


SOURCE = Path("benchmarks/results/cloud_retrieval.json")
DESTINATION = Path(
    "apps/web/src/data/retrieval-results.json"
)


def main() -> None:
    report = json.loads(SOURCE.read_text())

    modes = []

    for mode, metrics in report["summary"].items():
        modes.append(
            {
                "mode": mode,
                "precision_at_k": metrics["precision_at_5"],
                "recall_at_k": metrics["recall_at_5"],
                "ndcg_at_k": metrics["ndcg_at_5"],
                "mrr": metrics["mrr"],
                "mean_latency_ms": metrics["mean_latency_ms"],
                "p95_latency_ms": None,
            }
        )

    output = {
        "schema_version": "1.0",
        "benchmark": "common_pile",
        "environment": "cloud",
        "question_count": report["num_questions"],
        "k": 5,
        "embedding_model": "gemini-embedding-001",
        "collection": "documind_cloud",
        "modes": modes,
        "results": report["results"],
    }

    DESTINATION.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DESTINATION.write_text(
        json.dumps(output, indent=2) + "\n"
    )

    print(f"Exported: {DESTINATION}")


if __name__ == "__main__":
    main()
