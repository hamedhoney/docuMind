
import argparse
from pathlib import Path

from documind.config import get_settings
from documind.embeddings.encoder import create_encoder
from documind.evaluation.benchmark import run_benchmark
from documind.retrieval.bm25 import BM25Retriever
from documind.retrieval.hybrid import HybridRetriever
from documind.retrieval.reranker import Reranker
from documind.retrieval.semantic import SemanticRetriever
from documind.storage.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DocuMind retrieval strategies"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "benchmarks/datasets/common_pile_questions.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()
    settings = get_settings()

    output_path = args.output

    if output_path is None:
        output_path = Path(
            "benchmarks/results/cloud_retrieval.json"
            if settings.app_env == "cloud"
            else "benchmarks/results/local_retrieval.json"
        )

    encoder = create_encoder()
    store = VectorStore()

    try:
        store.initialize(dimension=encoder.dimension)

        semantic = SemanticRetriever(encoder, store)

        bm25 = BM25Retriever()
        bm25.build(store.get_all_chunks())

        reranker = (
            Reranker()
            if settings.reranker_provider == "local"
            else None
        )

        retriever = HybridRetriever(
            semantic=semantic,
            bm25=bm25,
            reranker=reranker,
        )

        print("DocuMind Retrieval Benchmark")
        print("=" * 45)
        print(f"Environment: {settings.app_env}")
        print(f"Embedding provider: {settings.embedding_provider}")
        print(f"Collection: {store.collection}")
        print(f"Dataset: {args.dataset}")
        print(f"Output: {output_path}")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = run_benchmark(
            retriever=retriever,
            dataset_path=args.dataset,
            output_path=output_path,
        )

        for mode, metrics in report["summary"].items():
            print(f"\n{mode}")

            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {name}: {value:.4f}")
                else:
                    print(f"  {name}: {value}")

        print(f"\nResults saved to: {output_path}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
