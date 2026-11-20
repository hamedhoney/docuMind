
import argparse
from pathlib import Path

from documind.embeddings.encoder import EmbeddingEncoder
from documind.evaluation.benchmark import run_benchmark
from documind.retrieval.bm25 import BM25Retriever
from documind.retrieval.hybrid import HybridRetriever
from documind.retrieval.reranker import Reranker
from documind.retrieval.semantic import SemanticRetriever
from documind.storage.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DocuMind retrieval strategies"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmarks/datasets/questions.json"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/latest.json"),
    )

    args = parser.parse_args()

    store = VectorStore()

    try:
        store.initialize()

        encoder = EmbeddingEncoder()
        semantic = SemanticRetriever(encoder, store)

        bm25 = BM25Retriever()
        bm25.build(store.get_all_chunks())

        reranker = Reranker()

        retriever = HybridRetriever(
            semantic=semantic,
            bm25=bm25,
            reranker=reranker,
        )

        report = run_benchmark(
            retriever=retriever,
            dataset_path=args.dataset,
            output_path=args.output,
        )

        print("\nDocuMind Retrieval Benchmark")
        print("=" * 45)

        for mode, metrics in report["summary"].items():
            print(f"\n{mode}")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")

        print(f"\nResults saved to: {args.output}")

    finally:
        store.close()


if __name__ == "__main__":
    main()
