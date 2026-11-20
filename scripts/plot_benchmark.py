import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("benchmarks/results/common_pile.json")
OUTPUT = Path("benchmarks/results")

report = json.loads(RESULTS.read_text())
summary = report["summary"]

modes = list(summary)

labels = {
    "dense": "Dense",
    "bm25": "BM25",
    "hybrid": "Hybrid",
    "hybrid_rerank": "Hybrid + Rerank",
}

metrics = [
    ("recall_at_5", "Recall@5"),
    ("ndcg_at_5", "nDCG@5"),
    ("mrr", "MRR"),
    ("mean_latency_ms", "Mean latency (ms)"),
]

OUTPUT.mkdir(parents=True, exist_ok=True)

for metric, title in metrics:
    values = [
        summary[mode][metric]
        for mode in modes
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        [labels.get(mode, mode) for mode in modes],
        values,
    )

    ax.set_title(f"DocuMind — {title}")
    ax.set_ylabel(title)

    if metric != "mean_latency_ms":
        ax.set_ylim(0, 1)

    ax.bar_label(bars, fmt="%.3f", padding=3)

    fig.tight_layout()

    filename = metric + ".png"

    fig.savefig(
        OUTPUT / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

print(f"Charts saved to {OUTPUT}")
