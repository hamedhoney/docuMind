import json
from pathlib import Path


def check_regression(
    baseline_path: Path,
    current_path: Path,
    tolerance: float = 0.02,
) -> list[str]:
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")

    baseline = json.loads(baseline_path.read_text())
    current = json.loads(current_path.read_text())

    if baseline["num_questions"] != current["num_questions"]:
        raise ValueError(
            "Benchmark datasets have different question counts"
        )

    metrics = (
        "recall_at_5",
        "ndcg_at_5",
        "mrr",
    )

    regressions = []

    for mode, baseline_metrics in baseline["summary"].items():
        if mode not in current["summary"]:
            regressions.append(f"Missing retrieval mode: {mode}")
            continue

        current_metrics = current["summary"][mode]

        for metric in metrics:
            previous = baseline_metrics[metric]
            latest = current_metrics[metric]
            change = latest - previous

            if change < -tolerance:
                regressions.append(
                    f"{mode}: {metric} dropped "
                    f"from {previous:.4f} to {latest:.4f} "
                    f"({change:+.4f})"
                )

    return regressions
