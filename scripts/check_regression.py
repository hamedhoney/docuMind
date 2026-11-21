import argparse
from pathlib import Path

from documind.evaluation.regression import check_regression


def main():
    parser = argparse.ArgumentParser(description="Check DocuMind retrieval regressions")

    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("benchmarks/baselines/common_pile.json"),
    )

    parser.add_argument(
        "--current",
        type=Path,
        default=Path("benchmarks/results/common_pile.json"),
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()

    regressions = check_regression(
        baseline_path=args.baseline,
        current_path=args.current,
        tolerance=args.tolerance,
    )

    if regressions:
        print("\nRetrieval regressions detected:")

        for regression in regressions:
            print(f"  - {regression}")

        raise SystemExit(1)

    print("No significant retrieval regressions detected.")


if __name__ == "__main__":
    main()
