
import json

from documind.evaluation.regression import check_regression


def make_report(recall: float) -> dict:
    return {
        "num_questions": 10,
        "summary": {
            "dense": {
                "recall_at_5": recall,
                "ndcg_at_5": 0.8,
                "mrr": 0.9,
            }
        },
    }


def test_regression_detected(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"

    baseline.write_text(
        json.dumps(make_report(0.8))
    )

    current.write_text(
        json.dumps(make_report(0.6))
    )

    regressions = check_regression(
        baseline,
        current,
    )

    assert len(regressions) == 1
    assert "recall_at_5" in regressions[0]


def test_no_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"

    baseline.write_text(
        json.dumps(make_report(0.8))
    )

    current.write_text(
        json.dumps(make_report(0.81))
    )

    assert check_regression(
        baseline,
        current,
    ) == []
