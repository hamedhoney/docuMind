import pytest

from documind.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c"}

    assert recall_at_k(retrieved, relevant, 2) == 0.5


def test_reciprocal_rank():
    retrieved = ["a", "b", "c"]
    relevant = {"c"}

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_ndcg_perfect_ranking():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}

    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_empty_relevance_rejected():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)
