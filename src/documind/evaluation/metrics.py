from math import log2


def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    return sum(chunk_id in relevant for chunk_id in retrieved[:k]) / k


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("Relevant set cannot be empty")

    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(
    retrieved: list[str],
    relevant: set[str],
) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("Relevant set cannot be empty")

    seen = set()
    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in relevant and chunk_id not in seen:
            dcg += 1.0 / log2(rank + 1)
            seen.add(chunk_id)

    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_count + 1))

    return dcg / idcg
