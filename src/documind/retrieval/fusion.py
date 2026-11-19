def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    if k <= 0:
        raise ValueError("k must be positive")

    combined: dict[str, dict] = {}

    for source_index, results in enumerate(result_lists):
        source = (
            ["dense", "bm25"][source_index]
            if source_index < 2
            else f"source_{source_index}"
        )

        seen = set()

        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            if chunk_id not in combined:
                combined[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "retrieved_by": [],
                }

            combined[chunk_id]["rrf_score"] += 1.0 / (k + rank)
            combined[chunk_id]["retrieved_by"].append(source)

    return sorted(
        combined.values(),
        key=lambda item: (
            -item["rrf_score"],
            item["chunk_id"],
        ),
    )
