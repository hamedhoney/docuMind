class Reranker:
    def __init__(
        self,
        model_name: str = ("cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        passages: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not passages:
            return []

        pairs = [(query, passage["text"]) for passage in passages]

        scores = self.model.predict(pairs)

        results = [
            {
                **passage,
                "rerank_score": float(score),
            }
            for passage, score in zip(passages, scores)
        ]

        return sorted(
            results,
            key=lambda item: item["rerank_score"],
            reverse=True,
        )[:top_k]
