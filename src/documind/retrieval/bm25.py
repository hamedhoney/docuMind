import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    def __init__(self):
        self.chunks: list[dict] = []
        self.index: BM25Okapi | None = None

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks

        if not chunks:
            self.index = None
            return

        corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.index = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_id: str | None = None,
    ) -> list[dict]:
        if self.index is None or not tokenize(query):
            return []

        scores = self.index.get_scores(tokenize(query))

        ranked = sorted(
            (
                (index, score)
                for index, score in enumerate(scores)
                if document_id is None
                or self.chunks[index]["document_id"] == document_id
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for index, score in ranked:
            if score <= 0:
                continue

            results.append(
                {
                    **self.chunks[index],
                    "score": float(score),
                }
            )

            if len(results) >= top_k:
                break

        return results
