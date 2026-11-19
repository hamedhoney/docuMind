
from documind.embeddings.encoder import EmbeddingEncoder
from documind.storage.vector_store import VectorStore


class SemanticRetriever:
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        store: VectorStore,
    ):
        self.encoder = encoder
        self.store = store

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        vector = self.encoder.encode([query])[0].tolist()

        result = self.store.client.query_points(
            collection_name=self.store.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "score": point.score,
                **(point.payload or {}),
            }
            for point in result.points
        ]
