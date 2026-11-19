from qdrant_client import models

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

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        vector = self.encoder.encode([query])[0].tolist()

        query_filter = None

        if document_id is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )

        result = self.store.client.query_points(
            collection_name=self.store.collection,
            query=vector,
            query_filter=query_filter,
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
