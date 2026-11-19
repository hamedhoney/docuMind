
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from documind.ingestion.chunker import Chunk


class VectorStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "documind",
    ):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def initialize(self, dimension: int = 384) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    def add(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk/vector count mismatch")

        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        if points:
            self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )
