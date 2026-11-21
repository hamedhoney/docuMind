from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from documind.config import get_settings
from documind.ingestion.chunker import Chunk


class VectorStore:
    def __init__(
        self,
        collection: str | None = None,
    ):
        settings = get_settings()

        self.collection = collection or settings.qdrant_collection

        if settings.app_env == "cloud":
            if not (settings.qdrant_url and settings.qdrant_api_key):
                raise ValueError("Cloud Qdrant credentials are missing")

            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=30,
            )
        else:
            settings.qdrant_path.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.client = QdrantClient(path=str(settings.qdrant_path))

    def initialize(self, dimension: int = 384) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        # Required for document-specific filtering in Qdrant Cloud.
        collection_info = self.client.get_collection(
            collection_name=self.collection,
        )

        if "document_id" not in collection_info.payload_schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
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

    def get_indexed_chunk_ids(self, document_id: str) -> set[str]:
        return {
            chunk["chunk_id"]
            for chunk in self.get_all_chunks()
            if chunk["document_id"] == document_id
        }

    def get_all_chunks(self) -> list[dict]:
        chunks = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                if point.payload:
                    chunks.append(
                        {
                            "chunk_id": point.payload["chunk_id"],
                            "document_id": point.payload["document_id"],
                            "page_number": point.payload["page_number"],
                            "text": point.payload["text"],
                        }
                    )

            if offset is None:
                break

        return chunks

    def close(self) -> None:
        self.client.close()
