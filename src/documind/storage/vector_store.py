from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from documind.ingestion.chunker import Chunk


class VectorStore:
    def __init__(
        self,
        collection: str = "documind",
        storage_path: str | None = None,
    ):
        project_root = Path(__file__).resolve().parents[3]

        if storage_path is None:
            storage_path = str(
                project_root / "data" / "qdrant"
            )

        Path(storage_path).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = QdrantClient(path=storage_path)
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
