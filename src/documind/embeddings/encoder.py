from typing import Protocol

import numpy as np

from documind.config import get_settings


class Encoder(Protocol):
    dimension: int

    def encode(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> np.ndarray: ...


class LocalEmbeddingEncoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        self.model = SentenceTransformer(settings.embedding_model)
        dimension = self.model.get_sentence_embedding_dimension()

        if dimension is None:
            raise ValueError("Unknown embedding dimension")

        self.dimension = dimension

    def encode(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        return np.asarray(
            self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )


class GeminiEmbeddingEncoder:
    def __init__(self):
        from google import genai

        settings = get_settings()

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")

        self.client = genai.Client(
            api_key=settings.gemini_api_key,
        )
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension

    def encode(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> np.ndarray:
        from google.genai import types

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = []

        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]

            response = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dimension,
                ),
            )

            if not response.embeddings:
                raise ValueError("Gemini returned no embeddings")

            vectors.extend(embedding.values for embedding in response.embeddings)

        array = np.asarray(vectors, dtype=np.float32)

        if array.shape != (len(texts), self.dimension):
            raise ValueError("Unexpected embedding dimensions")

        norms = np.linalg.norm(array, axis=1, keepdims=True)

        return array / np.maximum(norms, 1e-12)


def create_encoder() -> Encoder:
    settings = get_settings()

    if settings.embedding_provider == "gemini":
        return GeminiEmbeddingEncoder()

    return LocalEmbeddingEncoder()
