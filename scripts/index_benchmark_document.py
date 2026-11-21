import hashlib
import random
import re
import time
from pathlib import Path

from google.genai.errors import ClientError

from documind.embeddings.encoder import create_encoder
from documind.ingestion.chunker import chunk_pages
from documind.ingestion.pdf import extract_pdf
from documind.storage.vector_store import VectorStore

PDF_PATH = Path("benchmarks/documents/common_pile.pdf")
BATCH_SIZE = 8
MAX_RETRIES = 5


def get_retry_delay(error: ClientError, attempt: int) -> float:
    message = str(error)

    match = re.search(
        r"retry in ([\d.]+)s",
        message,
        flags=re.IGNORECASE,
    )

    if match:
        return float(match.group(1)) + 2.0

    match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)s", message)

    if match:
        return float(match.group(1)) + 2.0

    return min(2 ** attempt * 10, 120) + random.uniform(0, 2)


def encode_with_retry(encoder, texts: list[str]):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return encoder.encode(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
            )

        except ClientError as error:
            if error.code != 429:
                raise

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    "Gemini embedding quota still exceeded. "
                    "Run the script again after the quota resets."
                ) from error

            delay = get_retry_delay(error, attempt)

            print(
                f"Rate limit reached. "
                f"Waiting {delay:.1f} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError("Embedding retries exhausted")


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    document_id = hashlib.sha256(
        PDF_PATH.read_bytes()
    ).hexdigest()

    pages = extract_pdf(PDF_PATH, document_id)
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("No extractable text found")

    store = VectorStore()

    try:
        encoder = create_encoder()
        store.initialize(dimension=encoder.dimension)

        indexed_ids = store.get_indexed_chunk_ids(document_id)

        remaining = [
            chunk
            for chunk in chunks
            if chunk.chunk_id not in indexed_ids
        ]

        print(f"Document ID: {document_id}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Already indexed: {len(chunks) - len(remaining)}")
        print(f"Remaining: {len(remaining)}")

        for start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[start:start + BATCH_SIZE]

            vectors = encode_with_retry(
                encoder,
                [chunk.text for chunk in batch],
            )

            store.add(batch, vectors.tolist())

            completed = (
                len(chunks)
                - len(remaining)
                + min(start + BATCH_SIZE, len(remaining))
            )

            print(f"Indexed {completed}/{len(chunks)} chunks")

            # Reduce request bursts on the free tier.
            time.sleep(2)

        print("Indexing complete!")

    finally:
        store.close()


if __name__ == "__main__":
    main()
