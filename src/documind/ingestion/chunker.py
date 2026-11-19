from dataclasses import dataclass

from documind.ingestion.pdf import Page


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    page_number: int
    text: str


def chunk_pages(
    pages: list[Page],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        for start in range(0, len(page.text), step):
            text = page.text[start : start + chunk_size]

            if not text.strip():
                continue

            chunks.append(
                Chunk(
                    chunk_id=(f"{page.document_id}-{page.page_number}-{start}"),
                    document_id=page.document_id,
                    page_number=page.page_number,
                    text=text,
                )
            )

            if start + chunk_size >= len(page.text):
                break

    return chunks
