from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class Page:
    document_id: str
    page_number: int
    text: str


def extract_pdf(path: Path, document_id: str) -> list[Page]:
    pages = []

    with pymupdf.open(path) as pdf:
        for index, page in enumerate(pdf):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    Page(
                        document_id=document_id,
                        page_number=index + 1,
                        text=text,
                    )
                )

    return pages
