import json
from pathlib import Path

from documind.storage.vector_store import VectorStore

dataset_path = Path(
    "benchmarks/datasets/common_pile_questions.json"
)

questions = json.loads(dataset_path.read_text())

store = VectorStore()

try:
    store.initialize()
    chunks = store.get_all_chunks()

    print(f"Found {len(chunks)} indexed chunks")

    for question in questions:
        keywords = question["evidence_keywords"]

        matches = [
            chunk for chunk in chunks
            if all(
                keyword.lower() in chunk["text"].lower()
                for keyword in keywords
            )
        ]

        print(f"\n{question['id']}: {question['question']}")

        for chunk in matches[:5]:
            print("Document ID:", chunk["document_id"])
            print("Chunk ID:", chunk["chunk_id"])
            print("Page:", chunk["page_number"])
            print("Text:", chunk["text"][:300])
            print("---")

finally:
    store.close()