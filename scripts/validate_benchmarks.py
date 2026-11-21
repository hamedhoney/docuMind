import json
from pathlib import Path

from documind.storage.vector_store import VectorStore

dataset_path = Path("benchmarks/datasets/common_pile_questions.json")

questions = json.loads(dataset_path.read_text())

store = VectorStore()

try:
    store.initialize()
    chunks = store.get_all_chunks()

    indexed_ids = {chunk["chunk_id"] for chunk in chunks}

    errors = []

    for question in questions:
        relevant = question["relevant_chunk_ids"]

        if not relevant:
            errors.append(f"{question['id']}: no relevance labels")
            continue

        missing = set(relevant) - indexed_ids

        if missing:
            errors.append(f"{question['id']}: missing chunks {missing}")

    if errors:
        print("Dataset validation failed:")
        for error in errors:
            print(" -", error)
        raise SystemExit(1)

    print(f"Validated {len(questions)} questions")
    print(f"Indexed chunks: {len(chunks)}")

finally:
    store.close()
