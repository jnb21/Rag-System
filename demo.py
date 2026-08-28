"""Add documents to the RAG system one at a time and inspect what's produced.

Current stage: chunking + embedding + vector store. Retrieval and generation
are added later.
"""

import os

from rag_system import EMBEDDING_DIMS, VectorStore

DATA_DIR = "data"
FILES_IN_ORDER = [
    "01_python_basics.txt",
    "02_machine_learning.txt",
    "03_rag_systems.txt",
    "04_vector_databases.txt",
]


def main():
    store = VectorStore()

    for step, filename in enumerate(FILES_IN_ORDER, start=1):
        path = os.path.join(DATA_DIR, filename)
        print(f"\n{'=' * 70}")
        print(f"STEP {step}: adding {filename}")
        print("=" * 70)

        added = store.add_document(path)
        print(f"  + {added} chunk(s) produced from this file")
        print(f"  + vector store now holds {len(store)} chunk(s) total")

        new_entries = store.entries[-added:]
        for i, entry in enumerate(new_entries, start=1):
            preview = entry["text"][:80].replace("\n", " ")
            vector_preview = ", ".join(f"{v:.2f}" for v in entry["embedding"][:6])
            print(f"    chunk {i} ({len(entry['text'].split())} words): {preview}...")
            print(f"      embedding[{EMBEDDING_DIMS}] = [{vector_preview}, ...]")


if __name__ == "__main__":
    main()
