"""Add documents to the RAG system one at a time and inspect what's produced.

Current stage: chunking + embedding. Retrieval and generation are added later.
"""

import os

from rag_system import EMBEDDING_DIMS, chunk_text, embed_chunk

DATA_DIR = "data"
FILES_IN_ORDER = [
    "01_python_basics.txt",
    "02_machine_learning.txt",
    "03_rag_systems.txt",
    "04_vector_databases.txt",
]


def main():
    all_chunks = []

    for step, filename in enumerate(FILES_IN_ORDER, start=1):
        path = os.path.join(DATA_DIR, filename)
        print(f"\n{'=' * 70}")
        print(f"STEP {step}: adding {filename}")
        print("=" * 70)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = embed_chunk(chunk)
            all_chunks.append({"text": chunk, "source": filename, "embedding": embedding})

        print(f"  + {len(chunks)} chunk(s) produced from this file")
        print(f"  + index now holds {len(all_chunks)} chunk(s) total")
        for i, chunk in enumerate(chunks, start=1):
            preview = chunk[:80].replace("\n", " ")
            embedding = all_chunks[len(all_chunks) - len(chunks) + i - 1]["embedding"]
            vector_preview = ", ".join(f"{v:.2f}" for v in embedding[:6])
            print(f"    chunk {i} ({len(chunk.split())} words): {preview}...")
            print(f"      embedding[{EMBEDDING_DIMS}] = [{vector_preview}, ...]")


if __name__ == "__main__":
    main()
