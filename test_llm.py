"""Quick isolated check of the LLM answer-generation step.

Builds the vector store, then calls generate_answer() once and prints the
result -- a fast way to confirm the LLM piece works before running the full
step-by-step walkthrough in demo.py.
"""

import os

from demo import DATA_DIR, FILES_IN_ORDER, SAMPLE_QUESTION
from rag_system import VectorStore, generate_answer


def main():
    store = VectorStore()
    for filename in FILES_IN_ORDER:
        store.add_document(os.path.join(DATA_DIR, filename))
    print(f"Indexed {len(store)} chunk(s) from {len(FILES_IN_ORDER)} file(s).")

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is set -- using Claude for generation.\n")
    else:
        print("ANTHROPIC_API_KEY is not set -- using extractive fallback.\n")

    print(f"Question: {SAMPLE_QUESTION}")
    answer, results = generate_answer(store, SAMPLE_QUESTION, top_k=3)
    print(f"\nAnswer:\n{answer}")
    print("\nSources used:")
    for score, entry in results:
        print(f"  - {entry['source']} (score={score:.3f})")


if __name__ == "__main__":
    main()
