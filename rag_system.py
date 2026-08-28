"""RAG (Retrieval-Augmented Generation) system, built up step by step.

Current stage: chunking + embedding + vector store + retrieval + generation.
"""

import hashlib
import math
import os
import re

EMBEDDING_DIMS = 64


def chunk_text(text, max_words=120, overlap=20):
    """Split text into overlapping windows of words, breaking on blank lines first."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= max_words:
            chunks.append(" ".join(words))
            continue
        start = 0
        while start < len(words):
            end = start + max_words
            chunks.append(" ".join(words[start:end]))
            start = end - overlap
    return chunks


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def embed_chunk(text, dims=EMBEDDING_DIMS):
    """Turn text into a fixed-size dense vector using the hashing trick.

    Real embedding models learn this mapping from training data so that
    meaning, not just word overlap, determines proximity. This version hashes
    each token to a dimension and a sign, which is enough to demonstrate the
    core embedding property that matters for RAG: similar text ends up with
    similar vectors, so vector distance becomes a proxy for relevance.
    """
    vector = [0.0] * dims
    for token in tokenize(text):
        digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
        index = digest % dims
        sign = 1.0 if (digest // dims) % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


class VectorStore:
    """In-memory store holding each chunk's text, source file, and embedding.

    This is the layer a real system would swap for Pinecone, Weaviate,
    Qdrant, or pgvector -- something that persists vectors to disk and can
    search millions of them quickly. Here it's just a Python list, since the
    goal is to see the shape of the data before adding search over it.
    """

    def __init__(self):
        self.entries = []  # each: {"text": str, "source": str, "embedding": list[float]}

    def add_document(self, path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)

        chunks = chunk_text(text)
        for chunk in chunks:
            self.entries.append(
                {"text": chunk, "source": source, "embedding": embed_chunk(chunk)}
            )
        return len(chunks)

    def __len__(self):
        return len(self.entries)

    def search(self, query_text, top_k=3):
        """Return the top_k entries whose embedding is closest to the query's."""
        query_vector = embed_chunk(query_text)
        scored = [
            (_cosine_similarity(query_vector, entry["embedding"]), entry)
            for entry in self.entries
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]


def _cosine_similarity(vec_a, vec_b):
    # embed_chunk() already returns unit-length vectors, so the dot product
    # alone equals cosine similarity -- no need to divide by magnitudes.
    return sum(a * b for a, b in zip(vec_a, vec_b))


def build_prompt(question, results):
    """Assemble the retrieved chunks and the question into a single LLM prompt."""
    context = "\n\n".join(f"[{entry['source']}] {entry['text']}" for _, entry in results)
    return (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def generate_answer(store, question, top_k=3):
    """Retrieve relevant chunks, prompt an LLM with them, and return the answer."""
    results = store.search(question, top_k=top_k)
    if not results:
        return "No indexed content is relevant to that question yet.", results

    prompt = build_prompt(question, results)
    answer = _call_llm(prompt)
    if answer is None:
        answer = (
            "(extractive fallback, no ANTHROPIC_API_KEY set) "
            "Most relevant passage: " + results[0][1]["text"]
        )
    return answer, results


def _call_llm(prompt):
    """Call Claude with the assembled prompt, or return None if not configured."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
