"""RAG (Retrieval-Augmented Generation) system, built up step by step.

Current stage: chunking + embedding.
"""

import hashlib
import math
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
