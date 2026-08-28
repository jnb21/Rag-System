"""RAG (Retrieval-Augmented Generation) system, built up step by step.

Current stage: chunking only.
"""

import re


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
