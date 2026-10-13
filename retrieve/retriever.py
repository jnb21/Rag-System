"""Retrieval: given a built index, find the chunks most relevant to a question."""


def retrieve(index, question, top_k=3):
    """Embed the question and return the top_k nearest chunks (NodeWithScore)."""
    return index.as_retriever(similarity_top_k=top_k).retrieve(question)
