"""Augmentation: turn retrieved chunks + a question into a single grounded prompt."""

PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the context does not contain the answer, say so.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def _source_of(node):
    return node.node.metadata.get("file_name", "unknown")


def build_context(nodes):
    """Join retrieved nodes into one context block, tagged with their source file."""
    return "\n\n".join(f"[{_source_of(node)}] {node.node.get_content()}" for node in nodes)


def build_prompt(nodes, question):
    """Augment the question with retrieved context, producing the final LLM prompt."""
    return PROMPT_TEMPLATE.format(context=build_context(nodes), question=question)
