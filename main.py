"""End-to-end walkthrough of the pipeline, one call per R-A-G stage:

retrieve (LlamaIndex + Chroma) -> augment (build grounded prompt) ->
generate (LangChain + Claude).
"""

from augment.prompt import build_prompt
from generate.llm import generate
from retrieve.ingest import build_index
from retrieve.retriever import retrieve

SAMPLE_QUESTION = "How does a RAG system use vector similarity to retrieve relevant chunks?"


def main():
    print("Building index from data/ ...")
    index = build_index()

    nodes = retrieve(index, SAMPLE_QUESTION, top_k=3)

    print(f"\nQuestion: {SAMPLE_QUESTION}")
    print("\nRetrieved chunks:")
    for rank, node in enumerate(nodes, start=1):
        source = node.node.metadata.get("file_name", "unknown")
        preview = node.node.get_content()[:100].replace("\n", " ")
        print(f"  #{rank} score={node.score:.3f} source={source}")
        print(f"      {preview}...")

    if not nodes:
        answer = "No indexed content is relevant to that question yet."
    else:
        prompt = build_prompt(nodes, SAMPLE_QUESTION)
        answer = generate(prompt, fallback_text=nodes[0].node.get_content())

    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
