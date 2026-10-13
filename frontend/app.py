"""Interactive front end for the Retrieve -> Augment -> Generate pipeline.

Upload a folder of PDFs, index them, then ask questions grounded in them.
Run with: streamlit run frontend/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from augment.prompt import build_prompt
from generate.llm import generate
from retrieve.ingest import DATA_DIR, build_index
from retrieve.retriever import retrieve

DATA_PATH = ROOT / DATA_DIR

MUSE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F4EEE2;
    color: #3B342A;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif;
    color: #3B342A;
    font-weight: 600;
}

[data-testid="stFileUploader"], .stTextInput > div > div, [data-testid="stExpander"] {
    background-color: #FBF7EE;
    border-radius: 14px;
    border: 1px solid #E3D9C4;
}

.stButton > button {
    background-color: #C1694F;
    color: #FBF7EE;
    border-radius: 999px;
    border: none;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
}

.stButton > button:hover {
    background-color: #A8553F;
    color: #FBF7EE;
}

.stButton > button:disabled {
    background-color: #D9CBB3;
    color: #8A7F6A;
}

[data-testid="stExpander"] summary {
    font-family: 'Fraunces', serif;
}

.stAlert {
    border-radius: 14px;
}
</style>
"""

st.set_page_config(page_title="Document Insight", page_icon="\U0001F4DC", layout="centered")
st.markdown(MUSE_CSS, unsafe_allow_html=True)

if "index" not in st.session_state:
    st.session_state.index = None
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

st.title("Document Insight")
st.caption("Upload a folder of PDFs, then ask questions grounded in them.")

uploaded_files = st.file_uploader(
    "Upload PDF documents", type=["pdf"], accept_multiple_files=True
)

if st.button("Process documents", disabled=not uploaded_files):
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    for existing in DATA_PATH.glob("*.pdf"):
        existing.unlink()
    for uploaded in uploaded_files:
        (DATA_PATH / uploaded.name).write_bytes(uploaded.getvalue())

    with st.spinner("Chunking, embedding, and indexing..."):
        st.session_state.index = build_index()
        st.session_state.doc_names = [f.name for f in uploaded_files]

    st.success(f"Indexed {len(uploaded_files)} document(s).")

if st.session_state.doc_names:
    st.markdown("**Indexed documents:** " + ", ".join(st.session_state.doc_names))

st.divider()

question = st.text_input("Ask a question about the uploaded documents")

if st.button("Ask", disabled=st.session_state.index is None or not question):
    with st.spinner("Retrieving relevant passages..."):
        nodes = retrieve(st.session_state.index, question, top_k=3)

    if not nodes:
        st.warning("No indexed content is relevant to that question yet.")
    else:
        with st.spinner("Generating answer..."):
            prompt = build_prompt(nodes, question)
            answer = generate(prompt, fallback_text=nodes[0].node.get_content())

        st.markdown(f"### Answer\n{answer}")

        with st.expander("Sources used"):
            for rank, node in enumerate(nodes, start=1):
                source = node.node.metadata.get("file_name", "unknown")
                st.markdown(f"**#{rank} -- {source}** (score={node.score:.3f})")
                st.caption(node.node.get_content()[:300] + "...")
elif st.session_state.index is None:
    st.info("Upload and process PDF documents to get started.")
