"""Ingestion + indexing stage, built with LlamaIndex.

Loads every PDF in a directory, splits it into overlapping chunks, embeds
each chunk with a local sentence-transformers model, and stores the result
in a Chroma collection persisted to disk.
"""

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def _vector_store(fresh=False):
    """Open the persisted Chroma collection, optionally wiping it first."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    if fresh:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return ChromaVectorStore(chroma_collection=collection)


def build_index(data_dir=DATA_DIR):
    """Re-ingest every PDF in data_dir into a fresh Chroma collection."""
    documents = SimpleDirectoryReader(data_dir, required_exts=[".pdf"]).load_data()
    storage_context = StorageContext.from_defaults(vector_store=_vector_store(fresh=True))
    return VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )


def load_index():
    """Reopen the already-persisted collection without re-ingesting documents."""
    return VectorStoreIndex.from_vector_store(_vector_store())
