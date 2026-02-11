"""ChromaDB vector store — build from chunks or load from disk."""

from langchain_chroma import Chroma
from rag.embeddings import get_embedding_function
import config

COLLECTION_NAME = "neura_policies"


def build_vectorstore(chunks):
    """Create and persist a new vector store from document chunks."""
    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        collection_name=COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    print(f"[VectorStore] Built with {len(chunks)} chunks")
    return store


def load_vectorstore():
    """Load existing vector store from disk."""
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    print("[VectorStore] Loaded from disk")
    return store
