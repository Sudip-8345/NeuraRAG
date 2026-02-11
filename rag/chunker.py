"""Split documents into smaller chunks for embedding."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

"""
I have used RecursiveCharacterTextSplitter with Markdown-aware separators.
500 char chunks keep ~1 policy section each (good retrieval precision).
50 char overlap avoids losing context at chunk boundaries.
"""

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n---", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[Chunker] Created {len(chunks)} chunks")
    return chunks
