"""Retriever — semantic search + simple keyword reranking."""

import config


def retrieve_chunks(vectorstore, query):
    """Get top-k chunks from the vector store by similarity."""
    results = vectorstore.similarity_search_with_score(query, k=config.TOP_K)
    print(f"[Retriever] Found {len(results)} chunks")
    return results


def rerank_chunks(query, results):
    """Re-score chunks by keyword overlap with the query (simple reranking).

    Chunks with more query-word matches rank higher.
    Ties fall back to original similarity order.
    """
    query_words = set(query.lower().split())

    scored = []
    for i, (doc, sim_score) in enumerate(results):
        chunk_words = set(doc.page_content.lower().split())
        overlap = len(query_words & chunk_words)
        # Higher overlap = better, lower distance = better
        scored.append((doc, sim_score, overlap - sim_score, i))

    scored.sort(key=lambda x: (-x[2], x[3]))
    return [(doc, sim) for doc, sim, _, _ in scored]


def get_source_name(doc):
    """Extract just the filename from a document's source metadata."""
    source = doc.metadata.get("source", "unknown")
    return source.replace("\\", "/").split("/")[-1]


def format_context(results):
    """Format retrieved chunks into a numbered context string for the LLM."""
    parts = []
    for i, (doc, score) in enumerate(results, 1):
        name = get_source_name(doc)
        parts.append(f"[Source {i}: {name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)
