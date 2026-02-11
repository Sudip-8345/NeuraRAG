"""Retriever — semantic search + simple keyword reranking."""

import config


def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": config.TOP_K})
    return retriever


def rerank_chunks(query, results):
    """Re-score chunks by keyword overlap with the query.
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
    source = doc.metadata.get("source", "unknown")
    return source.replace("\\", "/").split("/")[-1]

