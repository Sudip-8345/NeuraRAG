from rank_bm25 import BM25Okapi

# Query
query = "python list comprehension memory optimization"

# Retrieved docs from vector DB
docs = [
    "Python memory management techniques",
    "List comprehension vs loops in Python",
    "Optimizing database queries",
    "Python list comprehension memory usage benchmark"
]

# Tokenize
tokenized_docs = [doc.lower().split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)

tokenized_query = query.lower().split()

# Keyword scores
scores = bm25.get_scores(tokenized_query)

# Rerank
ranked_docs = sorted(
    zip(docs, scores),
    key=lambda x: x[1],
    reverse=True
)

for doc, score in ranked_docs:
    print(score, doc)
