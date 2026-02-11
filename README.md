# NeuraRAG — Policy Q&A Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about company policy documents (Refund, Cancellation, Shipping/Delivery) using semantic retrieval and Groq-hosted LLaMA.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com/keys) (free tier available)
- [Google AI API Key](https://aistudio.google.com/apikey) (for embeddings)

### 2. Setup

```bash
# Clone the repo
git clone https://github.com/your-username/NeuraRAG.git
cd NeuraRAG

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API keys
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux
# Then edit .env and add your GROQ_API_KEY and GOOGLE_API_KEY
```

### 3. Build the Vector Store

```bash
python main.py --build
```

### 4. Ask Questions

```bash
# Interactive Q&A (default: prompt v2 with reranking)
python main.py

# Use prompt v1 for comparison
python main.py --prompt v1

# Disable reranking
python main.py --no-rerank
```

### 5. Run Evaluation

```bash
# Evaluate with prompt v2 (default)
python -m evaluation.evaluate

# Evaluate with prompt v1
python -m evaluation.evaluate --prompt v1

# Compare v1 vs v2
python -m evaluation.evaluate --compare
```

---

## Architecture Overview

```
NeuraRAG/
├── config.py                # Centralized settings (API keys, chunk size, model)
├── main.py                  # CLI entry point (build index / interactive Q&A)
├── .env                     # API keys (not committed)
├── .env.example             # Template for .env
├── requirements.txt         # Python dependencies
│
├── data/                    # Source policy documents (Markdown)
│   ├── cancellation_policy.md
│   ├── refund_policy.md
│   └── shipping_policy.md
│
├── rag/                     # RAG pipeline modules
│   ├── loader.py            # Load .md files using LangChain DirectoryLoader
│   ├── chunker.py           # Split docs with RecursiveCharacterTextSplitter
│   ├── embeddings.py        # Google Generative AI embeddings
│   ├── vectorstore.py       # ChromaDB build & load
│   ├── retriever.py         # Semantic search + keyword reranking
│   ├── prompts.py           # Prompt templates v1 & v2 with iteration notes
│   ├── chain.py             # Full RAG chain: retrieve → rerank → prompt → LLM
│   └── logger.py            # Basic query tracing to logs/rag_trace.log
│
├── evaluation/              # Evaluation pipeline
│   ├── questions.py         # 8 test questions (answerable / partial / unanswerable)
│   └── evaluate.py          # Automated scoring with keyword + hallucination checks
│
├── evaluation_results/      # JSON outputs from evaluation runs
├── chroma_db/               # Persisted vector store (auto-generated)
└── logs/                    # Query trace logs (auto-generated)
```

### Pipeline Flow

```
Documents (.md) → Loader → Chunker (500 chars) → Embeddings (Google GenAI)
    → ChromaDB (vector store) → Semantic Retrieval (top-3)
    → Keyword Reranking → Prompt Template → Groq LLaMA 3.1 → Answer
```

---

## Design Decisions

### Chunking Strategy (500 chars, 50 overlap)
- Policy docs are short (~60 lines each) with clear sections (headings, bullets).
- 500 chars keeps each chunk ≈ 1 policy section, preserving semantic coherence.
- 50 char overlap prevents cutting mid-sentence at chunk boundaries.
- Custom separators (`## `, `### `, `---`, `\n\n`) respect Markdown structure.

### Retrieval: Top-3 + Keyword Reranking
- Top-3 balances relevance vs. noise for these short docs.
- Keyword-overlap reranking re-scores chunks by query-word frequency — a simple but effective way to boost chunks that directly mention the user's topic.

### Embedding: Google Generative AI
- High-quality embeddings at no cost (free tier).
- Works well with LangChain's `GoogleGenerativeAIEmbeddings`.

### LLM: Groq LLaMA 3.1 8B Instant
- Fast inference via Groq's hardware.
- Temperature 0.1 for factual, deterministic answers.
- 8B parameter model is sufficient for structured policy Q&A.

---

## Prompts

### Prompt V1 — Initial Version

```
You are a helpful assistant for Neura Dynamics company policies.

Use the following context to answer the question. If the answer is not
in the context, say "I don't have enough information to answer this."

Context: {context}
Question: {question}
Answer:
```

**Issues observed:** No citations, occasional hallucination, unstructured output, weak handling of unanswerable questions.

### Prompt V2 — Improved Version

```
You are a precise policy assistant for Neura Dynamics. Your job is to
answer questions ONLY using the provided context from company policy documents.

RULES:
1. ONLY use information explicitly stated in the context below.
2. Do NOT add any information, assumptions, or details beyond the context.
3. If the context does not contain the answer, respond with:
   "This information is not available in the provided policy documents."
4. If only part of the question can be answered, answer what you can and
   clearly state which part cannot be answered from the available context.
5. Cite the source document for each piece of information using [Source: filename].
6. Use bullet points for multi-part answers.

CONTEXT: {context}
QUESTION: {question}

Respond in this format:
**Answer:** <your answer here, with [Source: filename] citations>
**Sources:** <list the source document(s) used>
**Confidence:** <High / Medium / Low>
```

**What changed and why:**

| Change | Why |
|--------|-----|
| Added numbered RULES section | Explicit constraints reduce hallucination more effectively than vague instructions |
| Required `[Source: filename]` citations | Enables users to verify answers against source documents |
| Structured output format (Answer/Sources/Confidence) | Consistent, parseable output; confidence helps users gauge reliability |
| Separate handling for unanswerable + partially answerable | V1 only handled "don't know"; V2 distinguishes partial answers |
| Bullet points instruction | Improves readability for multi-part answers |

---

## Evaluation

### Test Set (8 Questions)

| # | Question | Category |
|---|----------|----------|
| 1 | How can I cancel my monthly subscription? | Answerable |
| 2 | How long does it take to process a refund? | Answerable |
| 3 | What happens if I cancel a workshop within 24 hours? | Answerable |
| 4 | How are project deliverables shared with clients? | Answerable |
| 5 | What is the refund policy for annual subs and what discounts are available? | Partially Answerable |
| 6 | Can I get a refund after project starts, and who is my account manager? | Partially Answerable |
| 7 | What are the pricing tiers for the AI platform? | Unanswerable |
| 8 | Does Neura Dynamics offer a free trial period? | Unanswerable |

### Scoring Rubric

| Symbol | Meaning |
|--------|---------|
| ✅ PASS | Accurate, grounded, no hallucination |
| ⚠️ PARTIAL | Partially correct or missing nuance |
| ❌ FAIL | Wrong, hallucinated, or missed entirely |

### Evaluation Criteria

- **Accuracy:** Does the answer match expected info from the docs?
- **Hallucination Avoidance:** Does the answer invent info not in context?
- **Answer Clarity:** Is the answer well-structured and easy to understand?
- **Grounding:** For unanswerable Qs, does it correctly decline?

### Running the Evaluation

```bash
# Single prompt evaluation
python -m evaluation.evaluate --prompt v2

# Compare both prompts side by side
python -m evaluation.evaluate --compare
```

Results are saved as JSON in the `evaluation_results/` folder.

---

## Bonus Features Implemented

| Feature | Location |
|---------|----------|
| Prompt templating with LangChain `PromptTemplate` | `rag/prompts.py` |
| Simple keyword-overlap reranking | `rag/retriever.py` |
| Comparison between prompt v1 and v2 | `evaluation/evaluate.py --compare` |
| Basic query tracing / logging | `rag/logger.py` → `logs/rag_trace.log` |

---

## Key Trade-offs & Improvements with More Time

### Trade-offs Made
- **Keyword reranking vs. cross-encoder:** Used simple keyword overlap instead of a neural reranker. Cheaper and faster, but less accurate for semantic similarity.
- **Fixed top-k=3:** Works well for 3 small docs but would need tuning for larger corpora.
- **Rule-based evaluation vs. LLM-as-judge:** Manual keyword + heuristic scoring is transparent but doesn't capture semantic correctness fully.

### Improvements with More Time
1. **Cross-encoder reranking** (e.g., `ms-marco-MiniLM`) for better retrieval precision.
2. **LLM-as-judge evaluation** — use a second LLM to score answer quality.
3. **Hybrid search** — combine semantic + BM25 keyword search for better recall.
4. **Output schema validation** — enforce JSON output with Pydantic models.
5. **Streaming responses** for better UX in the CLI.
6. **Metadata filtering** — allow users to specify which policy to search.
7. **Conversation memory** — maintain context across multi-turn Q&A.
8. **More evaluation metrics** — ROUGE, BERTScore, faithfulness scores.

---

## License

MIT — see [LICENSE](LICENSE) for details.
