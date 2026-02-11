# NeuraRAG — Policy Q&A Assistant

A Retrieval-Augmented Generation (RAG) system built with LangGraph that answers questions about company policy documents (Refund, Cancellation, Shipping/Delivery, Pricing) using an agentic workflow with intent classification, semantic retrieval, keyword reranking, and LLM-based generation with Groq LLaMA and Google Gemini fallback.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com/keys)
- [Google AI API Key](https://aistudio.google.com/apikey)
- [OpenRouter API Key](https://openrouter.ai/keys) — used as LLM judge for DeepEval evaluation
- [Confident AI API Key](https://app.confident-ai.com/) *(optional)* — for evaluation dashboard

### 2. Setup

```bash
# Clone the repo
git clone https://github.com/your-username/NeuraRAG.git
cd NeuraRAG

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
copy .env.example .env       # Windows
# Then edit .env and add your API keys:
#   GROQ_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY, CONFIDENT_API_KEY
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

### 5. Run Evaluation (DeepEval + OpenRouter)

```bash
# Evaluate with prompt v2 (default)
python evaluation/deepeval_eval.py --prompt v2

# Evaluate with prompt v1
python evaluation/deepeval_eval.py --prompt v1

# Disable reranking
python evaluation/deepeval_eval.py --prompt v2 --no-rerank

# Compare v1 vs v2
python evaluation/deepeval_eval.py --compare
```

---

## Architecture Overview

```
NeuraRAG/
├── config.py                # Centralized settings (API keys, chunk size, model)
├── main.py                  # CLI entry point (build index / interactive Q&A)
├── ranking.py               # Standalone BM25 reranking experiment script
├── .env                     # API keys (not committed)
├── .env.example             # Template for .env
├── requirements.txt         # Python dependencies
│
├── data/                    # Source policy documents (Markdown)
│   ├── cancellation_policy.md
│   ├── pricing.md
│   ├── refund_policy.md
│   └── shipping_policy.md
│
├── agent/                   # LangGraph agentic workflow
│   ├── state.py             # AgentState TypedDict (messages, intents, context, etc.)
│   ├── nodes.py             # Node functions: intent_classifier, retrieve, greet, out_of_scope
│   └── workflow.py          # Graph construction, compilation, and `ask()` entry point
│
├── rag/                     # RAG pipeline modules
│   ├── loader.py            # Load .md files using LangChain DirectoryLoader
│   ├── chunker.py           # Split docs with RecursiveCharacterTextSplitter
│   ├── embeddings.py        # Google Generative AI embeddings
│   ├── vectorstore.py       # ChromaDB build & load
│   ├── retriever.py         # Semantic search + keyword reranking
│   ├── prompts.py           # Prompt templates v1, v2, and greeting
│   └── generate.py          # LLM generation node (Groq primary, Google fallback)
│
├── utils/                   # Shared utilities
│   ├── llms.py              # LLM factory functions (Groq + Google GenAI)
│   └── logger.py            # Query tracing to logs/rag_trace.log
│
├── evaluation/              # Evaluation pipeline
│   ├── questions.py         # 5 test questions (answerable / partial / unanswerable)
│   └── deepeval_eval.py     # DeepEval scoring with OpenRouter LLM judge
├── chroma_db/               # Persisted vector store (auto-generated)
└── logs/                    # Query trace logs (auto-generated)
```

### Agent Workflow (LangGraph)

```
User Query → Intent Classifier (Groq LLM)
                 │
                 ├── GREETING → Greeter Node → Response
                 ├── INQUIRY  → Retriever → Reranker → LLM Generate → Response
                 └── OUT_OF_SCOPE → Decline Handler → Response
```

The agent uses a **conditional routing** pattern:
1. **Intent Classifier** — LLM classifies the query as GREETING, INQUIRY, or OUT_OF_SCOPE
2. **Retriever** — Semantic similarity search (ChromaDB) + keyword reranking
3. **Generator** — Groq LLaMA (primary) with Google Gemini (fallback)
4. **Memory** — Conversation history via deque (last N turns)

### RAG Pipeline Flow

```
Documents (.md) -> Loader -> Chunker (500 chars) -> Embeddings (Google GenAI)
    -> ChromaDB (vector store) -> Semantic Retrieval (top-3)
    -> Keyword Reranking -> Prompt Template -> Groq LLaMA 3.3 70B -> Answer
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
- Uses `gemini-embedding-001` via LangChain's `GoogleGenerativeAIEmbeddings`.

### LLM: Groq LLaMA 3.3 70B Versatile (Primary) + Google Gemini 2.5 Flash (Fallback)
- Fast inference via Groq's hardware.
- Temperature 0.1 for factual, deterministic answers.
- Automatic fallback to Google Gemini if Groq fails.
- Raw context returned as last resort if both models fail.

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
---
<img width="848" height="229" alt="Screenshot 2026-02-11 212034" src="https://github.com/user-attachments/assets/181b63bb-c87b-421a-a2bb-117f5b5a2845" />
---

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
---
<img width="846" height="323" alt="Screenshot 2026-02-11 211854" src="https://github.com/user-attachments/assets/2f1c1fd4-b53e-4cf1-bfa6-1f3e7559397c" />

---
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

### Test Set (5 Questions)

| # | Question | Category |
|---|----------|----------|
| 1 | How long does it take to process a refund? | Answerable |
| 2 | What is the refund policy for annual subs and what discounts are available? | Partially Answerable |
| 3 | Can I get a refund after project starts, and who is my account manager? | Partially Answerable |
| 4 | Does Neura Dynamics offer a free trial period? | Unanswerable |
| 5 | What programming languages and tech stack does Neura Dynamics use internally? | Unanswerable |

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
python evaluation/deepeval_eval.py --prompt v2

# Compare both prompts side by side
python evaluation/deepeval_eval.py --compare
```

Results are saved as JSON in the `evaluation_results/` folder.

### v1 Results
---
<img width="1161" height="616" alt="Screenshot 2026-02-12 004912" src="https://github.com/Sudip-8345/NeuraRAG/blob/e6129c164f2914af11d2387cfc0f7a91e64c6852/evaluation_results/v1/Screenshot%202026-02-12%20004912.png" />

### v2 Results
---
<img width="1362" height="630" alt="Screenshot 2026-02-12 010021" src="https://github.com/Sudip-8345/NeuraRAG/blob/e6129c164f2914af11d2387cfc0f7a91e64c6852/evaluation_results/v2/Screenshot%202026-02-12%20010021.png" />

---

## Bonus Features Implemented

| Feature | Location |
|---------|----------|
| LangGraph agentic workflow with intent routing | `agent/workflow.py`, `agent/nodes.py` |
| Intent classification (GREETING / INQUIRY / OUT_OF_SCOPE) | `agent/nodes.py` |
| Prompt templating with v1, v2, and greeting prompts | `rag/prompts.py` |
| Simple keyword-overlap reranking | `rag/retriever.py` |
| Dual LLM with automatic fallback (Groq → Google) | `utils/llms.py`, `rag/generate.py` |
| Conversation memory (last N turns) | `main.py` |
| DeepEval evaluation with OpenRouter LLM judge | `evaluation/deepeval_eval.py` |
| Prompt v1 vs v2 comparison | `evaluation/deepeval_eval.py --compare` |
| Confident AI dashboard integration | via `CONFIDENT_API_KEY` |
| Basic query tracing / logging | `utils/logger.py` → `logs/rag_trace.log` |

---

## Key Trade-offs & Improvements with More Time

### Trade-offs Made
- **Keyword reranking vs. cross-encoder:** Used simple keyword overlap instead of a neural reranker. Cheaper and faster, but less accurate for semantic similarity.
- **Fixed top-k=3:** Works well for 3 small docs but would need tuning for larger corpora.
- **OpenRouter as eval judge:** Convenient single API for accessing many models, but adds a network dependency and rate limits compared to local evaluation.

### Improvements with More Time
1. **Cross-encoder reranking** (e.g., `ms-marco-MiniLM`) for better retrieval precision.
2. **Hybrid search** — combine semantic + BM25 keyword search for better recall.
3. **Output schema validation** — enforce JSON output with Pydantic models.
4. **Streaming responses** for better UX in the CLI.
5. **Metadata filtering** — allow users to specify which policy to search.
6. **Larger evaluation set** — more questions per category for statistically robust scores.
7. **Deploy** - It is not the task. But I will deploy it further.

---

## License

MIT — see [LICENSE](LICENSE) for details.
