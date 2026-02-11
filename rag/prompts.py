"""
Prompt templates v1 and v2 for the RAG pipeline.

--- ITERATION LOG ---

V1 (Initial): Simple instruction. Issues: no citations, occasional hallucination,
    no structured output, weak on unanswerable questions.

V2 (Improved): Added explicit RULES, required citations, structured output format,
    separate handling for unanswerable/partial questions. Result: more grounded,
    consistent, traceable answers.
"""

from langchain.prompts import PromptTemplate

# --- PROMPT V1: Simple & direct, but lacks guardrails ---
PROMPT_V1 = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant for Neura Dynamics company policies.

Use the following context to answer the question. If the answer is not
in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""
)

# --- PROMPT V2: Rules-based, with citations and structured output ---
# Changes: RULES block, [Source: filename] citations, Answer/Sources/Confidence format,
#          partial-answer handling, bullet points for clarity.
PROMPT_V2 = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise policy assistant for Neura Dynamics. Your job is to
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

CONTEXT:
{context}

QUESTION: {question}

Respond in this format:
**Answer:**
<your answer here, with [Source: filename] citations>

**Sources:** <list the source document(s) used>

**Confidence:** <High / Medium / Low — based on how well the context covers the question>"""
)

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def get_prompt(version=None):
    """Return the prompt template for the given version (default: v2)."""
    version = version or "v2"
    if version not in PROMPTS:
        raise ValueError(f"Unknown prompt version: {version}. Use: {list(PROMPTS.keys())}")
    return PROMPTS[version]
