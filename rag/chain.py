"""RAG chain — retrieve → rerank → prompt → LLM → answer.

Includes try/except with LLM fallback:
  - Primary: Groq LLaMA 3.1 8B Instant
  - Fallback: Groq LLaMA 3.1 70B Versatile (if primary fails)
  - Last resort: Return raw context with error message
"""

from langchain_groq import ChatGroq
from langchain.chains import LLMChain

from rag.retriever import retrieve_chunks, rerank_chunks, format_context, get_source_name
from rag.prompts import get_prompt
from rag.logger import log, log_query
import config

# Fallback model if the primary one fails
FALLBACK_MODEL = "llama-3.3-70b-versatile"


def get_llm(model=None):
    """Create a Groq LLM instance."""
    return ChatGroq(
        model=model or config.LLM_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE,
    )


def call_llm_with_fallback(prompt, context, question):
    """Try primary LLM, fall back to secondary, then return raw context on failure.

    Returns:
        tuple: (answer_text, model_used)
    """
    # Try 1: Primary model
    try:
        llm = get_llm(config.LLM_MODEL)
        chain = LLMChain(llm=llm, prompt=prompt)
        answer = chain.run(context=context, question=question)
        return answer, config.LLM_MODEL

    except Exception as e:
        log("LLM_ERROR", f"Primary model failed: {e}")
        print(f"[LLM] Primary model failed ({e}), trying fallback...")

    # Try 2: Fallback model
    try:
        llm = get_llm(FALLBACK_MODEL)
        chain = LLMChain(llm=llm, prompt=prompt)
        answer = chain.run(context=context, question=question)
        return answer, FALLBACK_MODEL

    except Exception as e:
        log("LLM_ERROR", f"Fallback model also failed: {e}")
        print(f"[LLM] Fallback model also failed ({e})")

    # Try 3: Return raw context as last resort
    fallback_answer = (
        "⚠️ LLM is currently unavailable. Here is the raw context retrieved:\n\n"
        f"{context}\n\n"
        "Please review the above context to find your answer."
    )
    return fallback_answer, "none (raw context)"


def ask(vectorstore, question, prompt_version=None, use_reranking=True):
    """Full RAG pipeline: retrieve → rerank → prompt → LLM → answer.

    Returns:
        dict with: answer, context, sources, prompt_version, model_used
    """
    log("QUERY", question)

    # Step 1: Retrieve
    results = retrieve_chunks(vectorstore, question)

    # Step 2: Rerank (optional)
    if use_reranking:
        results = rerank_chunks(question, results)

    # Step 3: Format context
    context = format_context(results)

    # Step 4: Get prompt
    prompt = get_prompt(prompt_version)
    version_used = prompt_version or "v2"

    # Step 5: Call LLM with fallback
    answer, model_used = call_llm_with_fallback(prompt, context, question)
    log("LLM", f"Answered using {model_used}")

    # Step 6: Log full trace
    log_query(question, context, answer, version_used)

    # Step 7: Collect unique source filenames
    sources = list(dict.fromkeys(get_source_name(doc) for doc, _ in results))

    return {
        "answer": answer,
        "context": context,
        "sources": sources,
        "prompt_version": version_used,
        "model_used": model_used,
    }
