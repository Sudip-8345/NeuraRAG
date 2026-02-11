"""RAG chain —> retrieve -> rerank -> prompt -> LLM -> answer.
"""

from agent.state import AgentState
from utils.llms import get_groq_llm, get_google_llm
from langchain_core.messages import SystemMessage
from rag.prompts import get_prompt
from utils.logger import log, log_query
import config


def run_llm(state: AgentState):
    
    question = state["messages"][-1].content if state["messages"] else ""
    log("QUERY", question)
    
    prompt = state["prompt_version"]
    messages = [SystemMessage(
        content=get_prompt(prompt).format(
            last_user_message=question, 
            context=state.get("context", "")))] + state["messages"]

    # Primary model (Groq)
    try:
        response = get_groq_llm().invoke(messages)
        log_query(question, state.get("context", ""), response.content, config.GROQ_MODEL)
        return {'messages': response.content, 'model_used': config.GROQ_MODEL}
    except Exception as e:
        log("LLM_ERROR", f"Groq model failed: {e}")
        print(f"[LLM] Primary model failed ({e}), trying fallback by Gemini...")

    # Fallback model (Google)
    try:
        response = get_google_llm().invoke(messages)
        log_query(question, state.get("context", ""), response.content, config.GOOGLE_MODEL)
        return {'messages': response.content, 'model_used': config.GOOGLE_MODEL}
    except Exception as e:
        log("LLM_ERROR", f"Google model failed: {e}")
        print(f"[LLM] Fallback model also failed ({e})")

    # Last resort: Return raw context
    fallback_answer = (
        "LLM is currently unavailable. Here is the raw context retrieved:\n\n"
        f"{state.get('context', '')}\n\n"
        "Please review the above context to find your answer."
    )
    return {'messages': fallback_answer, 'model_used': "none (raw context)"}