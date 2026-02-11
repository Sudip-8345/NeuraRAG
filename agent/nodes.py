"""Retrieval tool — looks up relevant docs from the vectorstore."""

from langchain_core.messages import SystemMessage
from rag.prompts import get_prompt
from rag.retriever import get_retriever, rerank_chunks, get_source_name
from utils.llms import get_groq_llm, get_google_llm
from agent.state import AgentState
from utils.logger import log
import config

_vectorstore = None
_use_reranking = True


def init_tool(vectorstore, use_reranking=True):
    global _vectorstore, _use_reranking
    _vectorstore = vectorstore
    _use_reranking = use_reranking


def retrieve(state: AgentState) -> dict:

    if _vectorstore is None:
        raise RuntimeError("Vectorstore not initialized. Call init_tool() first.")

    query = state["messages"][-1].content if state["messages"] else ""
    results = get_retriever(_vectorstore).invoke(query)
    if _use_reranking:
        # Convert to (doc, score) tuples for reranking, then extract back
        doc_tuples = [(doc, i) for i, doc in enumerate(results)]
        results_reranked = rerank_chunks(query, doc_tuples)
        context = " ".join([doc.page_content for doc, _ in results_reranked])
        sources = list(dict.fromkeys(get_source_name(doc) for doc, _ in results_reranked))
    else:
        context = " ".join([doc.page_content for doc in results])
        sources = list(dict.fromkeys(get_source_name(doc) for doc in results))

    return {"context": context, "sources": sources}


# ===============Intent Classifier Tool ==================
def intent_classifier(state: AgentState) -> dict:
    system_prompt = '''
    You are an intent classifier for queries to Neura Dynamics. Classify the user's intent into one of the following categories:
    - GREETING: Casual hello or hi messages or bye.
    - INQUIRY: Questions about products, pricing, features, or policies.
    - OUT_OF_SCOPE: Messages that are irrelevant to Neura Dynamics or cannot be answered.
    
    Respond with ONLY one word: GREETING, INQUIRY, or OUT_OF_SCOPE. No other text.
    '''
    message = [SystemMessage(content=system_prompt)] + state['messages']
    try:    
        response = get_groq_llm().invoke(message)
    except Exception as e:
        print(f"[LLM] Intent classifier model failed ({e}), using fallback response.")
        response = get_google_llm().invoke(message)
    
    # Extract intent keyword from response
    intent_text = response.content.strip().upper()
    if "OUT_OF_SCOPE" in intent_text:
        intent = "OUT_OF_SCOPE"
    elif "INQUIRY" in intent_text:
        intent = "INQUIRY"
    else:
        intent = "GREETING"
    
    return {'intents': intent}

def route_after_classify(state: AgentState) -> str:
    intent = state["intents"].upper()
    if intent == "OUT_OF_SCOPE":
        return "out_of_scope_handler"
    elif intent == "INQUIRY":
        return "retriever"
    else:
        return "greeter"
    
    
def greet(state):
    question = state["messages"][-1].content if state["messages"] else ""
    log("GREET", question)

    prompt = get_prompt("greet")
    messages = [SystemMessage(
        content=prompt.format(
            last_user_message=question))] + state["messages"] 

    try:
        response = get_groq_llm().invoke(messages)
        answer = response.content
        model = config.GROQ_MODEL
    except Exception as e:
        log("LLM_ERROR", f"Greeting model failed: {e}")
        print(f"[LLM] Greeting model failed ({e}), using fallback response.")
        response = get_google_llm().invoke(messages)
        answer = response.content
        model = config.GOOGLE_MODEL
        
    log("GREET_ANSWER", f"Answered using {model}")

    return {
        "messages": answer,
        "model_used": model,
    }

def out_of_scope_handler(state):
    question = state["messages"][-1].content if state["messages"] else ""
    log("OUT_OF_SCOPE", question)
    return {
        "messages": "I'm sorry, but I can't assist with that request. Please ask about Neura Dynamics products, pricing, features, or policies.",
        "model_used": "no model used (out of scope)",
    }
    
