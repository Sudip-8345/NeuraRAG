"""LangGraph workflow"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from agent.state import AgentState
from agent.nodes import intent_classifier, greet, retrieve, out_of_scope_handler, route_after_classify
from rag.generate import run_llm


def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", intent_classifier)
    graph.add_node("greeter", greet)
    graph.add_node("retriever", retrieve)
    graph.add_node("generate", run_llm)
    graph.add_node("out_of_scope_handler", out_of_scope_handler)

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_after_classify)
    graph.add_edge("retriever", "generate")
    graph.add_edge("greeter", END)
    graph.add_edge("generate", END)
    graph.add_edge("out_of_scope_handler", END)

    return graph.compile()


# Singleton compiled graph
_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(question: str, prompt_version: str = "v2", chat_history: list = None):
    graph = _get_graph()

    messages = list(chat_history) if chat_history else []
    messages.append(HumanMessage(content=question))

    initial_state = {
        "messages": messages,
        "intents": "",
        "prompt_version": prompt_version,
        "context": "",
        "sources": [],
        "model_used": "",
    }

    result = graph.invoke(initial_state)

    last_msg = result["messages"][-1]
    answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    return {
        "answer": answer,
        "sources": result.get("sources", []),
        "context": [result.get("context", "")],
        "model_used": result.get("model_used", "unknown"),
        "prompt_version": prompt_version,
    }
