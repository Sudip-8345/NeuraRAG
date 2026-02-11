from typing import TypedDict, List, Dict, Any, Annotated, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intents: str
    user_info: Dict[str, Any]
    model_used: str
    context: str
    use_reranking: bool
    prompt_version: str
    sources: List[str]