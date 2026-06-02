import operator
from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class CknotAgentState(TypedDict):
    """Mutable working memory of the graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    logs_file_path: Optional[str]
    parsed_issues: Optional[str]
    fix_result: Optional[str]
    agent_summary: Annotated[Dict[str, Dict[str, Any]], operator.ior]

class CKnotConfig(TypedDict):
    """Immutable configuration and context schema."""
    thread_id: str
    user_id: str
    session_id: str
    current_task: str
    agent_llms: Optional[dict[str, str]]
