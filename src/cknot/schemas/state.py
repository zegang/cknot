import operator
from typing import Annotated, TypedDict, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class ProgressInfo(TypedDict):
    """Standardized progress tracking for the CLI/UI."""
    step: str          # e.g., "RESEARCH", "ANALYSIS"
    description: str   # e.g., "Reading local logs..."
    status: str        # "working", "done", "error", "pending"
    percentage: float  # 0.0 to 100.0
    total_tokens: int

class SummaryReport(TypedDict):
    """Standardized report for the Boss agent to ingest."""
    content: str       # High-level summary of results
    status: str        # "SUCCESS", "FAILURE", "PARTIAL"
    timestamp: str     # ISO format or readable string
    artifacts: Dict[str, Any] # e.g., {"file_path": "output.md"}

class CknotAgentState(TypedDict):
    """Mutable working memory of the graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Shared Data: Structured outputs/artifacts namespaced by Agent Name.
    # Example: state["agent_data"]["LogParserAgent"]["parsed_errors"]
    agent_data: Annotated[Dict[str, Any], operator.ior]

    # Real-time Status: Metadata for the Live Panel.
    # Example: state["progress_report"]["ArticleWriterAgent"] -> ProgressInfo
    progress_report: Annotated[Dict[str, ProgressInfo], operator.ior]

    # Final Summaries: What the Boss reads to talk back to the user.
    # Example: state["agent_summary"]["DeepSearchAgent"] -> SummaryReport
    agent_summary: Annotated[Dict[str, SummaryReport], operator.ior]

class CKnotConfig(TypedDict):
    """Immutable configuration and context schema."""
    thread_id: str
    user_id: str
    session_id: str
    current_task: str
    agent_llms: Optional[dict[str, str]]
