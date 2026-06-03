import logging
import time
import json
from typing import Optional, List, Union, Dict, Any
from pydantic import Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import RunnableConfig

from cknot.schemas.state import CknotAgentState
from cknot.tools.file_ops import read_log_file
from cknot.agents.base import CKnotBaseAgent
from cknot.utils.llm_manager import LLMManager
from cknot.agents.registry import AgentRegistry
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy

logger = logging.getLogger(__name__)

LOG_PARSER_PROMPT = (
    "You are a DevOps log expert. Analyze the provided container logs and identify the root cause of any errors."
)

class LogParserAgent(CKnotBaseAgent):
    """Agent responsible for reading log files and identifying issues."""
    system_prompt: str = Field(default=LOG_PARSER_PROMPT)
    expert_in: List[str] = Field(default_factory=lambda: ["log analysis", "root cause identification", "DevOps debugging", "container logs"])
    avoid_for: List[str] = Field(default_factory=lambda: ["writing code", "web search", "user interaction"])

    async def ainvoke(self, state: CknotAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Asynchronous execution for the log parser."""
        # Retrieve from shared data or config
        file_path = state.get("agent_data", {}).get("cknot", {}).get("logs_file_path")
        if not file_path:
            return {"parsed_issues": "No log file path provided."}
        
        logs = read_log_file.invoke(file_path)
        
        # Inject the logs into the conversation state for the base ainvoke to pick up
        temp_state = state.copy()
        temp_state["messages"] = list(temp_state.get("messages", [])) + [
            HumanMessage(content=f"Logs from {file_path}:\n\n{logs}")
        ]

        result = await super().ainvoke(temp_state, config)
        
        content = result["messages"][-1].content
        return {
            **result,
            "agent_data": {
                self.name: {
                    "issues": content,
                    "log_source": file_path
                }
            }
        }
