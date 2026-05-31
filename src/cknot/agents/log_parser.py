import logging
import time
from typing import Optional, List, Union, Dict, Any
from pydantic import Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import RunnableConfig
from cknot.schemas.state import AgentState
from cknot.tools.file_ops import read_log_file
from cknot.agents.system_prompts import LOG_PARSER_PROMPT
from cknot.agents.base import CKnotBaseAgent
from cknot.agents.registry import AgentRegistry
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy

logger = logging.getLogger(__name__)

class LogParserAgent(CKnotBaseAgent):
    """Agent responsible for reading log files and identifying issues."""
    system_prompt: str = Field(default=LOG_PARSER_PROMPT)
    good_at: List[str] = Field(default_factory=lambda: ["log analysis", "root cause identification", "DevOps debugging", "container logs"])
    poor_at: List[str] = Field(default_factory=lambda: ["writing code", "web search", "user interaction"])

    async def ainvoke(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Asynchronous execution for the log parser."""
        file_path = state.get("logs_file_path")
        if not file_path:
            return {"parsed_issues": "No log file path provided."}
        
        logs = read_log_file.invoke(file_path)
        start_time = time.perf_counter()
        
        active_llm = self._select_llm_service(state)
        if not active_llm:
            raise ValueError(
                "The Log Parser agent has no enabled LLM services. "
                "Use '/llms' to check status."
            )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Logs from {file_path}:\n\n{logs}")
        ]
        
        response = await active_llm.ainvoke(messages)
        self._log_metrics(start_time, usage=getattr(response, "usage_metadata", None))
        
        response.name = "log_parser"
        return {
            "parsed_issues": response.content,
            "messages": [response]
        }
