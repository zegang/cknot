import logging
import time
from typing import Optional, List, Union, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
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
    def __init__(
        self, 
        llm_services: Optional[List[LLMService]] = None,
        llm_select_policy: Union[LLMSelectPolicy, str] = LLMSelectPolicy.FIRST
    ):
        super().__init__(
            system_prompt=LOG_PARSER_PROMPT, 
            llm_services=llm_services,
            llm_select_policy=llm_select_policy,
            good_at=["log analysis", "root cause identification", "DevOps debugging", "container logs"],
            poor_at=["writing code", "web search", "user interaction"]
        )

    async def ainvoke(self, state: AgentState, llm: Optional[BaseChatModel] = None, service_id: Optional[str] = None) -> Dict[str, Any]:
        """Asynchronous execution for the log parser."""
        file_path = state.get("logs_file_path")
        if not file_path:
            return {"parsed_issues": "No log file path provided."}
        
        logs = read_log_file.invoke(file_path)
        start_time = time.perf_counter()
        
        active_llm = self._select_llm_service(state, service_id) or llm
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
