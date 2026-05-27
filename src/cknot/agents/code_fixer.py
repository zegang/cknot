import logging
import time
from typing import Optional, List, Union, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import RunnableConfig
from cknot.schemas.state import AgentState
from cknot.agents.system_prompts import CODE_FIXER_PROMPT
from cknot.agents.base import CKnotBaseAgent
from cknot.agents.registry import AgentRegistry
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy

logger = logging.getLogger(__name__)

class CodeFixerAgent(CKnotBaseAgent):
    """Agent responsible for proposing code fixes based on identified issues."""
    def __init__(
        self, 
        llm_services: Optional[List[LLMService]] = None,
        llm_select_policy: Union[LLMSelectPolicy, str] = LLMSelectPolicy.FIRST
    ):
        super().__init__(
            system_prompt=CODE_FIXER_PROMPT, 
            llm_services=llm_services,
            llm_select_policy=llm_select_policy,
            good_at=["code fixing", "patching", "software engineering", "refactoring"],
            poor_at=["log parsing", "web search", "internet research"]
        )

    async def ainvoke(self, state: AgentState, llm: Optional[BaseChatModel] = None, service_id: Optional[str] = None) -> Dict[str, Any]:
        """Asynchronous execution for the code fixer."""
        issues = state.get("parsed_issues")
        if not issues or "Error" in issues:
            return {"fix_result": "Cannot fix: No valid issues identified."}

        start_time = time.perf_counter()
        active_llm = self._select_llm_service(state, service_id) or llm
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Identified Issues:\n{issues}")
        ]
        
        response = await active_llm.ainvoke(messages)
        self._log_metrics(start_time, usage=getattr(response, "usage_metadata", None))
        
        response.name = "code_fixer"
        return {
            "fix_result": response.content,
            "messages": [response]
        }
