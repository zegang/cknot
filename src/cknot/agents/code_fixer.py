import logging
import time
import json
from typing import List, Dict, Any
from pydantic import Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import RunnableConfig
from cknot.schemas.state import CknotAgentState
from cknot.agents.system_prompts import CODE_FIXER_PROMPT
from cknot.agents.base import CKnotBaseAgent
from cknot.utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class CodeFixerAgent(CKnotBaseAgent):
    """Agent responsible for proposing code fixes based on identified issues."""
    system_prompt: str = Field(default=CODE_FIXER_PROMPT)
    good_at: List[str] = Field(default_factory=lambda: ["code fixing", "patching", "software engineering", "refactoring"])
    poor_at: List[str] = Field(default_factory=lambda: ["log parsing", "web search", "internet research"])

    async def ainvoke(self, state: CknotAgentState, config: RunnableConfig) -> Dict[str, Any]:
        """Asynchronous execution for the code fixer."""
        issues = state.get("parsed_issues")
        if not issues or "Error" in issues:
            return {"fix_result": "Cannot fix: No valid issues identified."}

        start_time = time.perf_counter()
        active_llm = self._select_llm_service(state)
        if not active_llm:
            raise ValueError(
                "The Code Fixer agent has no enabled LLM services. "
                "Use '/llms' to check status."
            )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Identified Issues:\n{issues}")
        ]
        
        llm_client = LLMManager().get_llm_service_client(active_llm.id)
        response = await llm_client.ainvoke(messages)

        # Prepare JSON-style representations for logging
        req_json = json.dumps([m.dict() for m in messages], indent=2, ensure_ascii=False, default=str)
        res_json = json.dumps(response.dict(), indent=2, ensure_ascii=False, default=str)
        log_msg = f"LLM: {active_llm.model_name}\nRequest: {req_json}\nResponse: {res_json}"

        self._log_metrics(start_time, usage=getattr(response, "usage_metadata", None), message=log_msg)
        
        response.name = "code_fixer"
        return {
            "fix_result": response.content,
            "messages": [response]
        }
