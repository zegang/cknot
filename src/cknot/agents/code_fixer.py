import logging
import time
from typing import List, Dict, Any
from pydantic import Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import RunnableConfig
from cknot.schemas.state import AgentState
from cknot.agents.system_prompts import CODE_FIXER_PROMPT
from cknot.agents.base import CKnotBaseAgent

logger = logging.getLogger(__name__)

class CodeFixerAgent(CKnotBaseAgent):
    """Agent responsible for proposing code fixes based on identified issues."""
    system_prompt: str = Field(default=CODE_FIXER_PROMPT)
    good_at: List[str] = Field(default_factory=lambda: ["code fixing", "patching", "software engineering", "refactoring"])
    poor_at: List[str] = Field(default_factory=lambda: ["log parsing", "web search", "internet research"])

    async def ainvoke(self, state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
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
        
        response = await active_llm.ainvoke(messages)
        self._log_metrics(start_time, usage=getattr(response, "usage_metadata", None))
        
        response.name = "code_fixer"
        return {
            "fix_result": response.content,
            "messages": [response]
        }
