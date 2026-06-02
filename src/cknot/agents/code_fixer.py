import logging
from typing import List, Dict, Any
from pydantic import Field
from langchain_core.messages import HumanMessage
from langgraph.graph.state import RunnableConfig
from cknot.schemas.state import CknotAgentState
from cknot.agents.base import CKnotBaseAgent

logger = logging.getLogger(__name__)

CODE_FIXER_PROMPT = (
    "You are a Senior Software Engineer. Provide a detailed code fix or patch based on the identified issues."
)

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

        # Inject the issues into the conversation state for the base ainvoke to pick up
        temp_state = state.copy()
        temp_state["messages"] = list(temp_state.get("messages", [])) + [
            HumanMessage(content=f"Identified Issues:\n{issues}")
        ]

        result = await super().ainvoke(temp_state, config)
        result["fix_result"] = result["messages"][-1].content
        return result
