from typing import Optional, List, Union, Dict, Any
from pydantic import Field
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import RunnableConfig

from cknot.schemas.state import CknotAgentState
from cknot.tools.web_search import web_search
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy
from cknot.agents.registry import AgentRegistry
from cknot.agents.base import CKnotBaseAgent

DEEP_SEARCH_PROMPT = (
    "You are a Deep Research Specialist. Your workflow is:\n"
    "1. Parse and analyze the user's input to identify core research requirements.\n"
    "2. Use the web_search tool to perform comprehensive and deep internet searches.\n"
    "3. Synthesize the findings into a structured, insightful analysis."
)

class DeepSearchAgent(CKnotBaseAgent):
    """Dedicated agent for deep internet research and analysis."""
    system_prompt: str = Field(default=DEEP_SEARCH_PROMPT)
    tools: List[Any] = Field(default_factory=lambda: [web_search])
    good_at: List[str] = Field(default_factory=lambda: ["deep internet research", "web searching", "latest news", "public data"])
    poor_at: List[str] = Field(default_factory=lambda: ["local file access", "private database analysis", "code execution"])

    async def ainvoke(
        self,
        state: CknotAgentState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """
        Makes the class instance usable directly as a LangGraph node.
        Supports both direct LLM injection and service registry lookup.
        """
        return await super().ainvoke(state, config)
