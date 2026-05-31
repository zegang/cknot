from typing import Optional, List, Union, Dict, Any
from pydantic import Field
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import RunnableConfig
from cknot.schemas.state import AgentState
from cknot.tools.web_search import web_search
from cknot.agents.system_prompts import DEEP_SEARCH_PROMPT
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy
from cknot.agents.registry import AgentRegistry
from cknot.agents.base import CKnotBaseAgent

class DeepSearchAgent(CKnotBaseAgent):
    """Dedicated agent for deep internet research and analysis."""
    system_prompt: str = Field(default=DEEP_SEARCH_PROMPT)
    tools: List[Any] = Field(default_factory=lambda: [web_search])
    good_at: List[str] = Field(default_factory=lambda: ["deep internet research", "web searching", "latest news", "public data"])
    poor_at: List[str] = Field(default_factory=lambda: ["local file access", "private database analysis", "code execution"])

    async def ainvoke(
        self,
        state: AgentState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """
        Makes the class instance usable directly as a LangGraph node.
        Supports both direct LLM injection and service registry lookup.
        """
        return await super().ainvoke(state, config)
