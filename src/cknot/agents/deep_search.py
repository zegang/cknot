from typing import Optional, List, Union, Dict, Any
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import RunnableConfig
from cknot.schemas.state import AgentState
from cknot.tools.search import web_search
from cknot.agents.system_prompts import DEEP_SEARCH_PROMPT
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy
from cknot.agents.registry import AgentRegistry
from cknot.agents.base import CKnotBaseAgent

class DeepSearchAgent(CKnotBaseAgent):
    """Dedicated agent for deep internet research and analysis."""
    def __init__(
        self, 
        llm_services: Optional[List[LLMService]] = None,
        llm_select_policy: Union[LLMSelectPolicy, str] = LLMSelectPolicy.FIRST
    ):
        super().__init__(
            system_prompt=DEEP_SEARCH_PROMPT, 
            tools=[web_search],
            llm_services=llm_services,
            llm_select_policy=llm_select_policy,
            good_at=["deep internet research", "web searching", "latest news", "public data"],
            poor_at=["local file access", "private database analysis", "code execution"]
        )

    async def ainvoke(
        self,
        state: AgentState,
        llm: Optional[BaseChatModel] = None,
        service_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Makes the class instance usable directly as a LangGraph node.
        Supports both direct LLM injection and service registry lookup.
        """
        return await super().ainvoke(state, llm=llm, service_id=service_id)
