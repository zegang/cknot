import logging
import time
import uuid
from typing import List, Optional, Union, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph.state import RunnableConfig
from cknot.schemas.state import AgentState
from cknot.agents.system_prompts import CKNOT_BOSS_PROMPT
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy
from cknot.agents.registry import AgentRegistry
from cknot.agents.base import CKnotBaseAgent

logger = logging.getLogger(__name__)

class CKnotBossAgent(CKnotBaseAgent):
    """
    The central orchestrator (Boss Agent) that routes tasks.
    Utilizes astream for real-time output capabilities.
    """
    def __init__(
        self, 
        tools: Optional[List] = None,
        llm_services: Optional[List[LLMService]] = None,
        llm_select_policy: Union[LLMSelectPolicy, str] = LLMSelectPolicy.FIRST,
        sub_agents: Optional[List[CKnotBaseAgent]] = None
    ):
        super().__init__(
            system_prompt=CKNOT_BOSS_PROMPT, 
            tools=tools, 
            llm_services=llm_services,
            llm_select_policy=llm_select_policy
        )
        self.sub_agents = sub_agents or []

    def _get_messages(self, state: AgentState) -> List[BaseMessage]:
        """Overrides base to inject specialist capabilities into the Boss prompt."""
        team_manifest = "\n\nTEAM DIRECTORY & DELEGATION PROTOCOL:\n"
        for agent in self.sub_agents:
            name = agent.__class__.__name__
            good = ", ".join(agent.good_at) if agent.good_at else "General tasks"
            poor = ", ".join(agent.poor_at) if agent.poor_at else "None specified"
            
            # Ensure triggers match the hardcoded logic in orchestrator.py should_continue
            if "DeepSearch" in name:
                trigger = "TRIGGER_DEEP_SEARCH"
            elif "LogParser" in name:
                trigger = "TRIGGER_LOG_ANALYSIS"
            else:
                trigger = f"TRIGGER_{name.upper()}"
            
            team_manifest += f"- {name}: Expert in [{good}]. Avoid for [{poor}].\n"
            team_manifest += f"  TO DELEGATE: You must include the keyword '{trigger}' in your response.\n"
            
        enhanced_prompt = f"{self.system_prompt}{team_manifest}"
        
        current_messages = state.get("messages", []) if isinstance(state, dict) else state.messages
        return [SystemMessage(content=enhanced_prompt)] + current_messages

    async def ainvoke(
        self,
        state: AgentState,
        llm: Optional[BaseChatModel] = None,
        service_id: Optional[str] = None
    )-> Dict[str, Any]:
        """
        Asynchronous invocation for non-streaming responses.
        """
        return await super().ainvoke(state, llm=llm, service_id=service_id)

    async def astream(
        self, 
        state: AgentState,
        config: RunnableConfig,
        llm: Optional[BaseChatModel] = None, 
        service_id: Optional[str] = None
    ):
        """
        Streams agent responses as chunks.
        Uses the service_id to retrieve an LLM from the registry, falling back to the provided llm instance.
        """
        start_time = time.perf_counter()

        active_llm = self._select_llm_service(state, service_id) or llm
        if not active_llm:
            raise ValueError(f"No LLM provided or service_id '{service_id}' not found in registry.")

        llm_with_tools = active_llm.bind_tools(self.tools) if self.tools else active_llm
        messages = self._get_messages(state)
        final_chunk = None

        # Generate a stable ID for this turn to ensure chunks merge correctly in LangGraph
        turn_id = str(uuid.uuid4())

        async for chunk in llm_with_tools.astream(messages):
            chunk.id = turn_id
            # Accumulate chunks to compute final usage metadata
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk

            # Only yield chunks with content or tool calls to the graph state.
            # Trailing empty metadata chunks can cause routing logic to fail if they 
            # are treated as the 'last message' in a non-merging scenario.
            if chunk.content or chunk.tool_calls:
                yield {"messages": [chunk]}

        # Extract usage from the aggregated message if available
        usage = getattr(final_chunk, "usage_metadata", None)
        
        # Log summarized interaction for better clarity
        messages[-1].content if messages else "None"
        
        self._log_metrics(start_time, usage=usage, mode="streaming invocation")
