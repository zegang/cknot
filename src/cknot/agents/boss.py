import logging
import time
import uuid
from typing import List, Optional, Union, Dict, Any
from pydantic import Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph.state import RunnableConfig
from cknot.schemas.state import AgentState
from cknot.agents.system_prompts import CKNOT_BOSS_PROMPT
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy
from cknot.agents.registry import AgentRegistry
from cknot.agents.base import CKnotBaseAgent
from cknot.utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class CKnotBossAgent(CKnotBaseAgent):
    """
    The central orchestrator (Boss Agent) that routes tasks.
    Utilizes astream for real-time output capabilities.
    """
    system_prompt: str = Field(default=CKNOT_BOSS_PROMPT)
    sub_agents: List[CKnotBaseAgent] = Field(default_factory=list)

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
        config: RunnableConfig
    )-> Dict[str, Any]:
        """
        Asynchronous invocation for non-streaming responses.
        """
        return await super().ainvoke(state, config)

    async def astream(
        self, 
        state: AgentState,
        config: RunnableConfig
    ):
        """
        Streams agent responses as chunks.
        Uses the service_id to retrieve an LLM from the registry, falling back to the provided llm instance.
        """
        start_time = time.perf_counter()

        active_llm = self._select_llm_service(state)
        if not active_llm:
            raise ValueError(
                f"The Boss Orchestrator has no enabled LLM services. "
                "Please use '/llms' to enable a service or check your configuration."
            )

        llm_manager = LLMManager()
        llm_svc_client = llm_manager.get_llm_service_client(active_llm.id)

        llm_svc_clinet_with_tools = llm_svc_client.bind_tools(self.tools) if self.tools else llm_svc_client
        messages = self._get_messages(state)
        final_chunk = None

        # Generate a stable ID for this turn to ensure chunks merge correctly in LangGraph
        turn_id = str(uuid.uuid4())

        async for chunk in llm_svc_clinet_with_tools.astream(messages):
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
