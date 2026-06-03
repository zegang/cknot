import logging
import time
import json
import uuid
from typing import List, Optional, Union, Dict, Any
from pydantic import Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, HumanMessage
from langgraph.graph.state import RunnableConfig
from pydantic import Field
from cknot.schemas.state import CknotAgentState
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy
from cknot.agents.registry import AgentRegistry
from cknot.agents.base import CKnotBaseAgent
from cknot.utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

CKNOT_BOSS_PROMPT = (
    "You are 'cknot', the central orchestrator. Your primary responsibility is task triage and delegation.\n\n"
    "1. ANALYZE: Review the user's input and compare it against the 'Expert in' capabilities in your Team Directory.\n"
    "2. DELEGATE: If a specialist is better suited for the task, delegate immediately by including the phrase 'Agent [AgentName]'. "
        "Briefly explain that you are calling a specialist, then include the phrase.\n"
    "3. DIRECT ACTION: If no specialist matches, you can solve it with your own tools/knowledge while noting no specialist was called.\n\n"
    "Maintain a professional, authoritative, and efficient persona."
)

class CKnotBossAgent(CKnotBaseAgent):
    """
    The central orchestrator (Boss Agent) that routes tasks.
    Utilizes astream for real-time output capabilities.
    """
    system_prompt: str = Field(default=CKNOT_BOSS_PROMPT)
    sub_agents: List[CKnotBaseAgent] = Field(default_factory=list)

    def _get_messages(self, state: CknotAgentState) -> List[BaseMessage]:
        """Overrides base to inject specialist capabilities into the Boss prompt."""
        team_manifest = "\n\nTEAM DIRECTORY & DELEGATION PROTOCOL:\n"
        for agent in self.sub_agents:
            name = agent.__class__.__name__
            good = ", ".join(agent.expert_in) if agent.expert_in else "General tasks"
            poor = ", ".join(agent.avoid_for) if agent.avoid_for else "None specified"
            
            trigger = f"Agent {name}"
            team_manifest += f"- {name}: Expert in [{good}]. Avoid for [{poor}].\n"
            team_manifest += f"  TO DELEGATE: You must include the phrase '{trigger}' in your response.\n"

        # Inject summaries from specialists if any exist in the state
        agent_summaries = state.get("agent_summary", {}) if isinstance(state, dict) else getattr(state, "agent_summary", {})
        if agent_summaries:
            team_manifest += "\n\nREPORTS FROM SPECIALISTS (Use these to answer the user):\n"
            
            # Sort summaries chronologically by timestamp
            sorted_summaries = sorted(agent_summaries.items(), key=lambda x: x[1].get("timestamp", ""))
            for agent_key, data in sorted_summaries:
                content = data.get("content", "")
                ts = data.get("timestamp", "unknown time")
                status = data.get("status", "SUCCESS")
                team_manifest += f"- {agent_key} [{status}] (at {ts}): {content}\n"
            
        enhanced_prompt = f"{self.system_prompt}{team_manifest}"
        
        current_messages = state.get("messages", []) if isinstance(state, dict) else state.messages

        # Best Practice: Suffix with HumanMessage to force model generation
        full_messages = [SystemMessage(content=enhanced_prompt)] + current_messages
        if full_messages and isinstance(full_messages[-1], AIMessage):
            full_messages.append(
                HumanMessage(content="Analyze the team reports and user history above, then provide your next instruction or final answer.")
            )

        return full_messages

    async def ainvoke(
        self,
        state: CknotAgentState,
        config: RunnableConfig
    )-> Dict[str, Any]:
        """
        Asynchronous invocation for non-streaming responses.
        """
        return await super().ainvoke(state, config)

    async def astream(
        self, 
        state: CknotAgentState,
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

        llm_svc_client_with_tools = llm_svc_client.bind_tools(self.tools) if self.tools else llm_svc_client
        messages = self._get_messages(state)
        final_chunk = None

        # Generate a stable ID for this turn to ensure chunks merge correctly in LangGraph
        turn_id = str(uuid.uuid4())

        async for chunk in llm_svc_client_with_tools.astream(messages):
            chunk.id = turn_id
            # Accumulate chunks to compute final usage metadata
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk

            usage = getattr(chunk, "usage_metadata", None)
            total_tokens = usage.get("total_tokens", 0) if usage else 0
            # Only yield chunks with content or tool calls to the graph state.
            # Trailing empty metadata chunks can cause routing logic to fail if they 
            # are treated as the 'last message' in a non-merging scenario.
            if chunk.content or chunk.tool_calls:
                yield {
                    "messages": [chunk],
                    "progress_report": {
                        "cknot": {
                            "step": "TRIAGE",
                            "description": "cknot is thinking...",
                            "percentage": 0.0,
                            "total_tokens": total_tokens
                        }
                    }
                }

        # Extract usage from the aggregated message if available
        usage = getattr(final_chunk, "usage_metadata", None)
        
        # Log full interaction in JSON style for debugging
        log_msg = (
            f"LLM: {active_llm.model_name}\n"
            f"Request: {json.dumps([m.dict() for m in messages], indent=2, ensure_ascii=False, default=str)}\n"
            f"Response: {json.dumps(final_chunk.dict() if final_chunk else {}, indent=2, ensure_ascii=False, default=str)}"
        )
        
        self._log_metrics(start_time, usage=usage, mode="streaming invocation", message=log_msg)
