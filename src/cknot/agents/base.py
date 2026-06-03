import time
import random
import json
import logging
import re
from typing import List, Optional, Any, Dict, Union
import uuid
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from cknot.config.config import settings
from cknot.schemas.state import CknotAgentState
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy
from cknot.utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class CKnotBaseAgent(BaseModel):
    """
    Base class for all CKnot agents.
    Encapsulates system prompt management and standardized LLM invocation logic.
    """
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique technical identifier for the agent.")
    name: str = Field(default="")
    system_prompt: str = Field(default="")
    llm_services: List[LLMService] = Field(default_factory=list)
    llm_select_policy: LLMSelectPolicy = Field(default=LLMSelectPolicy.FIRST)
    expert_in: List[str] = Field(default_factory=list)
    avoid_for: List[str] = Field(default_factory=list)
    tools: List[Any] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def good_at(self) -> List[str]:
        """Alias for expert_in to maintain compatibility with CLI commands."""
        return self.expert_in

    @property
    def poor_at(self) -> List[str]:
        """Alias for avoid_for to maintain compatibility with CLI commands."""
        return self.avoid_for

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            self.name = self.__class__.__name__

    def _get_messages(self, state: CknotAgentState) -> List[BaseMessage]:
        """Prepares the message list with system prompt injection."""
        # Handle both list of messages or the state object
        current_messages = state.get("messages", []) if isinstance(state, dict) else state.messages

        if not current_messages:
            # If no messages are present, provide a default prompt for the LLM to ask the user for input.
            return [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content="I am ready to help. Please provide your instructions or ask a question.")
            ]

        # Filter out routing keywords (Boss delegation: "Agent [Name]") to prevent LLM confusion
        filtered_history = [
            m for m in current_messages 
            if not (isinstance(m, AIMessage) and re.search(r'Agent\s+\w+', m.content))
        ]

        # Best Practice: Ensure the conversation ends with a HumanMessage to prompt the LLM.
        # Local models (Ollama/Qwen) often return empty content if the history tails with an AIMessage.
        full_messages = [SystemMessage(content=self.system_prompt)] + filtered_history
        
        if full_messages and isinstance(full_messages[-1], AIMessage):
            full_messages.append(
                HumanMessage(content="Please provide your next response based on the conversation and instructions above.")
            )
            
        return full_messages

    def _log_metrics(self, start_time: float,
                     usage: Optional[Dict[str, Any]] = None,
                     mode: str = "invocation", message: Optional[str] = None):
        """Helper to log the time taken and token usage for a specific operation."""
        duration = time.perf_counter() - start_time
        usage_str = ""
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            usage_str = f" | Tokens: {total_tokens} (In: {input_tokens}, Out: {output_tokens})"

        logger.debug(
            f"Agent {self.name} ({self.agent_id[:8]}) {mode} took {duration:.4f} seconds.{usage_str}"
        )
        if message:
            logger.debug(message)

    def add_llm_service(self, service: LLMService):
        """Adds an LLM service to the agent's internal registry."""
        if not service:
            logger.debug(f'Invalid LLM service to add to agent {self.name} ({self.agent_id[:8]}).')
            return
        self.llm_services.append(service)
        logger.debug(f"LLM service '{service.id}' added to agent {self.name} ({self.agent_id[:8]}).")

    def remove_llm_service(self, service_id: str):
        """Removes an LLM service from the agent's internal registry."""
        initial_count = len(self.llm_services)
        self.llm_services = [s for s in self.llm_services if s.id != service_id]
        if len(self.llm_services) < initial_count:
            logger.debug(f"LLM service '{service_id}' removed from agent {self.name} ({self.agent_id[:8]}).")

    def _select_llm_service(self, state: CknotAgentState) -> Optional[LLMService]:
        """
        Selects an LLM service based on the configured policy.
        """
        enabled_services = [s for s in self.llm_services if s.is_enabled]
        if not enabled_services:
            if settings.USE_DEFAULT_LLM_FALLBACK:
                return LLMManager().get_llm_service(settings.DEFAULT_LLM_SERVICE)
            return None

        # Apply Policies
        if self.llm_select_policy == LLMSelectPolicy.RANDOM:
            return random.choice(enabled_services)

        if self.llm_select_policy == LLMSelectPolicy.WEIGHTED:
            # Attempt to extract 'weight' from extra fields, default to 1
            weights = [getattr(s, "weight", 1) for s in enabled_services]
            selected = random.choices(enabled_services, weights=weights, k=1)[0]
            return selected

        if self.llm_select_policy == LLMSelectPolicy.CAPABILITY:
            # Matches based on 'current_task' found in the AgentState (via config)
            # Or by checking tags in the LLMService metadata
            current_task = ""
            if isinstance(state, dict) and "current_task" in state:
                current_task = state.get("current_task", "").lower()
            
            for s in enabled_services:
                # Check if service ID or tags (if any) match the current task
                tags = getattr(s, "tags", [])
                if current_task in s.id.lower() or current_task in [t.lower() for t in tags]:
                    logger.debug(f"Capability match: {s.id} for task {current_task}")
                    return s

            # Fallback to keyword matching in the last message if no task match
            last_msg = state.get("messages", [])[-1].content.lower() if state.get("messages") else ""
            for s in enabled_services:
                if s.id.lower() in last_msg:
                     return s

        # Default: 'first' policy
        return enabled_services[0]

    async def ainvoke(
        self, 
        state: CknotAgentState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """
        Standard asynchronous invocation.
        Uses the service_id to retrieve an LLM from the registry, falling back to the provided llm instance.
        """
        start_time = time.perf_counter()

        # Resolve which LLM to use: registry first, then passed instance
        active_llm = self._select_llm_service(state)
        if not active_llm:
            raise ValueError(
                f"Agent '{self.name}' has no enabled LLM services. "
                "Please use '/llms' to check status or '/agents llm set' to assign one."
            )

        llm_manager = LLMManager()
        llm_svc_client = llm_manager.get_llm_service_client(active_llm.id)

        llm_svc_client_with_tools = llm_svc_client.bind_tools(self.tools) if self.tools else llm_svc_client
        messages = self._get_messages(state)

        # Generate a stable ID for logging consistency
        turn_id = str(uuid.uuid4())
        logger.debug(f"Agent {self.name} ({self.agent_id[:8]}) invoking LLM {active_llm.model_name} with turn_id: {turn_id}")
        response = await llm_svc_client_with_tools.ainvoke(messages)
        response.id = turn_id

        # Extract usage metadata if available (standard in newer LangChain versions)
        usage = getattr(response, "usage_metadata", None)

        # Prepare JSON-style representations for logging
        req_json = json.dumps([m.dict() for m in messages], indent=2, ensure_ascii=False, default=str)
        res_json = json.dumps(response.dict(), indent=2, ensure_ascii=False, default=str)

        self._log_metrics(
            start_time, 
            usage=usage, 
            mode="ainvoke",
            message=f'LLM: {active_llm.model_name}\nRequest: {req_json}\nResponse: {res_json}'
        )

        # Ensure we return a dictionary to update the graph state
        total_tokens = usage.get("total_tokens", 0) if usage else 0
        return {
            "messages": [response],
            "agent_data": {
                self.name: {
                    "raw_output": response.content,
                    "last_run": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            },
            "progress_report": {
                self.name: {
                    "step": "WORKING",
                    "description": f"{self.name} is working...",
                    "status": "done",
                    "percentage": 100.0,
                    "total_tokens": total_tokens
                }
            }
        }