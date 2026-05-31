import time
import random
import logging
from typing import List, Optional, Any, Dict, Union
import uuid
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from cknot.schemas.state import AgentState
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy
from cknot.utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class CKnotBaseAgent(BaseModel):
    """
    Base class for all CKnot agents.
    Encapsulates system prompt management and standardized LLM invocation logic.
    """
    system_prompt: str
    llm_services: List[LLMService] = Field(default_factory=list)
    llm_select_policy: LLMSelectPolicy = Field(default=LLMSelectPolicy.FIRST)
    good_at: List[str] = Field(default_factory=list)
    poor_at: List[str] = Field(default_factory=list)
    tools: List[Any] = Field(default_factory=list)
    name: str = Field(default="")
    
    _uuid: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            self.name = self.__class__.__name__

    def _get_messages(self, state: AgentState) -> List[BaseMessage]:
        """Prepares the message list with system prompt injection."""
        # Handle both list of messages or the state object
        current_messages = state.get("messages", []) if isinstance(state, dict) else state.messages

        if not current_messages:
            # If no messages are present, provide a default prompt for the LLM to ask the user for input.
            return [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content="I am ready to help. Please provide your instructions or ask a question.")
            ]

        # Use the last user input
        last_user_msg = next(
            (m for m in reversed(current_messages) if isinstance(m, HumanMessage)),
            HumanMessage(content="I am ready to help. Please provide your instructions or ask a question."))
        return [SystemMessage(content=self.system_prompt), last_user_msg]

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
            f"Agent {self.name} ({self._uuid[:8]}) {mode} took {duration:.4f} seconds.{usage_str}"
        )
        if message:
            logger.debug(message)

    def add_llm_service(self, service: LLMService):
        """Adds an LLM service to the agent's internal registry."""
        if not service:
            logger.debug(f'Invalid LLM service to add to agent {self.name} ({self._uuid[:8]}).')
            return
        self.llm_services.append(service)
        logger.debug(f"LLM service '{service.id}' added to agent {self.name} ({self._uuid[:8]}).")

    def remove_llm_service(self, service_id: str):
        """Removes an LLM service from the agent's internal registry."""
        initial_count = len(self.llm_services)
        self.llm_services = [s for s in self.llm_services if s.id != service_id]
        if len(self.llm_services) < initial_count:
            logger.debug(f"LLM service '{service_id}' removed from agent {self.name} ({self._uuid[:8]}).")

    def _select_llm_service(self, state: AgentState) -> Optional[LLMService]:
        """
        Selects an LLM service based on the configured policy.
        """
        if not self.llm_services:
            return None

        enabled_services = [s for s in self.llm_services if s.is_enabled]
        if not enabled_services:
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
        state: AgentState,
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

        llm_svc_clinet_with_tools = llm_svc_client.bind_tools(self.tools) if self.tools else llm_svc_client
        messages = self._get_messages(state)

        # Generate a stable ID for logging consistency
        turn_id = str(uuid.uuid4())
        logger.debug(f"Agent {self.name} ({self._uuid[:8]}) invoking LLM {active_llm.model_name} with turn_id: {turn_id}")
        response = await llm_svc_clinet_with_tools.ainvoke(messages)
        response.id = turn_id

        # Extract usage metadata if available (standard in newer LangChain versions)
        usage = getattr(response, "usage_metadata", None)
        self._log_metrics(
            start_time, 
            usage=usage, 
            mode="ainvoke",
            message=f'LLM: {active_llm.model_name}\nRequest: {messages}\nResponse: {response}'
        )

        # Ensure we return a dictionary to update the graph state
        return {
            "messages": [response]
        }