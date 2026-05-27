import logging
import inspect
from typing import Dict, Type, Any, List, Optional, Union
from cknot.agents.base import CKnotBaseAgent
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy

logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    A centralized registry for managing and instantiating CKnot agents.
    This allows for dynamic agent discovery and configuration.
    """
    _agents: Dict[str, Type[CKnotBaseAgent]] = {}
    _instances: Dict[str, CKnotBaseAgent] = {}

    @classmethod
    def register_agent(cls, name: str, agent_class: Type[CKnotBaseAgent]):
        """Registers an agent class with a given name."""
        if name in cls._agents:
            logger.warning(f"Agent '{name}' already registered. Overwriting.")
        cls._agents[name] = agent_class
        logger.info(f"Agent '{name}' ({agent_class.__name__}) registered.")

    @classmethod
    def get_agent_instance(
        cls, 
        name: str, 
        llm_services: Optional[List[LLMService]] = None,
        llm_select_policy: Union[LLMSelectPolicy, str] = LLMSelectPolicy.FIRST,
        **kwargs: Any
    ) -> CKnotBaseAgent:
        """Instantiates and returns an agent from the registry."""
        agent_class = cls._agents.get(name)
        if not agent_class:
            raise ValueError(f"Agent '{name}' not found in registry.")
        
        # Pass common configurations and any additional kwargs
        return agent_class(llm_services=llm_services, llm_select_policy=llm_select_policy, **kwargs)

    @classmethod
    def get_all_capabilities(cls) -> Dict[str, Dict[str, List[str]]]:
        """
        Returns a mapping of registered agent names to their 'good_at' and 'poor_at' capabilities.
        """
        capabilities = {}
        for name, agent_class in cls._agents.items():
            try:
                # We attempt a lightweight instantiation to access the metadata attributes.
                # Since some agents (like CKnotBossAgent) require 'tools', we inspect the 
                # signature to provide dummy arguments where necessary.
                sig = inspect.signature(agent_class.__init__)
                kwargs = {"tools": []} if "tools" in sig.parameters else {}
                
                dummy = agent_class(**kwargs)
                capabilities[name] = {
                    "good_at": getattr(dummy, "good_at", []),
                    "poor_at": getattr(dummy, "poor_at", [])
                }
            except Exception as e:
                logger.debug(f"Failed to retrieve capabilities for {name}: {e}")
                capabilities[name] = {"good_at": [], "poor_at": []}
        return capabilities