import logging
import inspect
from typing import Dict, Type, Any, List, Optional, Union
from cknot.agents.base import CKnotBaseAgent
from cknot.schemas.llm_service import LLMService, LLMSelectPolicy

logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    A centralized registry for managing CKnot agents.
    This allows for dynamic agent discovery and configuration.
    """
    _agents: Dict[str, CKnotBaseAgent] = {}

    @classmethod
    def register_agent(cls, agent: CKnotBaseAgent):
        """Registers an agent."""
        if agent.name in cls._agents:
            logger.warning(f"Agent '{agent.name}' already registered. Overwriting.")
        cls._agents[agent.name] = agent
        logger.info(f"Agent '{agent.name}' registered.")

    @classmethod
    def unregister_agent(cls, name: str):
        """Removes an agent class and its cached instance from the registry."""
        if name in cls._agents:
            del cls._agents[name]
            logger.info(f"Agent '{name}' unregistered.")
        else:
            logger.warning(f"Attempted to unregister non-existent agent '{name}'.")

    @classmethod
    def get_agent(cls, name: str) -> Optional[CKnotBaseAgent]:
        """Get agent by name."""
        return cls._agents.get(name, None)

    @classmethod
    def list_agents(cls) -> Dict[str, CKnotBaseAgent]:
        """Returns a list of all registered agent."""
        return cls._agents

    @classmethod
    def get_agent_status(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Returns the status and capabilities of a specific agent.
        """
        agent = cls.get_agent(name)
        if not agent:
            return None

        return {
            "name": name,
            "class": agent.__class__.__name__,
            "system_prompt": agent.system_prompt,
            "good_at": agent.good_at,
            "poor_at": agent.poor_at,
            "llm_select_policy": agent.llm_select_policy.value,
            "llm_services": [s.id for s in agent.llm_services],
            "tools": [getattr(t, "name", str(t)) for t in agent.tools]
        }

    @classmethod
    def get_all_details(cls) -> List[Dict[str, Any]]:
        """Returns full status and metadata for all registered agents."""
        details = []
        for name in cls.list_agents():
            status = cls.get_agent_status(name)
            if status:
                details.append(status)
        return details

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