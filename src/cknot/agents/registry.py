import logging
import inspect
import importlib.util
import os
import sys
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
            "expert_in": agent.expert_in,
            "avoid_for": agent.avoid_for,
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
        Returns a mapping of registered agent names to their 'expert_in' and 'avoid_for' capabilities.
        """
        capabilities = {}
        for name, agent in cls._agents.items():
            try:
                capabilities[name] = {
                    "expert_in": agent.expert_in,
                    "avoid_for": agent.avoid_for
                }
            except Exception as e:
                logger.debug(f"Failed to retrieve capabilities for {name}: {e}")
                capabilities[name] = {"expert_in": [], "avoid_for": []}
        return capabilities

    @classmethod
    def load_custom_agents(cls, directory: str):
        """Dynamically loads agents from a specific directory."""
        if not os.path.exists(directory):
            logger.warning(f"Plugin directory {directory} does not exist.")
            return

        for filename in os.listdir(directory):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                file_path = os.path.join(directory, filename)
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        
                        # Search for CKnotBaseAgent subclasses in the module
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, CKnotBaseAgent) and 
                                obj is not CKnotBaseAgent):
                                
                                # Instantiate and register
                                try:
                                    agent_instance = obj()
                                    cls.register_agent(agent_instance)
                                except Exception as e:
                                    logger.error(f"Failed to instantiate agent {name}: {e}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {filename}: {e}")