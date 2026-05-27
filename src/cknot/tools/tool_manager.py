import logging
from functools import wraps
from typing import List, Optional, Dict, Any, Callable, Union
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from cknot.schemas.tool_config import ToolConfig
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Manages tool configurations, persistence in Redis, and integration with the graph.
    """
    _instances: dict[str, 'ToolManager'] = {}

    def __new__(cls, redis_client: Union[Redis, AsyncRedis]):
        # Maintain separate singletons for sync and async clients to avoid IO conflicts
        client_type = "async" if isinstance(redis_client, AsyncRedis) else "sync"
        if client_type not in cls._instances:
            cls._instances[client_type] = super(ToolManager, cls).__new__(cls)
        return cls._instances[client_type]

    def __init__(self, redis_client: Union[Redis, AsyncRedis]):
        if getattr(self, "_initialized", False):
            return
        self._redis = redis_client
        self._prefix = "tool_manager:"
        # Registry to map tool IDs to actual LangChain tool objects
        self._registry: Dict[str, BaseTool] = {}
        # Detect if we are using an asynchronous client
        self._is_async = isinstance(redis_client, AsyncRedis)
        self._initialized = True

    def _wrap_tool_with_usage_tracking(self, tool_id: str, tool: BaseTool):
        """Wraps tool execution to automatically increment usage count in Redis."""
        original_run = tool._run
        original_arun = tool._arun

        @wraps(original_run)
        def wrapped_run(*args, **kwargs):
            if not self._is_async:
                self.increment_usage(tool_id)
            # Usage tracking via sync client is skipped if manager is in async mode
            # to avoid blocking tool execution threads.
            return original_run(*args, **kwargs)

        @wraps(original_arun)
        async def wrapped_arun(*args, **kwargs):
            if self._is_async:
                await self.aincrement_usage(tool_id)
            else:
                self.increment_usage(tool_id)
            return await original_arun(*args, **kwargs)

        # Replace the instance methods with the wrapped versions
        tool._run = wrapped_run
        tool._arun = wrapped_arun

    def register_tool_instance(self, tool_id: str, tool_instance: BaseTool):
        """Registers a live tool instance with an ID and syncs initial config to Redis."""
        # Automatically wrap the tool to track usage whenever it's called
        self._wrap_tool_with_usage_tracking(tool_id, tool_instance)

        self._registry[tool_id] = tool_instance
        
        # Initial sync to Redis (expects sync client during graph creation)
        if not self._is_async:
            if not self.get_tool_config(tool_id):
                config = ToolConfig(
                    id=tool_id,
                    name=tool_instance.name,
                    description=tool_instance.description
                )
                self.save_tool_config(config)
        else:
            logger.debug(f"Async ToolManager: Skipping initial sync for tool '{tool_id}'.")

    def save_tool_config(self, config: ToolConfig):
        """Saves tool configuration to Redis."""
        if self._is_async:
            raise RuntimeError("Cannot use sync save_tool_config with an async client. Use asave_tool_config.")
        key = f"{self._prefix}{config.id}"
        # Convert bool to string for Redis compatibility
        data = {
            k: (str(v) if isinstance(v, bool) else v)
            for k, v in config.model_dump().items()
        }
        self._redis.hset(key, mapping=data)

    def get_tool_config(self, tool_id: str) -> Optional[ToolConfig]:
        """Retrieves tool configuration from Redis."""
        if self._is_async:
            raise RuntimeError("Cannot use sync get_tool_config with an async client. Use aget_tool_config.")
        key = f"{self._prefix}{tool_id}"
        data = self._redis.hgetall(key)
        if not data:
            return None
        
        return self._parse_config_data(data)

    def list_tool_configs(self) -> List[ToolConfig]:
        """Lists all tool configurations stored in Redis."""
        if self._is_async:
            raise RuntimeError("Cannot use sync list_tool_configs with an async client. Use alist_tool_configs.")
        configs = []
        for key in self._redis.scan_iter(f"{self._prefix}*"):
            tool_id = key.replace(self._prefix, "")
            cfg = self.get_tool_config(tool_id)
            if cfg:
                configs.append(cfg)
        return configs

    def get_runnable_tools(self) -> List[BaseTool]:
        """Returns the list of tool instances that are currently enabled in configuration."""
        if self._is_async:
            logger.warning("get_runnable_tools called in async mode. This might fail if get_tool_config is sync.")

        runnable = []
        for tool_id, instance in self._registry.items():
            config = self.get_tool_config(tool_id)
            if config and config.is_enabled:
                runnable.append(instance)
        return runnable

    def increment_usage(self, tool_id: str):
        """Increments the call counter for a tool in Redis."""
        if self._is_async:
            return
        key = f"{self._prefix}{tool_id}"
        if self._redis.exists(key):
            self._redis.hincrby(key, "usage_count", 1)

    def delete_tool_config(self, tool_id: str):
        """Removes tool configuration from Redis."""
        if self._is_async:
            raise RuntimeError("Cannot use sync delete_tool_config with an async client. Use adelete_tool_config.")
        key = f"{self._prefix}{tool_id}"
        self._redis.delete(key)
        logger.info(f"Tool config for '{tool_id}' deleted from Redis.")

    # --- Asynchronous counterparts ---

    async def asave_tool_config(self, config: ToolConfig):
        """Saves tool configuration to Redis (Asynchronous)."""
        if not self._is_async:
            self.save_tool_config(config)
            return
        key = f"{self._prefix}{config.id}"
        data = {
            k: (str(v) if isinstance(v, bool) else v)
            for k, v in config.model_dump().items()
        }
        await self._redis.hset(key, mapping=data)

    async def aget_tool_config(self, tool_id: str) -> Optional[ToolConfig]:
        """Retrieves tool configuration from Redis (Asynchronous)."""
        if not self._is_async:
            return self.get_tool_config(tool_id)
        key = f"{self._prefix}{tool_id}"
        data = await self._redis.hgetall(key)
        if not data:
            return None
        return self._parse_config_data(data)

    async def alist_tool_configs(self) -> List[ToolConfig]:
        """Lists all tool configurations stored in Redis (Asynchronous)."""
        if not self._is_async:
            return self.list_tool_configs()
        configs = []
        async for key in self._redis.scan_iter(f"{self._prefix}*"):
            tool_id = key.replace(self._prefix, "")
            cfg = await self.aget_tool_config(tool_id)
            if cfg:
                configs.append(cfg)
        return configs

    async def aincrement_usage(self, tool_id: str):
        """Increments the call counter for a tool in Redis (Asynchronous)."""
        if not self._is_async:
            self.increment_usage(tool_id)
            return
        key = f"{self._prefix}{tool_id}"
        if await self._redis.exists(key):
            await self._redis.hincrby(key, "usage_count", 1)

    def _parse_config_data(self, data: dict) -> ToolConfig:
        """Internal helper to parse Redis hash data into a ToolConfig object."""
        # Basic type conversion for Pydantic (Redis hgetall returns strings with decode_responses=True)
        if "is_enabled" in data:
            data["is_enabled"] = str(data["is_enabled"]).lower() == "true"
        if "usage_count" in data:
            data["usage_count"] = int(data["usage_count"])
            
        return ToolConfig(**data)